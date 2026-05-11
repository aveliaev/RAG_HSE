"""
pipeline_comparison.py — Сравнение конфигураций пайплайна (накопительное)
=========================================================================

Тестируемые конфигурации (каждая добавляет компонент к предыдущей):
  1. Base RAG          — e5-base + ChromaDB, без доп. компонентов
  2. + Reranker        — добавляет cross-encoder переранжирование
  3. + Regex Rewrite   — добавляет нормализацию сленга (regex)
  4. + LLM Decomp.     — добавляет LLM-переформулирование и мульти-запросы
  5. + FAQ Cache       — добавляет трёхуровневый кеш перед пайплайном

Метрики:
  Retrieval : Hit@1, Hit@3, Hit@5, MRR@5, Precision@3, Context Recall
  Latency   : mean, median, p95 (в мс)
  Cache     : hit rate, latency при попадании / промахе

Запуск:
  python3 pipeline_comparison.py
  python3 pipeline_comparison.py --questions 50         # быстро, для отладки
  python3 pipeline_comparison.py --with-llm             # включить конф. 4
  python3 pipeline_comparison.py --output results/      # куда сохранять файлы
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import sys, os, re, json, time, argparse, logging, statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

# ─── пути ────────────────────────────────────────────────────────────────────
BOT_DIR     = Path(__file__).parent          # .../bot/
ROOT        = BOT_DIR.parent                 # .../ДЛЯ КЛОДА 2/
DATASET_DIR = ROOT / "dataset"
TEST_DIR    = ROOT / "test"
DEFAULT_QA  = TEST_DIR / "fkn_qa_test_all.md"

sys.path.insert(0, str(BOT_DIR))
os.chdir(BOT_DIR)

# ─── .env ────────────────────────────────────────────────────────────────────
for _ef in (BOT_DIR / ".env", ROOT / ".env"):
    if _ef.exists():
        for _l in _ef.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                k, _, v = _l.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

logging.basicConfig(level=logging.WARNING)

# ─── импорты из проекта ───────────────────────────────────────────────────────
from rag_engine import (
    _split_into_chunks, _rewrite_query, _rewrite_query_llm, _rerank,
)
from faq_cache import lookup as faq_lookup, _normalize as faq_normalize
from eval_rag import (
    parse_qa, tokenize,
    hit_at_k, reciprocal_rank, precision_at_k, context_recall,
)

# ─── утилита: симуляция прогретого кеша ───────────────────────────────────────
def warm_cache(qa_pairs: list[dict], fraction: float = 0.35) -> dict[str, str]:
    """
    Симуляция динамического кеша: добавляем `fraction` случайных вопросов из датасета
    как «уже подтверждённые пользователями ответы».

    В реальной системе кеш копится постепенно из лайкнутых ответов.
    Здесь мы имитируем состояние системы после N первых пользователей.

    НЕ трогает файл dynamic_cache.json.
    """
    import random
    rng = random.Random(42)   # фиксируем seed для воспроизводимости
    n_warm = int(len(qa_pairs) * fraction)
    sampled = rng.sample(qa_pairs, n_warm)
    warm: dict[str, str] = {}
    for item in sampled:
        norm = faq_normalize(item["question"])
        if norm:
            warm[norm] = item["answer"]
    return warm


def faq_lookup_with_warm(question: str, warm: dict[str, str]) -> tuple[str | None, str]:
    """
    Расширенный lookup: сначала статический FAQ, потом прогретый warm-кеш.
    Используется только в конфигурации 5.
    """
    answer, src = faq_lookup(question)
    if answer is not None:
        return answer, src
    norm = faq_normalize(question)
    if norm in warm:
        return warm[norm], "warm-cache"
    return None, "rag"

# ─── описание конфигураций ───────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    id:               int
    name:             str          # короткое имя для графика
    label:            str          # полная подпись
    embed_model:      str = "intfloat/multilingual-e5-base"
    use_prefix:       bool = True   # passage: / query: для E5
    enable_reranker:  bool = False
    enable_regex:     bool = False  # детерминированный slang-rewrite
    enable_llm:       bool = False  # LLM rewrite + decomposition (требует API)
    enable_faq:       bool = False  # FAQ-кеш как первый фильтр
    top_k:            int  = 5
    reranker_model:   str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


CONFIGS: list[PipelineConfig] = [
    PipelineConfig(
        id=1, name="Base RAG",
        label="1. Base RAG\n(e5-base)",
        enable_regex=False, enable_reranker=False,
    ),
    PipelineConfig(
        id=2, name="+ Reranker",
        label="2. + Reranker\n(cross-encoder)",
        enable_regex=False, enable_reranker=True,
    ),
    PipelineConfig(
        id=3, name="+ Regex Rewrite",
        label="3. + Regex\nRewrite",
        enable_regex=True, enable_reranker=True,
    ),
    PipelineConfig(
        id=4, name="+ LLM Decomp.",
        label="4. + LLM\nDecomp.",
        enable_regex=True, enable_reranker=True, enable_llm=True,
    ),
    PipelineConfig(
        id=5, name="+ FAQ Cache",
        label="5. + FAQ\nCache",
        enable_regex=True, enable_reranker=True, enable_faq=True,
    ),
]

# ─── загрузка документов и индекса ───────────────────────────────────────────
_DOCS_SKIP = ("qa_test", "_test", "test_")

def load_docs() -> dict[str, str]:
    docs = {}
    for p in sorted(DATASET_DIR.glob("*.md")):
        if any(pat in p.stem.lower() for pat in _DOCS_SKIP):
            continue
        docs[p.stem] = p.read_text(encoding="utf-8")
    return docs


def build_shared_index(embed_model: str, use_prefix: bool):
    """Строит один эфемерный индекс, используемый всеми конфигурациями."""
    import chromadb
    from chromadb.utils import embedding_functions

    print(f"\n  Загружаю модель эмбеддингов: {embed_model} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model, device="cpu",
    )
    client     = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="pipeline_cmp",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs   = load_docs()
    chunks: list[dict] = []
    for src, text in docs.items():
        chunks.extend(_split_into_chunks(text, src))

    prefix = "passage: " if use_prefix else ""
    collection.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=[prefix + c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks],
    )
    print(f"готово ({len(chunks)} чанков, {time.perf_counter()-t0:.1f}с)")
    return collection, len(chunks)


def load_reranker(model: str):
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Загружаю реранкер: {model} ...", end=" ", flush=True)
        r = CrossEncoder(model, device="cpu")
        print("готово")
        return r
    except Exception as e:
        print(f"\n  ⚠  Реранкер недоступен: {e}")
        return None

# ─── поиск для одного вопроса ─────────────────────────────────────────────────
def retrieve_one(
    question: str,
    collection,
    cfg: PipelineConfig,
    reranker,
    query_prefix: str,
    doc_prefix:   str,
) -> tuple[list[dict], float]:
    """
    Возвращает (chunks, retrieval_latency_ms).
    Latency включает rewrite + vector search + rerank.
    НЕ включает LLM-вызов и FAQ-кеш (они измеряются отдельно).
    """
    t0 = time.perf_counter()

    rewritten = _rewrite_query(question) if cfg.enable_regex else question

    n_cand = max(cfg.top_k * 2, 10)
    results = collection.query(
        query_texts=[query_prefix + rewritten],
        n_results=n_cand,
    )

    candidates = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        clean = doc[len(doc_prefix):] if doc.startswith(doc_prefix) else doc
        candidates.append({
            "text":    clean,
            "source":  meta["source"],
            "heading": meta["heading"],
        })

    if cfg.enable_reranker and reranker is not None:
        chunks = _rerank(rewritten, candidates, top_k=cfg.top_k)
    else:
        chunks = candidates[:cfg.top_k]

    latency_ms = (time.perf_counter() - t0) * 1000
    return chunks, latency_ms

# ─── прогон одной конфигурации ────────────────────────────────────────────────
@dataclass
class ConfigResult:
    cfg:          PipelineConfig
    n:            int
    hit1:         float
    hit3:         float
    hit5:         float
    mrr5:         float
    p3:           float
    ctx_recall:   float
    lat_mean:     float   # мс, среднее
    lat_median:   float
    lat_p95:      float
    # только для FAQ cache
    cache_hit_rate:  float = 0.0
    lat_cache_hit:   float = 0.0   # среднее для попаданий
    lat_cache_miss:  float = 0.0   # среднее для промахов
    n_cache_hits:    int   = 0
    raw_latencies:   list  = field(default_factory=list)


def run_config(
    cfg:        PipelineConfig,
    qa_pairs:   list[dict],
    collection,
    reranker,
    with_llm:   bool = False,
    verbose:    bool = False,
    warm:       dict | None = None,   # прогретый кеш для конфиг. 5
) -> Optional[ConfigResult]:

    if cfg.enable_llm and not with_llm:
        return None

    print(f"\n{'─'*60}")
    print(f"  Конфигурация {cfg.id}: {cfg.name}")

    query_prefix = "query: "   if cfg.use_prefix else ""
    doc_prefix   = "passage: " if cfg.use_prefix else ""

    h1_l, h3_l, h5_l, rr_l, p3_l, cr_l = [], [], [], [], [], []
    lat_all: list[float] = []
    lat_cache_hits:  list[float] = []
    lat_cache_miss:  list[float] = []
    n_cache_hits = 0

    for i, item in enumerate(qa_pairs, 1):
        q         = item["question"]
        ref_tok   = tokenize(item["answer"])

        # ── FAQ-кеш (только конфиг 5) ────────────────────────────────────────
        if cfg.enable_faq:
            t_faq = time.perf_counter()
            cached, src = faq_lookup_with_warm(q, warm or {})
            faq_ms = (time.perf_counter() - t_faq) * 1000

            if cached is not None:
                n_cache_hits += 1
                lat_all.append(faq_ms)
                lat_cache_hits.append(faq_ms)
                # для кеш-попаданий retrieval-метрики пропускаем
                # (кеш возвращает текстовый ответ, не чанки)
                h1_l.append(False)
                h3_l.append(False)
                h5_l.append(False)
                rr_l.append(0.0)
                p3_l.append(0.0)
                # Context Recall по тексту ответа кеша
                cr_l.append(context_recall(ref_tok, cached))
                continue

        # ── LLM rewrite / decomp ─────────────────────────────────────────────
        if cfg.enable_llm:
            t_llm = time.perf_counter()
            q_rewritten_llm = _rewrite_query_llm(
                _rewrite_query(q) if cfg.enable_regex else q
            )
            llm_ms = (time.perf_counter() - t_llm) * 1000
        else:
            llm_ms = 0.0
            q_rewritten_llm = q

        # для retrieval используем LLM-переписанный (или оригинальный) запрос
        q_for_retrieval = q_rewritten_llm if cfg.enable_llm else q

        # ── retrieval + rerank ────────────────────────────────────────────────
        chunks, ret_ms = retrieve_one(
            q_for_retrieval, collection, cfg, reranker, query_prefix, doc_prefix,
        )
        total_ms = ret_ms + llm_ms

        lat_all.append(total_ms)
        if cfg.enable_faq:
            lat_cache_miss.append(total_ms)

        h1_l.append(hit_at_k(chunks, ref_tok, k=1))
        h3_l.append(hit_at_k(chunks, ref_tok, k=3))
        h5_l.append(hit_at_k(chunks, ref_tok, k=5))
        rr_l.append(reciprocal_rank(chunks, ref_tok))
        p3_l.append(precision_at_k(chunks, ref_tok, k=3))
        cr_l.append(context_recall(ref_tok, "\n".join(c["text"] for c in chunks)))

        if verbose and i % 50 == 0:
            print(f"    {i}/{len(qa_pairs)} вопросов обработано...", flush=True)

    n = len(qa_pairs)
    lats = sorted(lat_all)

    result = ConfigResult(
        cfg       = cfg,
        n         = n,
        hit1      = round(sum(h1_l) / n, 3),
        hit3      = round(sum(h3_l) / n, 3),
        hit5      = round(sum(h5_l) / n, 3),
        mrr5      = round(sum(rr_l) / n, 3),
        p3        = round(sum(p3_l) / n, 3),
        ctx_recall= round(sum(cr_l) / n, 3),
        lat_mean  = round(statistics.mean(lats), 1),
        lat_median= round(statistics.median(lats), 1),
        lat_p95   = round(lats[int(len(lats) * 0.95)], 1) if lats else 0.0,
        raw_latencies = lat_all,
    )

    if cfg.enable_faq:
        result.cache_hit_rate  = round(n_cache_hits / n, 3)
        result.n_cache_hits    = n_cache_hits
        result.lat_cache_hit   = round(statistics.mean(lat_cache_hits), 2)  if lat_cache_hits  else 0.0
        result.lat_cache_miss  = round(statistics.mean(lat_cache_miss), 1)  if lat_cache_miss  else 0.0

    print(
        f"  → Hit@1={result.hit1:.3f}  Hit@3={result.hit3:.3f}  "
        f"Hit@5={result.hit5:.3f}  MRR@5={result.mrr5:.3f}\n"
        f"     P@3={result.p3:.3f}    CtxRec={result.ctx_recall:.3f}  "
        f"Lat mean={result.lat_mean:.0f}ms  p95={result.lat_p95:.0f}ms"
    )
    if cfg.enable_faq and n_cache_hits:
        print(
            f"     Cache hits: {n_cache_hits}/{n} ({result.cache_hit_rate*100:.1f}%)  "
            f"hit={result.lat_cache_hit:.1f}ms  miss={result.lat_cache_miss:.0f}ms"
        )
    return result

# ─── текстовая таблица ────────────────────────────────────────────────────────
def print_table(results: list[ConfigResult]) -> None:
    COL_W = 20
    COLS  = ["Hit@1","Hit@3","Hit@5","MRR@5","P@3","Ctx Rec","Lat mean","Lat p95"]

    def best(key: str) -> float:
        vals = [getattr(r, key) for r in results]
        return max(vals)

    bests = {
        "hit1": best("hit1"), "hit3": best("hit3"), "hit5": best("hit5"),
        "mrr5": best("mrr5"), "p3":   best("p3"),   "ctx_recall": best("ctx_recall"),
        "lat_mean": min(r.lat_mean for r in results),   # для latency меньше = лучше
        "lat_p95":  min(r.lat_p95  for r in results),
    }

    header = f"{'Конфигурация':{COL_W}}" + "".join(f"{c:^10}" for c in COLS)
    sep    = "─" * len(header)

    print(f"\n{'='*len(header)}")
    print("  СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in results:
        def f(val: float, key: str, low_better=False) -> str:
            is_best = (val == bests[key])
            s = f"{val:.3f}"
            return f"*{s}*" if is_best else f" {s} "
        def fl(val: float, key: str) -> str:
            is_best = (val == bests[key])
            s = f"{val:.0f}ms"
            return f"*{s}*" if is_best else f" {s} "

        row = (
            f"{r.cfg.name:{COL_W}}"
            f"{f(r.hit1,'hit1'):^10}"
            f"{f(r.hit3,'hit3'):^10}"
            f"{f(r.hit5,'hit5'):^10}"
            f"{f(r.mrr5,'mrr5'):^10}"
            f"{f(r.p3,'p3'):^10}"
            f"{f(r.ctx_recall,'ctx_recall'):^10}"
            f"{fl(r.lat_mean,'lat_mean'):^10}"
            f"{fl(r.lat_p95,'lat_p95'):^10}"
        )
        print(row)

        if r.cfg.enable_faq and r.n_cache_hits:
            print(
                f"  {'↳ cache':>{COL_W-2}}"
                f"  hit rate={r.cache_hit_rate*100:.1f}%"
                f"  hit={r.lat_cache_hit:.1f}ms"
                f"  miss={r.lat_cache_miss:.0f}ms"
            )

    print(sep)
    print(f"  (* = лучшее значение в столбце, n={results[0].n} вопросов)\n")

# ─── dashboard (matplotlib) ──────────────────────────────────────────────────
def build_dashboard(results: list[ConfigResult], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  ⚠  matplotlib не установлен, пропускаю dashboard")
        return

    PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
    names   = [r.cfg.name for r in results]
    N       = len(results)
    x       = np.arange(N)
    W       = 0.15   # ширина столбца

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Сравнение конфигураций RAG-пайплайна", fontsize=16, fontweight="bold", y=0.98)

    # ── 1. Hit@K ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 2, 1)
    for i, (metric, label) in enumerate([("hit1","Hit@1"),("hit3","Hit@3"),("hit5","Hit@5")]):
        vals = [getattr(r, metric) for r in results]
        bars = ax1.bar(x + i*W, vals, W, label=label, color=PALETTE[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x + W)
    ax1.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Доля запросов")
    ax1.set_title("Hit Rate@K", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. MRR@5 и P@3 ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 2)
    for i, (metric, label) in enumerate([("mrr5","MRR@5"),("p3","Precision@3")]):
        vals = [getattr(r, metric) for r in results]
        bars = ax2.bar(x + i*W*1.5, vals, W*1.4, label=label, color=PALETTE[i+3], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x + W*0.75)
    ax2.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Значение")
    ax2.set_title("MRR@5 и Precision@3", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. Context Recall ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(3, 2, 3)
    vals = [r.ctx_recall for r in results]
    bars = ax3.bar(names, vals, color=PALETTE, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Доля токенов эталона в контексте")
    ax3.set_title("Context Recall", fontweight="bold")
    ax3.tick_params(axis="x", labelsize=8, rotation=15)
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Latency (mean + p95) ──────────────────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 4)
    lat_mean = [r.lat_mean for r in results]
    lat_p95  = [r.lat_p95  for r in results]
    bars_m = ax4.bar(x - W/2, lat_mean, W*0.9, label="mean", color="#4C72B0", alpha=0.85)
    bars_p = ax4.bar(x + W/2, lat_p95,  W*0.9, label="p95",  color="#C44E52", alpha=0.85)
    for bar, v in zip(bars_m, lat_mean):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars_p, lat_p95):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax4.set_ylabel("Задержка (мс)")
    ax4.set_title("Latency retrieval (мс)", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. Latency distribution (box plot) ───────────────────────────────────
    ax5 = fig.add_subplot(3, 2, 5)
    bp = ax5.boxplot(
        [r.raw_latencies for r in results],
        labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel("Задержка (мс)")
    ax5.set_title("Распределение задержек (box plot, без выбросов)", fontweight="bold")
    ax5.tick_params(axis="x", labelsize=8, rotation=15)
    ax5.grid(axis="y", alpha=0.3)

    # ── 6. FAQ Cache — круговая + bar latency ────────────────────────────────
    ax6 = fig.add_subplot(3, 2, 6)
    faq_res = next((r for r in results if r.cfg.enable_faq), None)
    if faq_res and faq_res.n_cache_hits > 0:
        n_hits   = faq_res.n_cache_hits
        n_misses = faq_res.n - n_hits
        wedges, texts, autotexts = ax6.pie(
            [n_hits, n_misses],
            labels=[f"Cache hit\n({n_hits})", f"Cache miss\n({n_misses})"],
            autopct="%1.1f%%",
            colors=["#55A868","#C44E52"],
            startangle=90,
            wedgeprops=dict(alpha=0.8),
        )
        for at in autotexts:
            at.set_fontsize(9)

        # Добавляем latency сравнение как текст
        ax6.set_title(
            f"FAQ Cache (конф. 5)\n"
            f"hit={faq_res.lat_cache_hit:.1f}мс  "
            f"miss={faq_res.lat_cache_miss:.0f}мс  "
            f"mean={faq_res.lat_mean:.0f}мс",
            fontweight="bold", fontsize=9,
        )
    else:
        ax6.set_title("FAQ Cache — данных нет", fontweight="bold")
        ax6.text(0.5, 0.5, "Запустите с конфигурацией 5\nили нет совпадений в кеше",
                 ha="center", va="center", fontsize=10, transform=ax6.transAxes)
        ax6.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard сохранён: {out_path}")

# ─── сохранение результатов ───────────────────────────────────────────────────
def save_results(results: list[ConfigResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    data = []
    for r in results:
        d = {k: v for k, v in r.__dict__.items() if k not in ("cfg", "raw_latencies")}
        d["config_id"]   = r.cfg.id
        d["config_name"] = r.cfg.name
        data.append(d)
    json_path = out_dir / f"pipeline_comparison_{ts}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  JSON сохранён: {json_path}")

    # CSV
    import csv
    csv_path = out_dir / f"pipeline_comparison_{ts}.csv"
    fields = ["config_id","config_name","n","hit1","hit3","hit5","mrr5","p3",
              "ctx_recall","lat_mean","lat_median","lat_p95",
              "cache_hit_rate","n_cache_hits","lat_cache_hit","lat_cache_miss"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    print(f"  CSV сохранён: {csv_path}")

# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Накопительное сравнение конфигураций RAG-пайплайна с дашбордом"
    )
    parser.add_argument("--questions", "-n", type=int, default=0,
                        help="Число вопросов (0=все, 50 для быстрой проверки)")
    parser.add_argument("--with-llm", action="store_true",
                        help="Включить конф. 4: LLM Query Decomposition (нужен API-ключ)")
    parser.add_argument("--file", "-f", type=Path, default=DEFAULT_QA,
                        help="QA-файл")
    parser.add_argument("--output", "-o", type=str, default="results",
                        help="Папка для сохранения результатов (по умолчанию: results/)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ── QA-пары ───────────────────────────────────────────────────────────────
    qa_file = args.file if args.file.is_absolute() else ROOT / args.file
    if not qa_file.exists():
        sys.exit(f"❌ Файл не найден: {qa_file}")

    all_pairs = parse_qa(qa_file)
    _SKIP = ("блокировка","блокируем","blocked","офтопик","нецензур","грубость","грубые")
    qa_pairs = [p for p in all_pairs
                if not any(kw in p.get("category","").lower() for kw in _SKIP)]

    if args.questions > 0:
        qa_pairs = qa_pairs[:args.questions]

    print(f"\n{'='*60}")
    print("  СРАВНЕНИЕ КОНФИГУРАЦИЙ RAG-ПАЙПЛАЙНА")
    print(f"{'='*60}")
    print(f"  QA-файл    : {qa_file.name}")
    print(f"  Вопросов   : {len(qa_pairs)}")
    print(f"  LLM Decomp : {'включено' if args.with_llm else 'выключено (--with-llm)'}")

    # ── индекс (один для всех конфигураций) ───────────────────────────────────
    cfg0 = CONFIGS[0]
    collection, n_chunks = build_shared_index(cfg0.embed_model, cfg0.use_prefix)
    print(f"  Чанков в индексе: {n_chunks}")

    # ── реранкер (один для конфигураций 2–5) ─────────────────────────────────
    reranker = load_reranker(cfg0.reranker_model)

    # ── прогрев FAQ-кеша (симуляция) ─────────────────────────────────────────
    # Случайно отбираем 35% вопросов как «уже подтверждённые пользователями» —
    # имитация состояния динамического кеша после нескольких недель работы бота.
    # Тестируем на ВСЕХ вопросах: ~35% попадут в кеш, ~65% пойдут в пайплайн.
    WARM_FRACTION = 0.35
    warm_cache_data = warm_cache(qa_pairs, fraction=WARM_FRACTION)
    print(
        f"\n  FAQ-кеш: симуляция прогрева {int(len(qa_pairs)*WARM_FRACTION)} "
        f"пар (seed=42), тест на всех {len(qa_pairs)} вопросах"
    )

    # ── прогон конфигураций ───────────────────────────────────────────────────
    t_total = time.perf_counter()
    results: list[ConfigResult] = []

    for cfg in CONFIGS:
        if cfg.enable_llm and not args.with_llm:
            print(f"\n  Пропускаю конф. {cfg.id} '{cfg.name}' (требует --with-llm)")
            continue
        # Все конфигурации тестируются на одном наборе qa_pairs
        result = run_config(cfg, qa_pairs, collection, reranker,
                            with_llm=args.with_llm, verbose=args.verbose,
                            warm=warm_cache_data if cfg.enable_faq else None)
        if result:
            results.append(result)

    elapsed = time.perf_counter() - t_total

    # ── вывод таблицы ─────────────────────────────────────────────────────────
    print_table(results)
    print(f"  Общее время выполнения: {elapsed:.0f}с ({elapsed/60:.1f} мин)")

    # ── сохранение ────────────────────────────────────────────────────────────
    out_dir = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    ts = time.strftime("%Y%m%d_%H%M%S")

    save_results(results, out_dir)
    build_dashboard(results, out_dir / f"dashboard_{ts}.png")

    print(f"\n  Готово. Результаты в: {out_dir}/")


if __name__ == "__main__":
    main()
