"""
pipeline_comparison.py — Сравнение конфигураций пайплайна (накопительное)
=========================================================================

Тестируемые конфигурации (каждая добавляет компонент к предыдущей):
  1. Base RAG          — e5-base + ChromaDB, без доп. компонентов
  2. + Reranker        — добавляет cross-encoder переранжирование
  3. + Regex Rewrite   — добавляет нормализацию сленга (regex)
  4. + LLM Decomp      — разбивает запрос на подзапросы, реранк по исходному Q
  5. + FAQ Cache       — добавляет трёхуровневый кеш перед пайплайном

Запуск:
  python3 pipeline_comparison.py
  python3 pipeline_comparison.py --questions 50
  python3 pipeline_comparison.py --output results/

Внимание: конфигурация 4 делает LLM-вызов на каждый вопрос (~1-3с),
поэтому на 245 вопросах ожидаемое время ~15-20 мин.
"""

import sys, os, re, json, time, argparse, logging, statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

BOT_DIR     = Path(__file__).parent
ROOT        = BOT_DIR.parent
DATASET_DIR = ROOT / "dataset"
TEST_DIR    = ROOT / "test"
DEFAULT_QA  = TEST_DIR / "fkn_qa_test_all.md"

sys.path.insert(0, str(BOT_DIR))
os.chdir(BOT_DIR)

for _ef in (BOT_DIR / ".env", ROOT / ".env"):
    if _ef.exists():
        for _l in _ef.read_text().splitlines():
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                k, _, v = _l.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
        break

logging.basicConfig(level=logging.WARNING)

from rag_engine import _split_into_chunks, _rewrite_query, _rerank
from faq_cache import lookup as faq_lookup, _normalize as faq_normalize
from eval_rag import (
    parse_qa, tokenize,
    hit_at_k, reciprocal_rank, precision_at_k, context_recall,
)
from config import USE_YANDEX, GROQ_API_KEY


# ─── LLM-декомпозиция запроса ─────────────────────────────────────────────────

_DECOMPOSE_PROMPT = (
    "Ты — маршрутизатор поисковых запросов для базы знаний о поступлении "
    "на ФКН НИУ ВШЭ.\n"
    "Разбей вопрос пользователя на 1–3 конкретных поисковых подзапроса "
    "(каждый на отдельной строке, без нумерации и маркеров).\n"
    "Если вопрос уже конкретный — верни его одной строкой без изменений.\n"
    "Отвечай только подзапросами, без пояснений."
)


def _llm_decompose(question: str) -> str:
    if USE_YANDEX:
        try:
            import requests
            from config import YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL
            body = {
                "modelUri": f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
                "completionOptions": {"maxTokens": 120, "temperature": 0.1},
                "messages": [
                    {"role": "system", "text": _DECOMPOSE_PROMPT},
                    {"role": "user",   "text": question},
                ],
            }
            resp = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
                json=body, timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["result"]["alternatives"][0]["message"]["text"]
        except Exception as e:
            logging.warning("YandexGPT decompose error: %s", e)
    if GROQ_API_KEY:
        try:
            from groq import Groq
            from config import GROQ_MODEL
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_PROMPT},
                    {"role": "user",   "content": question},
                ],
                max_tokens=120, temperature=0.1,
            )
            return completion.choices[0].message.content or question
        except Exception as e:
            logging.warning("Groq decompose error: %s", e)
    return question


def decompose_query(question: str) -> list[str]:
    raw = _llm_decompose(question)
    lines = [re.sub(r"^[\d]+[.)]\s*|^[-•*]\s*", "", ln.strip())
             for ln in raw.strip().splitlines()]
    queries = [ln for ln in lines if ln]
    return queries if queries else [question]


def warm_cache(qa_pairs: list[dict], fraction: float = 0.35) -> dict[str, str]:
    import random
    rng = random.Random(42)
    n_warm = int(len(qa_pairs) * fraction)
    sampled = rng.sample(qa_pairs, n_warm)
    warm: dict[str, str] = {}
    for item in sampled:
        norm = faq_normalize(item["question"])
        if norm:
            warm[norm] = item["answer"]
    return warm


def faq_lookup_with_warm(question: str, warm: dict[str, str]) -> tuple[str | None, str]:
    answer, src = faq_lookup(question)
    if answer is not None:
        return answer, src
    norm = faq_normalize(question)
    if norm in warm:
        return warm[norm], "warm-cache"
    return None, "rag"


@dataclass
class PipelineConfig:
    id:               int
    name:             str
    label:            str
    embed_model:      str = "intfloat/multilingual-e5-base"
    use_prefix:       bool = True
    enable_reranker:  bool = False
    enable_regex:     bool = False
    enable_decomp:    bool = False
    enable_faq:       bool = False
    top_k:            int  = 5
    reranker_model:   str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


CONFIGS: list[PipelineConfig] = [
    PipelineConfig(
        id=1, name="Base RAG",
        label="1. Base RAG\n(e5-base)",
    ),
    PipelineConfig(
        id=2, name="+ Reranker",
        label="2. + Reranker\n(cross-encoder)",
        enable_reranker=True,
    ),
    PipelineConfig(
        id=3, name="+ Regex Rewrite",
        label="3. + Regex\nRewrite",
        enable_regex=True, enable_reranker=True,
    ),
    PipelineConfig(
        id=4, name="+ LLM Decomp",
        label="4. + LLM\nDecomp",
        enable_regex=True, enable_reranker=True, enable_decomp=True,
    ),
    PipelineConfig(
        id=5, name="+ FAQ Cache",
        label="5. + FAQ\nCache",
        enable_regex=True, enable_reranker=True, enable_decomp=True, enable_faq=True,
    ),
]

_DOCS_SKIP = ("qa_test", "_test", "test_")

def load_docs() -> dict[str, str]:
    docs = {}
    for p in sorted(DATASET_DIR.glob("*.md")):
        if any(pat in p.stem.lower() for pat in _DOCS_SKIP):
            continue
        docs[p.stem] = p.read_text(encoding="utf-8")
    return docs


def build_shared_index(embed_model: str, use_prefix: bool):
    import chromadb
    from chromadb.utils import embedding_functions
    print(f"\n  Загружаю модель: {embed_model} ...", end=" ", flush=True)
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
        print(f"\n  Реранкер недоступен: {e}")
        return None


def retrieve_one(question, collection, cfg, reranker, query_prefix, doc_prefix):
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
        candidates.append({"text": clean, "source": meta["source"], "heading": meta["heading"]})
    if cfg.enable_reranker and reranker is not None:
        chunks = _rerank(rewritten, candidates, top_k=cfg.top_k)
    else:
        chunks = candidates[:cfg.top_k]
    return chunks, (time.perf_counter() - t0) * 1000


def retrieve_decomposed(question, collection, cfg, reranker, query_prefix, doc_prefix):
    """
    Декомпозиция → поиск по каждому подзапросу → объединение пула →
    реранк по ИСХОДНОМУ вопросу → топ-K.
    """
    t0 = time.perf_counter()
    sub_queries = decompose_query(question)

    seen: set[str] = set()
    candidates: list[dict] = []
    n_cand = max(cfg.top_k * 2, 10)

    for sq in sub_queries:
        rewritten_sq = _rewrite_query(sq) if cfg.enable_regex else sq
        results = collection.query(
            query_texts=[query_prefix + rewritten_sq],
            n_results=n_cand,
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            key = f"{meta['source']}|{meta['heading']}"
            if key not in seen:
                seen.add(key)
                clean = doc[len(doc_prefix):] if doc.startswith(doc_prefix) else doc
                candidates.append({
                    "text": clean,
                    "source": meta["source"],
                    "heading": meta["heading"],
                })

    # финальный реранк по ИСХОДНОМУ вопросу — не по подзапросу!
    if cfg.enable_reranker and reranker is not None and candidates:
        chunks = _rerank(question, candidates, top_k=cfg.top_k)
    else:
        chunks = candidates[:cfg.top_k]

    return chunks, (time.perf_counter() - t0) * 1000, sub_queries


@dataclass
class ConfigResult:
    cfg:           PipelineConfig
    n:             int
    hit1:          float
    hit3:          float
    hit5:          float
    mrr5:          float
    p3:            float
    ctx_recall:    float
    lat_mean:      float
    lat_median:    float
    lat_p95:       float
    cache_hit_rate:  float = 0.0
    lat_cache_hit:   float = 0.0
    lat_cache_miss:  float = 0.0
    n_cache_hits:    int   = 0
    raw_latencies:   list  = field(default_factory=list)


def run_config(cfg, qa_pairs, collection, reranker, verbose=False, warm=None):
    print(f"\n{'─'*60}")
    print(f"  Конфигурация {cfg.id}: {cfg.name}")

    query_prefix = "query: "   if cfg.use_prefix else ""
    doc_prefix   = "passage: " if cfg.use_prefix else ""

    h1_l, h3_l, h5_l, rr_l, p3_l, cr_l = [], [], [], [], [], []
    lat_all, lat_cache_hits, lat_cache_miss = [], [], []
    n_cache_hits = 0

    for i, item in enumerate(qa_pairs, 1):
        q       = item["question"]
        ref_tok = tokenize(item["answer"])

        if cfg.enable_faq:
            t_faq = time.perf_counter()
            cached, src = faq_lookup_with_warm(q, warm or {})
            faq_ms = (time.perf_counter() - t_faq) * 1000
            if cached is not None:
                n_cache_hits += 1
                lat_all.append(faq_ms)
                lat_cache_hits.append(faq_ms)
                h1_l.append(False); h3_l.append(False); h5_l.append(False)
                rr_l.append(0.0); p3_l.append(0.0)
                cr_l.append(context_recall(ref_tok, cached))
                continue

        if cfg.enable_decomp:
            chunks, ms, _ = retrieve_decomposed(q, collection, cfg, reranker, query_prefix, doc_prefix)
        else:
            chunks, ms = retrieve_one(q, collection, cfg, reranker, query_prefix, doc_prefix)
        lat_all.append(ms)
        if cfg.enable_faq:
            lat_cache_miss.append(ms)

        h1_l.append(hit_at_k(chunks, ref_tok, k=1))
        h3_l.append(hit_at_k(chunks, ref_tok, k=3))
        h5_l.append(hit_at_k(chunks, ref_tok, k=5))
        rr_l.append(reciprocal_rank(chunks, ref_tok))
        p3_l.append(precision_at_k(chunks, ref_tok, k=3))
        cr_l.append(context_recall(ref_tok, "\n".join(c["text"] for c in chunks)))

        if verbose and i % 50 == 0:
            print(f"    {i}/{len(qa_pairs)}...", flush=True)

    n    = len(qa_pairs)
    lats = sorted(lat_all)
    res  = ConfigResult(
        cfg=cfg, n=n,
        hit1=round(sum(h1_l)/n,3), hit3=round(sum(h3_l)/n,3),
        hit5=round(sum(h5_l)/n,3), mrr5=round(sum(rr_l)/n,3),
        p3=round(sum(p3_l)/n,3),   ctx_recall=round(sum(cr_l)/n,3),
        lat_mean=round(statistics.mean(lats),1),
        lat_median=round(statistics.median(lats),1),
        lat_p95=round(lats[int(len(lats)*0.95)],1) if lats else 0.0,
        raw_latencies=lat_all,
    )
    if cfg.enable_faq:
        res.cache_hit_rate = round(n_cache_hits/n,3)
        res.n_cache_hits   = n_cache_hits
        res.lat_cache_hit  = round(statistics.mean(lat_cache_hits),2) if lat_cache_hits else 0.0
        res.lat_cache_miss = round(statistics.mean(lat_cache_miss),1) if lat_cache_miss else 0.0

    print(
        f"  → Hit@1={res.hit1:.3f}  Hit@3={res.hit3:.3f}  "
        f"Hit@5={res.hit5:.3f}  MRR@5={res.mrr5:.3f}\n"
        f"     P@3={res.p3:.3f}    CtxRec={res.ctx_recall:.3f}  "
        f"Lat mean={res.lat_mean:.0f}ms  p95={res.lat_p95:.0f}ms"
    )
    if cfg.enable_faq and n_cache_hits:
        print(f"     Cache hits: {n_cache_hits}/{n} ({res.cache_hit_rate*100:.1f}%)  "
              f"hit={res.lat_cache_hit:.1f}ms  miss={res.lat_cache_miss:.0f}ms")
    return res


def print_table(results):
    COL_W = 18
    COLS  = ["Hit@1","Hit@3","Hit@5","MRR@5","P@3","Ctx Rec","Lat mean","Lat p95"]
    bests = {
        "hit1": max(r.hit1 for r in results),
        "hit3": max(r.hit3 for r in results),
        "hit5": max(r.hit5 for r in results),
        "mrr5": max(r.mrr5 for r in results),
        "p3":   max(r.p3   for r in results),
        "ctx_recall": max(r.ctx_recall for r in results),
        "lat_mean": min(r.lat_mean for r in results),
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
        def f(val, key): s=f"{val:.3f}"; return f"*{s}*" if val==bests[key] else f" {s} "
        def fl(val, key): s=f"{val:.0f}ms"; return f"*{s}*" if val==bests[key] else f" {s} "
        row = (
            f"{r.cfg.name:{COL_W}}"
            f"{f(r.hit1,'hit1'):^10}{f(r.hit3,'hit3'):^10}{f(r.hit5,'hit5'):^10}"
            f"{f(r.mrr5,'mrr5'):^10}{f(r.p3,'p3'):^10}{f(r.ctx_recall,'ctx_recall'):^10}"
            f"{fl(r.lat_mean,'lat_mean'):^10}{fl(r.lat_p95,'lat_p95'):^10}"
        )
        print(row)
        if r.cfg.enable_faq and r.n_cache_hits:
            print(f"  {'↳ cache':>{COL_W-2}}  "
                  f"hit rate={r.cache_hit_rate*100:.1f}%  "
                  f"hit={r.lat_cache_hit:.1f}ms  miss={r.lat_cache_miss:.0f}ms")
    print(sep)
    print(f"  (* = лучшее, n={results[0].n} вопросов)\n")


def build_dashboard(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib не установлен, пропускаю dashboard")
        return

    PALETTE = ["#4C72B0","#DD8452","#55A868","#8172B2","#C44E52"]
    names   = [r.cfg.name for r in results]
    N       = len(results)
    x       = np.arange(N)
    W       = 0.18

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Сравнение конфигураций RAG-пайплайна", fontsize=15, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(3, 2, 1)
    for i, (metric, label) in enumerate([("hit1","Hit@1"),("hit3","Hit@3"),("hit5","Hit@5")]):
        vals = [getattr(r, metric) for r in results]
        bars = ax1.bar(x + i*W, vals, W, label=label, color=PALETTE[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x+W); ax1.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax1.set_ylim(0,1.1); ax1.set_ylabel("Доля запросов")
    ax1.set_title("Hit Rate@K", fontweight="bold"); ax1.legend(fontsize=8); ax1.grid(axis="y",alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    for i, (metric, label) in enumerate([("mrr5","MRR@5"),("p3","Precision@3")]):
        vals = [getattr(r, metric) for r in results]
        bars = ax2.bar(x+i*W*1.5, vals, W*1.4, label=label, color=PALETTE[i+2], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x+W*0.75); ax2.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax2.set_ylim(0,1.1); ax2.set_ylabel("Значение")
    ax2.set_title("MRR@5 и Precision@3", fontweight="bold"); ax2.legend(fontsize=8); ax2.grid(axis="y",alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 3)
    vals = [r.ctx_recall for r in results]
    bars = ax3.bar(names, vals, color=PALETTE, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_ylim(0,1.0); ax3.set_ylabel("Доля токенов эталона в контексте")
    ax3.set_title("Context Recall", fontweight="bold")
    ax3.tick_params(axis="x",labelsize=8,rotation=15); ax3.grid(axis="y",alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    lat_mean = [r.lat_mean for r in results]
    lat_p95  = [r.lat_p95  for r in results]
    bars_m = ax4.bar(x-W/2, lat_mean, W*0.9, label="mean", color="#4C72B0", alpha=0.85)
    bars_p = ax4.bar(x+W/2, lat_p95,  W*0.9, label="p95",  color="#C44E52", alpha=0.85)
    for bar, v in zip(bars_m, lat_mean):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars_p, lat_p95):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x); ax4.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax4.set_ylabel("Задержка (мс)"); ax4.set_title("Latency retrieval (мс)", fontweight="bold")
    ax4.legend(fontsize=8); ax4.grid(axis="y",alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 5)
    bp = ax5.boxplot(
        [r.raw_latencies for r in results],
        tick_labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax5.set_ylabel("Задержка (мс)"); ax5.set_title("Распределение задержек (box plot)", fontweight="bold")
    ax5.tick_params(axis="x",labelsize=8,rotation=15); ax5.grid(axis="y",alpha=0.3)

    ax6 = fig.add_subplot(3, 2, 6)
    faq_res = next((r for r in results if r.cfg.enable_faq), None)
    if faq_res and faq_res.n_cache_hits > 0:
        n_hits, n_misses = faq_res.n_cache_hits, faq_res.n - faq_res.n_cache_hits
        _, _, autotexts = ax6.pie(
            [n_hits, n_misses],
            labels=[f"Cache hit\n({n_hits})", f"Cache miss\n({n_misses})"],
            autopct="%1.1f%%", colors=["#55A868","#C44E52"],
            startangle=90, wedgeprops=dict(alpha=0.8),
        )
        for at in autotexts: at.set_fontsize(9)
        ax6.set_title(
            f"FAQ Cache\nhit={faq_res.lat_cache_hit:.1f}мс  "
            f"miss={faq_res.lat_cache_miss:.0f}мс  mean={faq_res.lat_mean:.0f}мс",
            fontweight="bold", fontsize=9,
        )
    else:
        ax6.text(0.5,0.5,"FAQ Cache\n(нет попаданий)",ha="center",va="center",
                 fontsize=11,transform=ax6.transAxes); ax6.axis("off")

    plt.tight_layout(rect=[0,0,1,0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard сохранён: {out_path}")


def save_results(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    data = []
    for r in results:
        d = {k:v for k,v in r.__dict__.items() if k not in ("cfg","raw_latencies")}
        d["config_id"] = r.cfg.id; d["config_name"] = r.cfg.name
        data.append(d)
    import csv
    csv_path = out_dir / f"pipeline_comparison_{ts}.csv"
    fields = ["config_id","config_name","n","hit1","hit3","hit5","mrr5","p3",
              "ctx_recall","lat_mean","lat_median","lat_p95",
              "cache_hit_rate","n_cache_hits","lat_cache_hit","lat_cache_miss"]
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(data)
    json_path = out_dir / f"pipeline_comparison_{ts}.json"
    json_path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(f"  CSV  сохранён: {csv_path}")
    print(f"  JSON сохранён: {json_path}")
    return out_dir / f"dashboard_{ts}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions","-n",type=int,default=0)
    parser.add_argument("--file","-f",type=Path,default=DEFAULT_QA)
    parser.add_argument("--output","-o",type=str,default="results")
    parser.add_argument("--verbose","-v",action="store_true")
    args = parser.parse_args()

    qa_file = args.file if args.file.is_absolute() else ROOT/args.file
    if not qa_file.exists():
        sys.exit(f"Файл не найден: {qa_file}")

    all_pairs = parse_qa(qa_file)
    _SKIP = ("блокировка","блокируем","blocked","офтопик","нецензур","грубость","грубые")
    qa_pairs = [p for p in all_pairs if not any(kw in p.get("category","").lower() for kw in _SKIP)]
    if args.questions > 0:
        qa_pairs = qa_pairs[:args.questions]

    print(f"\n{'='*60}")
    print("  СРАВНЕНИЕ КОНФИГУРАЦИЙ RAG-ПАЙПЛАЙНА")
    print(f"{'='*60}")
    print(f"  QA-файл  : {qa_file.name}")
    print(f"  Вопросов : {len(qa_pairs)}")

    cfg0 = CONFIGS[0]
    collection, n_chunks = build_shared_index(cfg0.embed_model, cfg0.use_prefix)
    print(f"  Чанков   : {n_chunks}")
    reranker = load_reranker(cfg0.reranker_model)

    WARM_FRACTION = 0.35
    warm_cache_data = warm_cache(qa_pairs, fraction=WARM_FRACTION)
    print(f"\n  FAQ-кеш: прогрев {int(len(qa_pairs)*WARM_FRACTION)} пар (seed=42), "
          f"тест на всех {len(qa_pairs)}")

    t_total = time.perf_counter()
    results = []
    for cfg in CONFIGS:
        res = run_config(cfg, qa_pairs, collection, reranker,
                         verbose=args.verbose,
                         warm=warm_cache_data if cfg.enable_faq else None)
        results.append(res)

    elapsed = time.perf_counter() - t_total
    print_table(results)
    print(f"  Время: {elapsed:.0f}с ({elapsed/60:.1f} мин)")

    out_dir = Path(args.output) if Path(args.output).is_absolute() else ROOT/args.output
    dashboard_path = save_results(results, out_dir)
    build_dashboard(results, dashboard_path)
    print(f"\n  Готово. Результаты в: {out_dir}/")


if __name__ == "__main__":
    main()
