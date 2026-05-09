import sys
import re
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from config import RAG_TOP_K
from rag_engine import build_index, retrieve, retrieve_k, _rewrite_query, _rewrite_query_llm, generate
from faq_cache import lookup
from rag_engine import needs_clarification, check_content

def build_index_clean() -> int:
    return build_index()

QA_FILE = Path(__file__).parent.parent / "test" / "fkn_qa_test_all.md"

def parse_qa(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    pairs = []
    category = ""
    q_id = q_text = a_text = ""

    for line in text.splitlines():
        line = line.rstrip()

        if line.startswith("## ") and ("категори" in line.lower() or "раздел" in line.lower()):
            category = line.strip("# ").strip()
            continue

        if re.match(r"^\*\*Q\d+\*\*$", line):
            if q_text and a_text and q_id:
                pairs.append({"id": q_id, "category": category, "question": q_text, "answer": a_text})
            q_id = line.strip("*")
            q_text = ""
            a_text = ""
            continue

        if not q_id:
            continue

        if line.startswith("> ") and not q_text:
            q_text = line[2:].strip()
            continue

        if re.match(r"^[Вв]опрос\s*:", line) and not q_text:
            q_text = re.sub(r"^[Вв]опрос\s*:\s*", "", line).strip()
            continue

        if line.startswith("**A:**"):
            a_text = line[6:].strip()
            continue

        if re.match(r"^[Оо]твет\s*:", line) and not a_text:
            a_text = re.sub(r"^[Оо]твет\s*:\s*", "", line).strip()
            continue

        if a_text and q_id and line.strip():
            skip = (line.startswith("---") or line.startswith("**Q") or
                    line.startswith("**A") or line.startswith("## ") or
                    line.startswith("# ") or re.match(r"^[Вв]опрос\s*:", line) or
                    re.match(r"^[Оо]твет\s*:", line))
            if not skip:
                a_text += " " + line.strip()

    if q_text and a_text and q_id:
        pairs.append({"id": q_id, "category": category, "question": q_text, "answer": a_text})

    return pairs

_STOP = {
    "а", "в", "и", "к", "на", "не", "но", "о", "от", "по", "с", "у",
    "я", "вы", "мы", "он", "она", "оно", "они", "это", "то",
    "как", "что", "где", "когда", "кто", "чем", "ли", "есть", "будет",
    "можно", "нужно", "нет", "да", "для", "при", "из", "до", "за", "со",
    "или", "если", "то", "же", "ещё", "уже", "тоже", "только",
}

def tokenize(text: str) -> set[str]:
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text.lower())
    return {t for t in text.split() if t and t not in _STOP and len(t) > 1}

def context_recall(answer_tokens: set[str], context: str) -> float:
    ctx_tokens = tokenize(context)
    if not answer_tokens:
        return 0.0
    found = len(answer_tokens & ctx_tokens)
    return found / len(answer_tokens)

def chunk_recall(answer_tokens: set[str], chunk_text: str) -> float:
    if not answer_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk_text)
    return len(answer_tokens & chunk_tokens) / len(answer_tokens)

def hit_at_k(chunks: list[dict], answer_tokens: set[str], k: int, threshold: float = 0.25) -> bool:
    for chunk in chunks[:k]:
        if chunk_recall(answer_tokens, chunk["text"]) >= threshold:
            return True
    return False

def reciprocal_rank(chunks: list[dict], answer_tokens: set[str], threshold: float = 0.25) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_recall(answer_tokens, chunk["text"]) >= threshold:
            return 1.0 / rank
    return 0.0

def precision_at_k(chunks: list[dict], answer_tokens: set[str], k: int, threshold: float = 0.25) -> float:
    relevant = sum(
        1 for chunk in chunks[:k]
        if chunk_recall(answer_tokens, chunk["text"]) >= threshold
    )
    return relevant / k if k else 0.0

def first_relevant_rank(chunks: list[dict], answer_tokens: set[str], threshold: float = 0.25) -> int:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_recall(answer_tokens, chunk["text"]) >= threshold:
            return rank
    return 0

class _CyrillicTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r'[а-яёa-z0-9]+', text.lower())

_rouge_scorer_instance = None

def _get_rouge_scorer():
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        from rouge_score import rouge_scorer as _rs
        _rouge_scorer_instance = _rs.RougeScorer(
            ["rouge1", "rougeL"],
            use_stemmer=False,
            tokenizer=_CyrillicTokenizer(),
        )
    return _rouge_scorer_instance

def rouge1_f(prediction: str, reference: str) -> float:
    score = _get_rouge_scorer().score(reference, prediction)
    return score["rouge1"].fmeasure

def rougeL_f(prediction: str, reference: str) -> float:
    score = _get_rouge_scorer().score(reference, prediction)
    return score["rougeL"].fmeasure

def determine_route(question: str) -> str:
    content_block = check_content(question)
    if content_block:
        return "blocked"

    faq_answer, source = lookup(question)
    if faq_answer is not None:
        return f"faq-{source}"

    clarification = needs_clarification(question, [])
    if clarification:
        return "clarification"

    return "rag"

def evaluate(qa_pairs: list[dict], with_llm: bool, verbose: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"  RAG Evaluation — {len(qa_pairs)} вопросов")
    print(f"  LLM generation: {'ON' if with_llm else 'OFF (только retrieval)'}")
    print(f"{'='*60}\n")

    print("Строю индекс ChromaDB (без тестового файла)...")
    n_chunks = build_index_clean()
    print(f"Индекс готов: {n_chunks} чанков\n")

    results = []
    route_counter: Counter = Counter()
    category_stats: dict = defaultdict(lambda: {"total": 0, "ctx_hit": 0, "ans_hit": 0})

    for item in qa_pairs:
        qid = item["id"]
        question = item["question"]
        ref_answer = item["answer"]
        category = item["category"]

        ref_tokens = tokenize(ref_answer)
        _t_start = time.perf_counter()

        content_block = check_content(question)
        clarification_msg = needs_clarification(question, []) if not content_block else None
        faq_answer, faq_source = lookup(question) if not content_block else (None, None)

        if content_block:
            route = "blocked"
        elif faq_answer is not None:
            route = f"faq-{faq_source}"
        elif clarification_msg:
            route = "clarification"
        else:
            route = "rag"
        route_counter[route] += 1

        top5_chunks = retrieve_k(question, k=5)
        rewritten = top5_chunks[0]["rewritten"] if top5_chunks else _rewrite_query(question)
        context = "\n\n---\n\n".join(
            f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in top5_chunks
        )

        c_recall = context_recall(ref_tokens, context)
        ctx_hit = c_recall >= 0.5
        h1 = hit_at_k(top5_chunks, ref_tokens, k=1)
        h3 = hit_at_k(top5_chunks, ref_tokens, k=3)
        h5 = hit_at_k(top5_chunks, ref_tokens, k=5)
        top1_source = top5_chunks[0]["source"] if top5_chunks else "?"

        chunk_recalls = [round(chunk_recall(ref_tokens, c["text"]), 3) for c in top5_chunks]

        rr   = reciprocal_rank(top5_chunks, ref_tokens)
        p1   = precision_at_k(top5_chunks, ref_tokens, k=1)
        p3   = precision_at_k(top5_chunks, ref_tokens, k=3)
        p5   = precision_at_k(top5_chunks, ref_tokens, k=5)
        frr  = first_relevant_rank(top5_chunks, ref_tokens)

        r1 = rL = None
        if content_block:
            predicted = content_block
        elif faq_answer is not None:
            predicted = faq_answer
        elif clarification_msg:
            predicted = clarification_msg
        else:
            predicted = ""

        if with_llm and route == "rag":
            try:
                predicted = generate(question, context, [], lang="ru")
                r1 = rouge1_f(predicted, ref_answer)
                rL = rougeL_f(predicted, ref_answer)
            except Exception as e:
                predicted = f"[ERROR: {e}]"
            time.sleep(15)

        latency_ms = round((time.perf_counter() - _t_start) * 1000, 1)

        row = {
            "id": qid,
            "category": category,
            "question": question,
            "route": route,
            "rewritten": rewritten if rewritten.lower() != question.lower() else "",
            "top1_source": top1_source,
            "context_recall": round(c_recall, 3),
            "ctx_hit": ctx_hit,
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5,
            "chunk_recalls": chunk_recalls,
            "rouge1_f": round(r1, 3) if r1 is not None else None,
            "rougeL_f": round(rL, 3) if rL is not None else None,
            "ans_hit": (r1 >= 0.3) if r1 is not None else None,
            "rr": round(rr, 4),
            "precision_at_1": round(p1, 3),
            "precision_at_3": round(p3, 3),
            "precision_at_5": round(p5, 3),
            "first_relevant_rank": frr,
            "latency_ms": latency_ms,
            "ref_answer": ref_answer,
            "predicted": predicted,
        }
        results.append(row)

        category_stats[category]["total"] += 1
        if ctx_hit:
            category_stats[category]["ctx_hit"] += 1
        if row["ans_hit"]:
            category_stats[category]["ans_hit"] += 1

        status = "✅" if h1 else ("⚡" if h3 else ("🔶" if h5 else "❌"))
        llm_status = f"  ROUGE1={r1:.2f}" if r1 is not None else ""
        print(f"  {status} {qid} [{route:12s}] H@1={int(h1)} H@3={int(h3)} H@5={int(h5)} recall={c_recall:.2f}  src={top1_source}{llm_status}")

        if verbose:
            print(f"      Q: {question}")
            print(f"      A(ref):  {ref_answer[:120]}...")
            if predicted:
                print(f"      A(pred): {predicted[:120]}...")
            print()

    n = len(results)
    n_rag = sum(1 for r in results if r["route"] == "rag")
    ctx_hits = sum(1 for r in results if r["ctx_hit"])
    rag_ctx_hits = sum(1 for r in results if r["route"] == "rag" and r["ctx_hit"])

    h1_total = sum(1 for r in results if r["route"] == "rag" and r["hit@1"])
    h3_total = sum(1 for r in results if r["route"] == "rag" and r["hit@3"])
    h5_total = sum(1 for r in results if r["route"] == "rag" and r["hit@5"])

    print(f"\n{'='*60}")
    print("  ИТОГОВЫЕ МЕТРИКИ")
    print(f"{'='*60}")
    print(f"  Всего вопросов:        {n}")
    print(f"  Маршруты:")
    for route, cnt in route_counter.most_common():
        print(f"    {route:20s}: {cnt}")
    print()
    print(f"  Retrieval (только RAG, n={n_rag}):")
    rag_results = [r for r in results if r["route"] == "rag"]
    if n_rag:
        mrr_val   = sum(r.get("rr", 0) for r in rag_results) / n_rag
        avg_p3    = sum(r.get("precision_at_3", 0) for r in rag_results) / n_rag
        avg_p5    = sum(r.get("precision_at_5", 0) for r in rag_results) / n_rag
        print(f"    Hit@1:  {h1_total}/{n_rag} = {h1_total/n_rag*100:.1f}%")
        print(f"    Hit@3:  {h3_total}/{n_rag} = {h3_total/n_rag*100:.1f}%")
        print(f"    Hit@5:  {h5_total}/{n_rag} = {h5_total/n_rag*100:.1f}%")
        print(f"    MRR@5:  {mrr_val:.3f}  (среднее 1/rank первого релевантного чанка)")
        print(f"    P@3:    {avg_p3*100:.1f}%  (доля релевантных среди top-3)")
        print(f"    P@5:    {avg_p5*100:.1f}%  (доля релевантных среди top-5)")
        print(f"    Context recall >= 0.5: {rag_ctx_hits}/{n_rag} = {rag_ctx_hits/n_rag*100:.1f}%")
    print(f"  Context recall >= 0.5 (все вопросы): {ctx_hits}/{n} = {ctx_hits/n*100:.1f}%" if n else "  Context recall >= 0.5 (все вопросы): n/a")

    rouge1_vals = [r["rouge1_f"] for r in results if r["rouge1_f"] is not None]
    if rouge1_vals:
        avg_r1 = sum(rouge1_vals) / len(rouge1_vals)
        rougeL_vals = [r["rougeL_f"] for r in results if r["rougeL_f"] is not None]
        avg_rL = sum(rougeL_vals) / len(rougeL_vals)
        ans_hits = sum(1 for r in results if r["ans_hit"])
        print(f"\n  Generation (n={len(rouge1_vals)}):")
        print(f"    Avg ROUGE-1:  {avg_r1:.3f}")
        print(f"    Avg ROUGE-L:  {avg_rL:.3f}")
        print(f"    Answer hit (R1>=0.3): {ans_hits}/{len(rouge1_vals)} = {ans_hits/len(rouge1_vals)*100:.1f}%")

    failures = [r for r in results if not r["hit@5"] and r["route"] == "rag"]
    partial = [r for r in results if r["hit@5"] and not r["hit@1"] and r["route"] == "rag"]
    if failures:
        print(f"\n  ❌ Полный промах hit@5=0 ({len(failures)} шт.):")
        for r in failures:
            print(f"    {r['id']} recall={r['context_recall']:.2f}  src={r['top1_source']}")
            print(f"      Q: {r['question']}")
    if partial:
        print(f"\n  🔶 Найдено в top3-5, но не в top1 ({len(partial)} шт.):")
        for r in partial:
            print(f"    {r['id']} H@1={int(r['hit@1'])} H@3={int(r['hit@3'])} H@5={int(r['hit@5'])}  Q: {r['question']}")

    faq_intercepts = [r for r in results if r["route"].startswith("faq")]
    if faq_intercepts:
        print(f"\n  ⚠️  FAQ-перехваты ({len(faq_intercepts)} шт.):")
        for r in faq_intercepts:
            print(f"    {r['id']} [{r['route']}]  Q: {r['question']}")

    print(f"\n{'='*60}")
    print("  ПО КАТЕГОРИЯМ  (hit@1 / hit@3 / hit@5)")
    print(f"{'='*60}")
    cat_hit = defaultdict(lambda: {"total": 0, "h1": 0, "h3": 0, "h5": 0})
    for r in results:
        if r["route"] != "rag":
            continue
        c = r["category"]
        cat_hit[c]["total"] += 1
        cat_hit[c]["h1"] += int(r["hit@1"])
        cat_hit[c]["h3"] += int(r["hit@3"])
        cat_hit[c]["h5"] += int(r["hit@5"])
    for cat, s in cat_hit.items():
        t = s["total"] or 1
        print(f"  {cat[:50]:50s}  {s['h1']}/{t}  {s['h3']}/{t}  {s['h5']}/{t}")

    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Детали сохранены: {out_path}")

    return {
        "n": n,
        "n_rag": n_rag,
        "routes": dict(route_counter),
        "hit@1": h1_total / n_rag if n_rag else None,
        "hit@3": h3_total / n_rag if n_rag else None,
        "hit@5": h5_total / n_rag if n_rag else None,
        "ctx_hit_rate": ctx_hits / n if n else None,
        "rag_ctx_hit_rate": rag_ctx_hits / n_rag if n_rag else None,
        "avg_rouge1": sum(rouge1_vals) / len(rouge1_vals) if rouge1_vals else None,
        "failures": [r["id"] for r in failures],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Генерировать ответы через LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--file", "-f", type=Path, default=QA_FILE, help="Путь к QA-файлу (по умолчанию fkn_qa_test (1).md)")
    args = parser.parse_args()

    qa_file = args.file if args.file.is_absolute() else Path(__file__).parent.parent / "test" / args.file
    qa_pairs = parse_qa(qa_file)
    print(f"Загружено {len(qa_pairs)} тестовых пар из {qa_file.name}")

    evaluate(qa_pairs, with_llm=args.with_llm, verbose=args.verbose)
