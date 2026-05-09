import sys
import re
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

try:
    from rouge_score import rouge_scorer as _rs_mod
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False

from rag_engine import build_index, retrieve, generate
from faq_cache import lookup

def parse_qa(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    pairs = []
    category = ""
    q_id = q_text = a_text = ""

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## Категория"):
            category = line.lstrip("# ").strip()
        elif re.match(r"^\*\*Q\d+\*\*$", line):
            q_id = line.strip("*")
            q_text = a_text = ""
        elif line.startswith("> ") and q_id and not q_text:
            q_text = line[2:].strip()
        elif line.startswith("**A:**") and q_id:
            a_text = line[6:].strip()
        elif a_text and q_id and line.strip() and not line.startswith(("---", "**Q", "## ", "# ")):
            a_text += " " + line.strip()

        if q_text and a_text and q_id:
            pairs.append({"id": q_id, "category": category, "question": q_text, "answer": a_text})
            q_id = q_text = a_text = ""

    return pairs

_STOP = {
    "а","в","и","к","на","не","но","о","от","по","с","у","я","вы","мы","он",
    "она","оно","они","это","то","как","что","где","когда","кто","чем","ли",
    "есть","будет","можно","нужно","нет","да","для","при","из","до","за","со",
    "или","если","же","ещё","уже","тоже","только",
}

def _tokens(text: str) -> set[str]:
    text = re.sub(r"[^а-яёa-z0-9\s]", " ", text.lower())
    return {t for t in text.split() if t and t not in _STOP and len(t) > 1}

def context_recall(ref: str, context: str) -> float:
    rt = _tokens(ref)
    if not rt:
        return 0.0
    return len(rt & _tokens(context)) / len(rt)

def rouge1(pred: str, ref: str) -> float:
    if not _ROUGE_OK or not pred:
        return 0.0
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    return s.score(ref, pred)["rouge1"].fmeasure

def rougeL(pred: str, ref: str) -> float:
    if not _ROUGE_OK or not pred:
        return 0.0
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return s.score(ref, pred)["rougeL"].fmeasure

def run(qa_pairs: list[dict], with_llm: bool, verbose: bool) -> None:
    print(f"\n{'='*62}")
    print(f"  RAG проверка  |  вопросов: {len(qa_pairs)}  |  LLM: {'вкл' if with_llm else 'выкл'}")
    print(f"{'='*62}\n")

    print("Строю индекс...")
    n = build_index()
    print(f"Готово: {n} чанков\n")

    rows = []
    cat_stats: dict = defaultdict(lambda: {"total": 0, "ctx_hit": 0, "ans_hit": 0})

    for item in qa_pairs:
        qid      = item["id"]
        question = item["question"]
        ref      = item["answer"]
        cat      = item["category"]

        context  = retrieve(question)
        c_rec    = context_recall(ref, context)
        ctx_hit  = c_rec >= 0.5

        faq_ans, faq_src = lookup(question)
        used_faq = faq_ans is not None

        predicted = ""
        r1 = rl = 0.0
        if with_llm and not used_faq:
            try:
                predicted = generate(question, context, [], lang="ru")
                r1 = rouge1(predicted, ref)
                rl = rougeL(predicted, ref)
            except Exception as e:
                predicted = f"[ОШИБКА: {e}]"
        elif used_faq:
            predicted = faq_ans
            r1 = rouge1(predicted, ref)
            rl = rougeL(predicted, ref)

        ans_hit = r1 >= 0.3 if predicted else None
        cat_stats[cat]["total"] += 1
        if ctx_hit:
            cat_stats[cat]["ctx_hit"] += 1
        if ans_hit:
            cat_stats[cat]["ans_hit"] += 1

        rows.append({
            "id": qid, "category": cat,
            "question": question, "ref_answer": ref,
            "predicted": predicted,
            "source": "faq" if used_faq else "rag",
            "context_recall": round(c_rec, 3),
            "ctx_hit": ctx_hit,
            "rouge1": round(r1, 3),
            "rougeL": round(rl, 3),
            "ans_hit": ans_hit,
        })

        ctx_icon = "✅" if ctx_hit else "❌"
        ans_icon = ("✅" if ans_hit else "❌") if ans_hit is not None else "  "
        llm_part = f"  R1={r1:.2f}" if predicted else ""
        src_tag  = f"[{faq_src}]" if used_faq else "[rag]"
        print(f"  {ctx_icon}{ans_icon} {qid}  ctx={c_rec:.2f}{llm_part}  {src_tag}")

        if verbose:
            print(f"       Q: {question}")
            print(f"  ожидаем: {ref[:110]}")
            if predicted:
                print(f"  ответил: {predicted[:110]}")
            print()

    n_total = len(rows)
    ctx_hits = sum(1 for r in rows if r["ctx_hit"])
    rag_rows = [r for r in rows if r["source"] == "rag"]
    rag_ctx  = sum(1 for r in rag_rows if r["ctx_hit"])

    r1_vals = [r["rouge1"] for r in rows if r["predicted"]]
    rl_vals = [r["rougeL"] for r in rows if r["predicted"]]
    ans_hits = sum(1 for r in rows if r["ans_hit"])
    n_pred   = len(r1_vals)

    print(f"\n{'='*62}")
    print("  ИТОГО")
    print(f"{'='*62}")
    print(f"  Вопросов всего:              {n_total}")
    print(f"  Через FAQ-кеш:               {n_total - len(rag_rows)}")
    print(f"  Через RAG:                   {len(rag_rows)}")
    print()
    print(f"  Context recall ≥0.5 (всё):  {ctx_hits}/{n_total} = {ctx_hits/n_total*100:.1f}%")
    if rag_rows:
        print(f"  Context recall ≥0.5 (RAG):  {rag_ctx}/{len(rag_rows)} = {rag_ctx/len(rag_rows)*100:.1f}%")
    if n_pred:
        print(f"\n  Ответов сгенерировано:       {n_pred}")
        print(f"  Avg ROUGE-1:                 {sum(r1_vals)/n_pred:.3f}")
        print(f"  Avg ROUGE-L:                 {sum(rl_vals)/n_pred:.3f}")
        print(f"  Answer hit (R1≥0.3):         {ans_hits}/{n_pred} = {ans_hits/n_pred*100:.1f}%")

    if not _ROUGE_OK:
        print("\n  ⚠  rouge_score не установлен — ROUGE не считался.")
        print("     Установите: pip install rouge-score")

    bad = [r for r in rows if not r["ctx_hit"] and r["source"] == "rag"]
    if bad:
        print(f"\n  ❌ Слабый retrieval ({len(bad)} шт.):")
        for r in bad:
            print(f"     {r['id']} recall={r['context_recall']}  {r['question'][:70]}")

    if any(r["category"] for r in rows):
        print(f"\n{'='*62}")
        print("  ПО КАТЕГОРИЯМ")
        print(f"{'='*62}")
        for cat, s in cat_stats.items():
            pct = s["ctx_hit"] / s["total"] * 100 if s["total"] else 0
            ans_pct = (s["ans_hit"] / s["total"] * 100) if s["total"] else 0
            ans_str = f"  ans={s['ans_hit']}/{s['total']}={ans_pct:.0f}%" if n_pred else ""
            print(f"  {cat[:50]:50s}  ctx={s['ctx_hit']}/{s['total']}={pct:.0f}%{ans_str}")

    out = Path(__file__).parent / "rag_check_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Детали → {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка RAG по QA-файлу")
    parser.add_argument("qa_file", help="Путь к .md файлу с тест-парами")
    parser.add_argument("--limit", type=int, default=0,
                        help="Проверить только первые N вопросов (0 = все)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Только retrieval, без генерации через Groq")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Печатать вопрос, ожидаемый и полученный ответ")
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        sys.exit(f"Файл не найден: {qa_path}")

    pairs = parse_qa(qa_path)
    if not pairs:
        sys.exit("Не удалось распарсить QA-пары из файла.")

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Загружено {len(pairs)} пар из {qa_path.name}")
    run(pairs, with_llm=not args.no_llm, verbose=args.verbose)
