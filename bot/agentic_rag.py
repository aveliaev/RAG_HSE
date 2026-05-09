import json
import logging
import re

from config import (
    USE_YANDEX, YANDEX_API_KEY, YANDEX_FOLDER_ID,
    GROQ_API_KEY, GROQ_MODEL,
)
from rag_engine import retrieve_k, generate, yandex_complete

log = logging.getLogger(__name__)

_groq_client = None
if GROQ_API_KEY:
    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY)

_DECOMPOSE_PROMPT = (
    "Ты — маршрутизатор поисковых запросов для базы знаний о поступлении на ФКН НИУ ВШЭ.\n"
    "База знаний охватывает: программы бакалавриата, баллы ЕГЭ, бюджет, скидки, общежитие, "
    "договор, военная кафедра/ВУЦ, академический процесс, ОВЗ, иностранные граждане, "
    "индивидуальные достижения, олимпиады, сроки подачи документов.\n\n"
    "Проанализируй вопрос и верни JSON-массив с 1–3 поисковыми запросами.\n\n"
    "Правила:\n"
    "- 1 запрос: простой фактический вопрос (один балл, одна дата, одно условие)\n"
    "- 2 запроса: вопрос пересекает две темы или программы\n"
    "- 3 запроса: агрегация по нескольким программам или сложный составной вопрос\n"
    "- Убери сленг, сформулируй точно\n"
    "- Не дублируй смысл запросов\n\n"
    "Примеры:\n"
    "Вопрос: «минималки на ПМИ» → [\"минимальные баллы ЕГЭ прикладная математика информатика\"]\n"
    "Вопрос: «какие конкурсы дают баллы на ДРИП» → "
    "[\"специальные конкурсы достижения ДРИП дизайн разработка\", "
    "\"индивидуальные достижения дизайн разработка информационных продуктов\"]\n"
    "Вопрос: «иностранец хочет на ПМИ — что сдавать и минбаллы» → "
    "[\"иностранные граждане вступительные испытания ПМИ минимальный балл\"]\n"
    "Вопрос: «у меня медаль и Лицей Яндекса — сколько суммарно баллов» → "
    "[\"аттестат с отличием медаль баллы индивидуальные достижения\", "
    "\"лицей академии яндекса дополнительные баллы программы ФКН\"]\n\n"
    "Верни ТОЛЬКО JSON-массив строк, без пояснений. Пример: [\"запрос 1\", \"запрос 2\"]"
)

def _llm_call(prompt: str, user_text: str, max_tokens: int = 120) -> str:
    if USE_YANDEX and YANDEX_API_KEY and YANDEX_FOLDER_ID:
        return yandex_complete(
            messages=[{"role": "user", "content": user_text}],
            system=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
        )

    if _groq_client:
        resp = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()

    return ""

def _parse_queries(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass
    return []

def decompose_query(question: str) -> list[str]:
    try:
        raw = _llm_call(_DECOMPOSE_PROMPT, f"Вопрос: {question}")
        queries = _parse_queries(raw)
        if queries:
            return queries[:3]
    except Exception as e:
        log.warning("decompose_query failed: %s", e)
    return [question]

def agentic_retrieve(question: str, chunks_per_query: int = 3) -> str:
    queries = decompose_query(question)

    seen: set[str] = set()
    unique_chunks: list[dict] = []

    for query in queries:
        chunks = retrieve_k(query, k=chunks_per_query, skip_llm_rewrite=True)
        for c in chunks:
            key = f"{c['source']}|{c['heading']}"
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)

    unique_chunks = unique_chunks[:8]

    if not unique_chunks:
        return ""

    return "\n\n---\n\n".join(
        f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in unique_chunks
    )

def agentic_ask(
    question: str,
    history: list[dict],
    lang: str = "ru",
    citizenship: str = "",
) -> tuple[str, dict]:
    queries = decompose_query(question)

    seen: set[str] = set()
    unique_chunks: list[dict] = []
    for query in queries:
        chunks = retrieve_k(query, k=3, skip_llm_rewrite=True)
        for c in chunks:
            key = f"{c['source']}|{c['heading']}"
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)
    unique_chunks = unique_chunks[:8]

    meta = {"sub_queries": queries, "n_chunks": len(unique_chunks)}

    if not unique_chunks:
        if lang == "en":
            answer = (
                "No information found in the knowledge base for this question. "
                "Please check ba.hse.ru or email abitur@hse.ru."
            )
        else:
            answer = (
                "В базе знаний нет информации по этому вопросу. "
                "Уточните на сайте ba.hse.ru или напишите на abitur@hse.ru."
            )
        return answer, meta

    context = "\n\n---\n\n".join(
        f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in unique_chunks
    )
    answer = generate(question, context, history, lang=lang, citizenship=citizenship)
    return answer, meta
