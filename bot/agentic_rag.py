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
    "\"лицей академии яндекса дополнительные баллы программы ФКН\"]\n"
    "Вопрос: «сколько нужно баллов на КНАД и сколько баллов даёт ГТО» → "
    "[\"минимальные баллы ЕГЭ компьютерные науки и анализ данных КНАД\", "
    "\"ГТО золотой серебряный дополнительные баллы индивидуальные достижения\"]\n"
    "Вопрос: «минбаллы на ПМИ и что дают за волонтёрство» → "
    "[\"минимальные баллы ЕГЭ прикладная математика и информатика\", "
    "\"волонтёрская деятельность дополнительные баллы индивидуальные достижения\"]\n\n"
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

_ACHIEVEMENT_KW = [
    "гто", "медаль", "волонтёр", "волонтер", "кмс", "мастер спорта",
    "абилимпикс", "аттестат", "отличи", "лицей яндекс", "яндекс лицей",
    "большая перемена", "конкурс", "достижени", "индивидуальн",
    "t-поколен", "т-поколен", "золот", "серебр",
]
_PROGRAM_SCORE_KW = [
    "пми", "пад", "кнад", "эад", "дрип", "прогинж", "робот",
    "прикладн", "компьютерн", "программн", "разработ", "экономик",
    "минимальн", "проходн", "балл", "поступ", "нужно набрать",
]

def _rule_split(question: str) -> list[str] | None:
    q = question.lower()
    has_achievement = any(kw in q for kw in _ACHIEVEMENT_KW)
    has_program_score = any(kw in q for kw in _PROGRAM_SCORE_KW)
    if not (has_achievement and has_program_score):
        return None

    # Detect "и сколько / а также / плюс" joining two questions
    split_patterns = [
        r"\s+и\s+сколько\s+",
        r"\s+и\s+что\s+дают\s+за\s+",
        r"\s+и\s+что\s+даёт\s+",
        r"\s+и\s+сколько\s+дают\s+за\s+",
        r"\s+а\s+также\s+",
        r"\s+плюс\s+",
        r"\s+и\s+ещё\s+",
        r"\s+и\s+еще\s+",
    ]
    for pat in split_patterns:
        parts = re.split(pat, q, maxsplit=1)
        if len(parts) == 2:
            return [parts[0].strip(), parts[1].strip()]
    return None

def decompose_query(question: str) -> list[str]:
    try:
        raw = _llm_call(_DECOMPOSE_PROMPT, f"Вопрос: {question}")
        queries = _parse_queries(raw)
        if queries and len(queries) >= 2:
            return queries[:3]
        # LLM returned 1 generic query — try rule-based split first
        split = _rule_split(question)
        if split:
            return split
        if queries:
            return queries[:3]
    except Exception as e:
        log.warning("decompose_query failed: %s", e)

    split = _rule_split(question)
    if split:
        return split
    return [question]

def agentic_retrieve(question: str, chunks_per_query: int = 4) -> str:
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
        chunks = retrieve_k(query, k=4, skip_llm_rewrite=True)
        for c in chunks:
            key = f"{c['source']}|{c['heading']}"
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)
    unique_chunks = unique_chunks[:10]

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
