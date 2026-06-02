

import json
import logging
import re

from config import AGENTIC_MAX_ITERS
from rag_engine import retrieve_k, generate
from agentic_rag import _llm_call, decompose_query

log = logging.getLogger(__name__)

_CHUNKS_PER_QUERY = 4
_MAX_CONTEXT_CHUNKS = 10

_GRADE_PROMPT = (
    "Ты оцениваешь, достаточно ли информации в КОНТЕКСТЕ, чтобы ПОЛНОСТЬЮ ответить "
    "на вопрос абитуриента о поступлении на ФКН НИУ ВШЭ.\n"
    "Верни ТОЛЬКО JSON-объект без пояснений:\n"
    '{"sufficient": true|false, "missing": "чего не хватает, кратко"}\n\n'
    "Правила:\n"
    "- sufficient=true, если контекст покрывает ВСЕ части вопроса.\n"
    "- sufficient=false, если хотя бы одна часть вопроса не покрыта — "
    "в поле missing коротко укажи, какую информацию нужно доискать.\n"
    "- Не придумывай: оценивай только то, что реально есть в контексте."
)

_REFINE_PROMPT = (
    "Контекст не полностью отвечает на вопрос абитуриента ФКН НИУ ВШЭ.\n"
    "Сформулируй ОДИН новый поисковый запрос к базе знаний, чтобы найти именно "
    "недостающую информацию. Убери сленг, сделай запрос точным и информативным.\n"
    "Верни ТОЛЬКО строку запроса — без кавычек, нумерации и пояснений."
)


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in chunks
    )


def _parse_grade(raw: str) -> dict:
    """Парсит JSON-оценку. При неудаче считаем контекст достаточным (завершаем цикл)."""
    for candidate in (raw, _extract_json_obj(raw)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "sufficient" in obj:
                return {
                    "sufficient": bool(obj.get("sufficient")),
                    "missing": str(obj.get("missing", "")).strip(),
                }
        except json.JSONDecodeError:
            continue
    return {"sufficient": True, "missing": ""}


def _extract_json_obj(raw: str) -> str | None:
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    return m.group() if m else None


def _grade_context(question: str, context: str) -> dict:
    try:
        user = f"Вопрос: {question}\n\nКОНТЕКСТ:\n{context[:3000]}"
        raw = _llm_call(_GRADE_PROMPT, user, max_tokens=120)
        if not raw.strip():
            return {"sufficient": True, "missing": ""}
        return _parse_grade(raw)
    except Exception as e:
        log.warning("grade_context failed: %s", e)
        return {"sufficient": True, "missing": ""}


def _refine_query(question: str, context: str, missing: str) -> str:
    try:
        user = (
            f"Вопрос абитуриента: {question}\n"
            f"Чего не хватает в контексте: {missing}\n\n"
            f"Уже найдено:\n{context[:1500]}"
        )
        raw = _llm_call(_REFINE_PROMPT, user, max_tokens=60)
        return raw.strip().strip('"').strip()
    except Exception as e:
        log.warning("refine_query failed: %s", e)
        return ""


def agentic_loop_ask(
    question: str,
    history: list[dict],
    lang: str = "ru",
    citizenship: str = "",
) -> tuple[str, dict]:
    seen: set[str] = set()
    chunks: list[dict] = []

    def _add(new_chunks: list[dict]) -> None:
        for c in new_chunks:
            key = f"{c['source']}|{c['heading']}"
            if key not in seen:
                seen.add(key)
                chunks.append(c)

    # Шаг 1: начальный retrieve через декомпозицию
    queries = decompose_query(question)
    for q in queries:
        _add(retrieve_k(q, k=_CHUNKS_PER_QUERY, skip_llm_rewrite=True))

    all_queries: list[str] = list(queries)
    grades: list[dict] = []

    # Шаги 2-4: цикл оценки и до-поиска
    for _ in range(max(0, AGENTIC_MAX_ITERS)):
        if not chunks:
            break
        grade = _grade_context(question, _format_context(chunks))
        grades.append(grade)
        if grade["sufficient"]:
            break

        refined = _refine_query(question, _format_context(chunks), grade["missing"])
        if not refined or refined.lower() in {q.lower() for q in all_queries}:
            log.info("agentic-loop: refine не дал нового запроса, останавливаюсь")
            break

        log.info("agentic-loop: до-поиск по '%s'", refined[:60])
        all_queries.append(refined)
        _add(retrieve_k(refined, k=_CHUNKS_PER_QUERY, skip_llm_rewrite=True))

    chunks = chunks[:_MAX_CONTEXT_CHUNKS]

    meta = {
        "sub_queries": all_queries,
        "n_chunks": len(chunks),
        "iterations": len(grades),
        "mode": "agentic-loop",
    }

    if not chunks:
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

    answer = generate(question, _format_context(chunks), history, lang=lang, citizenship=citizenship)
    return answer, meta
