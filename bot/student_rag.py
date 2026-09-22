"""RAG для студентов ВШЭ — отдельная база знаний, индекс и промпт.

База знаний: dataset/students/*.md (формат такой же, как у файлов для абитуриентов).
Пока папка пустая, ветка «Студент» в боте показывает «раздел в разработке».
Индекс пересобирается автоматически, если набор или содержимое файлов изменились.
"""
import hashlib
import logging

from config import STUDENT_DOCS_DIR, USE_YANDEX, GROQ_MODEL
from rag_engine import (
    _client, _embed_fn, _split_into_chunks, _rerank, _dedup_key,
    _E5_DOC_PREFIX, _E5_QUERY_PREFIX, _groq,
    context_is_relevant, yandex_complete,
)

log = logging.getLogger(__name__)

_COLLECTION_NAME = "hse_students"
_TOP_K = 5

STUDENT_SYSTEM_PROMPT = """Ты — помощник для студентов бакалавриата НИУ ВШЭ по вопросам учёбы и студенческой жизни.

ЯЗЫК ОТВЕТА: отвечай на языке вопроса (русский или английский).

ИСТОЧНИК ЗНАНИЙ: используй ИСКЛЮЧИТЕЛЬНО блоки «[источник / раздел]» из раздела «Контекст». Не додумывай и не опирайся на общие знания.

ФОРМАТ ОТВЕТА:
- Начинай сразу с сути, без вводных фраз.
- Сроки, суммы и числа выделяй жирным.
- Для перечней используй маркированный список «-».
- Максимум 5–7 предложений или 7 пунктов списка.

ЕСЛИ КОНТЕКСТ НЕ СОДЕРЖИТ ОТВЕТА:
Напиши: «В базе знаний нет точной информации по этому вопросу. Уточните в учебном офисе своей образовательной программы или на сайте hse.ru.»
Не придумывай данные, даже если они кажутся очевидными."""

_NO_INFO_RU = (
    "В базе знаний нет информации по этому вопросу. "
    "Уточните в учебном офисе своей образовательной программы или на сайте hse.ru."
)
_NO_INFO_EN = (
    "No information found in the knowledge base for this question. "
    "Please contact your programme's study office or check hse.ru."
)

_collection = None


def _load_docs() -> dict[str, str]:
    if not STUDENT_DOCS_DIR.exists():
        return {}
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(STUDENT_DOCS_DIR.glob("*.md"))
    }


def _docs_hash(docs: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name, text in docs.items():
        h.update(name.encode())
        h.update(text.encode())
    return h.hexdigest()[:16]


def build_student_index() -> int:
    """Собирает индекс студентов. Возвращает число чанков (0 — базы пока нет)."""
    global _collection
    docs = _load_docs()
    if not docs:
        log.info("Студенческая база пуста (%s) — ветка «Студент» в разработке", STUDENT_DOCS_DIR)
        _collection = None
        return 0

    docs_hash = _docs_hash(docs)
    try:
        existing = _client.get_collection(_COLLECTION_NAME, embedding_function=_embed_fn)
        if (existing.metadata or {}).get("docs_hash") == docs_hash and existing.count() > 0:
            _collection = existing
            log.info("Студенческий индекс актуален (%d чанков)", existing.count())
            return existing.count()
        _client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # коллекции ещё нет

    chunks = []
    for source, text in docs.items():
        chunks.extend(_split_into_chunks(text, source))
    if not chunks:
        log.warning("В %s есть файлы, но чанков не получилось", STUDENT_DOCS_DIR)
        _collection = None
        return 0

    _collection = _client.create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine", "docs_hash": docs_hash},
    )
    _collection.add(
        ids=[f"student_{i}" for i in range(len(chunks))],
        documents=[_E5_DOC_PREFIX + c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in chunks],
    )
    log.info("Студенческий индекс собран: %d чанков", len(chunks))
    return len(chunks)


def student_kb_available() -> bool:
    return _collection is not None and _collection.count() > 0


def _retrieve(question: str) -> list[dict]:
    n = min(_TOP_K * 3, _collection.count())
    res = _collection.query(query_texts=[_E5_QUERY_PREFIX + question], n_results=n)
    candidates, seen = [], set()
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        text = doc[len(_E5_DOC_PREFIX):] if doc.startswith(_E5_DOC_PREFIX) else doc
        key = _dedup_key(text)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"text": text, "source": meta["source"], "heading": meta["heading"]})
    return _rerank(question, candidates, top_k=_TOP_K)


def _generate(question: str, context: str, lang: str) -> str:
    system = STUDENT_SYSTEM_PROMPT
    if lang == "en":
        system += "\n\nCRITICAL: This user writes in English. Your ENTIRE response MUST be in English only."
    user_content = (
        f"=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===\n{context}\n=== КОНЕЦ КОНТЕКСТА ===\n\n"
        f"Вопрос студента: {question}\n\n"
        "Дай чёткий ответ строго по контексту выше."
    )
    if USE_YANDEX:
        return yandex_complete([{"role": "user", "content": user_content}],
                               system=system, max_tokens=600, temperature=0.3)
    resp = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_content}],
        max_tokens=600, temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def ask_student(question: str, lang: str = "ru") -> tuple[str, dict]:
    """Ответ на вопрос студента. Возвращает (ответ, meta) — как ask_rag у абитуриентов."""
    chunks = _retrieve(question)
    meta = {"n_chunks": len(chunks), "sources": sorted({c["source"] for c in chunks})}
    if not chunks or not context_is_relevant(chunks):
        meta["low_relevance"] = bool(chunks)
        return (_NO_INFO_EN if lang == "en" else _NO_INFO_RU), meta
    context = "\n\n---\n\n".join(f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in chunks)
    return _generate(question, context, lang), meta
