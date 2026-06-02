import logging
import re
import requests
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

from config import (
    GROQ_API_KEY, GROQ_MODEL, RAG_TOP_K, DOCS_DIR,
    YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL, USE_YANDEX,
    ENABLE_LLM_REWRITE, EMBED_MODEL, RERANKER_MODEL, ENABLE_RERANKER,
    CHROMA_DIR, ENABLE_LLM_CLARIFY,
)
from knowledge_base import SYSTEM_PROMPT

log = logging.getLogger(__name__)

_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL,
    device="cpu",
)

_E5_DOC_PREFIX = "passage: "
_E5_QUERY_PREFIX = "query: "

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name="fkn_rag",
    embedding_function=_embed_fn,
    metadata={"hnsw:space": "cosine"},
)

def _is_qa_format(text: str) -> bool:
    bold_lines = sum(1 for ln in text.splitlines() if re.match(r"^\*\*.+\*\*\s*$", ln))
    return bold_lines >= 5

def _split_qa_chunks(text: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    section = ""
    question = ""
    answer_lines: list[str] = []

    def _flush():
        if not question or not answer_lines:
            return
        answer = "\n".join(answer_lines).strip()
        if not answer or "не найден" in answer.lower() or len(answer) < 15:
            return
        q_clean = re.sub(r"\*+", "", question).strip().rstrip("?")
        chunks.append({
            "text": f"{section}\nВопрос: {q_clean}\nОтвет: {answer}",
            "heading": q_clean[:80],
            "source": source,
        })

    for line in text.splitlines():
        if line.startswith("## "):
            _flush()
            section = line
            question = ""
            answer_lines = []
        elif re.match(r"^\*\*.+\*\*\s*$", line):
            _flush()
            question = line
            answer_lines = []
        elif line.startswith("# ") or line.startswith("---") or line.startswith("*Источник"):
            pass
        elif question and line.strip():
            answer_lines.append(line.strip())

    _flush()
    return chunks

def _split_into_chunks(text: str, source: str) -> list[dict]:
    if _is_qa_format(text):
        return _split_qa_chunks(text, source)

    has_subheadings = any(ln.startswith("### ") for ln in text.splitlines())
    if has_subheadings:
        return _split_with_subheadings(text, source)

    chunks: list[dict] = []
    h1_context = ""
    current_heading = ""
    current_lines: list[str] = []

    def _flush_chunk():
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if len(content) <= 80:
            return
        prefix = f"{h1_context}\n" if h1_context else ""
        chunks.append({
            "text": f"{prefix}{current_heading}\n{content}",
            "heading": current_heading.strip("# "),
            "source": source,
        })

    for line in text.strip().splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            _flush_chunk()
            h1_context = line
            current_heading = ""
            current_lines = []
        elif line.startswith("## "):
            _flush_chunk()
            current_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    _flush_chunk()
    return chunks

def _split_with_subheadings(text: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    h1_context = ""
    h2_heading = ""
    sub_heading = ""
    current_lines: list[str] = []

    def _flush():
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if len(content) <= 50:
            return
        heading_label = sub_heading or h2_heading
        ctx_parts = []
        if h1_context:
            ctx_parts.append(h1_context)
        if sub_heading and h2_heading:
            ctx_parts.append(h2_heading)
        prefix = "\n".join(ctx_parts) + "\n" if ctx_parts else ""
        chunks.append({
            "text": f"{prefix}{heading_label}\n{content}",
            "heading": heading_label.strip("# "),
            "source": source,
        })

    for line in text.strip().splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            _flush()
            h1_context = line
            h2_heading = ""
            sub_heading = ""
            current_lines = []
        elif line.startswith("## ") and not line.startswith("### "):
            _flush()
            h2_heading = line
            sub_heading = ""
            current_lines = []
        elif line.startswith("### "):
            _flush()
            sub_heading = line
            current_lines = []
        else:
            current_lines.append(line)

    _flush()
    return chunks

_DOCS_EXCLUDE_PATTERNS = ("qa_test", "_test", "test_")

def _load_docs() -> dict[str, str]:
    docs = {}
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"Не найдено ни одного .md файла в {DOCS_DIR}\n"
            "Положите файлы fkn_programs_rag.md и другие рядом с папкой bot/"
        )
    for path in md_files:
        stem_lower = path.stem.lower()
        if any(pat in stem_lower for pat in _DOCS_EXCLUDE_PATTERNS):
            log.info("Пропускаю тестовый файл: %s", path.stem)
            continue
        text = path.read_text(encoding="utf-8")
        docs[path.stem] = text
    return docs

def build_index() -> int:
    existing = _collection.count()
    if existing > 0:
        log.info("ChromaDB: существующий индекс (%d чанков), пропускаю сборку", existing)
        return existing

    docs = _load_docs()
    all_chunks: list[dict] = []
    for source, text in docs.items():
        chunks = _split_into_chunks(text, source)
        all_chunks.extend(chunks)
        log.info("%s: %d чанков", source, len(chunks))

    if not all_chunks:
        raise ValueError("Все .md файлы оказались пустыми или слишком короткими.")

    prefixed_docs = [_E5_DOC_PREFIX + c["text"] for c in all_chunks]
    _collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=prefixed_docs,
        metadatas=[{"source": c["source"], "heading": c["heading"]} for c in all_chunks],
    )
    log.info("ChromaDB: добавлено %d чанков", len(all_chunks))
    return len(all_chunks)

_SLANG_REPLACEMENTS = [
    (r"\bпми\b",          "прикладная математика и информатика"),
    (r"\bпад\b",          "прикладной анализ данных"),
    (r"\bпрад\b",         "прикладной анализ данных"),
    (r"\bэад\b",          "экономика и анализ данных"),
    (r"\bкнад\b",         "компьютерные науки и анализ данных"),
    (r"\bпрогинж\b",      "программная инженерия"),
    (r"\bдрип\b",         "дизайн и разработка информационных продуктов"),
    (r"\bминималк[иуа]\b",   "минимальные баллы"),
    (r"\bдок[иов]\b",        "документы"),
    (r"\bинфу\b",            "информатику"),
    (r"\bинфа\b",            "информатика"),
    (r"\bматан\b",           "математика"),
    (r"\bвышк[аеу]\b",       "ВШЭ"),
    (r"\bдопбалл\w*\b",      "дополнительные баллы"),
    (r"\bпрогу\b",           "программу"),
    (r"\bфизр[ауе]\b",       "физику"),
    (r"\bзавал\w+\s+инфу\b", "не сдам информатику"),
    (r"\bволонтёр\w*\b",     "волонтёрская деятельность добровольцев"),
    (r"\bкмс\b",             "кандидат в мастера спорта"),
    (r"\bмшп\b",             "московская школа программистов"),
    (r"\bяндекс\s+лице[йя]\b", "лицей академии яндекса"),
    (r"\bбольш\w*\s+перемен\w*\b", "конкурс большая перемена"),
    (r"\bпрод\s+олимпиад\w*\b",    "олимпиада PROD промышленная разработка"),
]

def _rewrite_query(question: str) -> str:
    q = question.lower()
    for pattern, replacement in _SLANG_REPLACEMENTS:
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return q

_BAD_WORD_PATTERNS = [
    r"хуй|хуе|хуя|хуев|нахуй",
    r"пизд",
    r"ёбан|еблан|ёбн",
    r"блять\b",
    r"блядь|бляд",
    r"мразь",
    r"пиздец",
    r"залуп",
    r"мудак|мудил",
    r"долбоёб|долбоеб",
    r"сук[аиу]\b",
    r"ублюдк",
    r"пошли\s+вы|иди\s+нахуй|идите\s+нахуй",
]

_OFFTOPIC_PATTERNS = [
    r"рецепт\w*\s+\w+",
    r"как\s+приготов",
    r"прогноз\s+погод",
    r"(какая|какую|какое|какой)\s+погода?\b",
    r"погода\s+в\s+\w+",
    r"(фильм|сериал|аниме|книг)\w*\s+посовет",
    r"посовет[уй]\w*\s+(фильм|сериал|книг|игр)",
    r"что\s+посмотреть",
    r"напиши\s+(\w+\s+)?(сочинен|эссе|реферат)",
    r"напиши\s+(\w+\s+)?(код|программ)\s",
    r"сделай\s+(домашн|код|программ)",
    r"расскажи\s+анекдот",
    r"кто\s+так(ой|ая)\s+\w+",
    r"курс\s+(рубл|доллар|евро|валют)",
    r"(достопримечательност|куда\s+сходить|куда\s+поехать).{0,30}(в|по)",
    r"заебал[оа]?\b",
    r"(посовет|посоветуй|порекоменд).{0,20}(сериал|фильм|игр[уы]|книг|музык|рестор)",
]

_ADMISSION_KEYWORDS = [
    "поступ", "абитуриент", "зачислен", "приём", "прием",
    "егэ", "экзамен", "балл", "минимальн", "проходн",
    "бюджет", "платн", "место", "конкурс",
    "программ", "факультет", "фкн", "вшэ", "вышк", "вуз",
    "олимпиад", "достижен", "медаль", "гто", "бви",
    "документ", "договор", "оплат", "стоимост",
    "срок", "дедлайн", "июл", "август", "июн",
    "иностранец", "овз", "инвалид",
    "математик", "информатик", "физик",
    "пми", "пад", "прад", "эад", "кнад", "прогинж", "дрип",
    "волонтёр", "спортивн", "кмс",
    "спо", "колледж", "техникум",
    "апелляц", "согласи", "приоритет", "списк",
]

def check_content(question: str) -> str | None:
    q = question.lower()

    if any(re.search(pat, q) for pat in _BAD_WORD_PATTERNS):
        return "Пожалуйста, общайтесь вежливо. Я готов помочь с вопросами о поступлении на ФКН ВШЭ."

    if any(re.search(pat, q) for pat in _OFFTOPIC_PATTERNS):
        return (
            "Я специализируюсь только на вопросах поступления в ФКН НИУ ВШЭ. "
            "Задайте вопрос об экзаменах, баллах, программах или сроках. "
            "Контакты: abitur@hse.ru, ba.hse.ru."
        )

    return None

_PROGRAMS_HINT = (
    "ПМИ, ПАД, КНАД, ЭАД, Программная инженерия, "
    "Разработка игр, Робототехника, ДРИП"
)

_PROGRAM_PATTERNS = [
    r"\bпми\b", r"\bпад\b", r"\bпрад\b", r"\bэад\b", r"\bкнад\b", r"\bпрогинж\b", r"\bдрип\b",
    r"прикладн\w*\s+матем\w*", r"прикладн\w*\s+анализ",
    r"компьютерн\w*\s+науки", r"программн\w*\s+инженер\w*",
    r"разраб\w*\s+игр\w*", r"робототехник\w*",
    r"дизайн\s+и\s+разраб\w*", r"экономик\w+\s+и\s+анализ",
]

_CLARIFICATION_RULES = [
    {
        "triggers": ["егэ нужн", "что сдавать", "какие предмет", "какой предмет",
                     "какие экзамен", "какой экзамен", "егэ надо",
                     "что нужно сдат", "что надо сдат"],
        "excludes": [
            "овз", "инвалид", "слеп", "глух", "слабовид", "нарушен", "ассистент",
            "иностранц", "иностран",
        ],
        "needs_program": True,
        "question": f"На какую программу ФКН вас интересуют вступительные испытания?\n({_PROGRAMS_HINT})",
    },
    {
        "triggers": ["минималь", "минималк", "сколько балл", "проходн",
                     "мин балл", "нужно балл", "нужны балл", "какой балл нужен",
                     "какие балл", "балл для поступ"],
        "excludes": [
            "математик", "информатик", "русский", "физик", "английск", "обществознан",
            "достижен", "волонтёр", "спортив", "кмс", "мастер спорт",
            "медал", "аттестат", "олимпи", "конкурс", "гто", "военн",
            "абилимпикс", "лицей", "яндекс", "т-поколен", "prod", "высший пилотаж",
            "звани", "кандидат", "разряд", "чемпион",
            "дополнительн", "доп балл", "доп. балл", "инд", "за волонтёр",
            "совпадает", "реальн проходн", "реальный проходн", "чем отлич",
        ],
        "needs_program": True,
        "question": f"По какой программе ФКН вас интересуют минимальные баллы?\n({_PROGRAMS_HINT})",
    },
    {
        "triggers": ["всош", "олимпиад школьников", "призер олимпиад", "призёр олимпиад",
                     "диплом олимпиад", "бви", "без вступительных", "перечневая"],
        "excludes": [
            "что такое", "что это", "можно ли использ", "нескольк",
            "право", "условия", "кто имеет", "как работает",
            "а не бви", "не использ", "вместо бви", "за это баллы",
        ],
        "needs_program": True,
        "question": (
            f"Уточните, пожалуйста, на какую программу ФКН вы планируете поступать?\n({_PROGRAMS_HINT})\n\n"
            "Это важно — список засчитываемых олимпиад и условия БВИ отличаются по программам."
        ),
    },
    {
        "triggers": ["до какого числа", "когда подавать", "срок подачи", "дедлайн",
                     "последний день подач", "когда последн"],
        "excludes": [
            "бюджет", "платн", "договор", "зачислен",
            "приоритет", "госуслуг", "очерёдн", "порядок",
        ],
        "needs_program": False,
        "question": "Уточните:\n— Вы рассматриваете бюджетные места?\n— Или платные места?\n\nСроки подачи документов для них разные.",
    },
]

_ASKED_MARKERS = ["уточните", "на какую программу", "какую программу", "какие места"]

def _program_mentioned(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _PROGRAM_PATTERNS)

def needs_clarification(question: str, history: list[dict]) -> str | None:
    q = question.lower()
    dialog = " ".join(m["content"].lower() for m in history) + " " + q

    last_bot = next((m["content"].lower() for m in reversed(history) if m["role"] == "assistant"), "")
    if any(m in last_bot for m in _ASKED_MARKERS):
        return None

    for rule in _CLARIFICATION_RULES:
        if not any(t in q for t in rule["triggers"]):
            continue
        if any(e in q for e in rule["excludes"]):
            continue
        if rule["needs_program"] and _program_mentioned(dialog):
            continue
        if any(m in q for m in ["на фкн", "вообще", "все программы", "в целом", "перечисли",
                                "сравни", "список программ", "все программы", "по всем"]):
            continue
        return rule["question"]

    return None

_CLARIFY_PROMPT = (
    "Ты — фильтр уточняющих вопросов для бота-консультанта по поступлению на ФКН НИУ ВШЭ.\n"
    "Тебе дают вопрос абитуриента (и при наличии — предыдущий диалог).\n"
    "Реши: можно ли дать конкретный полезный ответ, или вопросу не хватает ключевой детали.\n\n"
    "Уточнение НУЖНО, если без детали ответ будет неоднозначным, например:\n"
    "- спрашивают про минимальные/проходные баллы, вступительные испытания, БВИ, олимпиады, "
    "скидки — но НЕ указана программа ФКН (ПМИ, ПАД, КНАД, ЭАД, Программная инженерия, "
    "Робототехника, ДРИП, Разработка игр);\n"
    "- спрашивают про сроки подачи или зачисление — но не ясно, бюджет или платное.\n\n"
    "Уточнение НЕ нужно, если:\n"
    "- вопрос общий по сути (что такое квазибюджет, как работает БВИ, перечисли программы, сравни X и Y);\n"
    "- нужная деталь уже есть в вопросе ИЛИ в предыдущем диалоге.\n\n"
    "Ответь СТРОГО так:\n"
    "- если всё понятно для ответа — ровно одно слово: OK\n"
    "- если нужно уточнение — одну короткую уточняющую фразу на языке вопроса, без преамбул и кавычек."
)

def _clarify_llm_call(prompt: str, user_text: str) -> str:
    if USE_YANDEX:
        return yandex_complete(
            messages=[{"role": "user", "content": user_text}],
            system=prompt, max_tokens=80, temperature=0.1,
        )
    resp = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        max_tokens=80, temperature=0.1,
    )
    return (resp.choices[0].message.content or "").strip()

def llm_clarify(question: str, history: list[dict], lang: str = "ru") -> str | None:
    """LLM читает вопрос и, если не хватает ключевой детали, возвращает уточняющий вопрос.

    Fallback после rule-based needs_clarification. Возвращает None, если уточнение не нужно.
    """
    if not ENABLE_LLM_CLARIFY or not (USE_YANDEX or GROQ_API_KEY):
        return None

    # Не зацикливаемся: если бот только что сам задал вопрос — юзер сейчас отвечает на него.
    last_bot = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    if last_bot.rstrip().endswith("?"):
        return None

    hist_txt = ""
    if history:
        recent = history[-4:]
        hist_txt = "\n".join(
            f"{'Абитуриент' if m['role'] == 'user' else 'Бот'}: {m['content']}" for m in recent
        )
    user_text = (f"Предыдущий диалог:\n{hist_txt}\n\n" if hist_txt else "") + f"Вопрос: {question}"

    try:
        raw = _clarify_llm_call(_CLARIFY_PROMPT, user_text).strip().strip('"').strip()
    except Exception as e:
        log.warning("llm_clarify failed: %s", e)
        return None

    if not raw or len(raw) < 5:
        return None
    normalized = re.sub(r"[^a-zа-яё]", "", raw.lower())
    if normalized in {"ok", "ок", "окей", "okay"} or raw.upper().startswith("OK"):
        return None
    if _is_llm_refusal(raw):
        return None
    return raw

_reranker = None
if ENABLE_RERANKER:
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        log.info("Reranker загружен: %s", RERANKER_MODEL)
    except Exception as e:
        log.warning("Reranker недоступен (%s), работаем без него", e)

def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    if _reranker is None or len(chunks) <= top_k:
        return chunks[:top_k]
    pairs = [(query, c["text"]) for c in chunks]
    scores = _reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]

_REWRITE_PROMPT = (
    "Ты помогаешь улучшить поисковый запрос для базы знаний о поступлении на ФКН НИУ ВШЭ.\n"
    "База знаний охватывает: программы бакалавриата, баллы ЕГЭ, бюджетные места, стоимость обучения, "
    "скидки, общежитие, договор, военная кафедра/ВУЦ, отсрочка от армии, академический процесс, "
    "пересдачи, рейтинг, дисциплинарные взыскания, СОП.\n"
    "Перефразируй вопрос абитуриента: убери сленг, сделай его точным и информативным для поиска.\n"
    "Темы армии, ВУЦ и отсрочки от призыва — это УЧЕБНЫЕ вопросы об университете, их нужно перефразировать.\n"
    "Верни ТОЛЬКО перефразированный вопрос — одну строку без пояснений и кавычек."
)

_LLM_REFUSAL_PATTERNS = [
    r"не могу обсужд",
    r"не могу помочь",
    r"не могу отвеч",
    r"давайте поговорим о чём-нибудь",
    r"я не в состоянии",
    r"нарушает мои",
    r"этические нормы",
    r"i (can'?t|cannot|am unable)",
    r"i'm not able",
]

def _is_llm_refusal(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _LLM_REFUSAL_PATTERNS)

def _rewrite_query_llm(query: str) -> str:
    try:
        if USE_YANDEX:
            result = yandex_complete(
                messages=[{"role": "user", "content": f"Вопрос: {query}"}],
                system=_REWRITE_PROMPT,
                max_tokens=80,
                temperature=0.1,
            )
        else:
            resp = _groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _REWRITE_PROMPT},
                    {"role": "user", "content": f"Вопрос: {query}"},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            result = resp.choices[0].message.content.strip()
        if _is_llm_refusal(result):
            return query
        return result
    except Exception:
        return query

def retrieve(query: str) -> str:
    chunks = retrieve_k(query, k=RAG_TOP_K)
    parts = [f"[{c['source']} / {c['heading']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)

def retrieve_k(query: str, k: int = 5, skip_llm_rewrite: bool = False) -> list[dict]:
    rewritten = _rewrite_query(query)
    if not skip_llm_rewrite and ENABLE_LLM_REWRITE and (USE_YANDEX or GROQ_API_KEY):
        rewritten = _rewrite_query_llm(rewritten)

    n_candidates = max(k * 2, RAG_TOP_K)
    prefixed_query = _E5_QUERY_PREFIX + rewritten
    results = _collection.query(query_texts=[prefixed_query], n_results=n_candidates)

    candidates = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        clean_doc = doc[len(_E5_DOC_PREFIX):] if doc.startswith(_E5_DOC_PREFIX) else doc
        candidates.append({
            "text": clean_doc,
            "source": meta["source"],
            "heading": meta["heading"],
            "rewritten": rewritten,
        })

    return _rerank(rewritten, candidates, top_k=k)

_YANDEX_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

def yandex_complete(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 600,
    temperature: float = 0.3,
) -> str:
    model_uri = f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}/latest"
    yandex_msgs = []
    if system:
        yandex_msgs.append({"role": "system", "text": system})
    for m in messages:
        yandex_msgs.append({"role": m["role"], "text": m.get("content", m.get("text", ""))})

    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        },
        "messages": yandex_msgs,
    }
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
    }
    resp = requests.post(_YANDEX_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"].strip()

_groq = Groq(api_key=GROQ_API_KEY)

def generate(
    question: str,
    context: str,
    history: list[dict],
    lang: str = "ru",
    citizenship: str = "",
) -> str:
    system = SYSTEM_PROMPT
    if lang == "en":
        system += "\n\nCRITICAL: This user writes in English. Your ENTIRE response MUST be in English only."

    citizenship_note = f"[Статус абитуриента: {citizenship}]\n" if citizenship else ""
    lang_note = "[ANSWER IN ENGLISH]\n" if lang == "en" else ""

    user_content = (
        f"=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===\n{context}\n=== КОНЕЦ КОНТЕКСТА ===\n\n"
        f"{citizenship_note}{lang_note}"
        f"Вопрос абитуриента: {question}\n\n"
        "Дай чёткий ответ строго по контексту выше."
    )

    if USE_YANDEX:
        msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
        msgs.append({"role": "user", "content": user_content})
        return yandex_complete(msgs, system=system, max_tokens=600, temperature=0.3)

    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def ask_rag(
    question: str,
    history: list[dict],
    lang: str = "ru",
    citizenship: str = "",
) -> str:
    context = retrieve(question)
    if not context.strip():
        if lang == "en":
            return (
                "No information found in the knowledge base for this question. "
                "Please check ba.hse.ru or email abitur@hse.ru."
            )
        return (
            "В базе знаний нет информации по этому вопросу. "
            "Уточните на сайте ba.hse.ru или напишите на abitur@hse.ru."
        )
    return generate(question, context, history, lang=lang, citizenship=citizenship)
