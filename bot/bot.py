import re
import json
import asyncio
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import time

from telegram import (
    Update, BotCommand,
    InlineKeyboardButton, InlineKeyboardMarkup,
    InlineQueryResultArticle, InputTextMessageContent,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    InlineQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction

from config import (
    TELEGRAM_TOKEN, MAX_HISTORY_PAIRS,
    RATE_LIMIT_MAX, RATE_LIMIT_WINDOW,
    DISLIKE_LOG, WEBHOOK_URL, WEBHOOK_PORT,
    ADMIN_USER_ID, QUARANTINE_FILE,
    USE_AGENTIC_RAG,
)
from faq_cache import lookup as faq_lookup, add_to_dynamic_cache, quarantine_answer, get_cache_stats
from rag_engine import build_index, check_content, needs_clarification, llm_clarify
from agentic_rag import agentic_ask


if USE_AGENTIC_RAG:
    from agentic_loop import agentic_loop_ask as ask_rag
else:
    ask_rag = agentic_ask
from calculator import try_calculate, calculate, SUBJ_OPTIONS, SUBJ_KW
from bot_events import log_interaction, log_vote
from speech import transcribe_ogg

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_history: dict[int, list[dict]] = defaultdict(list)

_vote_pending: dict[int, tuple[str, str, str]] = {}
_MAX_VOTE_PENDING = 500

_user_citizenship: dict[int, str] = {}

_pending_question: dict[int, str] = {}

_calc_state: dict[int, dict] = {}

_user_timestamps: dict[int, deque] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX))

_stats = {
    "start_time": datetime.now(),
    "cache_hits": defaultdict(int),
    "rag_calls": 0,
    "likes": 0,
    "dislikes": 0,
    "rate_limited": 0,
}

def _detect_lang(text: str) -> str:
    clean = re.sub(r"[^a-zа-яё]", "", text.lower())
    if not clean:
        return "ru"
    latin = sum(1 for c in clean if "a" <= c <= "z")
    return "en" if latin / len(clean) >= 0.6 and latin >= 3 else "ru"

def _to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text

_VOTE_KB = InlineKeyboardMarkup([[
    InlineKeyboardButton("👍 Верно", callback_data="vote_up"),
    InlineKeyboardButton("👎 Неверно", callback_data="vote_down"),
]])

_MAX_MSG_LEN = 4000

def _is_rate_limited(user_id: int) -> bool:
    now = time.monotonic()
    dq = _user_timestamps[user_id]
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX:
        _stats["rate_limited"] += 1
        return True
    dq.append(now)
    return False

_GREETING_WORDS = {
    "привет", "приветствую", "здравствуй", "здравствуйте",
    "добрый", "доброе", "хай", "хей", "hey", "hi", "hello", "yo",
}
_THANKS_WORDS = {
    "спасибо", "спасибки", "спс", "благодарю", "благодарен",
    "благодарна", "thanks", "thank", "сяп", "пасиб",
}

def _is_greeting(text: str) -> bool:
    words = set(re.sub(r"[^а-яёa-z\s]", " ", text.lower()).split())
    return bool(words & _GREETING_WORDS) and len(words) <= 5

def _is_thanks(text: str) -> bool:
    words = set(re.sub(r"[^а-яёa-z\s]", " ", text.lower()).split())
    return bool(words & _THANKS_WORDS)

def _add_to_history(user_id: int, role: str, content: str) -> None:
    history = _history[user_id]
    history.append({"role": role, "content": content})
    max_msgs = MAX_HISTORY_PAIRS * 2
    if len(history) > max_msgs:
        _history[user_id] = history[-max_msgs:]

def _get_history(user_id: int) -> list[dict]:
    return list(_history[user_id])

_CLARIFICATION_MARKERS = [
    "уточните", "на какую программу", "какую программу",
    "какие места", "вашу ситуацию", "что именно вас интересует",
]

def _build_effective_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    if not any(m in last_assistant.lower() for m in _CLARIFICATION_MARKERS):
        return question
    prev_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    return f"{prev_user} — {question}" if prev_user else question

def _log_dislike(question: str) -> None:
    try:
        with DISLIKE_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f"[{ts}] {question}\n")
    except Exception as e:
        log.warning("Не удалось записать дизлайк: %s", e)

_CITIZENSHIP_KB = InlineKeyboardMarkup([[
    InlineKeyboardButton("🇷🇺 Гражданин РФ", callback_data="citizen_rf"),
    InlineKeyboardButton("🌍 Иностранный", callback_data="citizen_foreign"),
]])

_CITIZENSHIP_KB_EN = InlineKeyboardMarkup([[
    InlineKeyboardButton("🇷🇺 Russian citizen", callback_data="citizen_rf"),
    InlineKeyboardButton("🌍 Foreign citizen", callback_data="citizen_foreign"),
]])

_ACTION_KB = InlineKeyboardMarkup([[
    InlineKeyboardButton("🧮 Калькулятор ЕГЭ", callback_data="action_calc"),
    InlineKeyboardButton("💬 Задать вопрос", callback_data="action_question"),
]])

_ACTION_KB_EN = InlineKeyboardMarkup([[
    InlineKeyboardButton("🧮 EGE Calculator", callback_data="action_calc"),
    InlineKeyboardButton("💬 Ask a question", callback_data="action_question"),
]])

def _build_subject_kb() -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(SUBJ_OPTIONS), 2):
        row = [
            InlineKeyboardButton(
                f"{em} {label}",
                callback_data=f"calc_subj_{sid}",
            )
            for sid, em, label, _ in SUBJ_OPTIONS[i:i+2]
        ]
        rows.append(row)
    rows.append([InlineKeyboardButton("❌ Отмена", callback_data="calc_cancel")])
    return InlineKeyboardMarkup(rows)

_CANCEL_KB = InlineKeyboardMarkup([[
    InlineKeyboardButton("❌ Отмена", callback_data="calc_cancel"),
]])

CITIZENSHIP_QUESTION = (
    "Прежде чем начать — уточните, пожалуйста:\n\n"
    "<b>Вы гражданин РФ или иностранный гражданин?</b>\n\n"
    "Это важно: у иностранных абитуриентов другой набор вступительных испытаний и отдельный конкурс."
)

CITIZENSHIP_QUESTION_EN = (
    "Before we start — please clarify:\n\n"
    "<b>Are you a Russian citizen or a foreign citizen?</b>\n\n"
    "This matters: foreign applicants have different entrance exams and a separate competition."
)

def _citizenship_prompt(lang: str) -> tuple[str, InlineKeyboardMarkup]:
    if lang == "en":
        return CITIZENSHIP_QUESTION_EN, _CITIZENSHIP_KB_EN
    return CITIZENSHIP_QUESTION, _CITIZENSHIP_KB

WELCOME_TEXT = (
    "Привет! Я бот-консультант по поступлению на "
    "<b>Факультет компьютерных наук НИУ ВШЭ</b> 🎓\n\n"
    "Могу ответить на вопросы о:\n"
    "• программах ФКН (ПМИ, ПАД, КНАД, ЭАД и другие)\n"
    "• минимальных баллах ЕГЭ и сроках подачи\n"
    "• БВИ, квазибюджете и зелёной волне\n"
    "• индивидуальных достижениях и скидках\n\n"
    "/reset — сбросить историю · /stats — статистика"
)

HELP_TEXT = (
    "<b>Команды:</b>\n"
    "/start — приветствие и выбор гражданства\n"
    "/reset — очистить историю\n"
    "/stats — статистика бота\n"
    "/help — справка\n\n"
    "<b>Как работает:</b>\n"
    "• Кешированные ответы — мгновенно\n"
    "• Остальные — поиск по базе знаний + Groq AI\n"
    "• Все ответы можно оценить 👍/👎\n"
    "• 👎 на кешированный ответ → карантин для проверки\n"
    "• Работает в inline-режиме: @бот вопрос"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _history[user_id].clear()
    _user_citizenship[user_id] = ""
    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"
    cq, ckb = _citizenship_prompt(lang)
    await update.message.reply_text(WELCOME_TEXT, parse_mode="HTML")
    await update.message.reply_text(cq, parse_mode="HTML", reply_markup=ckb)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _history[user_id].clear()
    _user_citizenship[user_id] = ""
    _pending_question.pop(user_id, None)
    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"
    cq, ckb = _citizenship_prompt(lang)
    prefix = "✅ History cleared.\n\n" if lang == "en" else "✅ История очищена.\n\n"
    await update.message.reply_text(prefix + cq, parse_mode="HTML", reply_markup=ckb)

async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not ADMIN_USER_ID or update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not QUARANTINE_FILE.exists():
        await update.message.reply_text("Карантин пуст.")
        return

    lines = QUARANTINE_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        await update.message.reply_text("Карантин пуст.")
        return

    recent = lines[-10:]
    text = f"<b>🔍 Карантин</b> (всего {len(lines)}, последние {len(recent)}):\n\n"
    for line in recent:
        try:
            rec = json.loads(line)
            q = rec["question"][:80]
            text += f"[{rec['ts']}] <i>{rec['source']}</i>\n<b>Q:</b> {q}\n\n"
        except Exception:
            continue

    await update.message.reply_text(text[:_MAX_MSG_LEN], parse_mode="HTML")

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uptime = datetime.now() - _stats["start_time"]
    h, rem = divmod(int(uptime.total_seconds()), 3600)
    m = rem // 60

    hits = _stats["cache_hits"]
    total_cache = sum(hits.values())
    total_rag = _stats["rag_calls"]
    total = total_cache + total_rag
    pct = round(total_cache / total * 100) if total else 0

    cs = get_cache_stats()
    pipeline = "agentic-loop" if USE_AGENTIC_RAG else "декомпозиция"
    text = (
        f"<b>📊 Статистика</b>\n\n"
        f"⏱ Uptime: {h}ч {m}м\n"
        f"⚙️ RAG: {pipeline}\n\n"
        f"<b>Запросы:</b>\n"
        f"  Всего: {total}\n"
        f"  Кеш: {total_cache} ({pct}%)\n"
        f"    static FAQ: {hits['hash'] + hits['fuzzy']}\n"
        f"    dynamic: {hits['dynamic']}\n"
        f"  RAG (Groq): {total_rag}\n\n"
        f"<b>Оценки:</b>\n"
        f"  👍 {_stats['likes']}  👎 {_stats['dislikes']}\n\n"
        f"<b>Кеш:</b>\n"
        f"  Сохранённых: {cs['dynamic_count']}\n"
        f"  В блэклисте: {cs['blacklist_count']}\n"
        f"  Rate-limited: {_stats['rate_limited']}"
    )
    await update.message.reply_text(text, parse_mode="HTML")

async def handle_citizenship(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    tg_lang = (query.from_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"

    if query.data == "citizen_rf":
        _user_citizenship[user_id] = "гражданин РФ"
        label = "🇷🇺 Гражданин РФ" if lang != "en" else "🇷🇺 Russian citizen"
    else:
        _user_citizenship[user_id] = "иностранный гражданин"
        label = "🌍 Иностранный гражданин" if lang != "en" else "🌍 Foreign citizen"

    if lang == "en":
        text = (
            f"✅ Status saved: <b>{label}</b>\n\n"
            "What would you like to do?"
        )
        kb = _ACTION_KB_EN
    else:
        text = (
            f"✅ Статус сохранён: <b>{label}</b>\n\n"
            "Что хотите сделать?"
        )
        kb = _ACTION_KB

    await query.edit_message_text(text, parse_mode="HTML", reply_markup=kb)
    log.info("Citizenship set for user %d: %s", user_id, _user_citizenship[user_id])

    pending_q = _pending_question.pop(user_id, None)
    if pending_q:
        log.info("Processing pending question after citizenship: '%s'", pending_q[:60])
        await _process_question(update.callback_query.message, user_id, pending_q)

async def handle_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    tg_lang = (query.from_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"

    await query.edit_message_reply_markup(reply_markup=None)

    if query.data == "action_calc":
        _calc_state[user_id] = {"step": "subject", "subj_id": None,
                                 "scores": None, "achievements": set()}
        await query.message.reply_text(
            "🧮 <b>Калькулятор конкурсного балла</b>\n\n"
            "Подберу топ‑3 программы ФКН и сравню с проходными баллами 2024 года.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Шаг 1 из 3</b> — Какой второй предмет ЕГЭ?",
            parse_mode="HTML",
            reply_markup=_build_subject_kb(),
        )
    else:
        text = (
            "Отлично! Задайте ваш вопрос о поступлении на ФКН — я отвечу."
            if lang != "en" else
            "Go ahead — ask your question about admission to FCS!"
        )
        await query.message.reply_text(text)

async def handle_calc_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data
    state = _calc_state.get(user_id)

    if data == "calc_noop":
        return

    if data == "calc_cancel":
        _calc_state.pop(user_id, None)
        await query.edit_message_text("✖️ Расчёт отменён. Можешь задать вопрос текстом.")
        return

    if state is None:
        await query.edit_message_text("Сессия истекла. Нажми /start или напиши вопрос.")
        return

    if data.startswith("calc_subj_"):
        subj_id = data.removeprefix("calc_subj_")
        subj_em, subj_name = next(
            ((em, lb) for s, em, lb, _ in SUBJ_OPTIONS if s == subj_id),
            ("", subj_id),
        )
        state["subj_id"] = subj_id
        state["subj_em"] = subj_em
        state["subj_name"] = subj_name
        state["step"] = "scores"
        _calc_state[user_id] = state
        await query.edit_message_text(
            f"🧮 <b>Калькулятор конкурсного балла</b>\n\n"
            f"Подберу топ‑3 программы ФКН и сравню с проходными баллами 2024 года.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Предмет: <b>{subj_em} {subj_name}</b>\n\n"
            f"<b>Шаг 2 из 3</b> — Введи три балла ЕГЭ одним сообщением:\n\n"
            f"  📐 математика  {subj_em} {subj_name[:5].lower()}  📝 русский\n\n"
            f"Например: <code>87 82 74</code>",
            parse_mode="HTML",
            reply_markup=_CANCEL_KB,
        )
        return

async def handle_vote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    msg_id = query.message.message_id
    pending = _vote_pending.pop(msg_id, None)

    await query.edit_message_reply_markup(reply_markup=None)

    if query.data == "vote_up":
        _stats["likes"] += 1
        log_vote(msg_id, "up")
        if pending:
            q, a, src = pending
            if src == "rag":
                add_to_dynamic_cache(q, a)
                log.info("Vote 👍 (RAG→cache): '%s'", q[:60])
            else:
                log.info("Vote 👍 (cache confirmed): '%s'", q[:60])
        await query.message.reply_text("✅ Спасибо за оценку!")

    else:
        _stats["dislikes"] += 1
        log_vote(msg_id, "down")
        if pending:
            q, a, src = pending
            if src == "rag":
                _log_dislike(q)
                log.info("Vote 👎 (RAG): '%s'", q[:60])
                await query.message.reply_text(
                    "👎 Понял. Попробуйте переформулировать или уточните детали.\n"
                    "Можно написать напрямую: abitur@hse.ru"
                )
            else:
                quarantine_answer(q, a, src)
                log.info("Vote 👎 (cache→quarantine [%s]): '%s'", src, q[:60])
                await query.message.reply_text(
                    "👎 Понял, этот ответ убран из кеша и отправлен на проверку. "
                    "Задайте вопрос ещё раз — поищу в базе знаний."
                )

async def handle_inline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query_text = (update.inline_query.query or "").strip()
    if len(query_text) < 3:
        await update.inline_query.answer([], cache_time=0)
        return

    answer, source = faq_lookup(query_text)
    if not answer:
        try:
            lang = _detect_lang(query_text)
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, ask_rag, query_text, [], lang, ""
                ),
                timeout=8.0,
            )
            answer, _ = result
        except asyncio.TimeoutError:
            answer = "Открой чат с ботом для полного ответа."

    results = [
        InlineQueryResultArticle(
            id=str(uuid.uuid4()),
            title=query_text[:60] + ("…" if len(query_text) > 60 else ""),
            description=re.sub(r"<[^>]+>", "", answer)[:120],
            input_message_content=InputTextMessageContent(
                _to_html(answer), parse_mode="HTML"
            ),
        )
    ]
    await update.inline_query.answer(results, cache_time=300)

async def _keep_typing(chat, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await chat.send_action(ChatAction.TYPING)
        except Exception:
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass

async def _process_question(message, user_id: int, question: str) -> None:
    lang = _detect_lang(question)
    citizenship = _user_citizenship.get(user_id, "")
    history = _get_history(user_id)
    t0 = time.monotonic()

    blocked = check_content(question)
    if blocked:
        await message.reply_text(_to_html(blocked), parse_mode="HTML")
        return

    calc_answer = try_calculate(question)
    if calc_answer is not None:
        log.info("Calculator hit for user %d", user_id)
        _add_to_history(user_id, "user", question)
        _add_to_history(user_id, "assistant", calc_answer)
        await message.reply_text(_to_html(calc_answer), parse_mode="HTML")
        return

    cached_answer, source = faq_lookup(question)
    if cached_answer:
        log.info("Cache hit [%s] user %d", source, user_id)
        _stats["cache_hits"][source] += 1
        _add_to_history(user_id, "user", question)
        _add_to_history(user_id, "assistant", cached_answer)
        sent = await _send_msg_with_vote(message, question, cached_answer, source, lang)
        log_interaction(
            user_id=user_id, question=question,
            route="faq", source=source,
            answer=cached_answer, sub_queries=[],
            lang=lang, latency_ms=(time.monotonic() - t0) * 1000,
            msg_id=sent.message_id,
        )
        return

    clarification = needs_clarification(question, history)
    if not clarification:
        clarification = await asyncio.get_running_loop().run_in_executor(
            None, llm_clarify, question, history, lang
        )
    if clarification:
        log.info("Clarification needed for user %d", user_id)
        _add_to_history(user_id, "user", question)
        _add_to_history(user_id, "assistant", clarification)
        await message.reply_text(_to_html(clarification), parse_mode="HTML")
        return

    effective_question = _build_effective_question(question, history)
    _stats["rag_calls"] += 1
    log.info("RAG [lang=%s, citizenship=%s] user %d", lang, citizenship or "?", user_id)

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.chat, stop_typing))
    meta: dict = {}
    try:
        answer, meta = await asyncio.get_running_loop().run_in_executor(
            None, ask_rag, effective_question, history, lang, citizenship
        )
    except Exception as e:
        log.error("RAG error for user %d: %s", user_id, e)
        answer = (
            "Произошла техническая ошибка. Попробуйте позже.\n"
            "Контакты: abitur@hse.ru · ba.hse.ru"
        )
    finally:
        stop_typing.set()
        typing_task.cancel()

    _add_to_history(user_id, "user", question)
    _add_to_history(user_id, "assistant", answer)
    sent = await _send_msg_with_vote(message, effective_question, answer, "rag", lang)
    log_interaction(
        user_id=user_id, question=question,
        route="rag", source="rag",
        answer=answer, sub_queries=meta.get("sub_queries", []),
        lang=lang, latency_ms=(time.monotonic() - t0) * 1000,
        msg_id=sent.message_id,
    )

_DISCLAIMER = (
    "\n\n<i>⚠️ Поступариум — это ИИ-консультант и может ошибаться. "
    "Проверяйте важную информацию на ba.hse.ru "
    "или в приёмной комиссии (abitur@hse.ru).</i>"
)
_DISCLAIMER_EN = (
    "\n\n<i>⚠️ Postuparium is an AI assistant and can make mistakes. "
    "Please verify important information at ba.hse.ru "
    "or with the admissions office (abitur@hse.ru).</i>"
)

async def _send_msg_with_vote(message, question: str, answer: str, source: str, lang: str = "ru"):
    disclaimer = _DISCLAIMER_EN if lang == "en" else _DISCLAIMER
    text = _to_html(answer)
    budget = _MAX_MSG_LEN - len(disclaimer)
    if len(text) > budget:
        cut = text.rfind("\n", 0, budget - 80)
        if cut < 200:
            cut = budget - 80
        text = text[:cut] + "\n\n<i>… ответ сокращён. Уточните детали в ba.hse.ru</i>"
    text += disclaimer
    msg = await message.reply_text(text, parse_mode="HTML", reply_markup=_VOTE_KB)
    if len(_vote_pending) >= _MAX_VOTE_PENDING:
        del _vote_pending[next(iter(_vote_pending))]
    _vote_pending[msg.message_id] = (question, answer, source)
    return msg

async def _handle_wizard_scores(message, user_id: int, question: str, calc: dict) -> None:
    nums = re.findall(r'\d+', question)
    if len(nums) < 3:
        await message.reply_text(
            "Нужны три числа — например: <code>87 82 74</code>",
            parse_mode="HTML",
        )
        return
    scores = [int(nums[0]), int(nums[1]), int(nums[2])]
    if not all(0 <= s <= 100 for s in scores):
        await message.reply_text(
            "Каждый балл должен быть от 0 до 100. Попробуй ещё раз."
        )
        return
    subj_em = calc.get("subj_em", "")
    subj_name = calc.get("subj_name", calc["subj_id"])
    calc["scores"] = scores
    calc["step"] = "achievements"
    _calc_state[user_id] = calc
    s0, s1, s2 = scores
    await message.reply_text(
        f"🧮 <b>Калькулятор конкурсного балла</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📐 Математика — <b>{s0}</b>\n"
        f"  {subj_em} {subj_name} — <b>{s1}</b>\n"
        f"  📝 Русский язык — <b>{s2}</b>\n"
        f"  ──────────────────\n"
        f"  <b>Σ ЕГЭ — {s0 + s1 + s2} баллов</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"<b>Шаг 3 из 3</b> — Есть индивидуальные достижения?\n\n"
        f"Напиши в свободной форме:\n"
        f"<i>медаль за особые успехи, Лицей Яндекса</i>\n"
        f"<i>ГТО золотой, КМС, Абилимпикс</i>\n\n"
        f"Если нет — напиши <b>нет</b>",
        parse_mode="HTML",
        reply_markup=_CANCEL_KB,
    )

async def _handle_wizard_achievements(message, user_id: int, question: str, calc: dict) -> None:
    ach_text = question.strip()
    scores = calc["scores"]
    subj_id = calc["subj_id"]
    _calc_state.pop(user_id, None)

    if re.match(r'^нет$|^no$|^0$|^-$', ach_text, re.I):
        ach_text = ""

    result = calculate(
        math=scores[0],
        second_subject=SUBJ_KW.get(subj_id, ""),
        second_score=scores[1],
        russian=scores[2],
        achievements_text=ach_text.lower(),
    )
    await message.reply_text(_to_html(result), parse_mode="HTML")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    addressed, _ = _is_addressed_in_group(update, context)
    if not addressed:
        return

    user_id = update.effective_user.id
    lang = _detect_lang(_user_citizenship.get(user_id, ""))

    if _is_rate_limited(user_id):
        msg = "⏳ Too many requests. Please wait a moment." if lang == "en" else "⏳ Слишком много запросов. Подождите немного."
        await update.message.reply_text(msg)
        return

    voice = update.message.voice
    if voice.duration > 60:
        await update.message.reply_text(
            "🎙 Голосовое сообщение слишком длинное (максимум 60 секунд). Напишите вопрос текстом."
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    tg_file = await context.bot.get_file(voice.file_id)
    audio_bytes = await tg_file.download_as_bytearray()

    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    stt_lang = "en-US" if tg_lang == "en" else "ru-RU"
    question = await transcribe_ogg(bytes(audio_bytes), lang=stt_lang)

    if not question:
        await update.message.reply_text(
            "😔 Не удалось распознать речь. Попробуйте ещё раз или напишите вопрос текстом."
        )
        return

    log.info("Voice→text user %d: %s", user_id, question[:80])

    if not _user_citizenship.get(user_id):
        _pending_question[user_id] = question
        cq, ckb = _citizenship_prompt("en" if tg_lang == "en" else "ru")
        await update.message.reply_text(cq, parse_mode="HTML", reply_markup=ckb)
        return

    await _process_question(update.message, user_id, question)


def _is_addressed_in_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, str]:
    """В группах бот отвечает только если его упомянули @ником или ответили на его сообщение.

    В личных чатах отвечает всегда. Возвращает (отвечать_ли, очищенный_от_@ника_текст).
    """
    msg = update.effective_message
    text = (msg.text or msg.caption or "").strip()
    chat = update.effective_chat
    if chat is None or chat.type not in ("group", "supergroup"):
        return True, text

    bot_username = context.bot.username or ""
    mention = f"@{bot_username}"
    reply_to_bot = (
        msg.reply_to_message is not None
        and msg.reply_to_message.from_user is not None
        and msg.reply_to_message.from_user.id == context.bot.id
    )
    if bot_username and mention.lower() in text.lower():
        text = re.sub(re.escape(mention), "", text, flags=re.IGNORECASE).strip()
        return True, text
    if reply_to_bot:
        return True, text
    return False, ""

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    addressed, question = _is_addressed_in_group(update, context)
    if not addressed:
        return
    question = question.strip()
    if not question:
        chat = update.effective_chat
        if chat is not None and chat.type in ("group", "supergroup"):
            bot_username = context.bot.username or "bot"
            await update.message.reply_text(
                f"Задайте вопрос вместе с упоминанием — например:\n"
                f"<code>@{bot_username} какие баллы нужны на ПМИ?</code>\n\n"
                f"Ask your question together with the mention, e.g.:\n"
                f"<code>@{bot_username} what scores do I need for the program?</code>",
                parse_mode="HTML",
            )
        return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "Абитуриент"
    lang = _detect_lang(question)

    if _is_rate_limited(user_id):
        msg = "⏳ Too many requests. Please wait a moment." if lang == "en" else "⏳ Слишком много запросов. Подождите немного."
        await update.message.reply_text(msg)
        return

    log.info("User %s (%d): %s", user_name, user_id, question[:80])
    await update.message.chat.send_action(ChatAction.TYPING)

    if _is_greeting(question):
        if lang == "en":
            await update.message.reply_text(
                f"Hello, {user_name}! Ask me anything about admission to HSE Faculty of Computer Science."
            )
        else:
            await update.message.reply_text(
                f"Привет, {user_name}! Задавайте вопросы о поступлении на ФКН ВШЭ."
            )
        if not _user_citizenship.get(user_id):
            cq, ckb = _citizenship_prompt(lang)
            await update.message.reply_text(cq, parse_mode="HTML", reply_markup=ckb)
        return

    if _is_thanks(question):
        if lang == "en":
            await update.message.reply_text("You're welcome! Feel free to ask more questions.")
        else:
            await update.message.reply_text("Рад помочь! Если появятся ещё вопросы — спрашивайте.")
        return

    if not _user_citizenship.get(user_id):
        _pending_question[user_id] = question
        cq, ckb = _citizenship_prompt(lang)
        await update.message.reply_text(cq, parse_mode="HTML", reply_markup=ckb)
        return

    calc = _calc_state.get(user_id)
    if calc and calc.get("step") == "scores":
        await _handle_wizard_scores(update.message, user_id, question, calc)
        return

    if calc and calc.get("step") == "achievements":
        await _handle_wizard_achievements(update.message, user_id, question, calc)
        return

    await _process_question(update.message, user_id, question)

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "Начать / выбор гражданства"),
        BotCommand("reset", "Сбросить историю"),
        BotCommand("stats", "Статистика"),
        BotCommand("help", "Справка"),
    ])

def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не задана!")

    log.info("RAG-пайплайн: %s", "agentic-loop" if USE_AGENTIC_RAG else "декомпозиция запроса")

    log.info("Строим ChromaDB-индекс...")
    n = build_index()
    log.info("Индекс готов: %d чанков", n)

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CallbackQueryHandler(handle_citizenship, pattern="^citizen_"))
    app.add_handler(CallbackQueryHandler(handle_action, pattern="^action_"))
    app.add_handler(CallbackQueryHandler(handle_calc_step, pattern="^calc_"))
    app.add_handler(CallbackQueryHandler(handle_vote, pattern="^vote_"))
    app.add_handler(InlineQueryHandler(handle_inline))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        log.info("Webhook: %s", WEBHOOK_URL)
        app.run_webhook(
            listen="0.0.0.0",
            port=WEBHOOK_PORT,
            url_path="/webhook",
            webhook_url=f"{WEBHOOK_URL}/webhook",
            drop_pending_updates=True,
        )
    else:
        log.info("Polling...")
        app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
