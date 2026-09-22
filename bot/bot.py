import io
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
    Update, BotCommand, InputFile,
    InlineKeyboardButton, InlineKeyboardMarkup,
    InlineQueryResultArticle, InputTextMessageContent, InlineQueryResultsButton,
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
from telegram.error import BadRequest

from config import (
    TELEGRAM_TOKEN, MAX_HISTORY_PAIRS,
    RATE_LIMIT_MAX, RATE_LIMIT_WINDOW,
    DISLIKE_LOG, WEBHOOK_URL, WEBHOOK_PORT,
    ADMIN_USER_ID, QUARANTINE_FILE,
    USE_AGENTIC_RAG,
)
from faq_cache import lookup as faq_lookup, add_to_dynamic_cache, quarantine_answer
from rag_engine import build_index, check_content, needs_clarification, llm_clarify
from agentic_rag import agentic_ask
from student_rag import build_student_index, student_kb_available, ask_student
from privacy import (
    hash_uid, redact_pii, has_consent, give_consent, delete_user_data, prune_old_logs,
    CONSENT_TEXT, CONSENT_TEXT_EN, privacy_document, privacy_date,
)


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
# httpx на INFO пишет каждый запрос с полным URL, а в URL Telegram API — токен бота
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

_history: dict[int, list[dict]] = defaultdict(list)

_vote_pending: dict[int, tuple[str, str, str]] = {}
_MAX_VOTE_PENDING = 500

# Выбор в начале диалога: университет → роль (абитуриент/студент) → (для абитуриента) гражданство
_user_university: dict[int, str] = {}
_user_role: dict[int, str] = {}

_user_citizenship: dict[int, str] = {}

_pending_question: dict[int, str] = {}

# Контекст уточнения: исходный вопрос + текст уточнения, который задал бот.
# На следующем сообщении пользователя мы НЕ уточняем повторно (иначе зацикливание),
# а склеиваем исходный вопрос с ответом-уточнением и идём в RAG. Эти же данные
# уходят в лог, чтобы дашборд показал весь тред (вопрос → уточнение → ответ → итог).
_clarify_pending: dict[int, dict] = {}

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

def _build_effective_question(question: str, history: list[dict]) -> str:
    # Раньше тут была склейка с предыдущим вопросом по маркерам ("уточните" и т.п.).
    # Она ошибочно срабатывала на тексте ОТКАЗА («Уточните на сайте ba.hse.ru»),
    # из-за чего предыдущий вопрос «протекал» в следующий ответ (дублирование).
    # Реальные уточнения теперь обрабатываются явным механизмом _clarify_pending,
    # поэтому здесь просто возвращаем вопрос как есть.
    return question

# Официальный график поступления — добавляем ссылку к ответам про сроки/даты/процесс подачи.
_SCHEDULE_LINK = "https://ba.hse.ru/entr"
_SCHEDULE_TRIGGERS = [
    "срок", "дедлайн", "до какого числа", "когда подавать", "когда подача",
    "когда поступ", "когда начина", "когда заканчива", "когда нести", "когда зачисл",
    "график поступл", "даты поступл", "дата подачи", "календарь поступл", "расписание поступл",
    "как поступить", "как поступать", "как мне поступить",
    "подать документ", "подавать документ", "подача документ", "подаю документ",
    "подаются документ", "подал документ", "подачи документ",
    "этапы поступл", "процесс поступл", "порядок поступл",
]

def _maybe_append_schedule_link(question: str, answer: str, lang: str) -> str:
    """К ответам про сроки/даты/процесс поступления добавляем официальный график ba.hse.ru/entr."""
    if "техническая ошибка" in answer.lower():
        return answer
    if not any(t in question.lower() for t in _SCHEDULE_TRIGGERS):
        return answer
    if _SCHEDULE_LINK in answer:
        return answer
    if lang == "en":
        return answer + f"\n\n📅 Official admission schedule and dates: {_SCHEDULE_LINK}"
    return answer + f"\n\n📅 Актуальный график и даты поступления: {_SCHEDULE_LINK}"

def _log_dislike(question: str) -> None:
    try:
        with DISLIKE_LOG.open("a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f"[{ts}] {redact_pii(question)}\n")
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

# Университеты, для которых есть бот. Новый вуз = новая строка (id, подпись ru, подпись en).
_UNIVERSITIES = [
    ("hse", "🎓 НИУ ВШЭ", "🎓 HSE University"),
]
_UNIVERSITY_NAMES = {uid: ru.split(" ", 1)[1] for uid, ru, _ in _UNIVERSITIES}

def _university_prompt(lang: str) -> tuple[str, InlineKeyboardMarkup]:
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(en if lang == "en" else ru, callback_data=f"uni_{uid}")
        for uid, ru, en in _UNIVERSITIES
    ]])
    if lang == "en":
        return "<b>Choose your university:</b>", kb
    return "<b>Выберите университет:</b>", kb

def _role_prompt(lang: str, university: str) -> tuple[str, InlineKeyboardMarkup]:
    name = _UNIVERSITY_NAMES.get(university, university)
    if lang == "en":
        return (
            f"✅ University: <b>{name}</b>\n\n<b>Who are you?</b>",
            InlineKeyboardMarkup([[
                InlineKeyboardButton("📝 Applicant", callback_data="role_applicant"),
                InlineKeyboardButton("🎓 Student", callback_data="role_student"),
            ]]),
        )
    return (
        f"✅ Университет: <b>{name}</b>\n\n<b>Вы абитуриент или студент?</b>",
        InlineKeyboardMarkup([[
            InlineKeyboardButton("📝 Абитуриент", callback_data="role_applicant"),
            InlineKeyboardButton("🎓 Студент", callback_data="role_student"),
        ]]),
    )

def _consent_prompt(lang: str) -> tuple[str, InlineKeyboardMarkup]:
    if lang == "en":
        return CONSENT_TEXT_EN, InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Accept", callback_data="consent_yes"),
            InlineKeyboardButton("📄 Details", callback_data="consent_more"),
        ]])
    return CONSENT_TEXT, InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Принимаю", callback_data="consent_yes"),
        InlineKeyboardButton("📄 Подробнее", callback_data="consent_more"),
    ]])

def _setup_prompt(user_id: int, lang: str) -> tuple[str, InlineKeyboardMarkup] | None:
    """Следующий незавершённый шаг (согласие → вуз → роль → гражданство) или None, если всё выбрано."""
    if not has_consent(user_id):
        return _consent_prompt(lang)
    university = _user_university.get(user_id)
    if not university:
        return _university_prompt(lang)
    role = _user_role.get(user_id)
    if not role:
        return _role_prompt(lang, university)
    if role == "applicant" and not _user_citizenship.get(user_id):
        return _citizenship_prompt(lang)
    return None

def _reset_user(user_id: int) -> None:
    _history[user_id].clear()
    _user_university.pop(user_id, None)
    _user_role.pop(user_id, None)
    _user_citizenship[user_id] = ""
    _pending_question.pop(user_id, None)
    _clarify_pending.pop(user_id, None)
    _calc_state.pop(user_id, None)

START_TEXT = (
    "Привет! Я <b>Поступариум</b> — ИИ-консультант для абитуриентов и студентов 🎓\n\n"
    "Для начала выберите университет и расскажите, кто вы.\n\n"
    "/start — начать заново · /reset — сбросить историю"
)
START_TEXT_EN = (
    "Hi! I'm <b>Postuparium</b> — an AI assistant for applicants and students 🎓\n\n"
    "First, choose your university and tell me who you are.\n\n"
    "/start — start over · /reset — clear history"
)

STUDENT_READY_TEXT = (
    "✅ Роль: <b>🎓 Студент ВШЭ</b>\n\n"
    "Задайте вопрос об учёбе: пересдачи, стипендии, академический отпуск, майноры и т. п."
)
STUDENT_READY_TEXT_EN = (
    "✅ Role: <b>🎓 HSE student</b>\n\n"
    "Ask a question about your studies: retakes, scholarships, academic leave, minors, etc."
)
STUDENT_WIP_TEXT = (
    "✅ Роль: <b>🎓 Студент ВШЭ</b>\n\n"
    "⚙️ Раздел для студентов сейчас в разработке — база знаний ещё наполняется.\n\n"
    "Если вы поступаете в ВШЭ — нажмите /start и выберите «Абитуриент»."
)
STUDENT_WIP_TEXT_EN = (
    "✅ Role: <b>🎓 HSE student</b>\n\n"
    "⚙️ The student section is under development — the knowledge base is being filled.\n\n"
    "If you're applying to HSE, press /start and choose “Applicant”."
)

def _student_intro(lang: str) -> str:
    if student_kb_available():
        return STUDENT_READY_TEXT_EN if lang == "en" else STUDENT_READY_TEXT
    return STUDENT_WIP_TEXT_EN if lang == "en" else STUDENT_WIP_TEXT

WELCOME_TEXT = (
    "Привет! Я бот-консультант по поступлению на "
    "<b>Факультет компьютерных наук НИУ ВШЭ</b> 🎓\n\n"
    "Могу ответить на вопросы о:\n"
    "• программах ФКН (ПМИ, ПАД, КНАД, ЭАД и другие)\n"
    "• минимальных баллах ЕГЭ и сроках подачи\n"
    "• БВИ, квазибюджете и зелёной волне\n"
    "• индивидуальных достижениях и скидках"
)

WELCOME_TEXT_EN = (
    "I'll help with admission to the "
    "<b>Faculty of Computer Science, HSE University</b> 🎓\n\n"
    "Ask me about programmes, exam scores, deadlines, olympiads and discounts."
)

HELP_TEXT = (
    "<b>Команды:</b>\n"
    "/start — выбор университета и роли (абитуриент / студент)\n"
    "/reset — очистить историю\n"
    "/help — справка\n"
    "/privacy — обработка персональных данных\n"
    "/deletedata — удалить мои данные\n\n"
    "<b>Как работает:</b>\n"
    "• Кешированные ответы — мгновенно\n"
    "• Остальные — поиск по базе знаний + Groq AI\n"
    "• Все ответы можно оценить 👍/👎\n"
    "• 👎 на кешированный ответ → карантин для проверки\n"
    "• Работает в inline-режиме: @бот вопрос"
)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _reset_user(user_id)
    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"
    uq, ukb = _setup_prompt(user_id, lang)
    await update.message.reply_text(START_TEXT_EN if lang == "en" else START_TEXT, parse_mode="HTML")
    await update.message.reply_text(uq, parse_mode="HTML", reply_markup=ukb)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _reset_user(user_id)
    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    lang = "en" if tg_lang == "en" else "ru"
    uq, ukb = _setup_prompt(user_id, lang)
    prefix = "✅ History cleared.\n\n" if lang == "en" else "✅ История очищена.\n\n"
    await update.message.reply_text(prefix + uq, parse_mode="HTML", reply_markup=ukb)

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

async def handle_consent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    lang = "en" if (query.from_user.language_code or "")[:2].lower() == "en" else "ru"

    if query.data == "consent_more":
        # Новым сообщением, а не правкой старого: правка выше по чату незаметна,
        # а повторное нажатие давало ошибку Telegram «Message is not modified».
        # Кнопки остаются только под новым сообщением.
        await query.edit_message_reply_markup(reply_markup=None)
        accept = InlineKeyboardMarkup([[InlineKeyboardButton(
            "✅ Accept" if lang == "en" else "✅ Принимаю", callback_data="consent_yes")]])
        await _send_privacy_document(query.message, lang, reply_markup=accept)
        return

    give_consent(user_id)
    log.info("Consent given by user %s", hash_uid(user_id))
    text, kb = _setup_prompt(user_id, lang) or _university_prompt(lang)
    done = "✅ Thank you!" if lang == "en" else "✅ Спасибо!"
    await query.edit_message_text(f"{done}\n\n{text}", parse_mode="HTML", reply_markup=kb)

async def _send_privacy_document(message, lang: str, reply_markup=None) -> None:
    filename, content = privacy_document(lang)
    caption = (f"📄 Privacy policy, in Russian (version of {privacy_date('en')})" if lang == "en"
               else f"📄 Политика обработки персональных данных (редакция от {privacy_date('ru')})")
    await message.reply_document(
        document=InputFile(io.BytesIO(content), filename=filename),
        caption=caption, reply_markup=reply_markup,
    )

async def cmd_privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = "en" if (update.effective_user.language_code or "")[:2].lower() == "en" else "ru"
    await _send_privacy_document(update.message, lang)

async def cmd_deletedata(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = "en" if (update.effective_user.language_code or "")[:2].lower() == "en" else "ru"
    if lang == "en":
        text = ("🗑 <b>Delete your data?</b>\n\nYour questions, the bot's answers, your ratings and "
                "your consent will be deleted. This cannot be undone.")
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🗑 Delete", callback_data="delete_yes"),
            InlineKeyboardButton("Cancel", callback_data="delete_no"),
        ]])
    else:
        text = ("🗑 <b>Удалить ваши данные?</b>\n\nБудут удалены ваши вопросы, ответы бота, оценки "
                "и согласие на обработку. Отменить это нельзя.")
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("🗑 Удалить", callback_data="delete_yes"),
            InlineKeyboardButton("Отмена", callback_data="delete_no"),
        ]])
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb)

async def handle_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    lang = "en" if (query.from_user.language_code or "")[:2].lower() == "en" else "ru"

    if query.data == "delete_no":
        await query.edit_message_text("Cancelled." if lang == "en" else "Отменено.")
        return

    removed = delete_user_data(user_id)
    _reset_user(user_id)
    log.info("User data deleted for %s: %d records", hash_uid(user_id), removed)
    await query.edit_message_text(
        f"✅ Your data has been deleted ({removed} records). Press /start to use the bot again."
        if lang == "en" else
        f"✅ Ваши данные удалены (записей: {removed}). Чтобы снова пользоваться ботом, нажмите /start."
    )

async def handle_university(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    lang = "en" if (query.from_user.language_code or "")[:2].lower() == "en" else "ru"

    if not has_consent(user_id):
        text, kb = _consent_prompt(lang)
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=kb)
        return

    university = query.data.removeprefix("uni_")
    _user_university[user_id] = university
    _user_role.pop(user_id, None)
    log.info("University set for user %s: %s", hash_uid(user_id), university)
    text, kb = _role_prompt(lang, university)
    await query.edit_message_text(text, parse_mode="HTML", reply_markup=kb)

async def handle_role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    lang = "en" if (query.from_user.language_code or "")[:2].lower() == "en" else "ru"

    if not _user_university.get(user_id):
        # кнопка из старого сообщения после /reset — начинаем выбор заново
        uq, ukb = _university_prompt(lang)
        await query.edit_message_text(uq, parse_mode="HTML", reply_markup=ukb)
        return

    role = query.data.removeprefix("role_")
    _user_role[user_id] = role
    _history[user_id].clear()
    log.info("Role set for user %s: %s", hash_uid(user_id), role)

    if role == "applicant":
        cq, ckb = _citizenship_prompt(lang)
        welcome = WELCOME_TEXT_EN if lang == "en" else WELCOME_TEXT
        await query.edit_message_text(welcome + "\n\n" + cq, parse_mode="HTML", reply_markup=ckb)
        return

    await query.edit_message_text(_student_intro(lang), parse_mode="HTML")
    pending_q = _pending_question.pop(user_id, None)
    if pending_q and student_kb_available():
        log.info("Processing pending student question: '%s'", pending_q[:60])
        await _process_question(query.message, user_id, pending_q)

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
    log.info("Citizenship set for user %s: %s", hash_uid(user_id), _user_citizenship[user_id])

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
        text = (
            "🧮 <b>Калькулятор ЕГЭ</b>\n\n"
            "⚙️ Этот раздел сейчас находится в разработке. "
            "Скоро здесь можно будет рассчитать конкурсный балл.\n\n"
            "А пока задайте мне любой вопрос о поступлении на ФКН — я с радостью отвечу."
            if lang != "en" else
            "🧮 <b>EGE Calculator</b>\n\n"
            "⚙️ This section is currently under development. "
            "You'll soon be able to calculate your competitive score here.\n\n"
            "Meanwhile, feel free to ask any question about admission to FCS."
        )
        await query.message.reply_text(text, parse_mode="HTML")
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
            if src == "student":
                log.info("Vote 👍 (student): '%s'", q[:60])
            elif src == "rag":
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
            if src == "student":
                _log_dislike(q)
                log.info("Vote 👎 (student): '%s'", q[:60])
                await query.message.reply_text(
                    "👎 Понял. Попробуйте переформулировать вопрос или уточните в учебном офисе."
                )
            elif src == "rag":
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
    if not has_consent(update.inline_query.from_user.id):
        await update.inline_query.answer(
            [], cache_time=0, is_personal=True,
            button=InlineQueryResultsButton(text="Открыть бота и принять условия", start_parameter="consent"),
        )
        return
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

async def _process_student_question(message, user_id: int, question: str) -> None:
    """Ветка «Студент»: своя база знаний, без кеша абитуриентов, уточнений и калькулятора."""
    lang = _detect_lang(question)
    t0 = time.monotonic()

    if check_content(question):
        await message.reply_text(
            "I only answer questions about studying at HSE. Please keep it polite and on topic."
            if lang == "en" else
            "Я отвечаю только на вопросы об учёбе в ВШЭ. Пожалуйста, сформулируйте вопрос вежливо и по теме."
        )
        return

    if not student_kb_available():
        await message.reply_text(_student_intro(lang), parse_mode="HTML")
        return

    _stats["rag_calls"] += 1
    log.info("Student RAG [lang=%s] user %s", lang, hash_uid(user_id))
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.chat, stop_typing))
    meta: dict = {}
    try:
        answer, meta = await asyncio.get_running_loop().run_in_executor(
            None, ask_student, question, lang
        )
    except Exception as e:
        log.error("Student RAG error for user %s: %s", hash_uid(user_id), e)
        answer = "Произошла техническая ошибка. Попробуйте позже."
    finally:
        stop_typing.set()
        typing_task.cancel()

    _add_to_history(user_id, "user", question)
    _add_to_history(user_id, "assistant", answer)
    sent = await _send_msg_with_vote(message, question, answer, "student", lang)
    log_interaction(
        user_id=user_id, question=question,
        route="student", source="student",
        answer=answer, sub_queries=[],
        lang=lang, latency_ms=(time.monotonic() - t0) * 1000,
        msg_id=sent.message_id,
    )

async def _process_question(message, user_id: int, question: str) -> None:
    if _user_role.get(user_id) == "student":
        await _process_student_question(message, user_id, question)
        return

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
        log.info("Calculator hit for user %s", hash_uid(user_id))
        _add_to_history(user_id, "user", question)
        _add_to_history(user_id, "assistant", calc_answer)
        await message.reply_text(_to_html(calc_answer), parse_mode="HTML")
        return

    cached_answer, source = faq_lookup(question)
    if cached_answer:
        log.info("Cache hit [%s] user %s", source, hash_uid(user_id))
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

    # Если это сообщение — ответ на ранее заданное уточнение, повторно НЕ уточняем
    # (иначе зацикливание), а отвечаем по сути, склеив исходный вопрос с уточнением.
    clarify_ctx = _clarify_pending.pop(user_id, None)

    if clarify_ctx is None:
        clarification = needs_clarification(question, history)
        if not clarification:
            clarification = await asyncio.get_running_loop().run_in_executor(
                None, llm_clarify, question, history, lang
            )
        if clarification:
            log.info("Clarification needed for user %s", hash_uid(user_id))
            # запоминаем ИСХОДНЫЙ вопрос и текст уточнения (для склейки и для лога)
            _clarify_pending[user_id] = {"original": question, "asked": clarification}
            _add_to_history(user_id, "user", question)
            _add_to_history(user_id, "assistant", clarification)
            await message.reply_text(_to_html(clarification), parse_mode="HTML")
            return

    # По умолчанию в лог уходит сам вопрос; при ответе на уточнение — исходный вопрос
    # + поля clarify_*, чтобы дашборд собрал весь тред в одну запись.
    logged_question = question
    clarify_asked = ""
    clarify_reply = ""
    if clarify_ctx:
        effective_question = f"{clarify_ctx['original']} (уточнение от пользователя: {question})"
        logged_question = clarify_ctx["original"]
        clarify_asked = clarify_ctx["asked"]
        clarify_reply = question
        log.info("Clarification answered, merged question for user %s", hash_uid(user_id))
    else:
        effective_question = _build_effective_question(question, history)
    _stats["rag_calls"] += 1
    log.info("RAG [lang=%s, citizenship=%s] user %s", lang, citizenship or "?", hash_uid(user_id))

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.chat, stop_typing))
    meta: dict = {}
    try:
        answer, meta = await asyncio.get_running_loop().run_in_executor(
            None, ask_rag, effective_question, history, lang, citizenship
        )
    except Exception as e:
        log.error("RAG error for user %s: %s", hash_uid(user_id), e)
        answer = (
            "Произошла техническая ошибка. Попробуйте позже.\n"
            "Контакты: abitur@hse.ru · ba.hse.ru"
        )
    finally:
        stop_typing.set()
        typing_task.cancel()

    answer = _maybe_append_schedule_link(question, answer, lang)
    _add_to_history(user_id, "user", question)
    _add_to_history(user_id, "assistant", answer)
    sent = await _send_msg_with_vote(message, effective_question, answer, "rag", lang)
    log_interaction(
        user_id=user_id, question=logged_question,
        route="rag", source="rag",
        answer=answer, sub_queries=meta.get("sub_queries", []),
        lang=lang, latency_ms=(time.monotonic() - t0) * 1000,
        msg_id=sent.message_id,
        clarify_asked=clarify_asked, clarify_reply=clarify_reply,
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

_STUDENT_DISCLAIMER = (
    "\n\n<i>⚠️ Поступариум — это ИИ-консультант и может ошибаться. "
    "Проверяйте важную информацию в учебном офисе или на hse.ru.</i>"
)
_STUDENT_DISCLAIMER_EN = (
    "\n\n<i>⚠️ Postuparium is an AI assistant and can make mistakes. "
    "Please verify important information with your study office or at hse.ru.</i>"
)

async def _send_msg_with_vote(message, question: str, answer: str, source: str, lang: str = "ru"):
    if source == "student":
        disclaimer = _STUDENT_DISCLAIMER_EN if lang == "en" else _STUDENT_DISCLAIMER
    else:
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

    # До согласия голос не отправляем на распознавание (это уже обработка данных)
    tg_lang = (update.effective_user.language_code or "")[:2].lower()
    if not has_consent(user_id):
        text, kb = _consent_prompt("en" if tg_lang == "en" else "ru")
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb)
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    tg_file = await context.bot.get_file(voice.file_id)
    audio_bytes = await tg_file.download_as_bytearray()

    stt_lang = "en-US" if tg_lang == "en" else "ru-RU"
    question = await transcribe_ogg(bytes(audio_bytes), lang=stt_lang)

    if not question:
        await update.message.reply_text(
            "😔 Не удалось распознать речь. Попробуйте ещё раз или напишите вопрос текстом."
        )
        return

    log.info("Voice→text user %s: %s", hash_uid(user_id), redact_pii(question[:80]))

    setup = _setup_prompt(user_id, "en" if tg_lang == "en" else "ru")
    if setup:
        _pending_question[user_id] = question
        await update.message.reply_text(setup[0], parse_mode="HTML", reply_markup=setup[1])
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

    log.info("User %s: %s", hash_uid(user_id), redact_pii(question[:80]))
    await update.message.chat.send_action(ChatAction.TYPING)

    if _is_greeting(question):
        student = _user_role.get(user_id) == "student"
        if lang == "en":
            topic = "studying at HSE" if student else "admission to HSE Faculty of Computer Science"
            await update.message.reply_text(f"Hello, {user_name}! Ask me anything about {topic}.")
        else:
            topic = "об учёбе в ВШЭ" if student else "о поступлении на ФКН ВШЭ"
            await update.message.reply_text(f"Привет, {user_name}! Задавайте вопросы {topic}.")
        setup = _setup_prompt(user_id, lang)
        if setup:
            await update.message.reply_text(setup[0], parse_mode="HTML", reply_markup=setup[1])
        return

    if _is_thanks(question):
        if lang == "en":
            await update.message.reply_text("You're welcome! Feel free to ask more questions.")
        else:
            await update.message.reply_text("Рад помочь! Если появятся ещё вопросы — спрашивайте.")
        return

    setup = _setup_prompt(user_id, lang)
    if setup:
        _pending_question[user_id] = question
        await update.message.reply_text(setup[0], parse_mode="HTML", reply_markup=setup[1])
        return

    calc = _calc_state.get(user_id)
    if calc and calc.get("step") == "scores":
        await _handle_wizard_scores(update.message, user_id, question, calc)
        return

    if calc and calc.get("step") == "achievements":
        await _handle_wizard_achievements(update.message, user_id, question, calc)
        return

    await _process_question(update.message, user_id, question)

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    # Повторное нажатие на ту же кнопку — Telegram отказывается «менять» сообщение на такое же
    if isinstance(err, BadRequest) and "message is not modified" in str(err).lower():
        return
    log.error("Ошибка при обработке апдейта: %s", err, exc_info=err)

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "Начать / выбор вуза и роли"),
        BotCommand("reset", "Сбросить историю"),
        BotCommand("help", "Справка"),
        BotCommand("privacy", "Персональные данные"),
        BotCommand("deletedata", "Удалить мои данные"),
    ])

def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не задана!")

    log.info("RAG-пайплайн: %s", "agentic-loop" if USE_AGENTIC_RAG else "декомпозиция запроса")

    log.info("Строим ChromaDB-индекс...")
    n = build_index()
    log.info("Индекс готов: %d чанков", n)
    build_student_index()
    prune_old_logs()

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("admin", cmd_admin))
    app.add_handler(CommandHandler("privacy", cmd_privacy))
    app.add_handler(CommandHandler("deletedata", cmd_deletedata))
    app.add_handler(CallbackQueryHandler(handle_consent, pattern="^consent_"))
    app.add_handler(CallbackQueryHandler(handle_delete, pattern="^delete_"))
    app.add_handler(CallbackQueryHandler(handle_university, pattern="^uni_"))
    app.add_handler(CallbackQueryHandler(handle_role, pattern="^role_"))
    app.add_handler(CallbackQueryHandler(handle_citizenship, pattern="^citizen_"))
    app.add_handler(CallbackQueryHandler(handle_action, pattern="^action_"))
    app.add_handler(CallbackQueryHandler(handle_calc_step, pattern="^calc_"))
    app.add_handler(CallbackQueryHandler(handle_vote, pattern="^vote_"))
    app.add_handler(InlineQueryHandler(handle_inline))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

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
