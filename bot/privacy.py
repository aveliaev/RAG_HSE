"""Персональные данные: согласие, псевдонимизация, маскирование, удаление и срок хранения логов."""
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta

from config import (
    LOG_SALT, LOG_RETENTION_DAYS, PRIVACY_CONTACT, CONSENTS_FILE,
    EVENTS_LOG, VOTES_LOG, USE_YANDEX,
)

log = logging.getLogger(__name__)

# Меняется при изменении текста политики — тогда согласие запрашивается заново
PRIVACY_VERSION = "2026-09-22"

# Соль, которой хешировались ID до появления LOG_SALT. Нужна, чтобы /deletedata
# находил и старые записи пользователя.
_LEGACY_SALT = "fkn_hse_bot"
if not LOG_SALT:
    log.warning("LOG_SALT не задан — ID в логах хешируются публичной солью и обратимы перебором. "
                "Задайте LOG_SALT в .env")


def hash_uid(user_id: int) -> str:
    return hashlib.sha256(f"{LOG_SALT or _LEGACY_SALT}{user_id}".encode()).hexdigest()[:12]


def _all_hashes(user_id: int) -> set[str]:
    return {hash_uid(user_id),
            hashlib.sha256(f"{_LEGACY_SALT}{user_id}".encode()).hexdigest()[:12]}


# --- Маскирование ---
_PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?:\+7|\b8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}\b"), "[телефон]"),
    (re.compile(r"\b\d{3}-\d{3}-\d{3}[\s-]\d{2}\b"), "[СНИЛС]"),
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "[номер карты]"),
    (re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b"), "[паспорт]"),
]


def redact_pii(text: str) -> str:
    """Заменяет в тексте email, телефоны, СНИЛС, номера карт и паспортов на метки."""
    if not text:
        return text
    for pattern, label in _PII_PATTERNS:
        text = pattern.sub(label, text)
    return text


def contains_pii(text: str) -> bool:
    return bool(text) and any(p.search(text) for p, _ in _PII_PATTERNS)


# --- Согласие ---
def _load_consents() -> dict:
    try:
        return json.loads(CONSENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


_consents: dict = _load_consents()


def _save_consents() -> None:
    try:
        CONSENTS_FILE.write_text(json.dumps(_consents, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("Не удалось сохранить согласия: %s", e)


def has_consent(user_id: int) -> bool:
    rec = _consents.get(hash_uid(user_id))
    return bool(rec) and rec.get("version") == PRIVACY_VERSION


def give_consent(user_id: int) -> None:
    _consents[hash_uid(user_id)] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "version": PRIVACY_VERSION,
    }
    _save_consents()


# --- Удаление и срок хранения ---
def _rewrite_jsonl(path, keep) -> int:
    """Перезаписывает jsonl, оставляя строки, для которых keep(record) истинно. Возвращает число удалённых."""
    if not path.exists():
        return 0
    kept, removed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if keep(rec):
            kept.append(line)
        else:
            removed += 1
    if removed:
        path.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    return removed


def delete_user_data(user_id: int) -> int:
    """Удаляет вопросы/ответы пользователя, его оценки и согласие. Возвращает число удалённых записей."""
    hashes = _all_hashes(user_id)
    msg_ids: set = set()

    def keep_event(rec):
        if rec.get("uid") in hashes:
            msg_ids.add(rec.get("msg_id"))
            return False
        return True

    removed = _rewrite_jsonl(EVENTS_LOG, keep_event)
    removed += _rewrite_jsonl(VOTES_LOG, lambda rec: rec.get("msg_id") not in msg_ids)
    for h in hashes:
        _consents.pop(h, None)
    _save_consents()
    return removed


def prune_old_logs() -> int:
    """Удаляет записи логов старше LOG_RETENTION_DAYS дней."""
    if LOG_RETENTION_DAYS <= 0:
        return 0
    cutoff = (datetime.now() - timedelta(days=LOG_RETENTION_DAYS)).isoformat(timespec="seconds")
    removed = sum(_rewrite_jsonl(p, lambda rec: rec.get("ts", "") >= cutoff) for p in (EVENTS_LOG, VOTES_LOG))
    if removed:
        log.info("Удалено %d записей логов старше %d дней", removed, LOG_RETENTION_DAYS)
    return removed


# --- Тексты ---
CONSENT_TEXT = (
    "🔒 <b>Персональные данные</b>\n\n"
    "Чтобы отвечать, бот обрабатывает ваши сообщения: они передаются в языковую модель, "
    "а вопросы и ответы сохраняются в обезличенном виде для улучшения качества.\n\n"
    "⚠️ Не отправляйте боту паспортные данные, СНИЛС, телефоны и другие личные сведения.\n\n"
    "Подробнее — /privacy. Удалить свои данные — /deletedata.\n\n"
    "Нажимая «Принимаю», вы соглашаетесь на обработку данных на этих условиях."
)
CONSENT_TEXT_EN = (
    "🔒 <b>Personal data</b>\n\n"
    "To answer, the bot processes your messages: they are sent to a language model, "
    "and questions and answers are stored in pseudonymised form to improve quality.\n\n"
    "⚠️ Do not send passport numbers, phone numbers or other personal details.\n\n"
    "Details — /privacy. Delete your data — /deletedata.\n\n"
    "By pressing “Accept” you agree to data processing on these terms."
)


def privacy_text(lang: str = "ru") -> str:
    llm_ru = ("YandexGPT и Yandex SpeechKit (Яндекс Облако, серверы в РФ)" if USE_YANDEX
              else "Groq (серверы в США — трансграничная передача)")
    llm_en = ("YandexGPT and Yandex SpeechKit (Yandex Cloud, servers in Russia)" if USE_YANDEX
              else "Groq (servers in the USA)")
    retention_ru = f"{LOG_RETENTION_DAYS} дней" if LOG_RETENTION_DAYS > 0 else "до удаления по запросу"
    retention_en = f"{LOG_RETENTION_DAYS} days" if LOG_RETENTION_DAYS > 0 else "until you request deletion"
    if lang == "en":
        contact = f"\n\n<b>Contact:</b> {PRIVACY_CONTACT}" if PRIVACY_CONTACT else ""
        return (
            "🔒 <b>Privacy policy</b>\n\n"
            "<b>What we process:</b> your messages (text and voice), your choices in the bot "
            "(university, role, citizenship) and your answer ratings.\n\n"
            "<b>Why:</b> to answer your questions and improve answer quality.\n\n"
            f"<b>Who processes it:</b> {llm_en}; Telegram delivers messages.\n\n"
            "<b>How it is stored:</b> your Telegram ID is replaced with a pseudonymous hash; "
            "phone numbers, emails, passport and card numbers are masked in logs. "
            "Voice messages are not stored. Your name and username are not stored.\n\n"
            f"<b>Retention:</b> {retention_en}.\n\n"
            "<b>Your rights:</b> /deletedata deletes your questions, answers and ratings; "
            "/privacy shows this notice."
            f"{contact}"
        )
    contact = f"\n\n<b>Контакт:</b> {PRIVACY_CONTACT}" if PRIVACY_CONTACT else ""
    return (
        "🔒 <b>Политика обработки персональных данных</b>\n\n"
        "<b>Что обрабатывается:</b> ваши сообщения (текст и голосовые), выбор в боте "
        "(университет, роль, гражданство) и оценки ответов.\n\n"
        "<b>Зачем:</b> чтобы отвечать на вопросы и улучшать качество ответов.\n\n"
        f"<b>Кто обрабатывает:</b> {llm_ru}; доставку сообщений обеспечивает Telegram.\n\n"
        "<b>Как хранится:</b> вместо Telegram ID в логах — обезличенный хеш; телефоны, почта, "
        "номера паспортов и карт в логах маскируются. Голосовые сообщения не сохраняются. "
        "Имя и username не сохраняются.\n\n"
        f"<b>Срок хранения:</b> {retention_ru}.\n\n"
        "<b>Ваши права:</b> /deletedata — удалить ваши вопросы, ответы и оценки; "
        "/privacy — это уведомление."
        f"{contact}"
    )
