"""Персональные данные: согласие, псевдонимизация, маскирование, удаление и срок хранения логов."""
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    LOG_SALT, LOG_RETENTION_DAYS, PRIVACY_CONTACT, PRIVACY_OPERATOR, CONSENTS_FILE,
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


_MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]


def privacy_date(lang: str = "ru") -> str:
    """Дата редакции политики (из PRIVACY_VERSION) в человекочитаемом виде."""
    d = datetime.strptime(PRIVACY_VERSION, "%Y-%m-%d")
    if lang == "en":
        return d.strftime("%d %B %Y")
    return f"{d.day} {_MONTHS_RU[d.month - 1]} {d.year} г."


# --- Документ политики (PDF) ---
_POLICY_TEMPLATE = Path(__file__).parent / "legal" / "privacy_policy.md"

# Шрифты с кириллицей: DejaVu ставится в Docker-образ, Arial — есть на macOS
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _policy_values() -> dict[str, str]:
    if USE_YANDEX:
        processors = (
            "- ООО «Яндекс.Облако» (Yandex Cloud) — размещение Бота на серверах в Российской Федерации, "
            "формирование ответов языковой моделью YandexGPT и распознавание голосовых сообщений "
            "(Yandex SpeechKit); сохранение запросов на стороне провайдера отключено;\n"
        )
        cross_border = (
            "9.1. Оператор не осуществляет трансграничную передачу персональных данных для формирования "
            "ответов: языковая модель и распознавание речи работают на серверах в Российской Федерации.\n\n"
        )
    else:
        processors = (
            "- ООО «Яндекс.Облако» (Yandex Cloud) — размещение Бота на серверах в Российской Федерации;\n"
            "- Groq, Inc. (США) — формирование ответов языковой моделью;\n"
        )
        cross_border = (
            "9.1. Для формирования ответов текст вопроса пользователя передаётся сервису Groq, Inc. (США). "
            "Трансграничная передача осуществляется на основании согласия пользователя и только в объёме, "
            "необходимом для формирования ответа; идентификатор пользователя при этом не передаётся.\n\n"
        )
    processors += (
        "- мессенджер Telegram — доставка сообщений между пользователем и Ботом;\n"
        "- при необходимости — провайдер защищённого сетевого подключения: только передача "
        "зашифрованного (TLS) трафика между сервером Бота и Telegram, без доступа к содержимому сообщений."
    )
    cross_border += (
        "9.2. Сообщения между пользователем и Ботом доставляются через мессенджер Telegram, серверы "
        "которого могут располагаться за пределами Российской Федерации. Пользователь самостоятельно "
        "выбирает Telegram как способ связи с Ботом и использует его на условиях политики "
        "конфиденциальности Telegram."
    )
    retention = (f"{LOG_RETENTION_DAYS} дней с даты записи" if LOG_RETENTION_DAYS > 0
                 else "до удаления по запросу пользователя")
    return {
        "date": privacy_date("ru"),
        "operator": f"{PRIVACY_OPERATOR} (далее — Оператор)" if PRIVACY_OPERATOR
                    else "администратор (владелец) Telegram-бота «Поступариум» (далее — Оператор)",
        "contact": PRIVACY_CONTACT or "через Бот — команды /privacy и /deletedata",
        "hosting": "Yandex Cloud, регион ru-central1",
        "processors": processors,
        "cross_border": cross_border,
        "retention": retention,
    }


def policy_markdown() -> str:
    text = _POLICY_TEMPLATE.read_text(encoding="utf-8")
    for key, value in _policy_values().items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _render_pdf(md: str) -> bytes | None:
    try:
        from fpdf import FPDF
    except ImportError:
        log.warning("fpdf2 не установлен — политика будет отправлена текстом")
        return None
    fonts = next(((r, b) for r, b in _FONT_CANDIDATES if Path(r).exists() and Path(b).exists()), None)
    if not fonts:
        log.warning("Не найден шрифт с кириллицей — политика будет отправлена текстом")
        return None

    title_lines = []
    date_line = f"Редакция от {privacy_date('ru')}"

    class PolicyPDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Main", size=8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, f"Политика обработки персональных данных · {date_line} · "
                             f"стр. {self.page_no()} из {{nb}}", align="C")

    pdf = PolicyPDF(format="A4")
    pdf.set_title("Политика обработки персональных данных — Поступариум")
    pdf.set_author(PRIVACY_OPERATOR or "Поступариум")
    pdf.add_font("Main", "", fonts[0])
    pdf.add_font("Main", "B", fonts[1])
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    lines = md.splitlines()
    i = 0
    # Шапка: «# Заголовок» и следующие за ним строки до первого раздела
    while i < len(lines) and not lines[i].startswith("## "):
        line = lines[i].strip()
        if line.startswith("# "):
            title_lines.append(line[2:])
        elif line:
            title_lines.append(line)
        i += 1
    pdf.set_font("Main", "B", 17)
    pdf.multi_cell(width, 9, title_lines[0], align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Main", "", 11)
    pdf.set_text_color(90, 90, 90)
    for line in title_lines[1:]:
        pdf.multi_cell(width, 6, line, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    for line in lines[i:]:
        line = line.rstrip()
        if not line:
            pdf.ln(1.5)
        elif line.startswith("## "):
            if pdf.get_y() > pdf.h - 45:  # заголовок раздела не оставляем одиноко внизу страницы
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font("Main", "B", 12.5)
            pdf.multi_cell(width, 7, line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        elif line.startswith("- "):
            pdf.set_font("Main", "", 10.5)
            if pdf.get_y() + 12 > pdf.page_break_trigger:  # маркер и первая строка — на одной странице
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_x(pdf.l_margin + 3)
            pdf.cell(4, 5.6, "•")
            pdf.set_xy(pdf.l_margin + 8, y)
            pdf.multi_cell(width - 8, 5.6, line[2:], markdown=True, align="J", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.8)
        else:
            pdf.set_font("Main", "", 10.5)
            pdf.multi_cell(width, 5.6, line, markdown=True, align="J", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.8)
    return bytes(pdf.output())


_document_cache: dict[str, tuple[str, bytes]] = {}


def privacy_document(lang: str = "ru") -> tuple[str, bytes]:
    """Политика для отправки в Telegram: (имя файла, содержимое). PDF, а если его не собрать — txt.
    Документ на русском (как требует 152-ФЗ); для англоязычных пользователей меняется только подпись."""
    if "doc" not in _document_cache:
        md = policy_markdown()
        pdf = _render_pdf(md)
        if pdf:
            _document_cache["doc"] = (f"Политика_ПДн_Поступариум_{PRIVACY_VERSION}.pdf", pdf)
        else:
            plain = re.sub(r"\*\*|^#+ ", "", md, flags=re.M)
            _document_cache["doc"] = (f"Политика_ПДн_Поступариум_{PRIVACY_VERSION}.txt",
                                      ("\ufeff" + plain).encode("utf-8"))
    return _document_cache["doc"]
