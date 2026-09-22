import os
from pathlib import Path

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

DOCS_DIR = Path(__file__).parent.parent / "dataset"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = "llama-3.3-70b-versatile"

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_MODEL = os.getenv("YANDEX_MODEL", "yandexgpt")

_use_yandex_env = os.getenv("USE_YANDEX", "")
USE_YANDEX = (
    _use_yandex_env.lower() == "true"
    if _use_yandex_env
    else bool(YANDEX_API_KEY and YANDEX_FOLDER_ID)
)

ENABLE_LLM_REWRITE = os.getenv("ENABLE_LLM_REWRITE", "true").lower() == "true"

USE_AGENTIC_RAG = os.getenv("USE_AGENTIC_RAG", "true").lower() == "true"

AGENTIC_MAX_ITERS = int(os.getenv("AGENTIC_MAX_ITERS", "2"))


ENABLE_LLM_CLARIFY = os.getenv("ENABLE_LLM_CLARIFY", "true").lower() == "true"

FAQ_SIMILARITY_THRESHOLD = 0.45

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

RAG_TOP_K = 8

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# Порог релевантности по score реранкера (cross-encoder). Если ЛУЧШИЙ найденный
# чанк набрал меньше — считаем, что в базе нет ответа, и честно отказываем,
# вместо того чтобы галлюцинировать по нерелевантному контексту.
# Калибровка по тестовому набору (grid search): score'ы релевантных и внебазовых
# вопросов перекрываются, поэтому это КОНСЕРВАТИВНЫЙ backstop, а не жёсткий фильтр.
#   -4.0  -> ложно режет ~1% валидных, ловит ~60% внебазовых (выбран баланс)
#   -2.0  -> ловит ~100% внебазовых, но рубит ~17% валидных — слишком агрессивно
# В проде запросы перед реранком чистит LLM-rewrite, поэтому реальный отсев ещё мягче.
RELEVANCE_MIN_SCORE = float(os.getenv("RELEVANCE_MIN_SCORE", "-4.0"))

MAX_HISTORY_PAIRS = 5

CACHE_FILE = Path(__file__).parent / "dynamic_cache.json"

DISLIKE_LOG = Path(__file__).parent / "disliked.log"

QUARANTINE_FILE = Path(__file__).parent / "quarantine.jsonl"

CHROMA_DIR = Path(__file__).parent / "chroma_db"

ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

RATE_LIMIT_MAX = 12
RATE_LIMIT_WINDOW = 60

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("PORT", "8443"))

EVENTS_LOG = Path(__file__).parent / "bot_events.jsonl"
VOTES_LOG = Path(__file__).parent / "bot_votes.jsonl"
