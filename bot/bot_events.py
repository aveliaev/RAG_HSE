import json
import hashlib
import logging
from datetime import datetime

from config import EVENTS_LOG, VOTES_LOG

log = logging.getLogger(__name__)

_SALT = "fkn_hse_bot"


def _hash_uid(user_id: int) -> str:
    return hashlib.sha256(f"{_SALT}{user_id}".encode()).hexdigest()[:12]


def log_interaction(
    *,
    user_id: int,
    question: str,
    route: str,
    source: str,
    answer: str,
    sub_queries: list[str],
    lang: str,
    latency_ms: float,
    msg_id: int,
) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "uid": _hash_uid(user_id),
        "msg_id": msg_id,
        "question": question,
        "route": route,
        "source": source,
        "answer": answer[:800],
        "sub_queries": sub_queries,
        "lang": lang,
        "latency_ms": round(latency_ms),
        "vote": None,
    }
    try:
        with EVENTS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Не удалось записать событие: %s", e)


def log_vote(msg_id: int, vote: str) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "msg_id": msg_id,
        "vote": vote,
    }
    try:
        with VOTES_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Не удалось записать голос: %s", e)
