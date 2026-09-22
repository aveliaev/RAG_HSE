import json
import logging
from datetime import datetime

from config import EVENTS_LOG, VOTES_LOG
from privacy import hash_uid, redact_pii

log = logging.getLogger(__name__)

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
    clarify_asked: str = "",
    clarify_reply: str = "",
) -> None:
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "uid": hash_uid(user_id),
        "msg_id": msg_id,
        "question": redact_pii(question),
        "route": route,
        "source": source,
        "answer": redact_pii(answer[:800]),
        "sub_queries": sub_queries,
        "lang": lang,
        "latency_ms": round(latency_ms),
        "vote": None,
    }
    # Если ответ дан после уточнения — сохраняем весь тред, чтобы дашборд показал
    # исходный вопрос → что бот переспросил → ответ пользователя → финальный ответ.
    if clarify_asked:
        record["clarify_asked"] = clarify_asked
        record["clarify_reply"] = redact_pii(clarify_reply)
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
