import logging
import aiohttp

from config import YANDEX_API_KEY, YANDEX_FOLDER_ID

log = logging.getLogger(__name__)

_STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


async def transcribe_ogg(audio_bytes: bytes, lang: str = "ru-RU") -> str | None:
    """Send OGG Opus audio bytes to Yandex SpeechKit STT, return transcript or None."""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        log.warning("YANDEX_API_KEY or YANDEX_FOLDER_ID not set — voice disabled")
        return None

    params = {
        "folderId": YANDEX_FOLDER_ID,
        "lang": lang,
        "topic": "general",
        "profanityFilter": "false",
    }
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "audio/ogg;codecs=opus",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _STT_URL, params=params, headers=headers, data=audio_bytes, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.error("SpeechKit error %d: %s", resp.status, body[:200])
                    return None
                data = await resp.json()
                return data.get("result") or None
    except Exception as e:
        log.error("SpeechKit request failed: %s", e)
        return None
