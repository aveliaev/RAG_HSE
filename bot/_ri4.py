"""Проверка разбиения QA-файла на чанки + полная переиндексация ChromaDB.

Запуск:  python _ri4.py [имя_файла_без_.md]   (по умолчанию fkn_faq_official_rag)
"""
import os
import sys
os.environ["USE_YANDEX"]="false"; os.environ["GROQ_API_KEY"]=""; os.environ["ENABLE_LLM_REWRITE"]="false"
import rag_engine as R
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "dataset"
source = sys.argv[1] if len(sys.argv) > 1 else "fkn_faq_official_rag"
path = DATASET / f"{source}.md"

if path.exists():
    txt = path.read_text(encoding="utf-8")
    ch = R._split_into_chunks(txt, source)
    print(f"{path.name}: QA-формат: {R._is_qa_format(txt)} | чанков: {len(ch)}")
else:
    print(f"Файл {path} не найден — пропускаю проверку, делаю только переиндексацию")

try:
    R._client.delete_collection("fkn_rag")
except Exception:
    pass  # коллекции ещё нет — создадим с нуля
R._collection = R._client.get_or_create_collection(name="fkn_rag", embedding_function=R._embed_fn, metadata={"hnsw:space":"cosine"})
n = R.build_index(); print(f"REINDEX: {n} чанков")
