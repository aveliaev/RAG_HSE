import os
os.environ["USE_YANDEX"] = "false"
os.environ["GROQ_API_KEY"] = ""
os.environ["ENABLE_LLM_REWRITE"] = "false"

import inspect
import faq_cache as FC
import rag_engine as RE
import knowledge_base as KB
from agentic_rag import _category_queries, decompose_query, _augment_categories

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok; FAIL += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if (detail and not ok) else ""))

print("\n== 1. КЕШ: короткие/мусорные запросы не отдаются из кеша ==")
for q in ["ПАД", "пми", "эад", "ami"]:
    ans, src = FC.lookup(q)
    check(f"lookup({q!r}) -> RAG (не кеш)", ans is None and src == "rag", f"src={src}")
ans, src = FC.lookup("какие баллы для эад")
check("валидный длинный запрос всё ещё ловится кешем", ans is not None and src == "dynamic", f"src={src}")

print("\n== 2. КЕШ: строгие правила добавления ==")
n0 = len(FC._DYNAMIC_NORM)
FC.add_to_dynamic_cache("пад", "что-то")
check("короткий запрос НЕ добавляется", len(FC._DYNAMIC_NORM) == n0)
FC.add_to_dynamic_cache("это нормальный длинный вопрос про что", "В базе знаний нет точной информации по этому вопросу.")
check("ответ-заглушка НЕ добавляется", len(FC._DYNAMIC_NORM) == n0)
FC.add_to_dynamic_cache("какие баллы для пми", "любой ответ")
check("блэклистнутый запрос НЕ добавляется", len(FC._DYNAMIC_NORM) == n0)
# валидное добавление и откат (без сохранения на диск)
FC._DYNAMIC_NORM_BACKUP = dict(FC._DYNAMIC_NORM)
added_key = FC._normalize("сколько бюджетных мест на робототехнике в москве")
FC._DYNAMIC_NORM[added_key] = "test"   # имитируем без записи на диск
check("валидный длинный запрос добавился бы", FC._normalize("сколько бюджетных мест на робототехнике в москве").split().__len__() >= FC._MIN_CACHE_TOKENS)
FC._DYNAMIC_NORM = FC._DYNAMIC_NORM_BACKUP

print("\n== 3. ИСТОРИЯ не передаётся в generate() ==")
src = inspect.getsource(RE.generate)
check("нет цикла по history в generate()", ("for msg in history" not in src) and ("for m in history" not in src))
check("в комментарии зафиксировано почему", "историю" in src.lower() or "history" in src.lower())

print("\n== 4. УТОЧНЕНИЯ ==")
def nc(q): return RE.needs_clarification(q, [])
check("'хочу на бюджет' -> переспрос программы", nc("хочу на бюджет") and "программу" in nc("хочу на бюджет"))
check("'хочу поступить' -> переспрос", bool(nc("хочу поступить")))
check("'хочу поступить на ПМИ' -> НЕ переспрашивает (программа есть)", nc("хочу поступить на ПМИ") is None)
check("'хочу на бюджет, баллы на КНАД' -> НЕ переспрашивает", nc("хочу на бюджет, какие баллы на КНАД") is None)
check("'какие минимальные баллы' (без программы) -> переспрос", bool(nc("какие минимальные баллы нужны")))
check("'минимальные баллы на ПАД' -> НЕ переспрашивает", nc("какие минимальные баллы на ПАД") is None)

print("\n== 5. КОНТЕНТ-ФИЛЬТР ==")
check("мат блокируется", RE.check_content("какого хуя") is not None)
check("оффтоп (погода) блокируется", RE.check_content("какая погода в москве") is not None)
check("нормальный вопрос проходит", RE.check_content("какие баллы на пми") is None)

print("\n== 6. CATEGORY-АУГМЕНТАЦИЯ (иностранцы/ОВЗ) ==")
Q = "Если я поступаю как иностранец с ОВЗ, какие документы и в какие сроки мне подавать?"
cats = _category_queries(Q)
check("оба категориальных под-запроса сгенерированы", len(cats) == 2, f"got {len(cats)}")
aug = _augment_categories(Q, ["базовый запрос"])
check("категориальные идут ПЕРВЫМИ", aug[0] in cats and aug[1] in cats)
check("только иностранец -> 1 категория", len(_category_queries("я иностранец, что сдавать")) == 1)
check("только ОВЗ -> 1 категория", len(_category_queries("я инвалид по слуху, какие условия")) == 1)
check("обычный вопрос -> 0 категорий", len(_category_queries("какие баллы на пми")) == 0)

print("\n== 7. СИСТЕМНЫЙ ПРОМПТ: новые директивы ==")
check("запрет выбирать программу", "ПРОГРАММА НЕ УКАЗАНА" in KB.SYSTEM_PROMPT)
check("запрет домысливать статус", "НЕ ДОМЫСЛИВАЙ СТАТУС" in KB.SYSTEM_PROMPT)
check("факты только из контекста", "ИСКЛЮЧИТЕЛЬНО" in KB.SYSTEM_PROMPT)

print("\n== 8. РЕТРИВЕР: новый материал в индексе ==")
def top_sources(q, k=4):
    return [c["source"] for c in RE.retrieve_k(q, k=k, skip_llm_rewrite=True)]
check("перечень документов -> fkn_documents_rag", "fkn_documents_rag" in top_sources("какие документы нужны при подаче заявления"))
check("признание диплома/IELTS -> fkn_foreign_rag", "fkn_foreign_rag" in top_sources("признают ли сертификат IELTS, нужно ли признание иностранного диплома"))
check("условия ОВЗ -> fkn_ovz_rag", "fkn_ovz_rag" in top_sources("особые условия вступительных испытаний для инвалидов"))

print("\n== 9. AGENTIC включён ==")
import config
check("USE_AGENTIC_RAG == True", config.USE_AGENTIC_RAG is True)

print(f"\n==== ИТОГО: PASS={PASS}  FAIL={FAIL} ====")
