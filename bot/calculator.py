import re
from dataclasses import dataclass

@dataclass
class Program:
    short: str
    name: str
    second_subjects: list
    min_math: int
    min_second: dict
    min_russian: int
    has_budget: bool
    passing_budget_2024: int | None
    budget_places: int
    notes: str = ""

PROGRAMS = [
    Program(
        short="ПМИ",
        name="Прикладная математика и информатика",
        second_subjects=["информатика"],
        min_math=75, min_second={"информатика": 75}, min_russian=60,
        has_budget=True, passing_budget_2024=294, budget_places=40,
    ),
    Program(
        short="Прогинж",
        name="Программная инженерия",
        second_subjects=["информатика"],
        min_math=70, min_second={"информатика": 70}, min_russian=60,
        has_budget=True, passing_budget_2024=282, budget_places=35,
    ),
    Program(
        short="КНАД",
        name="Компьютерные науки и анализ данных",
        second_subjects=["информатика", "физика"],
        min_math=70, min_second={"информатика": 65, "физика": 65}, min_russian=60,
        has_budget=True, passing_budget_2024=263, budget_places=30,
        notes="Онлайн-программа",
    ),
    Program(
        short="ЭАД",
        name="Экономика и анализ данных",
        second_subjects=["обществознание", "информатика", "иностранный"],
        min_math=70,
        min_second={"обществознание": 65, "информатика": 65, "иностранный": 65},
        min_russian=65,
        has_budget=True, passing_budget_2024=280, budget_places=30,
    ),
    Program(
        short="Робототехника",
        name="Проектирование интеллект. робототехнических систем",
        second_subjects=["информатика", "физика"],
        min_math=70, min_second={"информатика": 70, "физика": 70}, min_russian=60,
        has_budget=True, passing_budget_2024=268, budget_places=25,
    ),
    Program(
        short="ПАД",
        name="Прикладной анализ данных",
        second_subjects=["информатика", "физика"],
        min_math=70, min_second={"информатика": 60, "физика": 60}, min_russian=60,
        has_budget=False, passing_budget_2024=None, budget_places=0,
        notes="Только платное. Язык обучения — английский",
    ),
    Program(
        short="ДРИП",
        name="Дизайн и разработка информационных продуктов",
        second_subjects=["информатика", "физика"],
        min_math=70, min_second={"информатика": 60, "физика": 60}, min_russian=60,
        has_budget=False, passing_budget_2024=None, budget_places=0,
        notes="Только платное",
    ),
    Program(
        short="Игры",
        name="Разработка игр и цифровых продуктов",
        second_subjects=["информатика", "физика"],
        min_math=65, min_second={"информатика": 60, "физика": 60}, min_russian=60,
        has_budget=False, passing_budget_2024=None, budget_places=0,
        notes="Только платное",
    ),
]

_GENERAL_ACHIEVEMENTS: list[tuple] = [
    (re.compile(r'медал[ьея]|аттестат\s+с\s+отличи'), 3, "медаль/аттестат"),
    (re.compile(r'волонтёр|волонтер'), 2, "волонтёрство"),
    (re.compile(r'\bгто\b'), 2, "ГТО золотой"),
    (re.compile(r'большая\s+перемена'), 2, "Большая перемена"),
    (re.compile(r'доброволец|мывместе|#мывместе'), 2, "Доброволец России"),
    (re.compile(r'моя\s+стран|умножая\s+талант|эколог.*дело'), 2, "конкурс (+2)"),
    (re.compile(r'кмс|кандидат\s+в\s+мастер'), 5, "КМС"),
    (re.compile(r'мастер\s+спорта'), 7, "МС"),
    (re.compile(r'чемпион\s+(?:мира|европы|олимп)|олимпийских\s+игр|паралимпийск|сурдлимп'), 10, "чемпион ОИ/мира"),
    (re.compile(r'абилимпикс'), 10, "Абилимпикс"),
    (re.compile(r'военн|армия|мобилизац|\bсво\b'), 10, "военная служба/СВО"),
]

_OLYMPIAD_GENERAL: list[tuple] = [
    (re.compile(r'всош|всероссийск.*заключительн|заключительн.*этап.*олимп'), 10, "ВсОШ заключительный"),
    (re.compile(r'перечнев|диплом\s+олимпиад'), 6, "перечневая олимпиада"),
]

_SPECIAL: dict[str, list[tuple]] = {}

def _sp(*args) -> list[tuple]:
    return [(re.compile(pat, re.I), pts, name) for pat, pts, name in args]

_COMMON_SPECIAL = _sp(
    (r'лице[ейя]\s+яндекс', 5, "Лицей Яндекса"),
    (r'конкурс\s+фкн|конкурс.*абитур.*фкн|исследовательск.*фкн', 10, "Конкурс ФКН"),
    (r'т.поколени|тпоколени', 10, "Т-поколение"),
    (r'\bмшп\b|московск.*школ.*программ', 5, "МШП"),
    (r'\bprod\b|прод\b|промышленн.*разработ', 8, "PROD"),
)

for _p in ["ПМИ", "Прогинж", "КНАД", "ПАД", "Игры", "Робототехника"]:
    _SPECIAL[_p] = _COMMON_SPECIAL

_SPECIAL["ДРИП"] = _sp(
    (r'т.поколени|тпоколени', 10, "Т-поколение"),
    (r'\bмшп\b|московск.*школ.*программ', 5, "МШП"),
    (r'\bprod\b|прод\b|промышленн.*разработ', 8, "PROD"),
    (r'эйлер', 7, "Эйлер"),
    (r'келдыш', 8, "Келдыш"),
)

_SPECIAL["ЭАД"] = _sp(
    (r'региональн.*всош|всош.*региональн', 3, "ВсОШ региональный"),
    (r'экономическ.*школ|школ.*фэн', 8, "Экономическая школа ФЭН"),
    (r'высший\s+пилотаж', 8, "Высший пилотаж"),
)

def _extract_score(text: str, *patterns: str) -> int | None:
    for pat in patterns:
        m = re.search(rf'(\d{{2,3}})\s*(?:по\s*)?{pat}', text, re.I)
        if m:
            return int(m.group(1))
        m = re.search(rf'{pat}[:\s]+(\d{{2,3}})', text, re.I)
        if m:
            return int(m.group(1))
    return None

def parse_scores(text: str) -> dict:
    t = text.lower()

    math = _extract_score(t, r'мат(?:ематик[аеу])?')
    russian = _extract_score(t, r'рус(?:ск(?:ий|ого|ому))?(?:\s+яз(?:ык)?)?')

    second_subject = None
    second_score = None

    inf = _extract_score(t, r'инф(?:орматик[аеу]|у|а)?(?:\s+и\s+икт)?')
    phy = _extract_score(t, r'физик[аеу]?')
    soc = _extract_score(t, r'обществ(?:ознание|о)?')
    eng = _extract_score(t, r'(?:английский|англ(?:\.|ий)?|иностранный|немецкий|французский|испанский)')

    if inf is not None:
        second_subject, second_score = "информатика", inf
    elif phy is not None:
        second_subject, second_score = "физика", phy
    elif soc is not None:
        second_subject, second_score = "обществознание", soc
    elif eng is not None:
        second_subject, second_score = "иностранный", eng

    return {
        "math": math,
        "second_subject": second_subject,
        "second_score": second_score,
        "russian": russian,
        "achievements_text": t,
    }

def _best_match(patterns: list[tuple], text: str) -> tuple[int, str]:
    best_pts, best_name = 0, ""
    for rx, pts, name in patterns:
        if rx.search(text) and pts > best_pts:
            best_pts, best_name = pts, name
    return best_pts, best_name

def _compute_achievements(achievements_text: str, program_short: str) -> tuple[int, list[str]]:
    t = achievements_text

    best_a, name_a = _best_match(_GENERAL_ACHIEVEMENTS, t)
    best_b, name_b = _best_match(_OLYMPIAD_GENERAL, t)
    best_c, name_c = _best_match(_SPECIAL.get(program_short, []), t)

    total = min(best_a + best_b + best_c, 10)

    breakdown = []
    if best_a:
        breakdown.append(f"{name_a} +{best_a}")
    if best_b:
        breakdown.append(f"{name_b} +{best_b}")
    if best_c:
        breakdown.append(f"{name_c} +{best_c}")
    if not breakdown:
        breakdown.append("ИД не указаны (0)")

    return total, breakdown

_MEDAL = ["🥇", "🥈", "🥉"]

_SUBJ_DISPLAY = {
    "информатика":    ("💻", "Информатика"),
    "физика":         ("⚛️",  "Физика"),
    "обществознание": ("📚", "Обществознание"),
    "иностранный":    ("🌍", "Иностранный яз."),
}

def _pbar(score: int, passing: int, width: int = 10) -> str:
    pct = min(score / passing, 1.0)
    filled = round(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {round(pct * 100)}%"

def calculate(math: int, second_subject: str, second_score: int,
              russian: int, achievements_text: str = "") -> str:
    second_subject = second_subject.lower().strip()
    results = []

    for prog in PROGRAMS:
        if second_subject not in prog.second_subjects:
            continue

        min_s = prog.min_second.get(second_subject, 999)
        fails_min = []
        if math < prog.min_math:
            fails_min.append(f"математика {math} < {prog.min_math}")
        if second_score < min_s:
            fails_min.append(f"{second_subject} {second_score} < {min_s}")
        if russian < prog.min_russian:
            fails_min.append(f"русский {russian} < {prog.min_russian}")

        id_pts, id_breakdown = _compute_achievements(achievements_text, prog.short)
        total = math + second_score + russian + id_pts

        results.append({
            "prog": prog,
            "total": total,
            "id_pts": id_pts,
            "id_breakdown": id_breakdown,
            "fails_min": fails_min,
            "passes_min": len(fails_min) == 0,
        })

    results.sort(key=lambda r: (r["passes_min"], r["total"]), reverse=True)

    if not results:
        return (
            "По указанному предмету ни одна программа ФКН не подходит.\n"
            "Информатика — для большинства программ; физика — КНАД, Робото, Игры, ДРИП, ПАД; "
            "обществознание — только ЭАД."
        )

    ege_sum = math + second_score + russian
    s_em, s_name = _SUBJ_DISPLAY.get(second_subject, ("📋", second_subject.capitalize()))

    lines = [
        "🎓 <b>Конкурсный балл ФКН НИУ ВШЭ</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  📐 Математика — <b>{math}</b>",
        f"  {s_em} {s_name} — <b>{second_score}</b>",
        f"  📝 Русский язык — <b>{russian}</b>",
        "  ──────────────────",
        f"  <b>Σ ЕГЭ — {ege_sum} баллов</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "  🏆 <b>Топ программ для вас</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
    ]

    top3 = results[:3]
    for i, r in enumerate(top3):
        prog = r["prog"]
        total = r["total"]
        id_pts = r["id_pts"]
        medal = _MEDAL[i]

        lines.append(f"{medal} <b>{prog.short}</b>")
        lines.append(f"  <i>{prog.name}</i>")
        if prog.notes:
            lines.append(f"  ℹ️ <i>{prog.notes}</i>")
        lines.append("")

        if id_pts > 0:
            id_detail = "  ·  ".join(r["id_breakdown"])
            lines.append(f"  {ege_sum} + {id_pts} ИД = <b>{total} баллов</b>")
            lines.append(f"  <i>🏅 {id_detail}</i>")
        else:
            lines.append(f"  <b>{total} баллов</b>  <i>(без ИД)</i>")
        lines.append("")

        if not r["passes_min"]:
            lines.append("  ❌ Не проходите по минимальным баллам:")
            for fail in r["fails_min"]:
                lines.append(f"    · {fail}")
        elif prog.has_budget and prog.passing_budget_2024:
            diff = total - prog.passing_budget_2024
            lines.append(f"  {_pbar(total, prog.passing_budget_2024)}")
            if diff >= 0:
                lines.append(
                    f"  ✅ Бюджет вероятен"
                    f"  <i>(проходной {prog.passing_budget_2024}, запас +{diff} б.)</i>"
                )
            elif diff >= -10:
                lines.append(
                    f"  ⚠️ На грани бюджета"
                    f"  <i>(проходной {prog.passing_budget_2024}, не хватает {abs(diff)} б.)</i>"
                )
            else:
                lines.append(
                    f"  💰 Скорее платное"
                    f"  <i>(проходной {prog.passing_budget_2024}, не хватает {abs(diff)} б.)</i>"
                )
        else:
            lines.append("  💳 Только платное  <i>(бюджета нет)</i>")

        lines.append("")
        if i < len(top3) - 1:
            lines.append("╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌")
            lines.append("")

    if len(results) > 3:
        rest = results[3:]
        rest_ok = [r for r in rest if r["passes_min"]]
        rest_fail = [r for r in rest if not r["passes_min"]]
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        if rest_ok:
            lines.append("Также проходите: " + "  ·  ".join(r["prog"].short for r in rest_ok))
        if rest_fail:
            lines.append("Не прошли по минималкам: " + "  ·  ".join(r["prog"].short for r in rest_fail))
        lines.append("")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "<i>⚠️ Проходные — ориентировочные данные 2024 г.</i>",
        "<i>Актуальные конкурсные списки на ba.hse.ru с 27 июля 2026.</i>",
    ]

    return "\n".join(lines)

_CALC_KEYWORDS = re.compile(
    r"рассчита[йте]|посчита[йте]|калькул|мой\s+балл|"
    r"прохожу\s+ли|конкурсный\s+балл|"
    r"у\s+меня\s+\d+\s*(?:по\s*)?(?:мат|инф|физ|рус)|"
    r"\d+\s+(?:по\s*)?математик|"
    r"(?:математик[аеу]|мат)\s*[\s:]\s*\d+",
    re.I,
)

_REQUEST_TEMPLATE = (
    "Чтобы рассчитать конкурсный балл, напиши свои данные одним сообщением. Например:\n\n"
    "<i>Математика 87, информатика 82, русский 74. Есть медаль и победа в Лицее Яндекса.</i>\n\n"
    "Что можно указать:\n"
    "— Баллы ЕГЭ: математика, русский + один второй предмет "
    "(информатика / физика / обществознание / иностранный)\n"
    "— Достижения (необязательно): медаль, ГТО, Лицей Яндекса, Т-поколение, МШП, PROD, волонтёрство, "
    "КМС/МС, Абилимпикс, военная служба и др.\n\n"
    "Я подберу топ‑3 программы ФКН и сравню с проходными баллами 2024 года."
)

def try_calculate(text: str) -> str | None:
    if not _CALC_KEYWORDS.search(text):
        return None

    parsed = parse_scores(text)
    math = parsed["math"]
    second_subject = parsed["second_subject"]
    second_score = parsed["second_score"]
    russian = parsed["russian"]

    if math is None or second_subject is None or second_score is None or russian is None:
        return _REQUEST_TEMPLATE

    for score, name in [(math, "математика"), (second_score, second_subject), (russian, "русский")]:
        if not (0 <= score <= 100):
            return f"Балл по {name} должен быть от 0 до 100. Проверь данные и повтори."

    return calculate(
        math=math,
        second_subject=second_subject,
        second_score=second_score,
        russian=russian,
        achievements_text=parsed["achievements_text"],
    )

SUBJ_OPTIONS: list[tuple] = [
    ("info",  "💻", "Информатика",    "информатика"),
    ("phys",  "⚛️",  "Физика",         "физика"),
    ("soc",   "📚", "Обществознание", "обществознание"),
    ("eng",   "🌍", "Иностранный яз.", "иностранный"),
]

ACH_OPTIONS: list[tuple] = [
    ("medal",     "📜", "Медаль / Аттестат с отличием",     "медаль",             None),
    ("gto",       "🏅", "ГТО золотой знак",                  "гто",                None),
    ("vol",       "🤝", "Волонтёрство (100+ часов)",         "волонтёрство",       None),
    ("dobro",     "🫂", "Доброволец России / МЫВМЕСТЕ",      "доброволец",         None),
    ("peremen",   "🌟", "Большая перемена",                  "большая перемена",   None),
    ("other2",    "🏅", "Мой край / Умножая таланты / Экология",  "моя страна",    None),
    ("kms",       "🥈", "КМС",                               "кмс",                None),
    ("ms",        "🥇", "Мастер спорта и выше",              "мастер спорта",      None),
    ("olimp",     "🏆", "Чемпион ОИ / мира / Европы",       "чемпион мира",       None),
    ("abil",      "🎓", "Абилимпикс",                        "абилимпикс",         None),
    ("army",      "⚔️",  "Армия / СВО",                     "армия",              None),
    ("vsosh",     "🏆", "ВсОШ — заключительный этап",        "всош",               None),
    ("perechen",  "📋", "Перечневая олимпиада (без БВИ)",    "перечневая",         None),
    ("lyceum",    "🤖", "Лицей Яндекса (+5)",                "лицей яндекса",      {"info", "phys"}),
    ("fkn",       "🔬", "Конкурс ФКН (+10)",                 "конкурс фкн",        {"info", "phys"}),
    ("tpok",      "⚡", "Т-поколение (+10)",                  "т-поколение",        {"info", "phys"}),
    ("mshp",      "🖥", "МШП (+5)",                          "мшп",                {"info", "phys"}),
    ("prod",      "🏭", "PROD (+8)",                         "prod",               {"info", "phys"}),
    ("eiler",     "📐", "Эйлер — ДРИП (+7)",                 "эйлер",              {"info", "phys"}),
    ("keldysh",   "💡", "Келдыш — ДРИП (+8)",                "келдыш",             {"info", "phys"}),
    ("econ_sch",  "📊", "Экономическая школа ФЭН (+8)",      "экономическая школа", {"soc", "eng"}),
    ("vpilotazh", "✈️",  "Высший пилотаж (+8)",              "высший пилотаж",     {"soc", "eng"}),
    ("vsosh_reg", "🥉", "ВсОШ региональный — ЭАД (+3)",      "региональный всош",  {"soc", "eng"}),
]

_ACH_KW: dict[str, str] = {aid: kw for aid, _, _, kw, *_ in ACH_OPTIONS}
SUBJ_KW: dict[str, str] = {sid: kw for sid, _, _, kw in SUBJ_OPTIONS}

def ach_ids_to_text(ids: set) -> str:
    return " ".join(_ACH_KW[i] for i in ids if i in _ACH_KW)

def calculate_from_wizard(
    subj_id: str,
    scores: list[int],
    ach_ids: set[str],
) -> str:
    return calculate(
        math=scores[0],
        second_subject=SUBJ_KW.get(subj_id, ""),
        second_score=scores[1],
        russian=scores[2],
        achievements_text=ach_ids_to_text(ach_ids),
    )
