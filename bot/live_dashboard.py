import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EVENTS_LOG = Path(__file__).parent / "bot_events.jsonl"
VOTES_LOG = Path(__file__).parent / "bot_votes.jsonl"

st.set_page_config(
    page_title="Live Dashboard — ФКН бот",
    page_icon="🟢",
    layout="wide",
)

st.title("🟢 Live Dashboard — Бот ФКН НИУ ВШЭ")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Настройки")
    auto_refresh = st.toggle("Автообновление", value=True)
    refresh_sec = st.slider("Интервал (сек)", 5, 60, 10)
    n_last = st.slider("Последних событий", 20, 200, 50)
    show_answers = st.toggle("Показывать ответы бота", value=True)

    st.divider()
    route_filter = st.multiselect(
        "Фильтр по маршруту",
        options=["faq", "rag", "calc"],
        default=["faq", "rag", "calc"],
    )
    vote_filter = st.radio(
        "Фильтр по оценке",
        options=["Все", "👍 Лайки", "👎 Дизлайки", "Без оценки"],
        index=0,
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


@st.cache_data(ttl=refresh_sec)
def load_events() -> pd.DataFrame:
    records = _load_jsonl(EVENTS_LOG)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", ascending=False)
    return df


@st.cache_data(ttl=refresh_sec)
def load_votes() -> dict[int, str]:
    records = _load_jsonl(VOTES_LOG)
    result: dict[int, str] = {}
    for r in records:
        mid = r.get("msg_id")
        vote = r.get("vote")
        if mid is not None and vote:
            result[mid] = vote
    return result


df = load_events()
votes = load_votes()

if df.empty:
    st.warning(
        "Файл `bot_events.jsonl` не найден или пуст. "
        "Запустите бот и подождите первых сообщений."
    )
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
    st.stop()

# Join votes
df["vote"] = df["msg_id"].map(votes)

# Apply route filter
df_filtered = df[df["route"].isin(route_filter)].copy()

# Apply vote filter
if vote_filter == "👍 Лайки":
    df_filtered = df_filtered[df_filtered["vote"] == "up"]
elif vote_filter == "👎 Дизлайки":
    df_filtered = df_filtered[df_filtered["vote"] == "down"]
elif vote_filter == "Без оценки":
    df_filtered = df_filtered[df_filtered["vote"].isna()]


# ── Key metrics ───────────────────────────────────────────────────────────────
now = datetime.now()
today_df = df[df["ts"] >= pd.Timestamp(now.date())]
last_hour_df = df[df["ts"] >= pd.Timestamp(now - timedelta(hours=1))]

total = len(df)
today_total = len(today_df)
last_hour_total = len(last_hour_df)

n_likes = (df["vote"] == "up").sum()
n_dislikes = (df["vote"] == "down").sum()
n_voted = n_likes + n_dislikes
like_rate = n_likes / n_voted if n_voted else 0.0

faq_pct = (df["route"] == "faq").mean() if total else 0.0
avg_lat = df["latency_ms"].dropna().mean() if "latency_ms" in df.columns else 0

st.markdown("## Ключевые метрики")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Всего запросов", total)
col2.metric("Сегодня", today_total)
col3.metric("За последний час", last_hour_total)
col4.metric("Cache hit %", f"{faq_pct*100:.0f}%", help="Доля ответов из FAQ-кеша")
col5.metric(
    "👍 Рейтинг",
    f"{like_rate*100:.0f}%" if n_voted else "—",
    help=f"Лайков {n_likes} / Дизлайков {n_dislikes}",
)
col6.metric("Avg latency", f"{avg_lat:.0f} мс" if avg_lat else "—")

st.divider()


# ── Charts row ────────────────────────────────────────────────────────────────
ch1, ch2, ch3 = st.columns(3)

with ch1:
    st.markdown("### Запросов по часам (сегодня)")
    if not today_df.empty:
        hourly = (
            today_df.set_index("ts")
            .resample("1h")
            .size()
            .reset_index(name="count")
        )
        hourly["hour"] = hourly["ts"].dt.strftime("%H:00")
        fig_hr = px.bar(
            hourly, x="hour", y="count",
            color_discrete_sequence=["#636efa"],
            labels={"hour": "", "count": "Запросов"},
        )
        fig_hr.update_layout(height=260, margin=dict(t=10, b=30))
        st.plotly_chart(fig_hr, use_container_width=True)
    else:
        st.info("Нет данных за сегодня")

with ch2:
    st.markdown("### Маршруты")
    route_counts = df["route"].value_counts().reset_index()
    route_counts.columns = ["route", "count"]
    fig_pie = px.pie(
        route_counts, values="count", names="route",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.45,
    )
    fig_pie.update_layout(height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with ch3:
    st.markdown("### Оценки пользователей")
    vote_df = pd.DataFrame([
        {"label": "👍 Лайк", "count": int(n_likes)},
        {"label": "👎 Дизлайк", "count": int(n_dislikes)},
        {"label": "Без оценки", "count": int(total - n_voted)},
    ])
    fig_vote = px.bar(
        vote_df, x="label", y="count",
        color="label",
        color_discrete_map={
            "👍 Лайк": "#00cc96",
            "👎 Дизлайк": "#ef553b",
            "Без оценки": "#aaaaaa",
        },
        labels={"label": "", "count": ""},
    )
    fig_vote.update_layout(height=260, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_vote, use_container_width=True)

st.divider()


# ── Latency timeline ──────────────────────────────────────────────────────────
if "latency_ms" in df.columns and df["latency_ms"].notna().any():
    st.markdown("### ⏱ Latency по времени")
    lat_df = df[df["latency_ms"].notna()].copy()
    lat_df = lat_df.sort_values("ts")
    fig_lat = px.scatter(
        lat_df, x="ts", y="latency_ms",
        color="route",
        color_discrete_map={"faq": "#00cc96", "rag": "#636efa", "calc": "#ffa15a"},
        labels={"ts": "", "latency_ms": "мс", "route": "Маршрут"},
        opacity=0.7,
    )
    fig_lat.add_hline(
        y=lat_df["latency_ms"].mean(),
        line_dash="dash", line_color="orange",
        annotation_text=f"avg {lat_df['latency_ms'].mean():.0f}мс",
    )
    fig_lat.update_layout(height=250, margin=dict(t=10, b=30))
    st.plotly_chart(fig_lat, use_container_width=True)
    st.divider()


# ── Disliked questions ────────────────────────────────────────────────────────
disliked = df[df["vote"] == "down"][["ts", "question", "route", "source", "latency_ms"]]
if not disliked.empty:
    st.markdown(f"### 👎 Недовольные ответы ({len(disliked)})")
    st.dataframe(
        disliked.rename(columns={
            "ts": "Время", "question": "Вопрос",
            "route": "Маршрут", "source": "Источник", "latency_ms": "мс",
        }).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()


# ── Live feed ─────────────────────────────────────────────────────────────────
st.markdown(f"## Лента событий (последние {min(n_last, len(df_filtered))} из {len(df_filtered)})")

feed = df_filtered.head(n_last)

if feed.empty:
    st.info("Нет событий по выбранным фильтрам.")
else:
    for _, row in feed.iterrows():
        vote = row.get("vote")
        route = row.get("route", "?")
        lang = row.get("lang", "ru")
        latency = row.get("latency_ms")
        ts = row["ts"].strftime("%H:%M:%S") if pd.notna(row["ts"]) else "?"

        if vote == "up":
            vote_icon = "👍"
        elif vote == "down":
            vote_icon = "👎"
        else:
            vote_icon = "⬜"

        route_color = {"faq": "🟢", "rag": "🔵", "calc": "🟠"}.get(route, "⚪")
        lat_str = f"⏱ {latency:.0f}мс" if latency else ""
        label = (
            f"{vote_icon} {route_color} `{ts}` [{route}]  "
            f"{row['question'][:90]}{'…' if len(row['question']) > 90 else ''}  {lat_str}"
        )

        with st.expander(label, expanded=False):
            left, right = st.columns([1, 1])

            with left:
                st.markdown("**Вопрос пользователя:**")
                st.write(row["question"])

                sub_queries = row.get("sub_queries") or []
                if isinstance(sub_queries, list) and sub_queries:
                    st.markdown("**Декомпозиция запроса (RAG):**")
                    for i, q in enumerate(sub_queries, 1):
                        st.caption(f"{i}. {q}")

                meta_parts = []
                meta_parts.append(f"**Маршрут:** `{route}`")
                meta_parts.append(f"**Источник:** `{row.get('source', '?')}`")
                meta_parts.append(f"**Язык:** `{lang}`")
                if latency:
                    meta_parts.append(f"**Latency:** {latency:.0f} мс")
                st.markdown("  ·  ".join(meta_parts))

            with right:
                if show_answers:
                    answer = str(row.get("answer") or "").strip()
                    if answer:
                        st.markdown("**Ответ бота:**")
                        st.info(answer[:600] + ("…" if len(answer) > 600 else ""))
                    else:
                        st.caption("Ответ не записан")

                if vote == "up":
                    st.success("👍 Пользователь подтвердил ответ")
                elif vote == "down":
                    st.error("👎 Пользователь отметил ответ неверным")
                else:
                    st.caption("Оценка ещё не поставлена")


# ── Sub-queries breakdown ─────────────────────────────────────────────────────
rag_with_sq = df[
    (df["route"] == "rag") &
    df["sub_queries"].apply(lambda x: isinstance(x, list) and len(x) > 1)
]
if not rag_with_sq.empty:
    st.divider()
    st.markdown(f"### 🔍 Декомпозиция запросов (RAG, {len(rag_with_sq)} шт.)")
    sq_df = rag_with_sq[["ts", "question", "sub_queries", "vote"]].copy()
    sq_df["ts"] = sq_df["ts"].dt.strftime("%H:%M")
    sq_df["sub_queries"] = sq_df["sub_queries"].apply(
        lambda x: " → ".join(x) if isinstance(x, list) else ""
    )
    sq_df["vote"] = sq_df["vote"].map({"up": "👍", "down": "👎"}).fillna("—")
    sq_df.columns = ["Время", "Оригинальный вопрос", "Суб-запросы", "Оценка"]
    st.dataframe(sq_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# ── Footer / auto-refresh ─────────────────────────────────────────────────────
st.divider()
ts_now = datetime.now().strftime("%H:%M:%S")
col_info, col_btn = st.columns([8, 1])
col_info.caption(f"Обновлено: {ts_now} · событий всего: {total}")
if col_btn.button("🔄"):
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()
