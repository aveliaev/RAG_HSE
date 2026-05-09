import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path(__file__).parent / "eval_results.json"

st.set_page_config(
    page_title="RAG Dashboard — ФКН НИУ ВШЭ",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    for col in ["ctx_hit", "hit@1", "hit@3", "hit@5", "ans_hit"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df

df_all = load_results()

st.title("📊 RAG Evaluation Dashboard — ФКН НИУ ВШЭ")

if df_all.empty:
    st.error(
        "Файл `eval_results.json` не найден или пуст.\n\n"
        "Запустите сначала: `python bot/eval_rag.py`"
    )
    st.stop()

df_rag = df_all[df_all["route"] == "rag"].copy()
n_all = len(df_all)
n_rag = len(df_rag)

has_hitk = "hit@1" in df_all.columns

col_refresh = st.columns([8, 1])[1]
if col_refresh.button("🔄 Обновить"):
    st.cache_data.clear()
    st.rerun()

if not has_hitk:
    st.warning(
        "Колонки `hit@1 / hit@3 / hit@5` отсутствуют в `eval_results.json` — "
        "файл создан старой версией скрипта. "
        "Перезапустите eval и нажмите **Обновить**:\n\n"
        "```bash\ncd bot && python eval_rag.py\n```"
    )

st.markdown("## Ключевые метрики")

def _pct(val: float) -> str:
    return f"{val*100:.1f}%"

h1 = df_rag["hit@1"].mean() if (n_rag and has_hitk) else 0
h3 = df_rag["hit@3"].mean() if (n_rag and has_hitk) else 0
h5 = df_rag["hit@5"].mean() if (n_rag and has_hitk) else 0
ctx_recall_avg = df_rag["context_recall"].mean() if n_rag else 0
ctx_hit_rate = df_rag["ctx_hit"].mean() if n_rag else 0

has_rouge = df_all["rouge1_f"].notna().any()
avg_r1 = df_all["rouge1_f"].dropna().mean() if has_rouge else None
avg_rL = df_all["rougeL_f"].dropna().mean() if has_rouge else None

has_latency = "latency_ms" in df_all.columns and df_all["latency_ms"].notna().any()
avg_latency = df_all["latency_ms"].dropna().mean() if has_latency else None
p95_latency = df_all["latency_ms"].dropna().quantile(0.95) if has_latency else None

has_mrr = "rr" in df_rag.columns and df_rag["rr"].notna().any()
mrr_val = df_rag["rr"].mean() if has_mrr else None
has_prec = "precision_at_3" in df_rag.columns
avg_p3 = df_rag["precision_at_3"].mean() if has_prec else None
avg_p5 = df_rag["precision_at_5"].mean() if has_prec else None

st.markdown("#### Retrieval")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Hit@1", _pct(h1), help="Нужный чанк в top-1")
m2.metric("Hit@3", _pct(h3), help="Нужный чанк в top-3")
m3.metric("Hit@5", _pct(h5), help="Нужный чанк в top-5")
m4.metric("MRR@5", f"{mrr_val:.3f}" if mrr_val else "—",
          help="Mean Reciprocal Rank: среднее 1/rank первого релевантного чанка. 1.0 = всегда на первом месте")
m5.metric("Context Recall", _pct(ctx_recall_avg), help="Средний recall по top-5 чанкам в сумме")

st.markdown("#### Precision / Generation / Latency")
m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("P@3", _pct(avg_p3) if avg_p3 else "—",
          help="Precision@3: сколько из top-3 чанков релевантны")
m7.metric("P@5", _pct(avg_p5) if avg_p5 else "—",
          help="Precision@5: сколько из top-5 чанков релевантны")
m8.metric("ROUGE-1", _pct(avg_r1) if avg_r1 else "—", help="Средний ROUGE-1 (--with-llm)")
m9.metric("ROUGE-L", _pct(avg_rL) if avg_rL else "—")
m10.metric("Avg Latency", f"{avg_latency:.0f} мс" if avg_latency else "—",
           help="Среднее время retrieval без LLM")

st.markdown(f"*Всего вопросов: **{n_all}** / через RAG: **{n_rag}***")

st.divider()

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### Hit@K (RAG)")
    if not has_hitk:
        st.info("Нет данных — запустите `python eval_rag.py`")
    else:
        fig_hit = go.Figure(go.Bar(
            x=["Hit@1", "Hit@3", "Hit@5"],
            y=[h1 * 100, h3 * 100, h5 * 100],
            text=[f"{v*100:.1f}%" for v in [h1, h3, h5]],
            textposition="outside",
            marker_color=["#ef553b", "#ffa15a", "#00cc96"],
        ))
        fig_hit.update_layout(
            yaxis=dict(range=[0, 110], title="% вопросов"),
            margin=dict(t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig_hit, use_container_width=True)

with col_b:
    st.markdown("### Распределение маршрутов")
    route_counts = df_all["route"].value_counts().reset_index()
    route_counts.columns = ["route", "count"]
    fig_pie = px.pie(
        route_counts, values="count", names="route",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_pie.update_layout(margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_c:
    st.markdown("### Распределение Context Recall")
    fig_hist = px.histogram(
        df_rag, x="context_recall", nbins=20,
        color_discrete_sequence=["#636efa"],
        labels={"context_recall": "Context Recall"},
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="порог 0.5")
    fig_hist.update_layout(margin=dict(t=20, b=20), height=300, yaxis_title="Кол-во вопросов")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

st.markdown("## Hit@K по категориям")

if "category" in df_rag.columns and has_hitk:
    cat_stats = (
        df_rag.groupby("category")[["hit@1", "hit@3", "hit@5"]]
        .mean()
        .reset_index()
    )
    cat_stats["cat_short"] = cat_stats["category"].apply(
        lambda s: re.sub(r"Категория \d+ — ", "", s)[:40]
    )
    cat_melted = cat_stats.melt(
        id_vars=["cat_short"], value_vars=["hit@1", "hit@3", "hit@5"],
        var_name="Метрика", value_name="Значение",
    )
    cat_melted["Значение"] = cat_melted["Значение"] * 100

    fig_cat = px.bar(
        cat_melted, x="cat_short", y="Значение", color="Метрика",
        barmode="group",
        color_discrete_map={"hit@1": "#ef553b", "hit@3": "#ffa15a", "hit@5": "#00cc96"},
        labels={"cat_short": "Категория", "Значение": "%"},
    )
    fig_cat.update_layout(
        xaxis_tickangle=-30,
        margin=dict(t=20, b=100),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

has_frr = "first_relevant_rank" in df_rag.columns and df_rag["first_relevant_rank"].notna().any()
has_p1  = "precision_at_1" in df_rag.columns

if has_frr or has_prec:
    st.markdown("## Ранжирование и точность")

    rank_col, prec_col, mrr_cat_col = st.columns(3)

    with rank_col:
        st.markdown("### Ранг первого релевантного чанка")
        if has_frr:
            rank_counts = df_rag["first_relevant_rank"].value_counts().sort_index().reset_index()
            rank_counts.columns = ["rank", "count"]
            rank_counts["rank_label"] = rank_counts["rank"].apply(
                lambda x: "Не найдено" if x == 0 else f"Ранг {int(x)}"
            )
            order_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 0: 5}
            rank_counts["_order"] = rank_counts["rank"].map(lambda x: order_map.get(x, 6))
            rank_counts = rank_counts.sort_values("_order")

            bar_colors = [
                "#ef553b" if r == 0 else "#00cc96" if r == 1 else "#ffa15a"
                for r in rank_counts["rank"]
            ]
            fig_rank = go.Figure(go.Bar(
                x=rank_counts["rank_label"],
                y=rank_counts["count"],
                text=rank_counts["count"],
                textposition="outside",
                marker_color=bar_colors,
            ))
            fig_rank.update_layout(
                height=300, margin=dict(t=20, b=20),
                yaxis_title="Кол-во вопросов",
                xaxis_title="",
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        else:
            st.info("Нет данных `first_relevant_rank` — перезапустите eval_rag.py")

    with prec_col:
        st.markdown("### Precision@K (средняя)")
        if has_prec:
            p1_val = df_rag["precision_at_1"].mean() if has_p1 else h1
            prec_vals = [p1_val * 100, avg_p3 * 100, avg_p5 * 100]
            fig_prec = go.Figure(go.Bar(
                x=["P@1", "P@3", "P@5"],
                y=prec_vals,
                text=[f"{v:.1f}%" for v in prec_vals],
                textposition="outside",
                marker_color=["#636efa", "#ab63fa", "#19d3f3"],
            ))
            fig_prec.update_layout(
                height=300, margin=dict(t=20, b=20),
                yaxis=dict(range=[0, 110], title="%"),
            )
            st.plotly_chart(fig_prec, use_container_width=True)
        else:
            st.info("Нет данных Precision@K — перезапустите eval_rag.py")

    with mrr_cat_col:
        st.markdown("### MRR@5 по категориям")
        if has_mrr and "category" in df_rag.columns:
            mrr_cat = (
                df_rag.groupby("category")["rr"]
                .mean()
                .reset_index()
                .rename(columns={"rr": "MRR"})
            )
            mrr_cat["cat_short"] = mrr_cat["category"].apply(
                lambda s: re.sub(r"Категория \d+ — ", "", s)[:30]
            )
            mrr_cat = mrr_cat.sort_values("MRR", ascending=True)
            fig_mrr_cat = go.Figure(go.Bar(
                x=mrr_cat["MRR"],
                y=mrr_cat["cat_short"],
                orientation="h",
                text=[f"{v:.3f}" for v in mrr_cat["MRR"]],
                textposition="outside",
                marker_color="#ffa15a",
            ))
            fig_mrr_cat.update_layout(
                height=300, margin=dict(t=20, b=20, l=10, r=60),
                xaxis=dict(range=[0, 1.1], title="MRR@5"),
            )
            st.plotly_chart(fig_mrr_cat, use_container_width=True)
        else:
            st.info("Нет данных MRR или категорий")

    st.divider()

st.markdown("## Топ источников (top-1 chunk)")

col_src1, col_src2 = st.columns([1, 2])

with col_src1:
    src_counts = df_rag["top1_source"].value_counts().reset_index()
    src_counts.columns = ["source", "count"]
    fig_src = px.bar(
        src_counts, x="count", y="source", orientation="h",
        color_discrete_sequence=["#ab63fa"],
        labels={"source": "Файл", "count": "Кол-во раз в top-1"},
    )
    fig_src.update_layout(margin=dict(t=10, b=10), height=300, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_src, use_container_width=True)

with col_src2:
    if "category" in df_rag.columns:
        heat_df = (
            df_rag.groupby(["top1_source", "category"])
            .size()
            .reset_index(name="count")
        )
        heat_df["cat_short"] = heat_df["category"].apply(
            lambda s: re.sub(r"Категория \d+ — ", "", s)[:30]
        )
        heat_pivot = heat_df.pivot_table(
            index="top1_source", columns="cat_short", values="count", fill_value=0
        )
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="Blues",
            labels=dict(color="Кол-во"),
            aspect="auto",
        )
        fig_heat.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

st.markdown("## 🗺 Покрытие источников (Coverage)")

col_cov1, col_cov2 = st.columns([1, 1])

with col_cov1:
    st.markdown("### Источники в top-1 (Hit@1 ✅ vs Miss ❌)")
    if has_hitk and "top1_source" in df_rag.columns:
        cov_df = (
            df_rag.groupby("top1_source")["hit@1"]
            .agg(total="count", hits="sum")
            .reset_index()
        )
        cov_df["misses"] = cov_df["total"] - cov_df["hits"]
        cov_df["hit_rate"] = cov_df["hits"] / cov_df["total"]
        cov_df = cov_df.sort_values("hit_rate", ascending=True)

        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(
            name="Hit@1", x=cov_df["hits"], y=cov_df["top1_source"],
            orientation="h", marker_color="#00cc96",
            text=cov_df["hits"], textposition="inside",
        ))
        fig_cov.add_trace(go.Bar(
            name="Miss", x=cov_df["misses"], y=cov_df["top1_source"],
            orientation="h", marker_color="#ef553b",
            text=cov_df["misses"], textposition="inside",
        ))
        fig_cov.update_layout(
            barmode="stack",
            height=max(250, len(cov_df) * 28),
            margin=dict(t=20, b=20, l=10),
            xaxis_title="Кол-во вопросов",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cov, use_container_width=True)
    else:
        st.info("Нет данных hit@1 или top1_source")

with col_cov2:
    st.markdown("### Источники никогда НЕ в top-1 (потенциальные пробелы)")
    if "top1_source" in df_rag.columns:
        all_sources_in_top1 = set(df_rag["top1_source"].dropna().unique())
        low_coverage = (
            df_rag.groupby("top1_source")["hit@1"]
            .agg(total="count", hits="sum")
            .reset_index()
        )
        low_coverage["hit_rate"] = low_coverage["hits"] / low_coverage["total"]
        low_coverage = low_coverage[low_coverage["hit_rate"] < 0.5].sort_values("hit_rate")
        low_coverage.columns = ["Источник", "Всего вопросов", "Hit@1", "Hit Rate"]
        low_coverage["Hit Rate"] = low_coverage["Hit Rate"].map(lambda x: f"{x:.0%}")

        if low_coverage.empty:
            st.success("Все источники имеют Hit Rate ≥ 50% — отличное покрытие!")
        else:
            st.warning(f"{len(low_coverage)} источников с Hit Rate < 50%")
            st.dataframe(
                low_coverage.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

st.divider()

st.markdown("## Провалы: не найдено даже в top-5")

if not has_hitk:
    st.info("Нет данных hit@k — запустите `python eval_rag.py`")
else:
    failures = df_rag[~df_rag["hit@5"]].copy()
    if failures.empty:
        st.success("Нет полных провалов! Все вопросы нашлись хотя бы в top-5.")
    else:
        st.warning(f"{len(failures)} вопросов с hit@5 = 0")
        cols_show = ["id", "question", "context_recall", "top1_source", "rewritten"]
        cols_show = [c for c in cols_show if c in failures.columns]
        st.dataframe(failures[cols_show].reset_index(drop=True), use_container_width=True)

st.divider()

if has_rouge:
    st.markdown("## Generation — ROUGE метрики")
    df_gen = df_all[df_all["rouge1_f"].notna()].copy()

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_r1 = px.histogram(
            df_gen, x="rouge1_f", nbins=15, title="ROUGE-1 F",
            color_discrete_sequence=["#19d3f3"],
        )
        fig_r1.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="порог")
        fig_r1.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig_r1, use_container_width=True)
    with col_r2:
        fig_rL = px.histogram(
            df_gen, x="rougeL_f", nbins=15, title="ROUGE-L F",
            color_discrete_sequence=["#ff6692"],
        )
        fig_rL.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig_rL, use_container_width=True)

    st.divider()

if has_latency:
    st.markdown("## ⏱ Время ответа (latency)")

    lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(4)
    lat_col1.metric("Среднее", f"{avg_latency:.0f} мс", help="Среднее время на один вопрос")
    lat_col2.metric("Медиана", f"{df_all['latency_ms'].dropna().median():.0f} мс")
    lat_col3.metric("P95", f"{p95_latency:.0f} мс", help="95-й перцентиль — «почти худший случай»")
    lat_col4.metric("Макс.", f"{df_all['latency_ms'].dropna().max():.0f} мс")

    lat_a, lat_b, lat_c = st.columns(3)

    with lat_a:
        st.markdown("### Распределение latency")
        fig_lat_hist = px.histogram(
            df_all.dropna(subset=["latency_ms"]),
            x="latency_ms", nbins=30,
            color_discrete_sequence=["#ab63fa"],
            labels={"latency_ms": "Время (мс)"},
        )
        fig_lat_hist.add_vline(
            x=avg_latency, line_dash="dash", line_color="orange",
            annotation_text=f"avg {avg_latency:.0f}мс",
        )
        fig_lat_hist.add_vline(
            x=p95_latency, line_dash="dot", line_color="red",
            annotation_text=f"p95 {p95_latency:.0f}мс",
        )
        fig_lat_hist.update_layout(height=300, margin=dict(t=20, b=20), yaxis_title="Кол-во вопросов")
        st.plotly_chart(fig_lat_hist, use_container_width=True)

    with lat_b:
        st.markdown("### Latency по маршруту")
        lat_by_route = (
            df_all.dropna(subset=["latency_ms"])
            .groupby("route")["latency_ms"]
            .agg(["mean", "median", "max"])
            .reset_index()
            .rename(columns={"mean": "Среднее", "median": "Медиана", "max": "Макс."})
        )
        fig_route_lat = px.bar(
            lat_by_route, x="route", y="Среднее",
            error_y=lat_by_route["Макс."] - lat_by_route["Среднее"],
            color="route",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"route": "Маршрут", "Среднее": "Среднее время (мс)"},
        )
        fig_route_lat.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig_route_lat, use_container_width=True)

    with lat_c:
        st.markdown("### Latency vs Context Recall")
        scatter_df = df_rag.dropna(subset=["latency_ms", "context_recall"]).copy()
        scatter_df["hit_str"] = scatter_df["hit@1"].map({True: "Hit@1 ✅", False: "Miss ❌"}) if has_hitk else "N/A"
        fig_scatter = px.scatter(
            scatter_df, x="latency_ms", y="context_recall",
            color="hit_str",
            color_discrete_map={"Hit@1 ✅": "#00cc96", "Miss ❌": "#ef553b"},
            hover_data=["id", "question"],
            labels={"latency_ms": "Время (мс)", "context_recall": "Context Recall"},
            opacity=0.7,
        )
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="recall≥0.5")
        fig_scatter.update_layout(height=300, margin=dict(t=20, b=20), legend_title="")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Самые медленные запросы (топ-10)")
    slowest = (
        df_all.dropna(subset=["latency_ms"])
        .nlargest(10, "latency_ms")[["id", "question", "route", "latency_ms", "context_recall"]]
        .reset_index(drop=True)
    )
    st.dataframe(
        slowest,
        use_container_width=True,
        column_config={
            "latency_ms": st.column_config.NumberColumn("Время (мс)", format="%.0f мс"),
            "context_recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
        },
    )

    st.divider()

st.markdown("## Все вопросы — детальный просмотр")

filter_route = st.multiselect(
    "Маршрут",
    options=sorted(df_all["route"].unique()),
    default=sorted(df_all["route"].unique()),
)
filter_hit = st.radio(
    "Фильтр по Hit@1",
    options=["Все", "Hit@1 = True", "Hit@1 = False"],
    horizontal=True,
)

df_view = df_all[df_all["route"].isin(filter_route)].copy()
if "hit@1" in df_view.columns:
    if filter_hit == "Hit@1 = True":
        df_view = df_view[df_view["hit@1"] == True]
    elif filter_hit == "Hit@1 = False":
        df_view = df_view[df_view["hit@1"] == False]

display_cols = ["id", "category", "question", "route", "rewritten",
                "hit@1", "hit@3", "hit@5", "context_recall", "top1_source",
                "latency_ms", "rouge1_f", "rougeL_f"]
display_cols = [c for c in display_cols if c in df_view.columns]

st.dataframe(
    df_view[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=420,
    column_config={
        "hit@1": st.column_config.CheckboxColumn("H@1"),
        "hit@3": st.column_config.CheckboxColumn("H@3"),
        "hit@5": st.column_config.CheckboxColumn("H@5"),
        "ctx_hit": st.column_config.CheckboxColumn("Ctx Hit"),
        "context_recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
        "latency_ms": st.column_config.NumberColumn("Время (мс)", format="%.0f мс"),
        "rouge1_f": st.column_config.NumberColumn("ROUGE-1", format="%.3f"),
        "rougeL_f": st.column_config.NumberColumn("ROUGE-L", format="%.3f"),
    },
)

has_answers = "predicted" in df_view.columns and df_view["predicted"].fillna("").str.strip().any()

st.markdown("### Ответы модели")

if not has_answers:
    st.info("Ответы модели отсутствуют — запустите `python eval_rag.py --with-llm`")
else:
    for _, r in df_view.iterrows():
        predicted = str(r.get("predicted", "") or "").strip()
        h1_val = r.get("hit@1", None)
        recall_val = r.get("context_recall", None)

        if r["route"] == "blocked":
            icon = "🚫"
        elif h1_val is True:
            icon = "✅"
        elif r.get("hit@3") is True:
            icon = "⚡"
        elif r.get("hit@5") is True:
            icon = "🔶"
        else:
            icon = "❌"

        latency_val = r.get("latency_ms")
        latency_str = f"⏱ {latency_val:.0f}мс" if latency_val is not None else ""
        recall_str = f"recall={recall_val:.2f}" if recall_val is not None else ""
        label = f"{icon} **{r['id']}** — {r['question'][:80]}{'…' if len(r['question']) > 80 else ''}  `{r['route']}` {recall_str}  {latency_str}"

        with st.expander(label, expanded=False):
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Вопрос:**")
                st.write(r["question"])
                rewritten = str(r.get("rewritten", "") or "").strip()
                if rewritten:
                    st.markdown("**После перефразирования:**")
                    st.caption(rewritten)
                st.markdown(f"**Маршрут:** `{r['route']}` | **Источник:** `{r.get('top1_source', '?')}`")
                if recall_val is not None:
                    r1 = r.get("rouge1_f")
                    lat_part = f" | **Время:** {latency_val:.0f} мс" if latency_val is not None else ""
                    st.markdown(
                        f"**Context recall:** {recall_val:.3f}"
                        + (f" | **ROUGE-1:** {r1:.3f}" if r1 else "")
                        + lat_part
                    )

            with col_right:
                st.markdown("**Эталонный ответ:**")
                st.success(r.get("ref_answer", "—"))
                if predicted and not predicted.startswith("[ERROR"):
                    st.markdown("**Ответ модели:**")
                    st.warning(predicted)
                elif predicted.startswith("[ERROR"):
                    st.markdown("**Ответ модели:**")
                    st.error(predicted)
                else:
                    st.caption("*Ответ не генерировался*")

            chunk_recalls = r.get("chunk_recalls")
            if isinstance(chunk_recalls, list) and chunk_recalls:
                fig_cr = go.Figure(go.Bar(
                    x=[f"#{i+1}" for i in range(len(chunk_recalls))],
                    y=chunk_recalls,
                    marker_color=["#00cc96" if v >= 0.3 else "#ef553b" for v in chunk_recalls],
                    text=[f"{v:.2f}" for v in chunk_recalls],
                    textposition="outside",
                ))
                fig_cr.add_hline(y=0.3, line_dash="dash", line_color="orange")
                fig_cr.update_layout(height=200, margin=dict(t=10, b=10), yaxis=dict(range=[0, 1.1]))
                st.plotly_chart(fig_cr, use_container_width=True, key=f"cr_{r['id']}")

st.divider()
st.caption("RAG Dashboard · ФКН НИУ ВШЭ · данные из eval_results.json")
