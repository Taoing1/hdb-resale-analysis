"""Page 1: Data Overview — interactive filters, data table, statistics dashboard."""

import pandas as pd
import streamlit as st
import plotly.express as px

from utils.helpers import fmt_price, TOWN_COLORS


def run(df: pd.DataFrame):
    st.title("🏠 HDB 转售数据概览")
    st.markdown("交互式筛选与统计看板 — 探索新加坡 HDB 组屋转售数据全貌。")

    # ---- State Init ----
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False

    # ---- Sidebar: Multi-Condition Filters ----
    filters = _build_sidebar_filters(df)

    # ---- Apply Filters ----
    filtered = _apply_filters(df, filters)

    # ---- Filter Summary Chips ----
    _render_filter_chips(df, filtered, filters)

    # ====================== MAIN CONTENT ======================

    # ---- Row 1: Statistics Dashboard ----
    _render_stats_dashboard(df, filtered)

    st.divider()

    # ---- Row 2: Trend + Distribution Charts ----
    c1, c2 = st.columns([3, 2])
    with c1:
        _render_trend_chart(filtered)
    with c2:
        _render_volume_chart(filtered)

    # ---- Row 3: Price Distribution + Town/Type Matrix ----
    c1, c2 = st.columns(2)
    with c1:
        _render_price_distribution(filtered)
    with c2:
        _render_town_type_matrix(filtered)

    st.divider()

    # ---- Row 4: Data Table with Column Config ----
    _render_data_table(filtered)


# ======================== FILTERS ========================

def _build_sidebar_filters(df: pd.DataFrame) -> dict:
    """Build multi-condition sidebar filter controls."""

    with st.sidebar:
        st.subheader("🔍 数据筛选")

        # ---- Town ----
        st.caption("镇区")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ 全选", key="town_all", width='stretch'):
                st.session_state.town_sel = sorted(df["town"].unique())
        with c2:
            if st.button("🔄 清除", key="town_clr", width='stretch'):
                st.session_state.town_sel = []

        default_towns = st.session_state.get("town_sel", sorted(df["town"].unique()))
        sel_towns = st.multiselect(
            "镇区", sorted(df["town"].unique()),
            default=default_towns, key="town_multi",
            label_visibility="collapsed",
        )
        st.session_state.town_sel = sel_towns

        # ---- Year ----
        st.caption("交易年份")
        sel_years = st.slider(
            "年份", int(df["year"].min()), int(df["year"].max()),
            (int(df["year"].min()), int(df["year"].max())),
            key="year_slider", label_visibility="collapsed",
        )

        # ---- Flat Type ----
        st.caption("房型")
        flat_types = ["全部"] + sorted(df["flat_type"].unique())
        sel_type = st.selectbox(
            "房型", flat_types,
            key="type_select",
            label_visibility="collapsed",
        )
        sel_types = sorted(df["flat_type"].unique()) if sel_type == "全部" else [sel_type]

        # ---- Floor Area ----
        st.caption("面积范围 (sqm)")
        sel_area = st.slider(
            "面积", int(df["floor_area_sqm"].min()), int(df["floor_area_sqm"].max()),
            (int(df["floor_area_sqm"].min()), int(df["floor_area_sqm"].max())),
            key="area_slider", label_visibility="collapsed",
        )

        # ---- Price Range ----
        st.caption("价格范围 (新元)")
        sel_price = st.slider(
            "价格",
            int(df["resale_price"].min() // 10_000 * 10_000),
            int(df["resale_price"].max() // 10_000 * 10_000 + 10_000),
            (int(df["resale_price"].min() // 10_000 * 10_000),
             int(df["resale_price"].max() // 10_000 * 10_000 + 10_000)),
            10_000, key="price_slider", label_visibility="collapsed",
        )

        # ---- Remaining Lease ----
        lease_df = df.dropna(subset=["remaining_lease"])
        st.caption("剩余年限")
        sel_lease = st.slider(
            "年限", int(lease_df["remaining_lease"].min()), int(lease_df["remaining_lease"].max()),
            (int(lease_df["remaining_lease"].min()), int(lease_df["remaining_lease"].max())),
            key="lease_slider", label_visibility="collapsed",
        )

        # ---- Storey Range ----
        st.caption("楼层范围")
        storey_df = df.dropna(subset=["storey_mid"])
        sel_storey = st.slider(
            "楼层", int(storey_df["storey_low"].min()), int(storey_df["storey_high"].max()),
            (int(storey_df["storey_low"].min()), int(storey_df["storey_high"].max())),
            key="storey_slider", label_visibility="collapsed",
        )

        # ---- Reset All ----
        st.markdown("---")
        if st.button("🔃 重置全部筛选", width='stretch'):
            for k in ["town_sel"]:
                st.session_state.pop(k, None)
            st.rerun()

    return {
        "towns": sel_towns,
        "year_min": sel_years[0],
        "year_max": sel_years[1],
        "types": sel_types,
        "area_min": sel_area[0],
        "area_max": sel_area[1],
        "price_min": sel_price[0],
        "price_max": sel_price[1],
        "lease_min": sel_lease[0],
        "lease_max": sel_lease[1],
        "storey_min": sel_storey[0],
        "storey_max": sel_storey[1],
    }


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    """Apply multi-condition filters to dataframe."""
    mask = (
        df["town"].isin(f["towns"])
        & (df["year"].between(f["year_min"], f["year_max"]))
        & df["flat_type"].isin(f["types"])
        & (df["floor_area_sqm"].between(f["area_min"], f["area_max"]))
        & (df["resale_price"].between(f["price_min"], f["price_max"]))
    )
    if "remaining_lease" in df.columns:
        lease_ok = (
            df["remaining_lease"].isna()
            | df["remaining_lease"].between(f["lease_min"], f["lease_max"])
        )
        mask = mask & lease_ok
    if "storey_low" in df.columns and "storey_high" in df.columns:
        storey_ok = (
            df["storey_low"].isna()
            | ((df["storey_low"] >= f["storey_min"]) & (df["storey_high"] <= f["storey_max"]))
        )
        mask = mask & storey_ok

    return df[mask].copy()


# ==================== FILTER CHIPS ====================

def _render_filter_chips(df: pd.DataFrame, filtered: pd.DataFrame, f: dict):
    """Render active filter indicators and record count."""
    chips = []

    all_towns = sorted(df["town"].unique())
    if set(f["towns"]) != set(all_towns):
        chips.append(f"📍 {', '.join(f['towns'])}")
    if f["year_min"] != int(df["year"].min()) or f["year_max"] != int(df["year"].max()):
        chips.append(f"📅 {f['year_min']}–{f['year_max']}")
    all_types = sorted(df["flat_type"].unique())
    if f["types"] != all_types:
        chips.append(f"🏢 {', '.join(f['types'])}")
    if f["area_min"] != int(df["floor_area_sqm"].min()) or f["area_max"] != int(df["floor_area_sqm"].max()):
        chips.append(f"📐 {f['area_min']}–{f['area_max']} sqm")
    if f["price_min"] != int(df["resale_price"].min() // 10_000 * 10_000):
        chips.append(f"💰 {fmt_price(f['price_min'])}–{fmt_price(f['price_max'])}")

    pct = len(filtered) / len(df) * 100 if len(df) > 0 else 0

    c1, c2 = st.columns([3, 1])
    with c1:
        if chips:
            st.markdown(" **筛选条件:** " + " · ".join(chips))
        else:
            st.caption("当前显示全部数据")
    with c2:
        st.metric("筛选结果", f"{len(filtered):,} 条", f"{pct:.1f}% of total")


# ==================== STATS DASHBOARD ====================

def _render_stats_dashboard(df: pd.DataFrame, filtered: pd.DataFrame):
    """Dynamic statistics dashboard with metrics, deltas, and town breakdown."""
    st.subheader("📊 统计看板")

    # ---- Row 1: Required KPIs (3 cols, with deltas) ----
    avg_psm = filtered["price_per_sqm"].mean()
    all_avg_psm = df["price_per_sqm"].mean()
    cols = st.columns(3)
    metrics_row1 = [
        ("成交套数", f"{len(filtered):,}", None),
        ("平均单价", f"S${avg_psm:,.0f}", _delta_str(all_avg_psm, avg_psm)),
        ("平均总价", fmt_price(filtered["resale_price"].mean()), _delta_str(df["resale_price"].mean(), filtered["resale_price"].mean())),
    ]
    for i, (label, val, delta) in enumerate(metrics_row1):
        with cols[i]:
            st.metric(label, val, delta=delta)

    # ---- Row 2: Required KPIs continued (4 cols, no deltas) ----
    cols2 = st.columns(4)
    metrics_row2 = [
        ("最高单价", f"S${filtered['price_per_sqm'].max():,.0f}", None),
        ("最低单价", f"S${filtered['price_per_sqm'].min():,.0f}", None),
        ("中位总价", fmt_price(filtered["resale_price"].median()), None),
        ("总成交额", fmt_price(filtered["resale_price"].sum()), None),
    ]
    for i, (label, val, delta) in enumerate(metrics_row2):
        with cols2[i]:
            st.metric(label, val, delta=delta)

    # ---- Row 2: Per-Town Breakdown ----
    st.caption("镇区分组统计")
    town_stats = _town_breakdown(filtered)
    st.dataframe(
        town_stats, width='stretch', hide_index=True,
        column_config={
            "town": "镇区", "count": "套数", "avg_price": "均价",
            "median_price": "中位数", "min_price": "最低", "max_price": "最高",
            "avg_psm": "均价/sqm", "total": "成交总额",
        }
    )


def _delta_str(base: float, current: float) -> str:
    """Return delta percentage string comparing current to base."""
    if pd.isna(base) or base == 0:
        return None
    d = (current - base) / base * 100
    return f"{d:+.1f}%"


def _town_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-town statistics table."""
    if df.empty:
        return pd.DataFrame()
    stats = df.groupby("town").agg(
        count=("resale_price", "count"),
        avg_price=("resale_price", "mean"),
        median_price=("resale_price", "median"),
        min_price=("resale_price", "min"),
        max_price=("resale_price", "max"),
        avg_psm=("price_per_sqm", "mean"),
        total=("resale_price", "sum"),
    ).reset_index()
    for col in ["avg_price", "median_price", "min_price", "max_price", "avg_psm", "total"]:
        stats[col] = stats[col].apply(fmt_price)
    stats["count"] = stats["count"].apply(lambda x: f"{x:,}")
    return stats


# ==================== CHARTS ====================

def _render_trend_chart(filtered: pd.DataFrame):
    """Monthly average price trend line chart."""
    if filtered.empty:
        st.info("无数据")
        return
    trend = (filtered.groupby(["year", "month", "town"])["resale_price"]
             .mean().reset_index())
    fig = px.line(
        trend, x="month", y="resale_price", color="town",
        color_discrete_map=TOWN_COLORS,
        title="月度均价走势",
        labels={"month": "", "resale_price": "均价 (新元)", "town": "镇区"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')


def _render_volume_chart(filtered: pd.DataFrame):
    """Quarterly transaction volume bar chart."""
    if filtered.empty:
        st.info("无数据")
        return
    filtered_c = filtered.copy()
    filtered_c["quarter"] = filtered_c["month"].dt.to_period("Q").astype(str)
    vol = filtered_c.groupby(["quarter", "town"]).size().reset_index(name="count")
    fig = px.bar(
        vol, x="quarter", y="count", color="town",
        color_discrete_map=TOWN_COLORS,
        title="季度交易量",
        labels={"quarter": "", "count": "交易量 (套)", "town": "镇区"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')


def _render_price_distribution(filtered: pd.DataFrame):
    """Price distribution histogram by town."""
    if filtered.empty:
        st.info("无数据")
        return
    fig = px.histogram(
        filtered, x="resale_price", color="town", nbins=50,
        color_discrete_map=TOWN_COLORS,
        title="转售价格分布",
        labels={"resale_price": "价格 (新元)", "count": "数量"},
        opacity=0.75,
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')


def _render_town_type_matrix(filtered: pd.DataFrame):
    """Town × Flat Type average price heatmap."""
    if filtered.empty:
        st.info("无数据")
        return
    matrix = (filtered.groupby(["town", "flat_type"])["resale_price"]
              .mean().reset_index().pivot(index="town", columns="flat_type", values="resale_price"))
    fig = px.imshow(
        matrix, text_auto=",.0f", color_continuous_scale="Blues",
        title="镇区 × 房型 均价矩阵",
        labels=dict(x="房型", y="镇区", color="均价 (新元)"),
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')


# ==================== DATA TABLE ====================

def _render_data_table(filtered: pd.DataFrame):
    """Interactive data table with column visibility config and search."""
    st.subheader("📋 数据明细")

    # Column visibility selector
    all_cols = {
        "month": "月份", "town": "镇区", "flat_type": "房型",
        "floor_area_sqm": "面积(sqm)", "storey_range": "楼层范围",
        "remaining_lease": "剩余年限", "lease_commence_date": "建成年份",
        "resale_price": "转售价格", "price_per_sqm": "单价(sqm)",
        "flat_model": "户型", "storey_low": "楼层(低)", "storey_high": "楼层(高)",
    }
    available = [c for c in all_cols if c in filtered.columns]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption(f"共 {len(filtered):,} 条记录")
    with c2:
        default_cols = ["month", "town", "flat_type", "floor_area_sqm", "storey_range",
                        "remaining_lease", "resale_price", "price_per_sqm"]
        sel_cols = st.multiselect(
            "显示列", available, default=[c for c in default_cols if c in available],
            label_visibility="collapsed",
        )
    with c3:
        sort_col = st.selectbox(
            "排序", [all_cols.get(c, c) for c in sel_cols],
            index=sel_cols.index("month") if "month" in sel_cols else 0,
            label_visibility="collapsed",
        )
        sort_map = {v: k for k, v in all_cols.items()}
        sort_key = sort_map.get(sort_col, "month")

    if not sel_cols:
        sel_cols = available[:8]

    display = filtered[sel_cols].sort_values(sort_key, ascending=False).copy()

    # Format columns
    col_config = {}
    for c in sel_cols:
        label = all_cols.get(c, c)
        if c == "month":
            display[c] = display[c].dt.strftime("%Y-%m")
            col_config[c] = st.column_config.TextColumn(label, width="small")
        elif c == "resale_price":
            col_config[c] = st.column_config.NumberColumn(label, format="S$%d")
        elif c == "price_per_sqm":
            col_config[c] = st.column_config.NumberColumn(label, format="S$%d")
        elif c == "floor_area_sqm":
            col_config[c] = st.column_config.NumberColumn(label, format="%.1f sqm")
        elif c == "remaining_lease":
            display[c] = display[c].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
            col_config[c] = st.column_config.TextColumn(label, width="small")
        elif c in ("lease_commence_date", "storey_low", "storey_high"):
            col_config[c] = st.column_config.NumberColumn(label, format="%d")
        else:
            col_config[c] = st.column_config.TextColumn(label)

    st.dataframe(
        display, width='stretch', height=480,
        column_config=col_config, hide_index=True,
    )

    # Export
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ 导出当前筛选结果为 CSV", csv, "hdb_filtered.csv", "text/csv")
