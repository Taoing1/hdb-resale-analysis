"""Page 3: Factor Analysis — 6+ factors, mature vs non-mature, MRT/school proximity, purchase strategy."""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy import stats

from utils.helpers import TOWN_COLORS, fmt_price

# ---- MRT station coordinates (same as map page) ----
MRT_COORDS = {
    "PUNGGOL":    [(1.4052, 103.9022)],  # Punggol MRT
    "SENGKANG":   [(1.3924, 103.8951)],  # Sengkang MRT
    "HOUGANG":    [(1.3716, 103.8923)],  # Hougang MRT
}

# ---- Mature vs Non-Mature definitions ----
MATURE_TOWNS = ["ANG MO KIO", "TOA PAYOH", "QUEENSTOWN", "BEDOK", "BISHAN", "BUKIT MERAH",
                "CLEMENTI", "KALLANG/WHAMPOA", "GEYLANG", "MARINE PARADE", "BUKIT TIMAH"]
NON_MATURE_TOWNS = ["PUNGGOL", "SENGKANG", "HOUGANG", "WOODLANDS", "JURONG WEST",
                    "YISHUN", "BUKIT BATOK", "CHOA CHU KANG", "SEMBAWANG", "BUKIT PANJANG",
                    "PASIR RIS", "JURONG EAST", "SERANGOON"]

API_URL = "https://data.gov.sg/api/action/datastore_search"
RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"


# ====================== HELPER: distance ======================

def _haversine(lat1, lng1, lat2, lng2):
    """Return distance in km between two lat/lng pairs."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _nearest_mrt_dist(town: str) -> float:
    """Approximate distance from town center to nearest MRT station (km)."""
    from utils.helpers import TOWN_COORDS
    if town not in TOWN_COORDS:
        return 2.0
    t_lat, t_lng = TOWN_COORDS[town]
    mrt_list = MRT_COORDS.get(town, [(t_lat + 0.01, t_lng + 0.01)])
    return min(_haversine(t_lat, t_lng, mlat, mlng) for mlat, mlng in mrt_list)


# ====================== FETCH MATURE DATA ======================

@st.cache_data(ttl=86400)
def _fetch_mature_estate_data() -> pd.DataFrame:
    """Fetch mature estate HDB data for comparison analysis."""
    records = []
    import time
    for town in ["ANG MO KIO", "TOA PAYOH", "QUEENSTOWN", "BEDOK", "BISHAN"]:
        offset = 0
        while True:
            try:
                resp = requests.get(API_URL, params={
                    "resource_id": RESOURCE_ID, "limit": 500, "offset": offset,
                    "filters": json.dumps({"town": town}),
                }, timeout=30)
                resp.raise_for_status()
                batch = resp.json()["result"]["records"]
                if not batch:
                    break
                records.extend(batch)
                offset += len(batch)
                if len(batch) < 500:
                    break
                time.sleep(0.3)
            except Exception:
                break
        time.sleep(0.5)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["town"] = df["town"].str.upper().str.strip()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"] = df["month"].dt.year
    df = df[df["year"] >= 2020]
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["price_per_sqm"] = (df["resale_price"] / df["floor_area_sqm"]).round(2)
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    current_year = pd.Timestamp.now().year
    df["remaining_lease"] = df["lease_commence_date"].apply(
        lambda x: max(99 - (current_year - x), 0) if pd.notna(x) else None
    )
    df = df.dropna(subset=["resale_price", "floor_area_sqm", "month"])
    return df


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("📊 房价影响因素分析")
    st.markdown("面积 · 房型 · 楼层 · 剩余租约 · 房龄 · 镇区类型 — 含成熟区对比、配套设施影响与购房策略验证。")

    # Prepare derived columns
    df = df.copy()
    df["floor_age"] = df["lease_commence_date"].apply(
        lambda x: pd.Timestamp.now().year - x if pd.notna(x) else None
    )
    df["storey_bin"] = pd.cut(
        df["storey_mid"], bins=[0, 6, 15, 60],
        labels=["低层 (1-6)", "中层 (7-15)", "高层 (16+)"]
    )
    df["town_type"] = df["town"].apply(
        lambda t: "成熟区" if t in MATURE_TOWNS else "非成熟区"
    )
    # For our 3 target towns, all are non-mature
    df["mrt_dist_km"] = df["town"].apply(_nearest_mrt_dist)
    df["mrt_zone"] = pd.cut(
        df["mrt_dist_km"], bins=[0, 0.5, 1.0, 5.0],
        labels=["<500m", "500m–1km", ">1km"]
    )

    tabs = st.tabs([
        "① 面积", "② 房型", "③ 楼层", "④ 剩余租约",
        "⑤ 房龄", "⑥ 镇区类型", "🏘️ 成熟区对比", "🚇 配套设施", "💡 策略验证"
    ])

    with tabs[0]:
        _analyze_area(df)
    with tabs[1]:
        _analyze_flat_type(df)
    with tabs[2]:
        _analyze_storey(df)
    with tabs[3]:
        _analyze_lease(df)
    with tabs[4]:
        _analyze_age(df)
    with tabs[5]:
        _analyze_town_type(df)
    with tabs[6]:
        _analyze_mature_vs_non_mature(df)
    with tabs[7]:
        _analyze_amenities(df)
    with tabs[8]:
        _analyze_strategy(df)


# ======================== ① AREA vs UNIT PRICE ========================

def _analyze_area(df: pd.DataFrame):
    st.subheader("① 面积 vs 单价 — 散点图 + 回归线")
    df_clean = df.dropna(subset=["floor_area_sqm", "price_per_sqm"])
    sample = df_clean.sample(min(5000, len(df_clean)), random_state=42)

    fig = px.scatter(
        sample, x="floor_area_sqm", y="price_per_sqm", color="town",
        color_discrete_map=TOWN_COLORS, opacity=0.5, trendline="ols",
        title="面积 (sqm) vs 单价 (新元/sqm)",
        labels={"floor_area_sqm": "面积 (sqm)", "price_per_sqm": "单价 (新元/sqm)", "town": "镇区"},
    )
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    r, p = stats.pearsonr(df_clean["floor_area_sqm"], df_clean["price_per_sqm"])
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pearson r", f"{r:.3f}")
    with c2:
        st.caption(f"p = {p:.1e}  |  {'显著相关' if p < 0.05 else '不显著'}")

    st.caption(
        f"面积与单价呈 {'正' if r > 0 else '负'}相关 (r={r:.3f})。"
        f"{'大户型享有单价溢价，可能因大面积单位通常为较好的户型/楼层。' if r > 0 else '小户型单价更高（总价门槛效应），买家愿为低总价房产支付更高单价。'}"
    )


# ======================== ② FLAT TYPE vs AVG PRICE ========================

def _analyze_flat_type(df: pd.DataFrame):
    st.subheader("② 房型 vs 均价 — 柱状图")
    type_order = sorted(df["flat_type"].unique(),
                        key=lambda x: df[df["flat_type"] == x]["resale_price"].mean())
    type_stats = df.groupby("flat_type").agg(
        均价=("resale_price", "mean"), 中位数=("resale_price", "median"),
        套数=("resale_price", "count"),
    ).reindex(type_order).reset_index()

    fig = px.bar(
        type_stats, x="flat_type", y="均价", color="flat_type",
        text=type_stats["均价"].apply(lambda v: f"S${v:,.0f}"),
        title="各房型均价对比",
        labels={"flat_type": "房型", "均价": "均价 (新元)"},
    )
    fig.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')

    # Table
    display = type_stats.copy()
    display["均价"] = display["均价"].apply(fmt_price)
    display["中位数"] = display["中位数"].apply(fmt_price)
    st.dataframe(display, width='stretch', hide_index=True)

    # ANOVA
    groups = [df[df["flat_type"] == t]["resale_price"].values for t in df["flat_type"].unique()]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        st.caption(f"ANOVA: F={f_stat:.1f}, p={p_val:.1e}  |  房型间价格差异极显著 (p<0.001)")


# ======================== ③ STOREY vs PRICE ========================

def _analyze_storey(df: pd.DataFrame):
    st.subheader("③ 楼层 vs 均价 — 箱线图 (低/中/高层)")
    df_clean = df.dropna(subset=["storey_bin"])

    fig = px.box(
        df_clean, x="storey_bin", y="resale_price", color="town",
        color_discrete_map=TOWN_COLORS,
        title="低/中/高楼层价格分布",
        labels={"storey_bin": "楼层分组", "resale_price": "转售价格 (新元)", "town": "镇区"},
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    bin_stats = df_clean.groupby("storey_bin").agg(
        均价=("resale_price", "mean"), 套数=("resale_price", "count"),
    ).reset_index()
    for _, row in bin_stats.iterrows():
        st.caption(f"{row['storey_bin']}: 均价 {fmt_price(row['均价'])} ({row['套数']:,} 套)")

    # Kruskal-Wallis
    groups = [df_clean[df_clean["storey_bin"] == b]["resale_price"].values
              for b in df_clean["storey_bin"].unique()]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        st.caption(f"Kruskal-Wallis: H={h:.1f}, p={p:.1e}  |  {'显著' if p<0.05 else '不显著'}")


# ======================== ④ REMAINING LEASE vs UNIT PRICE ========================

def _analyze_lease(df: pd.DataFrame):
    st.subheader("④ 剩余租约 vs 单价 — 散点图")
    df_clean = df.dropna(subset=["remaining_lease", "price_per_sqm"])
    sample = df_clean.sample(min(5000, len(df_clean)), random_state=42)

    fig = px.scatter(
        sample, x="remaining_lease", y="price_per_sqm", color="town",
        color_discrete_map=TOWN_COLORS, opacity=0.5, trendline="ols",
        title="剩余年限 vs 单价",
        labels={"remaining_lease": "剩余年限 (年)", "price_per_sqm": "单价 (新元/sqm)", "town": "镇区"},
    )
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    r, p = stats.pearsonr(df_clean["remaining_lease"], df_clean["price_per_sqm"])
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pearson r", f"{r:.3f}")
    with c2:
        st.metric("平均单位租约折旧", f"S${abs(r) * df_clean['price_per_sqm'].std() / df_clean['remaining_lease'].std() * 10:,.0f}/10年")


# ======================== ⑤ FLOOR AGE vs UNIT PRICE ========================

def _analyze_age(df: pd.DataFrame):
    st.subheader("⑤ 房龄 vs 单价 — 散点图")
    df_clean = df.dropna(subset=["floor_age", "price_per_sqm"])
    sample = df_clean.sample(min(5000, len(df_clean)), random_state=42)

    fig = px.scatter(
        sample, x="floor_age", y="price_per_sqm", color="town",
        color_discrete_map=TOWN_COLORS, opacity=0.5, trendline="ols",
        title="房龄 vs 单价",
        labels={"floor_age": "房龄 (年)", "price_per_sqm": "单价 (新元/sqm)", "town": "镇区"},
    )
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    r, p = stats.pearsonr(df_clean["floor_age"], df_clean["price_per_sqm"])
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Pearson r", f"{r:.3f}")
    with c2:
        st.caption(f"p = {p:.1e}  |  房龄每增10年，单价约变动 S${abs(r) * df_clean['price_per_sqm'].std() / df_clean['floor_age'].std() * 10:,.0f}")

    st.caption(
        f"房龄与单价 {'正' if r > 0 else '负'}相关 (r={r:.3f})。"
        f"{'较新房产享有更高单价溢价。' if r < 0 else '部分老区因地段优越仍维持较高单价。'}"
    )


# ======================== ⑥ TOWN TYPE vs PRICE ========================

def _analyze_town_type(df: pd.DataFrame):
    st.subheader("⑥ 镇区类型 vs 均价 — 柱状图")

    # Fetch mature data for comparison
    with st.spinner("补充获取成熟区数据用于对比…"):
        mature_df = _fetch_mature_estate_data()
    if not mature_df.empty:
        mature_df["town_type"] = "成熟区"
        df["town_type"] = "非成熟区"
        combined = pd.concat([df, mature_df[df.columns.intersection(mature_df.columns)]], ignore_index=True)
    else:
        df["town_type"] = "非成熟区"
        combined = df
        st.warning("成熟区数据暂不可用，仅展示当前镇区。")

    type_stats = combined.groupby("town_type").agg(
        均价=("resale_price", "mean"), 中位数=("resale_price", "median"),
        套数=("resale_price", "count"), 平均单价=("price_per_sqm", "mean"),
    ).reset_index()

    fig = px.bar(
        type_stats, x="town_type", y="均价", color="town_type",
        text=type_stats["均价"].apply(lambda v: f"S${v:,.0f}"),
        title="成熟区 vs 非成熟区 均价对比",
        labels={"town_type": "镇区类型", "均价": "均价 (新元)"},
        color_discrete_map={"成熟区": "#FF6B6B", "非成熟区": "#4ECDC4"},
    )
    fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')

    # Stats table
    display = type_stats.copy()
    display["均价"] = display["均价"].apply(fmt_price)
    display["中位数"] = display["中位数"].apply(fmt_price)
    display["平均单价"] = display["平均单价"].apply(lambda x: f"S${x:,.0f}/sqm")
    st.dataframe(display, width='stretch', hide_index=True)

    # t-test
    if "成熟区" in combined["town_type"].values and "非成熟区" in combined["town_type"].values:
        mature_prices = combined[combined["town_type"] == "成熟区"]["resale_price"]
        non_mature_prices = combined[combined["town_type"] == "非成熟区"]["resale_price"]
        if len(mature_prices) > 10 and len(non_mature_prices) > 10:
            t_stat, p_val = stats.ttest_ind(mature_prices, non_mature_prices, equal_var=False)
            premium = (mature_prices.mean() - non_mature_prices.mean()) / non_mature_prices.mean() * 100
            st.caption(f"Welch t-test: t={t_stat:.1f}, p={p_val:.1e}  |  成熟区溢价约 {premium:+.1f}%")


# ======================== MATURE vs NON-MATURE ========================

def _analyze_mature_vs_non_mature(df: pd.DataFrame):
    st.subheader("🏘️ 成熟区 vs 非成熟区 — 对比分析")

    with st.expander("📖 成熟区/非成熟区定义", expanded=False):
        st.markdown("""
        **成熟组屋区 (Mature Estates)**：指开发历史久、基础设施完善、交通便利、社区成熟度高的镇区。
        典型代表：**Queenstown、Toa Payoh、Ang Mo Kio、Bedok、Clementi、Bishan** 等。
        特点：均价较高、新盘供应少、以转售市场为主、学校/医疗等配套齐全。

        **非成熟区 (Non-Mature Estates)**：指仍在发展中的新兴镇区，基础设施持续完善中。
        典型代表：**Punggol、Sengkang、Hougang、Woodlands、Jurong West** 等。
        特点：均价相对较低、新盘较多、规划和配套持续改善中、增值潜力受关注。

        *参考: HDB 官方镇区分类。*
        """)

    with st.spinner("补充获取成熟区对比数据…"):
        mature_df = _fetch_mature_estate_data()

    if mature_df.empty:
        st.info("成熟区数据暂不可用")
        return

    mature_df["town_type"] = "成熟区"
    df["town_type"] = "非成熟区"
    combined = pd.concat([df, mature_df[df.columns.intersection(mature_df.columns)]], ignore_index=True)
    combined = combined.dropna(subset=["price_per_sqm"])

    # 1. Price trend comparison
    st.subheader("📈 价格走势对比")
    trend = combined.groupby(["year", "town_type"])["resale_price"].mean().reset_index()
    fig = px.line(
        trend, x="year", y="resale_price", color="town_type", markers=True,
        color_discrete_map={"成熟区": "#FF6B6B", "非成熟区": "#4ECDC4"},
        title="年均价走势",
        labels={"year": "年份", "resale_price": "均价 (新元)", "town_type": "镇区类型"},
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    # 2. Flat type distribution
    st.subheader("🏢 主流户型分布")
    dist = combined.groupby(["town_type", "flat_type"]).size().reset_index(name="count")
    dist["pct"] = dist.groupby("town_type")["count"].transform(lambda x: x / x.sum() * 100)
    fig2 = px.bar(
        dist, x="flat_type", y="pct", color="town_type", barmode="group",
        color_discrete_map={"成熟区": "#FF6B6B", "非成熟区": "#4ECDC4"},
        text=dist["pct"].apply(lambda v: f"{v:.1f}%"),
        title="户型占比 (%) — 成熟区 vs 非成熟区",
        labels={"flat_type": "房型", "pct": "占比 (%)", "town_type": "镇区类型"},
    )
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, width='stretch')

    # 3. Key metrics
    st.subheader("📊 关键指标对比")
    for tt in ["成熟区", "非成熟区"]:
        sub = combined[combined["town_type"] == tt]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{tt} 均价", fmt_price(sub["resale_price"].mean()))
        with c2:
            st.metric(f"{tt} 单价", f"S${sub['price_per_sqm'].mean():,.0f}/sqm")
        with c3:
            st.metric(f"{tt} 交易量", f"{len(sub):,}")
        with c4:
            st.metric(f"{tt} 平均面积", f"{sub['floor_area_sqm'].mean():.0f} sqm")


# ======================== AMENITIES PROXIMITY ========================

def _analyze_amenities(df: pd.DataFrame):
    st.subheader("🚇 配套设施对价格的影响")
    st.markdown("基于镇区中心到最近 MRT 站点的近似距离（镇区级分析）。")

    df_clean = df.dropna(subset=["mrt_zone", "price_per_sqm"])

    # 1. MRT zone comparison
    c1, c2 = st.columns(2)
    with c1:
        zone_stats = df_clean.groupby("mrt_zone").agg(
            均价=("resale_price", "mean"), 平均单价=("price_per_sqm", "mean"),
            套数=("resale_price", "count"),
        ).reset_index()
        fig = px.bar(
            zone_stats, x="mrt_zone", y="平均单价", color="mrt_zone",
            text=zone_stats["平均单价"].apply(lambda v: f"S${v:,.0f}"),
            title="MRT 距离 vs 单价",
            labels={"mrt_zone": "距 MRT 距离", "平均单价": "单价 (新元/sqm)"},
        )
        fig.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')

    with c2:
        # MRT distance vs unit price scatter
        sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)
        fig2 = px.scatter(
            sample, x="mrt_dist_km", y="price_per_sqm", color="town",
            color_discrete_map=TOWN_COLORS, opacity=0.5, trendline="ols",
            title="MRT 距离 (km) vs 单价",
            labels={"mrt_dist_km": "距 MRT (km)", "price_per_sqm": "单价 (新元/sqm)"},
        )
        fig2.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width='stretch')

    # 2. Price difference: MRT < 500m vs > 1km
    near = df_clean[df_clean["mrt_zone"] == "<500m"]["price_per_sqm"]
    far = df_clean[df_clean["mrt_zone"] == ">1km"]["price_per_sqm"]
    if len(near) > 5 and len(far) > 5:
        diff_pct = (near.mean() - far.mean()) / far.mean() * 100
        t_stat, p_val = stats.ttest_ind(near, far, equal_var=False)
        st.caption(
            f"距 MRT <500m 比 >1km 单价高 **{diff_pct:+.1f}%** (t={t_stat:.1f}, p={p_val:.1e})。"
            f"靠近地铁站显著提升房价。"
        )
    else:
        st.caption("当前数据范围不足以区分 <500m 和 >1km 组别（镇区级近似分析）。")


# ======================== PURCHASE STRATEGY ========================

def _analyze_strategy(df: pd.DataFrame):
    st.subheader("💡 购房策略与保值验证分析")
    st.markdown("使用 2020–2023 年数据提出策略，2024 年至今数据验证表现。")

    split_year = 2024
    train = df[df["year"] < split_year].copy()
    test = df[df["year"] >= split_year].copy()

    if len(train) < 100 or len(test) < 50:
        st.warning("数据不足以进行策略验证。")
        return

    # ---- Strategy selection ----
    st.markdown("### 🎯 策略选择")
    strategy = st.radio(
        "选择购房策略",
        ["非成熟区成长策略", "低总价入门策略", "长剩余租约策略", "4-Room 中产家庭策略"],
        horizontal=True,
    )

    # Strategy parameters
    if strategy == "非成熟区成长策略":
        strategy_desc = "关注非成熟区的增值潜力，选择 Punggol/Sengkang 4-Room/5-Room，预算 ≤60万新元。"
        budget_max = 600_000
        target_types = ["4-Room", "5-Room"]
        target_towns = ["PUNGGOL", "SENGKANG"]
        strategy_name = "非成熟区成长"

    elif strategy == "低总价入门策略":
        strategy_desc = "控制总价 ≤45万新元，优先 3-Room/4-Room，适合首次购房者。"
        budget_max = 450_000
        target_types = ["3-Room", "4-Room"]
        target_towns = ["PUNGGOL", "SENGKANG", "HOUGANG"]
        strategy_name = "低总价入门"

    elif strategy == "长剩余租约策略":
        strategy_desc = "选择剩余年限 ≥70 年的房产，降低折旧风险，4-Room 为主。"
        budget_max = 650_000
        target_types = ["4-Room"]
        target_towns = ["PUNGGOL", "SENGKANG", "HOUGANG"]
        strategy_name = "长租约"

    else:  # 4-Room 中产家庭策略
        strategy_desc = "4-Room 房型供需最活跃，兼顾面积与预算，适合中等收入家庭。"
        budget_max = 550_000
        target_types = ["4-Room"]
        target_towns = ["PUNGGOL", "SENGKANG", "HOUGANG"]
        strategy_name = "4-Room 中产"

    st.info(strategy_desc)

    # Apply strategy filters
    strat_train = train[
        train["flat_type"].isin(target_types)
        & (train["resale_price"] <= budget_max)
        & train["town"].isin(target_towns)
    ]
    strat_test = test[
        test["flat_type"].isin(target_types)
        & (test["resale_price"] <= budget_max)
        & test["town"].isin(target_towns)
    ]

    # Baseline: all data with same type filter but no budget/town constraint
    base_train = train[train["flat_type"].isin(target_types)]
    base_test = test[test["flat_type"].isin(target_types)]

    if len(strat_train) < 50:
        st.warning(f"策略组训练数据不足 ({len(strat_train)} 条)，请放宽约束。")
        return

    # ---- Strategy Results ----
    st.markdown("### 📊 策略回测对比")

    def _calc_metrics(train_set, test_set, label):
        train_avg = train_set["resale_price"].mean()
        test_avg = test_set["resale_price"].mean()
        years = max(test_set["year"].mean() - train_set["year"].mean(), 0.5)
        cagr = (test_avg / train_avg) ** (1 / years) - 1 if train_avg > 0 else 0

        # Volatility: CV of quarterly avg
        test_set_c = test_set.copy()
        test_set_c["quarter"] = test_set_c["month"].dt.to_period("Q")
        quarterly = test_set_c.groupby("quarter")["resale_price"].mean()
        cv = quarterly.std() / quarterly.mean() if quarterly.mean() > 0 else 0

        # Max drawdown
        cummax = quarterly.cummax()
        drawdown = (quarterly - cummax) / cummax
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Volume stability
        vol = test_set_c.groupby("quarter").size()
        vol_cv = vol.std() / vol.mean() if vol.mean() > 0 else 0

        return {
            "分组": label,
            "训练期均价": train_avg,
            "验证期均价": test_avg,
            "CAGR": cagr,
            "波动率(CV)": cv,
            "最大回撤": max_dd,
            "成交量CV": vol_cv,
            "训练样本": len(train_set),
            "验证样本": len(test_set),
        }

    strat_metrics = _calc_metrics(strat_train, strat_test, f"策略组: {strategy_name}")
    base_metrics = _calc_metrics(base_train, base_test, "基准组: 同房型全量")

    # Comparison table
    compare = pd.DataFrame([strat_metrics, base_metrics])
    compare_display = compare.copy()
    for col in ["训练期均价", "验证期均价"]:
        compare_display[col] = compare_display[col].apply(fmt_price)
    for col in ["CAGR", "波动率(CV)", "最大回撤", "成交量CV"]:
        compare_display[col] = compare_display[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(compare_display, width='stretch', hide_index=True)

    # Chart: CAGR comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="年化涨幅", x=compare["分组"], y=[m["CAGR"]*100 for _, m in compare.iterrows()],
        text=[f"{m['CAGR']*100:.2f}%" for _, m in compare.iterrows()],
        textposition="outside", marker_color=["#FF6B6B", "#888888"],
    ))
    fig.update_layout(
        title="策略组 vs 基准组 — 年化涨幅对比", height=380,
        margin=dict(l=0, r=0, t=30, b=0), yaxis_title="CAGR (%)",
    )
    st.plotly_chart(fig, width='stretch')

    # Multi-metric radar
    metrics_names = ["年化涨幅", "稳定性(1-CV)", "低回撤(1-MDD)", "流动性(1-VC)"]
    strat_vals = [strat_metrics["CAGR"]*100, (1-strat_metrics["波动率(CV)"])*100,
                  (1-strat_metrics["最大回撤"])*100, (1-strat_metrics["成交量CV"])*100]
    base_vals = [base_metrics["CAGR"]*100, (1-base_metrics["波动率(CV)"])*100,
                 (1-base_metrics["最大回撤"])*100, (1-base_metrics["成交量CV"])*100]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=strat_vals, theta=metrics_names, fill="toself",
                                        name=f"策略组: {strategy_name}"))
    fig_radar.add_trace(go.Scatterpolar(r=base_vals, theta=metrics_names, fill="toself",
                                        name="基准组"))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                            title="四维雷达对比", height=420,
                            margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_radar, width='stretch')

    # ---- Conclusion ----
    st.markdown("### 📝 策略适用性分析")
    cagr_diff = strat_metrics["CAGR"] - base_metrics["CAGR"]
    cv_diff = strat_metrics["波动率(CV)"] - base_metrics["波动率(CV)"]

    st.markdown(f"""
    **{strategy_name}策略** 适用分析：

    | 对比维度 | 策略组 | 基准组 | 差异 |
    |---------|:---:|:---:|:---:|
    | 年化涨幅 | {strat_metrics['CAGR']*100:.2f}% | {base_metrics['CAGR']*100:.2f}% | {cagr_diff*100:+.2f}% |
    | 波动率 | {strat_metrics['波动率(CV)']*100:.2f}% | {base_metrics['波动率(CV)']*100:.2f}% | {cv_diff*100:+.2f}% |
    | 最大回撤 | {strat_metrics['最大回撤']*100:.2f}% | {base_metrics['最大回撤']*100:.2f}% | — |
    | 验证样本 | {strat_metrics['验证样本']} 套 | {base_metrics['验证样本']} 套 | — |

    **适合人群**：{'预算有限、追求增值潜力的年轻家庭' if '成长' in strategy_name or '入门' in strategy_name else '稳定型购房者，注重长期持有和租金稳定性'}
    """)
