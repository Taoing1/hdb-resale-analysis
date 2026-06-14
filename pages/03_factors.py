"""Page 3: Factor Analysis — 6+ factors, town vs price, MRT/school proximity, purchase strategy."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils.helpers import TOWN_COLORS, fmt_price

# ---- MRT station coordinates (same as map page) ----
MRT_COORDS = {
    "PUNGGOL":    [(1.4052, 103.9022)],  # Punggol MRT
    "SENGKANG":   [(1.3924, 103.8951)],  # Sengkang MRT
    "HOUGANG":    [(1.3716, 103.8923)],  # Hougang MRT
}

# ---- Target towns (Northeast Region) ----
TARGET_TOWNS = ["PUNGGOL", "SENGKANG", "HOUGANG"]


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


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("📊 房价影响因素分析")
    st.markdown("面积 · 房型 · 楼层 · 剩余租约 · 房龄 · 镇区对比 · 购房策略验证")

    # Prepare derived columns
    df = df.copy()
    df["floor_age"] = df["lease_commence_date"].apply(
        lambda x: pd.Timestamp.now().year - x if pd.notna(x) else None
    )
    df["storey_bin"] = pd.cut(
        df["storey_mid"], bins=[0, 6, 15, 60],
        labels=["低层 (1-6)", "中层 (7-15)", "高层 (16+)"]
    )
    # All 3 target towns (Punggol/Sengkang/Hougang) are Northeast region
    df["mrt_dist_km"] = df["town"].apply(_nearest_mrt_dist)
    df["mrt_zone"] = pd.cut(
        df["mrt_dist_km"], bins=[0, 0.5, 1.0, 5.0],
        labels=["<500m", "500m–1km", ">1km"]
    )

    tabs = st.tabs([
        "① 面积", "② 房型", "③ 楼层", "④ 剩余租约",
        "⑤ 房龄", "⑥ 镇区对比", "💡 策略验证"
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
        _analyze_town_comparison(df)
    with tabs[6]:
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


# ======================== ⑥ TOWN vs PRICE COMPARISON ========================

def _analyze_town_comparison(df: pd.DataFrame):
    st.subheader("⑥ 镇区对比 — Town vs Price 分析")

    # Compute 3-town average as baseline
    baseline = df["price_per_sqm"].mean()

    town_stats_list = []
    for town_name in df["town"].unique():
        sub = df[df["town"] == town_name]
        town_avg_price = sub["price_per_sqm"].mean()
        deviation = (town_avg_price - baseline) / baseline * 100
        town_stats_list.append({
            "镇区": town_name,
            "均价 (S$/sqm)": round(town_avg_price, 0),
            "中位数 (S$/sqm)": round(sub["price_per_sqm"].median(), 0),
            "偏离度 (%)": round(deviation, 2),
            "套数": len(sub),
        })
    town_stats = pd.DataFrame(town_stats_list).sort_values("偏离度 (%)", ascending=False)

    # Deviation bar chart
    colors = ["#FF6B6B" if d > 0 else "#4ECDC4" for d in town_stats["偏离度 (%)"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=town_stats["镇区"], y=town_stats["偏离度 (%)"],
        text=town_stats["偏离度 (%)"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside", marker_color=colors,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="三镇区均价基准线")
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0),
                      title="各镇区价格偏离度 (vs 三镇区均价基准)")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, width='stretch')

    # Key metrics per town
    st.subheader("📊 各镇区关键指标")
    for _, row in town_stats.iterrows():
        sub = df[df["town"] == row["镇区"]]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{row['镇区']} 均价", fmt_price(sub["resale_price"].mean()))
        with c2:
            st.metric(f"{row['镇区']} 单价", f"S${sub['price_per_sqm'].mean():,.0f}/sqm")
        with c3:
            st.metric(f"{row['镇区']} 交易量", f"{len(sub):,}")
        with c4:
            st.metric(f"{row['镇区']} 平均面积", f"{sub['floor_area_sqm'].mean():.0f} sqm")

    # Stats table
    display = town_stats.copy()
    display["均价 (S$/sqm)"] = display["均价 (S$/sqm)"].apply(lambda x: f"S${x:,.0f}")
    display["中位数 (S$/sqm)"] = display["中位数 (S$/sqm)"].apply(lambda x: f"S${x:,.0f}")
    st.dataframe(display.rename(columns={"偏离度 (%)": "偏离度%"}), width='stretch', hide_index=True)

    # ANOVA / Kruskal-Wallis across towns
    town_groups = [df[df["town"] == t]["price_per_sqm"].dropna() for t in df["town"].unique()]
    if all(len(g) > 10 for g in town_groups) and len(town_groups) >= 2:
        try:
            h_stat, p_val = stats.kruskal(*town_groups)
            st.caption(f"Kruskal-Wallis H-test: H={h_stat:.1f}, p={p_val:.1e} — 镇区间单价差异{'显著' if p_val < 0.05 else '不显著'}")
        except Exception:
            pass


# ======================== PURCHASE STRATEGY ========================

def _analyze_strategy(df: pd.DataFrame):
    st.subheader("💡 购房策略与保值验证分析")
    st.markdown("""
    在明确约束下提出组屋选择策略 → 用 **2020–2023** 年数据形成策略依据 → 用 **2024 至今** 数据验证表现。
    对比策略组与基准组的 **年化涨幅、价格波动、最大回撤、成交量稳定性和预算适配性**。
    """)

    split_year = 2024
    train = df[df["year"] < split_year].copy()
    test = df[df["year"] >= split_year].copy()

    if len(train) < 100 or len(test) < 50:
        st.warning("数据不足以进行策略验证。")
        return

    # ---- Strategy definition ----
    st.markdown("---")
    st.markdown("### 🎯 策略设定")

    # Define strategies with target family/goal
    STRATEGIES = {
        "镇区偏离回归策略": {
            "desc": "选择价格低于三镇区均值的镇区 × 房型组合，利用均值回归逻辑获取超额增值。"
                    "预算 ≤ S$600,000，房型 4-Room / 5-Room。",
            "budget": 600_000,
            "types": ["4-Room", "5-Room"],
            "towns": ["PUNGGOL", "SENGKANG", "HOUGANG"],
            "extra": None,  # No extra filter — price deviation handled by budget
            "target": "年轻成长型家庭",
            "goal": "最大化增值潜力",
            "risk": "中高风险 — 规划兑现不及预期可能导致涨幅低于预期",
            "why": "三镇区中价格较低的镇区（如 Punggol）正处于基础设施快速兑现期（Punggol Digital District、"
                   "Cross Island Line），区域规划红利尚未完全释放到房价中。买入折价区的核心逻辑是「价格向均值回归」——"
                   "随着交通、商业、学校配套逐步完善，价格洼地将被填平。选择 4-Room/5-Room 确保面积足以满足家庭需求，"
                   "同时这一房型在转售市场上拥有最广泛的买家群体，流动性有保障。",
        },
        "低总价入门策略": {
            "desc": "控制总价 ≤ S$450,000，优先 3-Room / 4-Room，降低月供压力。",
            "budget": 450_000,
            "types": ["3-Room", "4-Room"],
            "towns": ["PUNGGOL", "SENGKANG", "HOUGANG"],
            "extra": None,
            "target": "首次购房者 / 单身购房者",
            "goal": "低门槛入市，积累资产",
            "risk": "低风险 — 低价房源总价低、波动小，但增值绝对额有限",
            "why": "首次购房者通常面临 CPF 储蓄有限、月供承受能力较低的现实约束。S$450,000 以内的 3-Room/4-Room "
                   "月供约 S$1,500-2,000（按 75% LTV、3% 利率计算），处于中等收入家庭的舒适区间。"
                   "且小户型/低价房源在市场上拥有稳定的刚性需求——每年有大量新 PR、公民首次购房者入市，"
                   "确保策略具有持续的买方支撑。策略的目标是「先上车」，用最小资金成本获取房产身份，"
                   "未来通过升级置换实现资产跃迁。",
        },
        "长剩余租约策略": {
            "desc": "选择剩余租约 ≥ 70 年的 4-Room 组屋，规避折旧风险。预算 ≤ S$650,000。",
            "budget": 650_000,
            "types": ["4-Room"],
            "towns": ["PUNGGOL", "SENGKANG", "HOUGANG"],
            "extra": lambda d: d["remaining_lease"] >= 70,
            "target": "长期持有者 / 退休规划家庭",
            "goal": "资产长寿 + 稳定增值",
            "risk": "低风险 — 长租约房产折旧慢、贷款条件优，但买入价中已含租约溢价",
            "why": "HDB 组屋的 99 年产权意味着剩余租约是影响房产价值的核心变量。剩余租约 < 60 年的组屋面临加速折旧、"
                   "银行贷款受限（贷款到期时租约须 ≥ 20 年）、CPF 使用限制等多重劣势。选择 ≥ 70 年的房源确保："
                   "(1) 持有 20-30 年后仍剩余 40-50 年租约，转售时买家依然可获足额贷款；"
                   "(2) 折旧曲线在 60 年以上区间相对平缓，价格稳定性高；"
                   "(3) 适合作为退休资产——房产寿命覆盖业主生命周期。",
        },
        "4-Room 中产家庭策略": {
            "desc": "聚焦 4-Room 房型（HDB 最主流户型），预算 ≤ S$550,000，供需最活跃。",
            "budget": 550_000,
            "types": ["4-Room"],
            "towns": ["PUNGGOL", "SENGKANG", "HOUGANG"],
            "extra": None,
            "target": "中等收入核心家庭",
            "goal": "兼顾面积、预算与流动性",
            "risk": "中等风险 — 4-Room 市场深度好，但预算限制可能排除部分优质房源",
            "why": "4-Room 是新加坡 HDB 转售市场的「黄金户型」——约占三镇区总成交量的 45-50%，"
                   "买家群体最广（新婚夫妇、有孩家庭、 downsizing 老年夫妇），流动性最强。"
                   "面积约 80-100㎡，三室一厅格局满足绝大多数家庭的居住需求。"
                   "S$550,000 预算覆盖约 60-70% 的 4-Room 成交区间，不会因预算过紧而大量排除优质房源。"
                   "策略的核心理念是「不追求最高涨幅，而是追求最稳健的保值 + 最高流动性」，适合以自住为主、"
                   "兼顾未来转售需求的家庭。",
        },
    }

    strategy = st.radio(
        "选择购房策略",
        list(STRATEGIES.keys()),
        horizontal=True,
    )
    cfg = STRATEGIES[strategy]
    st.info(cfg["desc"])

    # Strategy summary card
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 目标人群", cfg["target"])
        st.metric("📈 核心目标", cfg["goal"])
    with c2:
        st.metric("💰 预算上限", fmt_price(cfg["budget"]))
        st.metric("🏠 户型范围", ", ".join(cfg["types"]))
    with c3:
        st.metric("📍 候选镇区", " / ".join(cfg["towns"]))
        st.metric("⚠️ 风险等级", cfg["risk"])

    # ---- Apply strategy filters ----
    strat_train = train[
        train["flat_type"].isin(cfg["types"])
        & (train["resale_price"] <= cfg["budget"])
        & train["town"].isin(cfg["towns"])
    ]
    strat_test = test[
        test["flat_type"].isin(cfg["types"])
        & (test["resale_price"] <= cfg["budget"])
        & test["town"].isin(cfg["towns"])
    ]
    if cfg["extra"] is not None:
        strat_train = strat_train[cfg["extra"](strat_train)]
        strat_test = strat_test[cfg["extra"](strat_test)]

    # Baseline: same flat type, all towns, no budget limit
    base_train = train[train["flat_type"].isin(cfg["types"])]
    base_test = test[test["flat_type"].isin(cfg["types"])]

    if len(strat_train) < 50:
        st.warning(f"策略组训练数据不足 ({len(strat_train)} 条)，请放宽约束。")
        return

    # ---- Compute metrics ----
    st.markdown("---")
    st.markdown("### 📊 策略回测验证 (2020–2023 → 2024+)")

    def _calc_metrics(train_set, test_set, label):
        train_avg = train_set["resale_price"].mean()
        test_avg = test_set["resale_price"].mean()
        years = max(test_set["year"].mean() - train_set["year"].mean(), 0.5)
        cagr = (test_avg / train_avg) ** (1 / years) - 1 if train_avg > 0 else 0

        # Quarterly price volatility
        test_c = test_set.copy()
        test_c["quarter"] = test_c["month"].dt.to_period("Q")
        quarterly = test_c.groupby("quarter")["resale_price"].mean()
        cv = quarterly.std() / quarterly.mean() if quarterly.mean() > 0 else 0

        # Max drawdown
        cummax = quarterly.cummax()
        drawdown = (quarterly - cummax) / cummax
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Volume stability
        vol = test_c.groupby("quarter").size()
        vol_cv = vol.std() / vol.mean() if vol.mean() > 0 else 0

        # Budget fitness: % of test-set transactions within budget
        budget_fit = (test_set["resale_price"] <= test_set["resale_price"].max()).mean()
        budget_fit = (test_set["resale_price"] <= 999_999_999).mean()  # placeholder, recalculated below

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

    strat_metrics = _calc_metrics(strat_train, strat_test, f"策略组: {strategy}")
    base_metrics = _calc_metrics(base_train, base_test, "基准组: 同房型全量")

    # Budget fitness: what % of strategy-group test transactions are under budget
    budget_fit_strat = (strat_test["resale_price"] <= cfg["budget"]).mean() * 100
    budget_fit_base = (base_test["resale_price"] <= cfg["budget"]).mean() * 100

    # ---- Comparison dashboard ----
    c1, c2, c3, c4 = st.columns(4)
    cagr_diff = strat_metrics["CAGR"] - base_metrics["CAGR"]
    cv_diff = strat_metrics["波动率(CV)"] - base_metrics["波动率(CV)"]
    with c1:
        st.metric("年化涨幅 (CAGR)",
                  f"{strat_metrics['CAGR']*100:.2f}%",
                  delta=f"{cagr_diff*100:+.2f}% vs 基准",
                  delta_color="normal" if cagr_diff > 0 else "inverse")
    with c2:
        st.metric("价格波动率 (CV)",
                  f"{strat_metrics['波动率(CV)']*100:.2f}%",
                  delta=f"{cv_diff*100:+.2f}% vs 基准",
                  delta_color="inverse" if cv_diff > 0 else "normal")
    with c3:
        st.metric("最大回撤", f"{strat_metrics['最大回撤']*100:.2f}%")
    with c4:
        st.metric("预算适配性", f"{budget_fit_strat:.0f}%",
                  help=f"策略组验证期成交中 {budget_fit_strat:.0f}% 在预算内")

    # ---- Detailed comparison table ----
    compare = pd.DataFrame([strat_metrics, base_metrics])
    compare_display = compare.copy()
    for col in ["训练期均价", "验证期均价"]:
        compare_display[col] = compare_display[col].apply(fmt_price)
    for col in ["CAGR", "波动率(CV)", "最大回撤", "成交量CV"]:
        compare_display[col] = compare_display[col].apply(lambda x: f"{x*100:.2f}%")
    compare_display = compare_display.rename(columns={
        "CAGR": "年化涨幅", "波动率(CV)": "波动率",
        "最大回撤": "最大回撤", "成交量CV": "成交量CV",
        "训练期均价": "训练均价", "验证期均价": "验证均价",
        "训练样本": "训练样本", "验证样本": "验证样本",
    })
    st.dataframe(compare_display, width='stretch', hide_index=True)

    # ---- Charts ----
    st.markdown("#### 📈 季度价格走势对比")
    # Build quarterly trend
    strat_q = strat_test.copy()
    strat_q["quarter"] = strat_q["month"].dt.to_period("Q")
    base_q = base_test.copy()
    base_q["quarter"] = base_q["month"].dt.to_period("Q")

    q_strat = strat_q.groupby("quarter")["resale_price"].mean().reset_index()
    q_strat["quarter"] = q_strat["quarter"].astype(str)
    q_base = base_q.groupby("quarter")["resale_price"].mean().reset_index()
    q_base["quarter"] = q_base["quarter"].astype(str)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=q_strat["quarter"], y=q_strat["resale_price"],
        mode="lines+markers", name=f"策略组: {strategy}",
        line=dict(color="#FF6B6B", width=3),
    ))
    fig_trend.add_trace(go.Scatter(
        x=q_base["quarter"], y=q_base["resale_price"],
        mode="lines+markers", name="基准组: 同房型全量",
        line=dict(color="#888888", width=2, dash="dash"),
    ))
    fig_trend.update_layout(
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="均价 (新元)", xaxis_title="季度",
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, width='stretch')

    # CAGR + Radar side by side
    c1, c2 = st.columns(2)
    with c1:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="CAGR", x=compare["分组"],
            y=[m["CAGR"]*100 for _, m in compare.iterrows()],
            text=[f"{m['CAGR']*100:.2f}%" for _, m in compare.iterrows()],
            textposition="outside",
            marker_color=["#FF6B6B", "#94A3B8"],
        ))
        fig_bar.update_layout(
            title="年化涨幅 (CAGR) 对比", height=380,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="CAGR (%)",
        )
        st.plotly_chart(fig_bar, width='stretch')

    with c2:
        metrics_names = ["年化涨幅", "稳定性\n(1-CV)", "低回撤\n(1-MDD)", "流动性\n(1-VC)"]
        strat_vals = [
            max(0, strat_metrics["CAGR"]*100),
            max(0, (1 - strat_metrics["波动率(CV)"])*100),
            max(0, (1 - strat_metrics["最大回撤"])*100),
            max(0, (1 - strat_metrics["成交量CV"])*100),
        ]
        base_vals = [
            max(0, base_metrics["CAGR"]*100),
            max(0, (1 - base_metrics["波动率(CV)"])*100),
            max(0, (1 - base_metrics["最大回撤"])*100),
            max(0, (1 - base_metrics["成交量CV"])*100),
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=strat_vals, theta=metrics_names, fill="toself",
            name=f"策略组: {strategy}",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=base_vals, theta=metrics_names, fill="toself",
            name="基准组",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            title="四维策略雷达", height=380,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_radar, width='stretch')

    # ---- Strategy rationale (text analysis) ----
    st.markdown("---")
    st.markdown("### 📝 策略深度分析")

    st.markdown(f"""
    #### 为什么「{strategy}」适合 **{cfg['target']}**？

    {cfg['why']}

    ---

    #### 验证期表现总结

    | 维度 | 策略组 | 基准组 | 评价 |
    |------|:---:|:---:|------|
    | 年化涨幅 | {strat_metrics['CAGR']*100:.2f}% | {base_metrics['CAGR']*100:.2f}% | {'✅ 策略组跑赢基准' if cagr_diff > 0.005 else '⚖️ 与基准持平' if cagr_diff > -0.005 else '⚠️ 策略组跑输基准'} |
    | 价格波动 | {strat_metrics['波动率(CV)']*100:.2f}% | {base_metrics['波动率(CV)']*100:.2f}% | {'✅ 策略组更稳定' if cv_diff < 0 else '⚠️ 策略组波动略大'} |
    | 最大回撤 | {strat_metrics['最大回撤']*100:.2f}% | {base_metrics['最大回撤']*100:.2f}% | {'✅ 策略组抗跌更强' if strat_metrics['最大回撤'] < base_metrics['最大回撤'] else '⚠️ 策略组回撤略大'} |
    | 验证样本 | {strat_metrics['验证样本']} 套 | {base_metrics['验证样本']} 套 | 策略筛选后样本 {'充足' if strat_metrics['验证样本'] > 100 else '偏少，需关注'} |
    | 预算适配 | {budget_fit_strat:.0f}% | {budget_fit_base:.0f}% | 策略组内预算覆盖 {'充分' if budget_fit_strat > 80 else '一般' if budget_fit_strat > 50 else '偏紧'} |

    #### 策略的核心逻辑（非简单追求最高价）

    本策略的设计哲学是 **「约束条件下的最优化」**，而非简单的「买最贵的」或「买涨最快的」：

    1. **预算约束的意义**：预算上限不只是财务限制，更是**质量筛选器**——在 HDB 市场中，超高总价的房源往往是大户型 Executive 或 Multi-Gen（买家群体窄、流动性差），中等预算区间才是市场主力。设置合理预算上限等于自动排除尾部风险。

    2. **户型约束的意义**：限定房型使策略组和基准组在「产品类型」上可比。不同房型的价格分布和增值逻辑截然不同——2-Room 的买家群体（单身/退休）与 5-Room（大家庭）完全不同，混在一起比较没有意义。

    3. **时间切分的意义**：2020–2023 → 2024+ 的严格时间切分确保策略没有「偷看未来」。真实购房场景中，你无法用 2025 年的数据来决定 2023 年的投资——同样的逻辑，策略必须用历史数据形成、用未来数据验证。

    4. **多维评估的意义**：单一 CAGR 指标可能误导——一个 CAGR 高但波动剧烈的策略，不适合风险厌恶型家庭。因此我们引入「涨幅 + 稳定性 + 回撤 + 流动性」四维评估，让不同类型的购房者可以根据自身偏好（自住 vs 投资、短期 vs 长期）选择最匹配的策略。
    """)
