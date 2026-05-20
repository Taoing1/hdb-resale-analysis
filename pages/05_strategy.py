"""Page 5: Purchase Strategy — budget/type constraints, 3-dimension scoring with train/test backtest, policy stubs."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import TOWN_COLORS, fmt_price


def run(df: pd.DataFrame):
    st.title("💡 购房策略分析")
    st.markdown("预算约束 + 房型筛选 → 三维度 (涨幅/稳定性/流动性) 量化评分 → 推荐排名。")

    # ---- Sidebar: Constraints ----
    filters = _build_sidebar(df)

    # ---- Filter Candidates ----
    candidates = df[
        df["flat_type"].isin(filters["types"])
        & (df["resale_price"].between(filters["budget_min"], filters["budget_max"]))
        & (df["floor_area_sqm"].between(filters["area_min"], filters["area_max"]))
    ].copy()

    if len(candidates) < 50:
        st.warning(f"当前约束下仅有 {len(candidates)} 条记录，建议放宽预算或房型条件。")
        return

    st.markdown(f"符合条件: **{len(candidates):,}** 条 ({filters['budget_min']/1000:.0f}K–{filters['budget_max']/1000:.0f}K SGD, "
                f"{', '.join(filters['types'])})")

    # ---- Train/Test Split for Backtesting ----
    split_year = int(df["year"].quantile(0.75))
    train = candidates[candidates["year"] <= split_year]
    test = candidates[candidates["year"] > split_year]

    if len(test) < 30:
        st.warning(f"测试集仅 {len(test)} 条记录 (>{split_year})，回测结果仅供参考。")

    st.caption(f"训练集 (≤{split_year}): {len(train):,} 条 | 测试集 (>{split_year}): {len(test):,} 条")

    # ==================== 3 DIMENSIONS ====================

    st.header("📈 三维度量化评分")

    appreciation_scores = _calc_appreciation(train, test)
    stability_scores = _calc_stability(candidates)
    liquidity_scores = _calc_liquidity(candidates, filters["types"])

    w = filters["weights"]
    composite = _build_composite(appreciation_scores, stability_scores, liquidity_scores, w)

    tabs = st.tabs(["📈 增值潜力", "🛡️ 价格稳定性", "💧 市场流动性", "🏆 综合排名", "📊 雷达对比"])

    with tabs[0]:
        _render_appreciation(appreciation_scores)
    with tabs[1]:
        _render_stability(stability_scores, w)
    with tabs[2]:
        _render_liquidity(liquidity_scores)
    with tabs[3]:
        _render_composite(composite, w)
    with tabs[4]:
        _render_radar(composite)

    # ==================== RECOMMENDATION ====================

    st.divider()
    st.header("🎯 策略推荐")
    _render_recommendation(composite, candidates, filters)

    # ==================== AFFORDABILITY ====================

    with st.expander("💰 购房能力测算", expanded=False):
        _render_affordability(filters)

    # ==================== POLICY STUB ====================

    with st.expander("📜 政策影响分析 (接口预留)", expanded=False):
        _policy_analysis_stub(filters)


# ======================== SIDEBAR ========================

def _build_sidebar(df: pd.DataFrame) -> dict:
    """Build sidebar constraint controls and return filter dict."""
    st.sidebar.subheader("🎯 购房约束")

    budget_min = st.sidebar.number_input("最低预算 (SGD)", 0, 2_000_000, 300_000, 10_000)
    budget_max = st.sidebar.number_input("最高预算 (SGD)", 0, 2_000_000, 600_000, 10_000)

    st.sidebar.markdown("---")
    target_types = st.sidebar.multiselect(
        "目标房型", sorted(df["flat_type"].unique()),
        default=["4-Room", "5-Room"]
    )
    area_min = st.sidebar.number_input("最小面积 (sqm)", 30, 300, 60)
    area_max = st.sidebar.number_input("最大面积 (sqm)", 30, 300, 150)

    st.sidebar.markdown("---")
    st.sidebar.caption("评分权重 (三维度)")
    w_app = st.sidebar.slider("涨幅权重", 0.0, 1.0, 0.4, 0.05, key="w_app")
    w_stab = st.sidebar.slider("稳定性权重", 0.0, 1.0, 0.3, 0.05, key="w_stab")
    w_liq = st.sidebar.slider("流动性权重", 0.0, 1.0, 0.3, 0.05, key="w_liq")
    total_w = w_app + w_stab + w_liq
    if total_w > 0:
        w_app, w_stab, w_liq = w_app / total_w, w_stab / total_w, w_liq / total_w
    st.sidebar.caption(f"归一化权重: 涨幅 {w_app:.2f} | 稳定性 {w_stab:.2f} | 流动性 {w_liq:.2f}")

    return {
        "budget_min": budget_min, "budget_max": budget_max,
        "types": target_types, "area_min": area_min, "area_max": area_max,
        "weights": {"appreciation": w_app, "stability": w_stab, "liquidity": w_liq},
    }


# ==================== DIMENSION 1: APPRECIATION ====================

def _calc_appreciation(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Compute CAGR-based appreciation scores using train/test split.

    Train period avg price → Test period avg price → annualized return (CAGR).
    This simulates: if you bought during the train period, what CAGR would you
    have realized by the test period?
    """
    groups = []
    for (town, ftype), grp in train.groupby(["town", "flat_type"]):
        train_avg = grp["resale_price"].mean()
        test_grp = test[(test["town"] == town) & (test["flat_type"] == ftype)]
        if len(test_grp) < 5 or pd.isna(train_avg):
            continue
        test_avg = test_grp["resale_price"].mean()
        train_year = grp["year"].mean()
        test_year = test_grp["year"].mean()
        years = max(test_year - train_year, 0.5)
        cagr = (test_avg / train_avg) ** (1 / years) - 1 if train_avg > 0 else 0
        groups.append({
            "town": town, "flat_type": ftype,
            "train_avg": train_avg, "test_avg": test_avg,
            "cagr": cagr, "train_n": len(grp), "test_n": len(test_grp),
        })

    result = pd.DataFrame(groups)
    if result.empty:
        return pd.DataFrame(columns=["town", "flat_type", "cagr", "score"])

    # Normalize to 0–100
    mn, mx = result["cagr"].min(), result["cagr"].max()
    spread = mx - mn if mx > mn else 1e-6
    result["score"] = ((result["cagr"] - mn) / spread * 100).round(1)
    return result.sort_values("score", ascending=False)


def _render_appreciation(scores: pd.DataFrame):
    """Display appreciation dimension with chart and table."""
    if scores.empty:
        st.info("数据不足，无法计算涨幅评分。")
        return

    display = scores.copy()
    display["train_avg"] = display["train_avg"].apply(fmt_price)
    display["test_avg"] = display["test_avg"].apply(fmt_price)
    display["cagr"] = display["cagr"].apply(lambda x: f"{x*100:+.2f}%")
    st.dataframe(
        display.rename(columns={"town": "镇区", "flat_type": "房型", "train_avg": "训练期均价",
                                "test_avg": "测试期均价", "cagr": "年化涨幅", "score": "评分"}),
        width='stretch', hide_index=True,
        column_order=["town", "flat_type", "train_avg", "test_avg", "cagr", "score", "train_n", "test_n"],
    )

    fig = px.bar(
        scores, x="town", y="cagr", color="flat_type", barmode="group",
        title="各镇区/房型 年化涨幅 (CAGR)",
        labels={"town": "镇区", "cagr": "年化涨幅", "flat_type": "房型"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width='stretch')

    st.caption(
        f"涨幅基于训练集/测试集时间切分回测：训练期均价 → 测试期均价计算年化复合增长率 (CAGR)。"
        f"CAGR 越高，历史增值表现越好。"
    )


# ==================== DIMENSION 2: STABILITY ====================

def _calc_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Price stability: lower coefficient of variation → higher score."""
    stats = df.groupby(["town", "flat_type"])["resale_price"].agg(["std", "mean", "count"]).reset_index()
    stats.columns = ["town", "flat_type", "std", "mean", "count"]
    stats["cv"] = (stats["std"] / stats["mean"]).round(4)

    mn, mx = stats["cv"].min(), stats["cv"].max()
    spread = mx - mn if mx > mn else 1e-6
    stats["score"] = ((1 - (stats["cv"] - mn) / spread) * 100).round(1)
    # Penalize groups with too few samples
    stats.loc[stats["count"] < 20, "score"] = stats.loc[stats["count"] < 20, "score"] * 0.7
    return stats.sort_values("score", ascending=False)


def _render_stability(scores: pd.DataFrame, weights: dict):
    """Display stability dimension."""
    display = scores.copy()
    display["std"] = display["std"].apply(fmt_price)
    display["mean"] = display["mean"].apply(fmt_price)
    st.dataframe(
        display.rename(columns={"town": "镇区", "flat_type": "房型", "std": "标准差",
                                "mean": "均价", "cv": "变异系数", "score": "评分", "count": "样本数"}),
        width='stretch', hide_index=True,
    )

    fig = px.bar(
        scores, x="town", y="score", color="flat_type", barmode="group",
        text=scores["score"].apply(lambda x: f"{x:.0f}"),
        title="各镇区/房型 稳定性评分 (CV 越低 → 评分越高)",
        labels={"town": "镇区", "score": "稳定性评分", "flat_type": "房型"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    st.caption("变异系数 (CV = std/mean) 衡量价格离散程度。CV 越小，该细分市场的价格越稳定，买方议价空间越小。")


# ==================== DIMENSION 3: LIQUIDITY ====================

def _calc_liquidity(df: pd.DataFrame, target_types: list) -> pd.DataFrame:
    """Liquidity: based on transaction volume within target types, normalized."""
    counts = df[df["flat_type"].isin(target_types)].groupby(
        ["town", "flat_type"]).size().reset_index(name="volume")

    if counts.empty:
        return pd.DataFrame(columns=["town", "flat_type", "volume", "score"])

    mn, mx = counts["volume"].min(), counts["volume"].max()
    spread = mx - mn if mx > mn else 1e-6
    counts["score"] = ((counts["volume"] - mn) / spread * 100).round(1)
    return counts.sort_values("score", ascending=False)


def _render_liquidity(scores: pd.DataFrame):
    """Display liquidity dimension."""
    if scores.empty:
        st.info("数据不足。")
        return

    st.dataframe(
        scores.rename(columns={"town": "镇区", "flat_type": "房型",
                               "volume": "交易量", "score": "评分"}),
        width='stretch', hide_index=True,
    )

    fig = px.bar(
        scores, x="town", y="volume", color="flat_type", barmode="group",
        title="各镇区/房型 交易活跃度 (成交量)",
        labels={"town": "镇区", "volume": "成交量 (套)", "flat_type": "房型"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width='stretch')

    st.caption("成交量是市场流动性的代理指标 — 交易越活跃，买入/卖出越容易找到对手方。")


# ==================== COMPOSITE ====================

def _build_composite(app: pd.DataFrame, stab: pd.DataFrame, liq: pd.DataFrame,
                     weights: dict) -> pd.DataFrame:
    """Merge 3 dimensions into weighted composite score, keeping numeric scores intact."""

    def _safe_merge(left, right, on):
        return left[on + ["score"]].merge(right[on + ["score"]], on=on, how="outer", suffixes=("_l", "_r"))

    # Start with all combos
    all_combos = app[["town", "flat_type"]].copy()
    all_combos = all_combos.merge(stab[["town", "flat_type"]], on=["town", "flat_type"], how="outer")
    all_combos = all_combos.merge(liq[["town", "flat_type"]], on=["town", "flat_type"], how="outer")

    merged = all_combos.merge(app[["town", "flat_type", "score"]].rename(columns={"score": "appreciation"}),
                               on=["town", "flat_type"], how="left")
    merged = merged.merge(stab[["town", "flat_type", "score"]].rename(columns={"score": "stability"}),
                          on=["town", "flat_type"], how="left")
    merged = merged.merge(liq[["town", "flat_type", "score"]].rename(columns={"score": "liquidity"}),
                          on=["town", "flat_type"], how="left")

    for col in ["appreciation", "stability", "liquidity"]:
        merged[col] = merged[col].fillna(0)

    w = weights
    merged["composite"] = (
        merged["appreciation"] * w["appreciation"]
        + merged["stability"] * w["stability"]
        + merged["liquidity"] * w["liquidity"]
    ).round(2)

    merged["label"] = merged["town"] + " — " + merged["flat_type"]
    return merged.sort_values("composite", ascending=False)


def _render_composite(composite: pd.DataFrame, weights: dict):
    """Render composite ranking."""
    display = composite[["town", "flat_type", "appreciation", "stability", "liquidity", "composite"]].copy()
    display = display.rename(columns={
        "town": "镇区", "flat_type": "房型",
        "appreciation": "涨幅", "stability": "稳定性",
        "liquidity": "流动性", "composite": "综合评分",
    })
    st.dataframe(display, width='stretch', hide_index=True)

    fig = go.Figure()
    top = composite.head(12)
    fig.add_trace(go.Bar(
        y=top["label"], x=top["composite"], orientation="h",
        marker=dict(color=top["composite"], colorscale="Viridis", showscale=True,
                     colorbar=dict(title="综合评分")),
        text=[f"{v:.1f}" for v in top["composite"]], textposition="outside",
    ))
    fig.update_layout(
        title="综合评分排名 (Top 12)",
        height=420, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="综合评分", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width='stretch')

    st.caption(f"权重: 涨幅={weights['appreciation']:.0%} | "
               f"稳定性={weights['stability']:.0%} | 流动性={weights['liquidity']:.0%}")


# ==================== RADAR CHART ====================

def _render_radar(composite: pd.DataFrame):
    """Multi-dimensional radar chart for top candidates."""
    top = composite.head(6)
    categories = ["涨幅", "稳定性", "流动性"]

    fig = go.Figure()
    for _, row in top.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["appreciation"], row["stability"], row["liquidity"]],
            theta=categories, fill="toself", name=row["label"],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100])),
        title="三维度雷达图 (Top 6)",
        height=500, margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, width='stretch')


# ==================== RECOMMENDATION ====================

def _render_recommendation(composite: pd.DataFrame, candidates: pd.DataFrame, filters: dict):
    """Generate strategy recommendation based on top-ranked groups."""
    if composite.empty:
        st.warning("数据不足，无法生成推荐。")
        return

    top = composite.iloc[0]
    top_town, top_type = top["town"], top["flat_type"]

    # Subset candidates matching the top recommendation
    best_deals = candidates[
        (candidates["town"] == top_town) & (candidates["flat_type"] == top_type)
    ]

    st.markdown(f"""
### 首选推荐: **{top_town} — {top_type}**

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 涨幅 | **{top['appreciation']:.1f}** | 历史 CAGR 表现评分 |
| 稳定性 | **{top['stability']:.1f}** | 价格波动性评分 (越高越稳) |
| 流动性 | **{top['liquidity']:.1f}** | 市场交易活跃度评分 |
| **综合** | **{top['composite']:.1f}** | 加权综合评分 |

**当前市场中匹配的房源**: {len(best_deals)} 套
- 均价: {fmt_price(best_deals['resale_price'].mean())}
- 价格区间: {fmt_price(best_deals['resale_price'].min())} – {fmt_price(best_deals['resale_price'].max())}
- 平均单价: {fmt_price(best_deals['price_per_sqm'].mean())}/sqm
""")

    # Alternative recommendations
    if len(composite) >= 3:
        st.subheader("备选方案")
        alts = composite.iloc[1:4]
        cols = st.columns(len(alts))
        for i, (_, alt) in enumerate(alts.iterrows()):
            alt_deals = candidates[
                (candidates["town"] == alt["town"]) & (candidates["flat_type"] == alt["flat_type"])
            ]
            with cols[i]:
                st.metric(
                    f"#{i+2} {alt['town']} {alt['flat_type']}",
                    f"{alt['composite']:.1f} 分",
                    f"{len(alt_deals)} 套在售",
                )

    st.divider()
    st.markdown("""
    **策略建议:**
    - **自住需求**: 优先考虑稳定性评分高的选项，减少买入后价格波动风险
    - **投资需求**: 优先考虑涨幅评分高的选项，关注历史增值趋势
    - **短期持有 (3–5年)**: 加权流动性，确保转手便利
    - **长期持有 (10年+)**: 加权涨幅，地段和面积是核心
    """)


# ==================== AFFORDABILITY ====================

def _render_affordability(filters: dict):
    """Affordability calculator: down payment, monthly mortgage estimate."""
    st.markdown("### 月供与首付估算")

    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.number_input("目标房价 (SGD)", 200_000, 2_000_000, int(filters["budget_max"]), 10_000)
        down_pct = st.slider("首付比例 (%)", 10, 50, 25)
    with c2:
        rate = st.number_input("年利率 (%)", 0.1, 8.0, 4.0, 0.1)
        tenure = st.slider("贷款年限", 5, 30, 25)
    with c3:
        cpf_oa = st.number_input("CPF OA 月缴 (SGD)", 0, 5_000, 800, 100)

    down = price * down_pct / 100
    loan = price - down
    monthly_rate = rate / 100 / 12
    n_payments = tenure * 12
    if monthly_rate > 0:
        monthly = loan * monthly_rate * (1 + monthly_rate) ** n_payments / ((1 + monthly_rate) ** n_payments - 1)
    else:
        monthly = loan / n_payments

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("首付", fmt_price(down))
    with c2:
        st.metric("贷款额", fmt_price(loan))
    with c3:
        st.metric("月供", fmt_price(monthly))
    with c4:
        st.metric("CPF 覆盖后月现金", fmt_price(max(monthly - cpf_oa, 0)))
    st.caption("以上为粗略估算，不含印花税、律师费、保险等。")


# ==================== POLICY ANALYSIS STUB ====================

def _policy_stub_bsd(price: float) -> dict:
    """计算买方印花税 (Buyer's Stamp Duty) — 待接入 IRAS 最新税率表.

    Args:
        price: 成交价或市场价 (SGD).

    Returns:
        dict with keys: bsd_amount, effective_rate, notes.

    当前税率 (IRAS 2024):
      - 首个 S$180,000: 1%
      - 次个 S$180,000: 2%
      - 次个 S$640,000: 3%
      - 余额: 4%

    接入方式: 调用 IRAS 公开税率表或硬编码最新税率。
    """
    # Placeholder logic
    if price <= 180_000:
        bsd = price * 0.01
    elif price <= 360_000:
        bsd = 180_000 * 0.01 + (price - 180_000) * 0.02
    elif price <= 1_000_000:
        bsd = 180_000 * 0.01 + 180_000 * 0.02 + (price - 360_000) * 0.03
    else:
        bsd = 180_000 * 0.01 + 180_000 * 0.02 + 640_000 * 0.03 + (price - 1_000_000) * 0.04
    return {"bsd_amount": bsd, "effective_rate": bsd / price * 100 if price > 0 else 0}


def _policy_stub_absd(citizenship: str, property_count: int) -> dict:
    """计算额外买方印花税 (Additional Buyer's Stamp Duty) — 待接入 IRAS 最新政策.

    Args:
        citizenship: 'SC' (Singapore Citizen), 'SPR' (Permanent Resident), 'Foreigner'.
        property_count: 名下已有住宅数量。

    Returns:
        dict with keys: absd_rate, notes.

    当前 ABSD 税率 (2024):
      - SC 首套: 0%, 二套: 20%, 三套+: 30%
      - SPR 首套: 5%, 二套: 30%, 三套+: 35%
      - Foreigner: 60%

    接入方式: 调用 IRAS 公开税率表，或通过 API 查询最新政策。
    """
    rates = {
        ("SC", 0): 0.00, ("SC", 1): 0.20, ("SC", 2): 0.30,
        ("SPR", 0): 0.05, ("SPR", 1): 0.30, ("SPR", 2): 0.35,
        ("Foreigner", 0): 0.60,
    }
    rate = rates.get((citizenship, min(property_count, 2)), 0.60)
    return {"absd_rate": rate, "effective": f"{rate*100:.0f}%"}


def _policy_stub_hdb_eligibility(household_income: float, citizenship: str) -> dict:
    """检查 HDB 购买资格 (BTO/转售) — 待接入 HDB 官网最新资格标准.

    Args:
        household_income: 家庭月收入 (SGD).
        citizenship: 'SC', 'SC+SC', 'SC+SPR', 'SC+Foreigner'.

    Returns:
        dict with keys: bto_eligible, resale_eligible, grants_available, notes.

    HDB 资格规则:
      - BTO: 至少 1 位 SC + 1 位 SC/SPR, 家庭收入上限 S$14,000 (4-5房)
      - 转售: SC+SC, SC+SPR 均可, 无收入上限
      - CPF Housing Grant: 视收入与公民身份而定

    接入方式: 调用 HDB 官网 API / 定期更新资格表。
    """
    eligible = citizenship in ("SC", "SC+SC", "SC+SPR")
    return {
        "bto_eligible": False,  # Not applicable for resale
        "resale_eligible": eligible,
        "grants_available": ["CPF Housing Grant (EHG)", "Proximity Housing Grant (PHG)"],
        "notes": "HDB 转售购房资格 — 待接入 HDB 官网实时数据。",
    }


def _policy_analysis_stub(filters: dict):
    """政策影响分析模块 — 整合 BSD/ABSD/HDB 资格, 展示政策因素对购房决策的影响.

    Args:
        filters: 当前筛选条件 (预算、房型等)。
    """
    st.markdown("""
    ### 政策影响分析 (待接入实时数据)

    此模块预留接口用于分析以下政策因素对购房策略的影响:

    | 政策因素 | 影响 | 数据来源 |
    |---------|------|---------|
    | **BSD (买方印花税)** | 增加购房前期成本 1–4% | IRAS |
    | **ABSD (额外买方印花税)** | SC 二套 +20%, SPR 首套 +5%, 外国人 +60% | IRAS |
    | **LTV (贷款价值比)** | HDB 贷款上限 80%, 银行 75% | MAS |
    | **CPF Housing Grant** | EHG 最高 S$120,000 (视收入) | HDB |
    | **MSR (按揭偿还率)** | 月供 ≤ 30% 家庭月收入 | MAS |
    | **MOP (最低居住期)** | 新组屋 5 年内不可转售 | HDB |
    | **BTO 供应** | 影响转售市场需求与价格 | HDB |
    | **降温措施** | 额外 ABSD, 贷款收紧等 | MND/MAS |

    **接入方式**: 调用 IRAS/HDB/MAS 公开 API 或定期更新税率表与资格规则。
    """)

    # Example: BSD calculation for the user's budget midpoint
    mid_price = (filters["budget_min"] + filters["budget_max"]) / 2
    bsd_info = _policy_stub_bsd(mid_price)
    absd_info = _policy_stub_absd("SC", 0)
    hdb_info = _policy_stub_hdb_eligibility(8000, "SC+SC")

    st.caption(f"示例 (预算中位数 {fmt_price(mid_price)}): "
               f"BSD ≈ {fmt_price(bsd_info['bsd_amount'])} ({bsd_info['effective_rate']:.1f}%), "
               f"ABSD (SC 首套) = {absd_info['effective']}, "
               f"可能的补贴: {', '.join(hdb_info['grants_available'])}")
