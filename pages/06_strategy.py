"""Page 5: Strategy Validation — 购房策略与保值验证分析 (Rubric 9.1–9.3)."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2

from utils.helpers import TOWN_COORDS, fmt_price

# ---- Target towns (Northeast Region) ----
MRT_COORDS = {
    "PUNGGOL":  [(1.4052, 103.9022)],
    "SENGKANG": [(1.3924, 103.8951)],
    "HOUGANG":  [(1.3716, 103.8923)],
}
TOWN_CENTERS = {
    "PUNGGOL":  (1.4043, 103.9028),
    "SENGKANG": (1.3917, 103.8942),
    "HOUGANG":  (1.3714, 103.8923),
}


# ====================== HELPERS ======================

def _haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def _town_mrt_dist(town: str) -> float:
    tc = TOWN_CENTERS.get(town)
    mrt_list = MRT_COORDS.get(town)
    if tc is None or mrt_list is None:
        return 2.0
    return min(_haversine(tc[0], tc[1], mlat, mlng) for mlat, mlng in mrt_list)


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("🏆 策略验证")
    st.markdown("策略形成 (2020–2023) → 验证 (2024+) → 四维分项得分 — 数据驱动的购房决策。")

    # ---- Prepare data ----
    df = df.copy()
    df["mrt_dist_km"] = df["town"].apply(_town_mrt_dist)
    df = df.dropna(subset=["resale_price", "month", "mrt_dist_km"])

    # ---- 9.1 核心规则：严格时间切分 ----
    train = df[df["year"] <= 2023].copy()
    test = df[df["year"] >= 2024].copy()

    if len(train) < 200 or len(test) < 50:
        st.error(f"数据不足: 训练集 {len(train)} 条, 测试集 {len(test)} 条")
        return

    all_towns = sorted(df["town"].unique())
    all_types = sorted(df["flat_type"].unique())

    # ==================== 9.2.1 STRATEGY SETTINGS ====================

    st.header("📋 策略设定 (9.2.1)")

    st.markdown("""
    **策略形成期**：2020–2023 年成交数据用于形成策略依据
    **验证期**：2024 年至今成交数据用于检验策略保值能力
    **约束**：每组必须设置预算约束和户型约束，避免直接选择高价中心区
    """)

    c1, c2 = st.columns(2)
    with c1:
        strategy_type = st.radio(
            "策略类型",
            ["镇区偏离回归策略", "MRT 便利策略",
             "长剩余租约策略", "低总价入门策略"],
            key="strat_type",
        )
    with c2:
        budget_max = st.slider(
            "预算上限 (新元)", 300_000, 800_000, 550_000, 10_000, key="strat_budget",
        )
        sel_types = st.multiselect(
            "户型范围", all_types,
            default=["4-Room", "5-Room"], key="strat_types",
        )

    # ---- 选择规则 ----
    if strategy_type == "MRT 便利策略":
        max_mrt, min_lease, min_train_vol = 0.5, 65, 50
    elif strategy_type == "长剩余租约策略":
        max_mrt, min_lease, min_train_vol = 5.0, 70, 50
    elif strategy_type == "低总价入门策略":
        max_mrt, min_lease, min_train_vol = 5.0, 60, 80
    else:  # 镇区偏离回归策略
        max_mrt, min_lease, min_train_vol = 5.0, 65, 60

    default_towns = ["PUNGGOL", "SENGKANG", "HOUGANG"]
    sel_towns = st.multiselect(
        "候选镇区", all_towns,
        default=[t for t in default_towns if t in all_towns],
        key="strat_towns",
    )

    if not sel_types or not sel_towns:
        st.warning("请至少选择一个户型和一个镇区。")
        return

    # 策略设定表
    st.markdown(f"""
    | 设定项 | 值 |
    |--------|-----|
    | 策略类型 | **{strategy_type}** |
    | 预算上限 | {fmt_price(budget_max)} 新元 |
    | 户型范围 | {', '.join(sel_types)} |
    | 候选镇区 | {', '.join(sel_towns)} |
    | 选择规则 | 距 MRT < {max_mrt}km、剩余租约 ≥ {min_lease} 年、训练期成交量 ≥ {min_train_vol} 套/组 |
    """)

    # ---- Apply strategy filters to training data ----
    train_vol = train.groupby(["town", "flat_type"]).size().reset_index(name="train_vol")

    strat_train = train[
        train["flat_type"].isin(sel_types)
        & (train["resale_price"] <= budget_max)
        & train["town"].isin(sel_towns)
        & (train["remaining_lease"] >= min_lease)
        & (train["mrt_dist_km"] < max_mrt)
    ].copy()
    strat_train = strat_train.merge(train_vol, on=["town", "flat_type"], how="left")
    strat_train = strat_train[strat_train["train_vol"] >= min_train_vol]

    if len(strat_train) < 50:
        st.error(f"策略组训练数据不足 ({len(strat_train)} 条)，请放宽约束。")
        st.stop()

    # Apply SAME filters to test set
    valid_combos = strat_train[["town", "flat_type"]].drop_duplicates()
    strat_test = test[
        test["flat_type"].isin(sel_types)
        & (test["resale_price"] <= budget_max)
        & test["town"].isin(sel_towns)
        & (test["remaining_lease"] >= min_lease)
        & (test["mrt_dist_km"] < max_mrt)
    ].copy()
    strat_test = strat_test.merge(valid_combos, on=["town", "flat_type"], how="inner")

    if len(strat_test) < 20:
        st.error(f"策略组验证数据不足 ({len(strat_test)} 条)。")
        st.stop()

    # ---- Baseline group ----
    base_train = train[
        train["flat_type"].isin(sel_types)
        & train["town"].isin(sel_towns)
    ].copy()
    base_test = test[
        test["flat_type"].isin(sel_types)
        & test["town"].isin(sel_towns)
    ].copy()

    st.success(
        f"策略组: 训练 {len(strat_train):,} 套 → 验证 {len(strat_test):,} 套 | "
        f"基准组: 训练 {len(base_train):,} 套 → 验证 {len(base_test):,} 套"
    )

    # ==================== 9.2.2 VALIDATION PERFORMANCE ====================

    st.header("📊 验证期表现 (9.2.2)")

    def _compute_metrics(tr_df, te_df):
        tr_avg = tr_df["resale_price"].mean()
        te_avg = te_df["resale_price"].mean()

        te = te_df.copy()
        te["quarter"] = te["month"].dt.to_period("Q")
        q_prices = te.groupby("quarter")["resale_price"].mean().sort_index()

        # 总涨幅
        total_return = (q_prices.iloc[-1] / q_prices.iloc[0] - 1) if len(q_prices) >= 2 else 0.0

        # 年化涨幅
        te_years = te["year"].mean()
        tr_years = tr_df["year"].mean()
        yrs = max(te_years - tr_years, 0.5)
        cagr = (te_avg / tr_avg) ** (1 / yrs) - 1 if tr_avg > 0 else 0.0

        # 价格波动
        price_vol = float(q_prices.std() / q_prices.mean()) if q_prices.mean() > 0 else 0.0

        # 最大回撤
        cummax = q_prices.cummax()
        dd = (q_prices - cummax) / cummax
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        # 成交量稳定性
        q_counts = te.groupby("quarter").size()
        vol_cv = float(q_counts.std() / q_counts.mean()) if q_counts.mean() > 0 else 0.0

        # 预算适配性
        budget_fit = float((te["resale_price"] <= budget_max).mean() * 100)

        # 年度涨幅标准差
        yearly = te.groupby("year")["resale_price"].mean().sort_index()
        yoy_changes = yearly.pct_change().dropna()
        yoy_std = float(yoy_changes.std()) if len(yoy_changes) > 0 else 0.0

        return {
            "训练期均价": tr_avg, "验证期均价": te_avg,
            "总涨幅": total_return, "年化涨幅 (CAGR)": cagr,
            "价格波动 (CV)": price_vol, "年度涨幅标准差": yoy_std,
            "最大回撤": max_dd, "成交量 CV": vol_cv,
            "预算适配性": budget_fit,
            "训练样本": len(tr_df), "验证样本": len(te_df),
        }

    sm = _compute_metrics(strat_train, strat_test)
    bm = _compute_metrics(base_train, base_test)

    # ---- Metrics table ----
    metrics_table = pd.DataFrame([
        {"指标": "训练期均价", "策略组": fmt_price(sm["训练期均价"]),
         "基准组": fmt_price(bm["训练期均价"])},
        {"指标": "验证期均价", "策略组": fmt_price(sm["验证期均价"]),
         "基准组": fmt_price(bm["验证期均价"])},
        {"指标": "总涨幅", "策略组": f"{sm['总涨幅']*100:+.2f}%",
         "基准组": f"{bm['总涨幅']*100:+.2f}%"},
        {"指标": "年化涨幅 (CAGR)", "策略组": f"{sm['年化涨幅 (CAGR)']*100:+.2f}%",
         "基准组": f"{bm['年化涨幅 (CAGR)']*100:+.2f}%"},
        {"指标": "价格波动 (CV)", "策略组": f"{sm['价格波动 (CV)']*100:.2f}%",
         "基准组": f"{bm['价格波动 (CV)']*100:.2f}%"},
        {"指标": "年度涨幅标准差", "策略组": f"{sm['年度涨幅标准差']*100:.2f}%",
         "基准组": f"{bm['年度涨幅标准差']*100:.2f}%"},
        {"指标": "最大回撤", "策略组": f"{sm['最大回撤']*100:.2f}%",
         "基准组": f"{bm['最大回撤']*100:.2f}%"},
        {"指标": "成交量 CV", "策略组": f"{sm['成交量 CV']*100:.2f}%",
         "基准组": f"{bm['成交量 CV']*100:.2f}%"},
        {"指标": "预算适配性", "策略组": f"{sm['预算适配性']:.1f}%",
         "基准组": f"{bm['预算适配性']:.1f}%"},
        {"指标": "验证样本数", "策略组": str(sm["验证样本"]),
         "基准组": str(bm["验证样本"])},
    ])
    st.dataframe(metrics_table, width='stretch', hide_index=True)

    # ==================== 9.2.3 STRATEGY vs BASELINE ====================

    st.header("📈 策略组 vs 基准组 (9.2.3)")

    # Quarterly trend
    st.subheader("验证期季度均价走势")
    strat_q = strat_test.copy()
    strat_q["quarter"] = strat_q["month"].dt.to_period("Q")
    strat_q["group"] = "策略组"
    base_q = base_test.copy()
    base_q["quarter"] = base_q["month"].dt.to_period("Q")
    base_q["group"] = "基准组"

    all_q = pd.concat([strat_q, base_q])
    q_trend = all_q.groupby(["quarter", "group"])["resale_price"].mean().reset_index()
    q_trend["quarter_str"] = q_trend["quarter"].astype(str)

    fig_trend = px.line(
        q_trend, x="quarter_str", y="resale_price", color="group",
        markers=True,
        color_discrete_map={"策略组": "#FF6B6B", "基准组": "#94A3B8"},
        labels={"quarter_str": "", "resale_price": "均价 (新元)", "group": ""},
    )
    fig_trend.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_tickangle=45, hovermode="x unified")
    st.plotly_chart(fig_trend, width='stretch')

    # 关键指标差异
    st.subheader("关键指标差异 (策略组 − 基准组)")
    cagr_diff = sm["年化涨幅 (CAGR)"] - bm["年化涨幅 (CAGR)"]

    diff_data = pd.DataFrame([
        {"指标": "年化涨幅", "差异": cagr_diff * 100, "单位": "pp"},
        {"指标": "价格波动", "差异": (bm["价格波动 (CV)"] - sm["价格波动 (CV)"]) * 100, "单位": "pp"},
        {"指标": "最大回撤", "差异": (bm["最大回撤"] - sm["最大回撤"]) * 100, "单位": "pp"},
        {"指标": "成交量 CV", "差异": (bm["成交量 CV"] - sm["成交量 CV"]) * 100, "单位": "pp"},
        {"指标": "预算适配性", "差异": sm["预算适配性"] - bm["预算适配性"], "单位": "pp"},
    ])
    fig_diff = px.bar(
        diff_data, x="指标", y="差异", color="指标",
        text=diff_data.apply(lambda r: f"{r['差异']:+.1f}{r['单位']}", axis=1),
        title="正值 = 策略组更优",
    )
    fig_diff.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    fig_diff.update_traces(textposition="outside")
    st.plotly_chart(fig_diff, width='stretch')

    # ==================== 9.2.4 STRATEGY REFLECTION ====================

    st.header("📝 策略说明与反思 (9.2.4)")

    better = "优于" if cagr_diff > 0 else "略低于"
    stab_label = "更稳定" if sm["价格波动 (CV)"] < bm["价格波动 (CV)"] else "波动略高"
    strat_towns_str = ", ".join(sel_towns)

    st.write(f"""
### 策略反思：为什么选择「{strategy_type}」？

**一、策略选择理由与训练期数据支撑**

我选择 **{strategy_type}**，预算上限 **{fmt_price(budget_max)}** 新元，目标户型 **{', '.join(sel_types)}**，候选镇区 **{strat_towns_str}**。这一策略在训练期（2020–2023）提供了以下数据支撑：

1. **价格定位**：策略组训练期均价为 {fmt_price(sm['训练期均价'])}，{'低于' if sm['训练期均价'] < bm['训练期均价'] else '接近'}基准组（{fmt_price(bm['训练期均价'])}）。{'策略有效筛选出了价格合理的优质房源——在预算内买到合理定价的房产，而非简单的"买便宜货"。' if sm['训练期均价'] < bm['训练期均价'] else '策略组的约束条件并未显著拉低均价，说明约束聚焦于质量筛选而非单纯压价。'}

2. **成交量保障**：训练期策略组有 **{sm['训练样本']}** 套成交记录（基准组 {bm['训练样本']} 套），说明该细分市场流动性充足，不存在"有价无市"的困境。足够的样本量也确保了价格统计的统计意义。

3. **租约与 MRT 筛选**：通过剩余租约 ≥ {min_lease} 年、距 MRT < {max_mrt}km 的选择规则，策略从源头上剔除了折旧风险高和交通不便的房源，提升了资产基本面的质量。

**二、策略组相对基准组的优势与不足**

进入验证期（2024 至今）：

- **增值维度**：策略组年化涨幅为 **{sm['年化涨幅 (CAGR)']*100:+.2f}%**，{better} 基准组的 **{bm['年化涨幅 (CAGR)']*100:+.2f}%**。{'策略组在涨幅上具有优势，说明约束条件筛选出的房源确实享有更高的市场认可。' if cagr_diff > 0 else '策略组涨幅略低，但这可能是因为策略放弃了部分高波动高涨幅的组合，换取更稳定的回报。'}
- **稳定性维度**：策略组价格波动 **{sm['价格波动 (CV)']*100:.2f}%**，{stab_label}（基准组 {bm['价格波动 (CV)']*100:.2f}%）。最大回撤 **{sm['最大回撤']*100:.2f}%** {'优于' if sm['最大回撤'] < bm['最大回撤'] else '高于'} 基准组的 **{bm['最大回撤']*100:.2f}%**，{'策略组的抗跌能力更强。' if sm['最大回撤'] < bm['最大回撤'] else '需关注极端行情下的尾部风险。'}
- **流动性维度**：成交量 CV 为 **{sm['成交量 CV']*100:.2f}%**，{'优于' if sm['成交量 CV'] < bm['成交量 CV'] else '高于'} 基准组（{bm['成交量 CV']*100:.2f}%），{'验证期内交易节奏稳定，转售便利。' if sm['成交量 CV'] < bm['成交量 CV'] else '成交波动较大，该细分市场可能受宏观环境影响更敏感。'}
- **预算适配性**：**{sm['预算适配性']:.1f}%** 的成交在预算范围内，{'预算约束切实可行。' if sm['预算适配性'] > 50 else '预算偏紧，建议适当放宽。'}

**三、2024 年以来的市场变化及影响**

2024 年以来，新加坡 HDB 转售市场经历了显著变化：(1) BTO 供应逐步恢复，部分缓解了转售市场需求压力；(2) 美联储开启降息周期，新加坡利率（SORA）随之下行，降低了购房贷款成本；(3) 政府继续实施降温措施，包括提高 ABSD 税率和收紧 LTV 比率。{('镇区偏离回归策略的均值回归逻辑依然成立——基础设施建设持续推进，价格偏低的镇区有望向均值收敛。' if '偏离' in strategy_type else '') + ('MRT 沿线房产的溢价在降息环境中可能进一步扩大——低利率鼓励购房者加杠杆，愿意为便利的交通支付更高溢价。' if 'MRT' in strategy_type else '') + ('长租约房产在利率下行周期中优势相对减弱——买家融资成本降低，对租约长度的敏感度下降，可能更关注地段和户型。' if '租约' in strategy_type else '') + ('低总价入门策略受惠于降息——首次购房者月供压力降低，门槛更易负担，可能刺激这一细分市场需求。' if '入门' in strategy_type else '')}

**四、验证期表现总结与改进方向**

综合来看，该策略在验证期 **{'表现良好' if cagr_diff > -0.01 and sm['最大回撤'] < bm['最大回撤'] else '有优化空间'}**。如果重新制定策略，我会做以下调整：

1. **细化镇区选择**：不按镇区名称粗筛，而是按具体镇区的基础设施规划（如 Cross Island Line 站点）、价格偏离度和 CAGR 趋势进行细化筛选；
2. **加入时序动量因子**：关注训练期内涨幅已有加速趋势的 (镇区, 房型) 组合，利用动量效应提升验证期收益；
3. **动态预算约束**：将预算与利率挂钩——利率下降时适当提高预算，利率上升时收紧，以反映真实的购房能力变化；
4. **微观地段变量**：如能获取 block 级坐标，应以 block 到 MRT 的实际距离替代镇区级近似距离，显著提升策略精准度。
    """)
