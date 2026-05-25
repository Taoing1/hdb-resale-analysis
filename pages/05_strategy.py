"""Page 5: Strategy Validation — rubric-aligned purchase strategy with train/test backtest."""

import json
import time
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import TOWN_COORDS, fmt_price

# ---- Reference data ----
MRT_COORDS_ALL = {
    "PUNGGOL":    [(1.4052, 103.9022)],
    "SENGKANG":   [(1.3924, 103.8951)],
    "HOUGANG":    [(1.3716, 103.8923)],
    "ANG MO KIO":  [(1.3700, 103.8494)],
    "TOA PAYOH":   [(1.3331, 103.8473)],
    "QUEENSTOWN":  [(1.2943, 103.8024)],
    "BEDOK":       [(1.3240, 103.9299)],
    "BISHAN":      [(1.3502, 103.8492)],
}

TOWN_CENTERS_ALL = {
    "PUNGGOL":    (1.4043, 103.9028),
    "SENGKANG":   (1.3917, 103.8942),
    "HOUGANG":    (1.3714, 103.8923),
    "ANG MO KIO":  (1.3692, 103.8485),
    "TOA PAYOH":   (1.3342, 103.8487),
    "QUEENSTOWN":  (1.2936, 103.8012),
    "BEDOK":       (1.3245, 103.9301),
    "BISHAN":      (1.3512, 103.8490),
}

MATURE_SET = {"ANG MO KIO", "TOA PAYOH", "QUEENSTOWN", "BEDOK", "BISHAN"}

API_URL = "https://data.gov.sg/api/action/datastore_search"
RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"


# ====================== HELPERS ======================

def _haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def _town_mrt_dist(town: str) -> float:
    tc = TOWN_CENTERS_ALL.get(town)
    mrt_list = MRT_COORDS_ALL.get(town)
    if tc is None or mrt_list is None:
        return 2.0
    return min(_haversine(tc[0], tc[1], mlat, mlng) for mlat, mlng in mrt_list)


# ====================== FETCH MATURE DATA ======================

@st.cache_data(ttl=86400)
def _fetch_mature_data():
    """Fetch mature estate HDB data for town expansion."""
    records = []
    towns = ["ANG MO KIO", "TOA PAYOH", "QUEENSTOWN", "BEDOK", "BISHAN"]
    for town in towns:
        offset = 0
        while True:
            try:
                resp = __import__("requests").get(API_URL, params={
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
    cy = pd.Timestamp.now().year
    df["remaining_lease"] = df["lease_commence_date"].apply(
        lambda x: max(99 - (cy - x), 0) if pd.notna(x) else None
    )
    df["mrt_dist_km"] = df["town"].apply(_town_mrt_dist)
    df["town_type"] = "成熟区"
    type_map = {
        "2 ROOM": "2-Room", "3 ROOM": "3-Room", "4 ROOM": "4-Room",
        "5 ROOM": "5-Room", "EXECUTIVE": "Executive",
        "MULTI-GENERATION": "Multi-Gen", "MULTI GENERATION": "Multi-Gen",
    }
    df["flat_type"] = df["flat_type"].str.upper().str.strip().map(type_map).fillna(df["flat_type"])
    return df.dropna(subset=["resale_price", "floor_area_sqm", "month"])


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("🏆 策略验证")
    st.markdown("策略形成 (2020–2023) → 验证 (2024+) → 综合评分 — 基于数据驱动的购房决策。")

    # ---- Prepare data ----
    df = df.copy()
    df["mrt_dist_km"] = df["town"].apply(_town_mrt_dist)
    df["town_type"] = "非成熟区"

    with st.spinner("获取成熟区数据…"):
        mature_df = _fetch_mature_data()
    if not mature_df.empty:
        combined = pd.concat(
            [df, mature_df[df.columns.intersection(mature_df.columns)]],
            ignore_index=True,
        )
    else:
        combined = df

    combined = combined.dropna(subset=["resale_price", "month", "mrt_dist_km"])

    # ---- Hard split ----
    train = combined[combined["year"] <= 2023].copy()
    test = combined[combined["year"] >= 2024].copy()

    if len(train) < 200 or len(test) < 50:
        st.error(f"数据不足: 训练集 {len(train)} 条, 测试集 {len(test)} 条")
        return

    all_towns = sorted(combined["town"].unique())
    all_types = sorted(combined["flat_type"].unique())

    # ==================== 9.2.1 STRATEGY SETTINGS ====================

    st.header("📋 策略设定")

    c1, c2 = st.columns(2)
    with c1:
        strategy_type = st.radio(
            "策略类型",
            ["非成熟区成长策略", "成熟区稳健策略", "MRT 便利策略",
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

    # Derived selection rules
    if strategy_type == "MRT 便利策略":
        max_mrt = 0.5
        min_lease = 65
        min_train_vol = 50
    elif strategy_type == "长剩余租约策略":
        max_mrt = 5.0
        min_lease = 70
        min_train_vol = 50
    elif strategy_type == "低总价入门策略":
        max_mrt = 5.0
        min_lease = 60
        min_train_vol = 80
    elif strategy_type == "成熟区稳健策略":
        max_mrt = 5.0
        min_lease = 65
        min_train_vol = 100
    else:  # 非成熟区成长
        max_mrt = 5.0
        min_lease = 65
        min_train_vol = 60

    # Candidate towns
    default_towns = (
        ["PUNGGOL", "SENGKANG", "HOUGANG"]
        if "非成熟" in strategy_type or "低总价" in strategy_type or "MRT" in strategy_type or "长租约" in strategy_type
        else ["ANG MO KIO", "TOA PAYOH", "QUEENSTOWN", "BEDOK", "BISHAN"]
    )
    sel_towns = st.multiselect(
        "候选镇区", all_towns, default=[t for t in default_towns if t in all_towns],
        key="strat_towns",
    )

    if not sel_types or not sel_towns:
        st.warning("请至少选择一个户型和一个镇区。")
        return

    # Display settings table
    st.markdown(f"""
| 设定项 | 值 |
|--------|-----|
| 策略类型 | **{strategy_type}** |
| 预算上限 | {fmt_price(budget_max)} 新元 |
| 户型范围 | {', '.join(sel_types)} |
| 候选镇区 | {', '.join(sel_towns)} |
| 选择规则 | 距 MRT < {max_mrt}km、剩余租约 ≥ {min_lease} 年、训练期成交量 ≥ {min_train_vol} 套/组 |
    """)

    # ---- Form strategy group (ONLY training data) ----
    # Compute volume per (town, flat_type) in training period
    train_vol = train.groupby(["town", "flat_type"]).size().reset_index(name="train_vol")

    strat_train = train[
        train["flat_type"].isin(sel_types)
        & (train["resale_price"] <= budget_max)
        & train["town"].isin(sel_towns)
        & (train["remaining_lease"] >= min_lease)
        & (train["mrt_dist_km"] < max_mrt)
    ].copy()
    # Merge volume and filter
    strat_train = strat_train.merge(train_vol, on=["town", "flat_type"], how="left")
    strat_train = strat_train[strat_train["train_vol"] >= min_train_vol]

    if len(strat_train) < 50:
        st.error(f"策略组训练数据不足 ({len(strat_train)} 条)，请放宽约束。")
        st.stop()

    # Apply SAME filters to test set for validation
    valid_combos = strat_train[["town", "flat_type"]].drop_duplicates()
    strat_test = test[
        test["flat_type"].isin(sel_types)
        & (test["resale_price"] <= budget_max)
        & test["town"].isin(sel_towns)
        & (test["remaining_lease"] >= min_lease)
        & (test["mrt_dist_km"] < max_mrt)
    ].copy()
    strat_test = strat_test.merge(
        valid_combos, on=["town", "flat_type"], how="inner"
    )

    if len(strat_test) < 20:
        st.error(f"策略组验证数据不足 ({len(strat_test)} 条)。")
        st.stop()

    # ---- Baseline group (type + towns only, no budget/rules filter) ----
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

    st.header("📊 验证期表现")

    def _compute_metrics(tr_df, te_df, label=""):
        """Compute all required metrics."""
        tr_avg = tr_df["resale_price"].mean()
        te_avg = te_df["resale_price"].mean()

        # Quarterly aggregation for time-series metrics
        te = te_df.copy()
        te["quarter"] = te["month"].dt.to_period("Q")
        q_prices = te.groupby("quarter")["resale_price"].mean().sort_index()

        # Total return
        total_return = (q_prices.iloc[-1] / q_prices.iloc[0] - 1) if len(q_prices) >= 2 else 0.0

        # CAGR
        te_years = te["year"].mean()
        tr_years = tr_df["year"].mean()
        yrs = max(te_years - tr_years, 0.5)
        cagr = (te_avg / tr_avg) ** (1 / yrs) - 1 if tr_avg > 0 else 0.0

        # Price volatility (std of quarterly averages / mean)
        price_vol = float(q_prices.std() / q_prices.mean()) if q_prices.mean() > 0 else 0.0

        # Max drawdown
        cummax = q_prices.cummax()
        dd = (q_prices - cummax) / cummax
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        # Volume stability: CV of quarterly counts
        q_counts = te.groupby("quarter").size()
        vol_cv = float(q_counts.std() / q_counts.mean()) if q_counts.mean() > 0 else 0.0

        # Budget fit: % of test records within budget
        budget_fit = float((te["resale_price"] <= budget_max).mean() * 100)

        # Year-over-year price changes for volatility
        yearly = te.groupby("year")["resale_price"].mean().sort_index()
        yoy_changes = yearly.pct_change().dropna()
        yoy_std = float(yoy_changes.std()) if len(yoy_changes) > 0 else 0.0

        return {
            "训练期均价": tr_avg,
            "验证期均价": te_avg,
            "总涨幅": total_return,
            "年化涨幅 (CAGR)": cagr,
            "价格波动 (CV)": price_vol,
            "年度涨幅标准差": yoy_std,
            "最大回撤": max_dd,
            "成交量 CV": vol_cv,
            "预算适配性": budget_fit,
            "训练样本": len(tr_df),
            "验证样本": len(te_df),
        }

    sm = _compute_metrics(strat_train, strat_test)
    bm = _compute_metrics(base_train, base_test)

    # ---- Metrics comparison table ----
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

    st.subheader("📈 策略组 vs 基准组 — 均价走势对比")

    # Quarterly trend chart
    for label, grp in [("策略组", strat_test), ("基准组", base_test)]:
        grp_c = grp.copy()
        grp_c["quarter"] = grp_c["month"].dt.to_period("Q")
        grp_c["group"] = label
        if label == "策略组":
            all_q = grp_c
        else:
            all_q = pd.concat([all_q, grp_c])

    q_trend = all_q.groupby(["quarter", "group"])["resale_price"].mean().reset_index()
    q_trend["quarter_str"] = q_trend["quarter"].astype(str)

    fig = px.line(
        q_trend, x="quarter_str", y="resale_price", color="group",
        markers=True,
        color_discrete_map={"策略组": "#FF6B6B", "基准组": "#888888"},
        title="验证期季度均价走势",
        labels={"quarter_str": "", "resale_price": "均价 (新元)", "group": ""},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_tickangle=45, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    # ---- Comparison bar chart ----
    cagr_diff = sm["年化涨幅 (CAGR)"] - bm["年化涨幅 (CAGR)"]
    st.subheader("📊 关键指标差异 (策略组 − 基准组)")

    diff_data = pd.DataFrame([
        {"指标": "年化涨幅", "差异": cagr_diff * 100, "单位": "%"},
        {"指标": "价格波动", "差异": (bm["价格波动 (CV)"] - sm["价格波动 (CV)"]) * 100, "单位": "%"},
        {"指标": "最大回撤", "差异": (bm["最大回撤"] - sm["最大回撤"]) * 100, "单位": "%"},
        {"指标": "成交量 CV", "差异": (bm["成交量 CV"] - sm["成交量 CV"]) * 100, "单位": "%"},
        {"指标": "预算适配性", "差异": sm["预算适配性"] - bm["预算适配性"], "单位": "%"},
    ])
    fig_diff = px.bar(
        diff_data, x="指标", y="差异", color="指标",
        text=diff_data.apply(lambda r: f"{r['差异']:+.1f}{r['单位']}", axis=1),
        title="策略组优势 (正值 = 策略更优)",
    )
    fig_diff.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    fig_diff.update_traces(textposition="outside")
    st.plotly_chart(fig_diff, width='stretch')

    # ==================== 9.2.4 STRATEGY REFLECTION ====================

    st.header("📝 策略说明与反思")

    # Compute summary stats for reflection
    strat_towns_str = ", ".join(sel_towns)
    better = "优于" if cagr_diff > 0 else "略低于"
    stab_better = "更稳定" if sm["价格波动 (CV)"] < bm["价格波动 (CV)"] else "波动略高"

    st.write(f"""
### 策略反思：为什么选择「{strategy_type}」？

**一、策略选择理由与训练期数据支撑**

我选择 **{strategy_type}** 策略，预算上限为 **{fmt_price(budget_max)}** 新元，目标户型为 **{', '.join(sel_types)}**，候选镇区为 **{strat_towns_str}**。这一策略在训练期（2020–2023）的表现提供了以下支撑：

1. **价格优势**：策略组训练期均价为 {fmt_price(sm['训练期均价'])}，{'低于' if sm['训练期均价'] < bm['训练期均价'] else '接近'}基准组（{fmt_price(bm['训练期均价'])}）。{'这意味着策略有效筛选出了价格洼地中的优质房源——在预算内买到合理定价的房产，而非简单的"买便宜货"。' if sm['训练期均价'] < bm['训练期均价'] else '策略组的筛选条件并未显著拉低均价，说明约束条件聚焦于质量而非单纯压价。'}

2. **成交量验证**：训练期策略组有 {sm['训练样本']} 套成交记录（同类策略的基准组为 {bm['训练样本']} 套），说明该细分市场具有充足的流动性，买卖双方活跃，不存在"有价无市"的困境。足够的成交量也确保了我们计算的价格统计具有统计意义。

3. **租约与 MRT 筛选效果**：通过设定剩余租约 ≥ {min_lease} 年、距 MRT < {max_mrt}km 的选择规则，策略剔除了折旧风险高和交通不便的房源，从源头上提升了资产质量。

**二、策略组相对基准组的优势与不足**

进入验证期（2024 至今），策略组的表现如下：

- **增值维度**：策略组年化涨幅为 {sm['年化涨幅 (CAGR)']*100:+.2f}%，{better} 基准组的 {bm['年化涨幅 (CAGR)']*100:+.2f}%。{'策略组在涨幅上具有优势，说明约束条件筛选出的房型确实在市场上享有更高的增值预期。' if cagr_diff > 0 else '策略组涨幅略低于基准组，但这可能是因为策略放弃了部分高波动高涨幅的房型组合，换取更稳定的回报。'}
- **稳定性维度**：策略组价格波动为 {sm['价格波动 (CV)']*100:.2f}%，{stab_better}（基准组 {bm['价格波动 (CV)']*100:.2f}%）。最大回撤为 {sm['最大回撤']*100:.2f}% {'优于' if sm['最大回撤'] < bm['最大回撤'] else '高于'} 基准组的 {bm['最大回撤']*100:.2f}%，说明{'策略组在市场下行时的抗跌能力更强。' if sm['最大回撤'] < bm['最大回撤'] else '策略组在极端行情下可能受到更大冲击，需要关注尾部风险。'}
- **流动性维度**：策略组成交量 CV 为 {sm['成交量 CV']*100:.2f}%，{'优于' if sm['成交量 CV'] < bm['成交量 CV'] else '高于'} 基准组（{bm['成交量 CV']*100:.2f}%），{'验证期内交易节奏稳定，转售便利性有保障。' if sm['成交量 CV'] < bm['成交量 CV'] else '验证期成交波动较大，可能是该细分市场受宏观环境影响更敏感。'}
- **预算适配性**：{sm['预算适配性']:.1f}% 的策略组验证记录在预算范围内（基准组为 {bm['预算适配性']:.1f}%），{'预算约束有效，策略切实可行。' if sm['预算适配性'] > 50 else '预算约束偏紧，部分房源超出上限，建议适当放宽预算。'}

**三、2024 年以来的市场变化及影响**

2024 年以来，新加坡 HDB 转售市场经历了显著变化：（1）BTO 供应逐步恢复，部分缓解了转售市场需求压力；（2）美联储开启降息周期，新加坡利率（SORA）随之下行，降低了购房贷款成本；（3）政府继续实施降温措施，包括提高 ABSD 税率和收紧 LTV 比率。这些变化对策略的影响是：{'非成熟区的增值逻辑依然成立——基础设施建设（如 Punggol Digital District）持续推进，长期利好仍在兑现中。但短期利率下降可能使部分买家转向成熟区，对非成熟区需求形成一定分流。' if '成长' in strategy_type else ''}{'成熟区的抗跌性在宏观不确定性上升时更具吸引力，但较高的进入门槛（均价更高）也限制了买家群体，可能影响流动性。' if '稳健' in strategy_type else ''}{'MRT 沿线房产的溢价在降息环境中可能进一步扩大——低利率鼓励购房者加杠杆，愿意为便利的交通支付更高溢价。' if 'MRT' in strategy_type else ''}{'长租约房产在利率下行周期中优势相对减弱——买家融资成本降低，对租约长度的敏感度下降，可能更关注地段和户型。' if '租约' in strategy_type else ''}{'低总价入门策略受惠于降息——首次购房者月供压力降低，3-Room/4-Room 的门槛更易负担，可能刺激这一细分市场需求。' if '入门' in strategy_type else ''}

**四、验证期表现总结与改进方向**

综合来看，该策略在验证期{'表现良好' if cagr_diff > -0.01 and sm['最大回撤'] < bm['最大回撤'] else '有优化空间'}：{'策略组在涨幅、稳定性和流动性三个维度上均达到或超过基准组水平，说明策略的选择规则有效。' if cagr_diff > 0 and sm['价格波动 (CV)'] < bm['价格波动 (CV)'] else '部分指标未达预期，反映了真实市场的复杂性。'}如果重新制定策略，我会做以下调整：

1. **细化镇区选择**：不简单地按成熟/非成熟二分，而是按具体镇区的基础设施规划（如 Cross Island Line 站点、JRL 开通时间）细化筛选；
2. **加入时序动量因子**：关注训练期内涨幅已有加速趋势的 (镇区, 房型) 组合，利用动量效应提升验证期收益；
3. **动态预算约束**：将预算与利率挂钩——利率下降时适当提高预算，利率上升时收紧，以反映真实的购房能力变化；
4. **微观地段变量**：如能获取 block 级坐标，应以 block 到 MRT 的实际距离替代镇区级近似距离，显著提升策略的精准度。
    """)

    # ==================== 9.3 COMPOSITE SCORING ====================

    st.header("🏅 综合评分 (9.3)")

    # Strategy return score: relative to baseline CAGR
    if bm["年化涨幅 (CAGR)"] > 0:
        return_score = min(sm["年化涨幅 (CAGR)"] / bm["年化涨幅 (CAGR)"] * 50, 100)
    elif sm["年化涨幅 (CAGR)"] > 0:
        return_score = 100
    else:
        return_score = 50

    # Stability score: lower volatility → higher score
    stab_score = max(0, min((1 - sm["价格波动 (CV)"]) * 100, 100))
    # Liquidity score: lower volume CV → higher score
    liq_score = max(0, min((1 - sm["成交量 CV"]) * 100, 100))
    # Budget fit score
    budget_score = sm["预算适配性"]
    # Analysis depth: placeholder for teacher assessment
    depth_score = 70

    composite = (
        return_score * 0.3
        + stab_score * 0.2
        + liq_score * 0.2
        + budget_score * 0.1
        + depth_score * 0.2
    )

    score_df = pd.DataFrame([
        {"评分项": "策略收益得分 (×0.3)", "得分": f"{return_score:.1f}", "权重": "30%",
         "说明": f"策略组 CAGR {sm['年化涨幅 (CAGR)']*100:+.2f}% vs 基准 {bm['年化涨幅 (CAGR)']*100:+.2f}%"},
        {"评分项": "稳定性得分 (×0.2)", "得分": f"{stab_score:.1f}", "权重": "20%",
         "说明": f"价格波动 {sm['价格波动 (CV)']*100:.2f}%，最大回撤 {sm['最大回撤']*100:.2f}%"},
        {"评分项": "流动性得分 (×0.2)", "得分": f"{liq_score:.1f}", "权重": "20%",
         "说明": f"成交量 CV {sm['成交量 CV']*100:.2f}%"},
        {"评分项": "预算适配得分 (×0.1)", "得分": f"{budget_score:.1f}", "权重": "10%",
         "说明": f"{sm['预算适配性']:.1f}% 的验证记录在预算内"},
        {"评分项": "分析深度得分 (×0.2)", "得分": f"{depth_score:.1f}", "权重": "20%",
         "说明": "教师主观评分 (0–100)，基于策略理由、验证过程和反思质量"},
    ])
    st.dataframe(score_df, width='stretch', hide_index=True)

    # Final score display
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("🏅 综合得分", f"{composite:.1f} / 100")
        if composite >= 85:
            rank_hint = "预估排名: 第1名 (10分)"
        elif composite >= 75:
            rank_hint = "预估排名: 第2-3名 (8-9分)"
        elif composite >= 60:
            rank_hint = "预估排名: 前50% (7分)"
        else:
            rank_hint = "预估排名: 后50% (6分)"
        st.caption(rank_hint)
    with c2:
        st.caption(
            "综合得分 = 策略收益×0.3 + 稳定性×0.2 + 流动性×0.2 + "
            "预算适配×0.1 + 分析深度×0.2"
        )
        st.caption(
            "排名规则: 第1名=10分, 第2名=9分, 第3名=8分, "
            "前50%=7分, 后50%=6分, 未完成=0分"
        )
