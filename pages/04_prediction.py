"""Page 4: Price Prediction — time-split, feature engineering, per-segment evaluation, interactive estimator."""

from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from utils.helpers import TOWN_COORDS, fmt_price

# ---- Hardcoded reference data ----
MRT_COORDS = {
    "PUNGGOL":  [(1.4052, 103.9022)],
    "SENGKANG": [(1.3924, 103.8951)],
    "HOUGANG":  [(1.3716, 103.8923)],
}

SCHOOLS_BY_TOWN = {
    "PUNGGOL":  ["Punggol Green Primary", "Mee Toh School", "Punggol Secondary"],
    "SENGKANG": ["Sengkang Green Primary", "Sengkang Secondary", "Nan Chiau High"],
    "HOUGANG":  ["Hougang Primary", "Hougang Secondary", "Xinmin Secondary"],
}

FLAT_TYPE_ORDINAL = {
    "2-Room": 1, "3-Room": 2, "4-Room": 3,
    "5-Room": 4, "Executive": 5, "Multi-Gen": 6,
}


# ====================== HELPERS ======================

def _haversine(lat1, lng1, lat2, lng2):
    """Return distance in km between two lat/lng pairs."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _nearest_mrt_dist(town: str) -> float:
    """Approximate distance from town centre to nearest MRT station (km)."""
    if town not in TOWN_COORDS:
        return 2.0
    t_lat, t_lng = TOWN_COORDS[town]
    mrt_list = MRT_COORDS.get(town, [(t_lat + 0.01, t_lng + 0.01)])
    return min(_haversine(t_lat, t_lng, mlat, mlng) for mlat, mlng in mrt_list)


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("🤖 房价预测模型")
    st.markdown("特征工程 → 时间切分训练 → 分房型评估 → 交互式预估。")

    # ---- 4.1 Feature Engineering ----
    df = df.copy()
    current_year = pd.Timestamp.now().year
    df["floor_age"] = df["lease_commence_date"].apply(
        lambda x: current_year - x if pd.notna(x) else None
    )
    df["flat_type_ordinal"] = df["flat_type"].map(FLAT_TYPE_ORDINAL).fillna(0).astype(int)
    df["mrt_dist_km"] = df["town"].apply(_nearest_mrt_dist)
    df["school_count"] = df["town"].apply(lambda t: len(SCHOOLS_BY_TOWN.get(t, [])))
    # Punggol / Sengkang / Hougang are all Northeast region towns

    feature_cols = [
        "floor_area_sqm", "remaining_lease", "floor_age", "storey_mid",
        "flat_type_ordinal", "mrt_dist_km", "school_count",
        "town", "year",
    ]
    target_col = "price_per_sqm"

    model_df = df.dropna(subset=feature_cols + [target_col]).copy()

    with st.expander("📋 特征工程概览", expanded=False):
        st.markdown(f"""
        | 特征 | 处理方式 | 说明 |
        |------|---------|------|
        | 面积 | StandardScaler | `floor_area_sqm` |
        | 剩余租约年限 | StandardScaler | `remaining_lease` 数值 |
        | 房龄 | StandardScaler | {current_year} − `lease_commence_date` |
        | 楼层 | StandardScaler | `storey_mid` 中位值 |
        | 房型 | StandardScaler (序数编码) | 2-Room=1, 3-Room=2, 4-Room=3, 5-Room=4, Executive=5, Multi-Gen=6 |
        | MRT 特征 | StandardScaler | 镇区中心到最近 MRT 站距离 (km) |
        | 学校特征 | StandardScaler | 镇区内学校数量 |
                | 镇区 | OneHotEncoder | Punggol / Sengkang / Hougang |
        | 成交年份 | StandardScaler | 从 `month` 提取 |
        """)
        st.caption(f"预测目标: **单价 (新元/sqm)** | 有效样本: **{len(model_df):,}** 条")

    # ---- 4.2 Train / Test Split (time-based) ----
    X = model_df[feature_cols + ["flat_type"]]
    y = model_df[target_col]

    X_train, X_test = X[X["year"] <= 2023].copy(), X[X["year"] >= 2024].copy()
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

    if len(X_train) < 100 or len(X_test) < 30:
        st.error(f"数据不足: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")
        return

    st.markdown(f"**时间切分**: 训练集 (2020–2023) **{len(X_train):,}** 条 | 测试集 (2024+) **{len(X_test):,}** 条")

    # ---- Preprocessor ----
    numeric_cols = [
        "floor_area_sqm", "remaining_lease", "floor_age", "storey_mid",
        "flat_type_ordinal", "mrt_dist_km", "school_count", "year",
    ]
    categorical_cols = ["town"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ])

    # ---- Sidebar: Model Selection + Hyperparameters ----
    st.sidebar.subheader("🤖 模型设置")

    use_lr = st.sidebar.checkbox("Linear Regression", value=False, key="use_lr")
    use_ridge = st.sidebar.checkbox("Ridge Regression", value=True, key="use_ridge")
    use_rf = st.sidebar.checkbox("Random Forest", value=True, key="use_rf")
    use_gb = st.sidebar.checkbox("Gradient Boosting", value=False, key="use_gb")

    params = {}
    if use_ridge:
        params["ridge_alpha"] = st.sidebar.slider("Ridge alpha", 0.01, 10.0, 1.0, 0.01, key="ridge_alpha")
    if use_rf:
        params["rf_n"] = st.sidebar.slider("RF n_estimators", 50, 500, 250, 50, key="rf_n")
        params["rf_depth"] = st.sidebar.slider("RF max_depth", 5, 30, 18, key="rf_depth")
    if use_gb:
        params["gb_n"] = st.sidebar.slider("GB n_estimators", 50, 500, 200, 50, key="gb_n")
        params["gb_lr"] = st.sidebar.slider("GB learning_rate", 0.01, 0.50, 0.10, 0.01, key="gb_lr")
        params["gb_depth"] = st.sidebar.slider("GB max_depth", 3, 15, 6, key="gb_depth")

    # ---- Fit preprocessor ----
    prep = preprocessor.fit(X_train)

    def _train_models():
        models = {}
        X_tr = prep.transform(X_train)
        X_te = prep.transform(X_test)

        if use_lr:
            m = LinearRegression()
            m.fit(X_tr, y_train)
            models["Linear Regression"] = {"pipeline": m, "pred": m.predict(X_te)}

        if use_ridge:
            m = Ridge(alpha=params.get("ridge_alpha", 1.0))
            m.fit(X_tr, y_train)
            models["Ridge Regression"] = {"pipeline": m, "pred": m.predict(X_te)}

        if use_rf:
            m = RandomForestRegressor(
                n_estimators=params.get("rf_n", 250),
                max_depth=params.get("rf_depth", 18),
                min_samples_leaf=5, random_state=42, n_jobs=-1,
            )
            m.fit(X_tr, y_train)
            models["Random Forest"] = {"pipeline": m, "pred": m.predict(X_te)}

        if use_gb:
            m = GradientBoostingRegressor(
                n_estimators=params.get("gb_n", 200),
                learning_rate=params.get("gb_lr", 0.1),
                max_depth=params.get("gb_depth", 6),
                min_samples_leaf=5, random_state=42,
            )
            m.fit(X_tr, y_train)
            models["Gradient Boosting"] = {"pipeline": m, "pred": m.predict(X_te)}

        return models

    if not any([use_lr, use_ridge, use_rf, use_gb]):
        st.warning("请在侧边栏至少选择一个模型。")
        return

    with st.spinner("训练模型中…"):
        models = _train_models()

    # ==================== 4.3 PER-SEGMENT EVALUATION ====================

    st.header("📊 分房型预测对比")

    segments = {
        "小户型 (≤3-Room)":     X_test["flat_type_ordinal"] <= 2,
        "中户型 (4-Room)":       X_test["flat_type_ordinal"] == 3,
        "大户型 (≥5-Room/Exec)": X_test["flat_type_ordinal"] >= 4,
        "老旧组屋 (租约<60年)":   X_test["remaining_lease"] < 60,
        "新近组屋 (租约≥80年)":   X_test["remaining_lease"] >= 80,
        "MRT沿线 (<500m)":       X_test["mrt_dist_km"] < 0.5,
    }

    all_seg_rows = []
    for seg_name, seg_mask in segments.items():
        n_seg = seg_mask.sum()
        if n_seg < 5:
            continue
        y_seg = y_test[seg_mask]
        for model_name, m in models.items():
            pred_seg = m["pred"][seg_mask.values]
            mae = float(np.mean(np.abs(y_seg - pred_seg)))
            mape_val = _mape(y_seg.values, pred_seg)
            r2_val = float(1 - np.sum((y_seg - pred_seg) ** 2) / np.sum((y_seg - y_seg.mean()) ** 2)) if y_seg.std() > 0 else 0
            all_seg_rows.append({
                "房源类型": seg_name, "模型": model_name,
                "样本数": n_seg, "MAE": mae, "MAPE(%)": mape_val, "R²": r2_val,
            })

    if all_seg_rows:
        seg_df = pd.DataFrame(all_seg_rows)
        st.dataframe(
            seg_df.style.format({
                "MAE": "S${:,.0f}", "MAPE(%)": "{:.1f}%", "R²": "{:.4f}",
            }),
            width='stretch', hide_index=True,
        )

        # Highlight large-error segments
        high_err = seg_df[seg_df["MAPE(%)"] > seg_df["MAPE(%)"].quantile(0.75)]
        if len(high_err) > 0:
            st.caption(
                "误差较大的类别可能因该类房型定价更复杂（如老旧组屋受翻新状态影响）"
                "或样本量偏小导致模型泛化不足。"
            )

    # ==================== 4.4 OVERALL EVALUATION ====================

    st.header("📐 整体评估")

    # ---- Metrics cards ----
    cols = st.columns(len(models))
    for i, (name, m) in enumerate(models.items()):
        mae = float(np.mean(np.abs(y_test - m["pred"])))
        rmse = float(np.sqrt(np.mean((y_test - m["pred"]) ** 2)))
        r2 = float(1 - np.sum((y_test - m["pred"]) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        with cols[i]:
            st.metric(f"MAE — {name}", f"S${mae:,.0f}/sqm")
            st.metric(f"RMSE — {name}", f"S${rmse:,.0f}/sqm")
            st.metric(f"R² — {name}", f"{r2:.4f}")
            st.metric(f"MAPE — {name}", f"{_mape(y_test.values, m['pred']):.1f}%")

    # ---- Cross-Validation ----
    with st.expander("🔍 交叉验证 (5-fold CV, 训练集)", expanded=False):
        X_tr = prep.transform(X_train)
        for name, m in models.items():
            pipe = m["pipeline"]
            cv_mae = -cross_val_score(pipe, X_tr, y_train, cv=5, scoring="neg_mean_absolute_error")
            cv_r2 = cross_val_score(pipe, X_tr, y_train, cv=5, scoring="r2")
            st.caption(
                f"{name}: CV MAE = S${cv_mae.mean():,.0f} ± S${cv_mae.std():,.0f}/sqm, "
                f"CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
            )

    # ---- Predicted vs Actual (colored by flat_type) ----
    st.subheader("预测值 vs 实际值")
    for name, m in models.items():
        plot_df = pd.DataFrame({
            "实际单价 (新元/sqm)": y_test.values,
            "预测单价 (新元/sqm)": m["pred"],
            "房型": X_test["flat_type"].values,
            "误差率 (%)": np.abs(y_test.values - m["pred"]) / y_test.values * 100,
        })
        fig = px.scatter(
            plot_df, x="实际单价 (新元/sqm)", y="预测单价 (新元/sqm)",
            color="房型",
            title=f"{name} — 预测 vs 实际 (按房型着色)",
            opacity=0.55,
        )
        fig.add_scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines", name="完美预测", line=dict(dash="dash", color="gray"),
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width='stretch')

    # ---- Residual Analysis ----
    with st.expander("📉 残差诊断", expanded=False):
        for name, m in models.items():
            residuals = y_test.values - m["pred"]
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.histogram(
                    residuals, nbins=50, title=f"{name} 残差分布",
                    labels={"value": "残差 (新元/sqm)", "count": "频数"},
                )
                fig1.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig1, width='stretch')
            with c2:
                fig2 = px.scatter(
                    x=m["pred"], y=residuals, opacity=0.4,
                    title=f"{name} 残差 vs 预测值",
                    labels={"x": "预测单价 (新元/sqm)", "y": "残差 (新元/sqm)"},
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2, width='stretch')

    # ---- Feature Importance ----
    st.header("⭐ 特征重要性")

    if "Random Forest" in models:
        rf_model = models["Random Forest"]["pipeline"]
        cat_features = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_cols))
        all_features = numeric_cols + cat_features

        importances = rf_model.feature_importances_
        # Aggregate OneHot back to original features
        agg_imp = {}
        for col in numeric_cols:
            idx = all_features.index(col)
            agg_imp[col] = float(importances[idx])
        for col in categorical_cols:
            col_indices = [i for i, f in enumerate(cat_features) if f.startswith(f"{col}_")]
            agg_imp[col] = float(sum(importances[[all_features.index(cat_features[i]) for i in col_indices]]))

        imp_df = pd.DataFrame({
            "特征": list(agg_imp.keys()), "重要性": list(agg_imp.values()),
        }).sort_values("重要性", ascending=True)

        fig_imp = px.bar(
            imp_df, x="重要性", y="特征", orientation="h",
            title="Random Forest 特征重要性 (聚合)",
            text=imp_df["重要性"].apply(lambda x: f"{x:.4f}"),
        )
        fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, width='stretch')

    # Ridge / Linear coefficients
    linear_model = None
    if "Ridge Regression" in models:
        linear_model = models["Ridge Regression"]["pipeline"]
    elif "Linear Regression" in models:
        linear_model = models["Linear Regression"]["pipeline"]

    if linear_model is not None and hasattr(linear_model, "coef_"):
        cat_features = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_cols))
        all_features = numeric_cols + cat_features
        coefs = linear_model.coef_
        coef_df = pd.DataFrame({
            "特征": all_features, "|系数|": np.abs(coefs),
        }).sort_values("|系数|", ascending=True).tail(15)
        fig_coef = px.bar(
            coef_df, x="|系数|", y="特征", orientation="h",
            title="Ridge |系数| (标准化后)",
            labels={"|系数|": "|系数|", "特征": "特征"},
        )
        fig_coef.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_coef, width='stretch')

    # ==================== INTERACTIVE PREDICTION ====================

    st.divider()
    st.header("🎯 房价预估器")

    with st.form("prediction_form"):
        st.markdown("输入房产属性，模型实时预估单价与总价。")

        c1, c2, c3 = st.columns(3)
        with c1:
            u_town = st.selectbox("镇区", sorted(df["town"].unique()), key="pred_town")
            u_type = st.selectbox("房型", sorted(df["flat_type"].unique()), key="pred_type")
            u_area = st.number_input("面积 (sqm)", 30.0, 250.0, 90.0, 1.0, key="pred_area")
        with c2:
            u_lease = st.number_input("剩余租约年限", 0.0, 99.0, 70.0, 1.0, key="pred_lease")
            u_commence = st.number_input("建成年份", 1960, 2030, max(1960, int(current_year - u_lease)), key="pred_commence")
            u_age = current_year - u_commence
            st.caption(f"→ 房龄约 {u_age} 年")
        with c3:
            u_storey = st.slider("楼层 (中位)", 1, 50, 10, key="pred_storey")
            u_year = st.number_input("交易年份", 2020, 2030, current_year, key="pred_year")

        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
            submitted = st.form_submit_button("💰 预测价格", use_container_width=True)

    if submitted:
        u_type_ord = FLAT_TYPE_ORDINAL.get(u_type, 0)
        u_mrt = _nearest_mrt_dist(u_town)
        u_schools = len(SCHOOLS_BY_TOWN.get(u_town, []))

        input_data = pd.DataFrame([{
            "floor_area_sqm": u_area, "remaining_lease": u_lease,
            "floor_age": u_age, "storey_mid": u_storey,
            "flat_type_ordinal": u_type_ord, "mrt_dist_km": u_mrt,
            "school_count": u_schools,
            "town": u_town, "year": u_year,
        }])

        input_tr = prep.transform(input_data)

        st.subheader("预测结果")
        results = {}
        cols = st.columns(len(models))
        for i, (name, m) in enumerate(models.items()):
            psm = float(m["pipeline"].predict(input_tr)[0])
            total = psm * u_area
            results[name] = psm
            with cols[i]:
                st.metric(
                    label=f"{name}",
                    value=f"S${total:,.0f}",
                    delta=f"S${psm:,.0f}/sqm",
                )

        # Prediction interval via RF individual trees
        if "Random Forest" in models:
            rf = models["Random Forest"]["pipeline"]
            if hasattr(rf, "estimators_"):
                tree_preds = np.array([t.predict(input_tr)[0] for t in rf.estimators_])
                lo, hi = np.percentile(tree_preds, [5, 95])
                st.caption(
                    f"RF 90% 预测区间: S${lo:,.0f}/sqm – S${hi:,.0f}/sqm "
                    f"(总价 S${lo*u_area:,.0f} – S${hi*u_area:,.0f})"
                )

        # Similar transactions — progressive relaxation to find matches
        st.subheader("📋 相似历史成交")

        def _find_similar(data, town, flat_type, area, lease):
            """Progressively relax filters until we find matches."""
            for area_range, lease_range in [(10, 5), (15, 10), (20, 15), (30, 99)]:
                mask = (
                    (data["flat_type"] == flat_type)
                    & (data["town"] == town)
                    & (data["floor_area_sqm"].between(
                        max(0, area - area_range), area + area_range))
                )
                candidates = data[mask]
                if "remaining_lease" in candidates.columns and lease_range < 99:
                    candidates = candidates[
                        candidates["remaining_lease"].between(
                            max(0, lease - lease_range), lease + lease_range
                        )
                    ]
                if len(candidates) >= 3:
                    return candidates.nlargest(10, "month"), area_range, lease_range
            # Final fallback: same flat_type + town only, no area/lease filter
            fallback = data[
                (data["flat_type"] == flat_type) & (data["town"] == town)
            ]
            return fallback.nlargest(10, "month"), None, None

        similar, a_rng, l_rng = _find_similar(df, u_town, u_type, u_area, u_lease)

        if len(similar) > 0:
            if a_rng is not None:
                st.caption(f"匹配条件: 面积 ±{a_rng}sqm, 租约 ±{l_rng}年 → 找到 {len(similar)} 条")
            else:
                st.caption(f"放宽至同镇区+同房型 → 找到 {len(similar)} 条")
            sim_disp = similar[[
                "month", "town", "flat_type", "floor_area_sqm",
                "remaining_lease", "resale_price", "price_per_sqm",
            ]].copy()
            sim_disp["month"] = sim_disp["month"].dt.strftime("%Y-%m")
            for col in ["resale_price", "price_per_sqm"]:
                sim_disp[col] = sim_disp[col].apply(lambda x: f"S${x:,.0f}")
            st.dataframe(sim_disp, width='stretch', hide_index=True)
        else:
            st.caption("未找到足够相似的历史交易记录，请尝试调整输入参数。")
