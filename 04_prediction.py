"""Page 4: Price Prediction — enriched features, RidgeCV, Hybrid RF/ExtraTrees, time-split."""

from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils.helpers import TOWN_COORDS, fmt_price

# ---- Constants ----
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
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def _nearest_mrt_dist(town: str) -> float:
    if town not in TOWN_COORDS:
        return 2.0
    t_lat, t_lng = TOWN_COORDS[town]
    mrt_list = MRT_COORDS.get(town, [(t_lat+0.01, t_lng+0.01)])
    return min(_haversine(t_lat, t_lng, mlat, mlng) for mlat, mlng in mrt_list)


def _mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)


# ====================== FEATURE ENGINEERING ======================

def _add_engineered_features(df, train_df=None, enc_maps=None):
    """Add rich engineered features. Uses train_df for target encoding (prevents leakage)."""
    df = df.copy()
    cy = pd.Timestamp.now().year

    # Basic features (skip if already present)
    if "floor_age" not in df.columns and "lease_commence_date" in df.columns:
        df["floor_age"] = df["lease_commence_date"].apply(
            lambda x: cy - x if pd.notna(x) else None)
    if "flat_type_ordinal" not in df.columns:
        df["flat_type_ordinal"] = df["flat_type"].map(FLAT_TYPE_ORDINAL).fillna(0).astype(int)
    if "mrt_dist_km" not in df.columns:
        df["mrt_dist_km"] = df["town"].apply(_nearest_mrt_dist)
    if "school_count" not in df.columns:
        df["school_count"] = df["town"].apply(lambda t: len(SCHOOLS_BY_TOWN.get(t, [])))

    # Polynomial / squared
    df["area_sq"] = df["floor_area_sqm"]**2
    df["lease_sq"] = df["remaining_lease"]**2
    df["storey_sq"] = df["storey_mid"]**2
    df["age_sq"] = df["floor_age"]**2

    # Interactions
    df["area_x_lease"] = df["floor_area_sqm"] * df["remaining_lease"]
    df["area_x_flat_type"] = df["floor_area_sqm"] * df["flat_type_ordinal"]
    df["lease_x_storey"] = df["remaining_lease"] * df["storey_mid"]
    df["age_x_area"] = df["floor_age"] * df["floor_area_sqm"]
    df["lease_x_flat_type"] = df["remaining_lease"] * df["flat_type_ordinal"]
    df["storey_x_flat_type"] = df["storey_mid"] * df["flat_type_ordinal"]

    # Ratios
    df["area_per_room"] = df["floor_area_sqm"] / df["flat_type_ordinal"].clip(lower=1)
    df["lease_ratio"] = df["remaining_lease"] / 99.0
    df["age_ratio"] = df["floor_age"] / 99.0

    # Temporal
    df["year_since_2020"] = df["year"] - 2020
    if "quarter" not in df.columns:
        df["quarter"] = df["month"].dt.quarter.astype(int) if "month" in df.columns else 2
    df["months_since_2020"] = (df["year"]-2020)*12 + df["quarter"]*3

    # Binned
    df["lease_bucket"] = pd.cut(
        df["remaining_lease"], bins=[0,40,60,80,99], labels=[0,1,2,3]).astype(int)
    df["area_bucket"] = pd.cut(
        df["floor_area_sqm"], bins=[0,60,90,120,300], labels=[0,1,2,3]).astype(int)
    df["storey_bucket"] = pd.cut(
        df["storey_mid"], bins=[0,5,10,20,50], labels=[0,1,2,3]).astype(int)

    # Target encoding (from training data only)
    if train_df is not None:
        tf_avg = train_df.groupby(["town","flat_type"])["price_per_sqm"].mean()
        ty_avg = train_df.groupby(["town","year"])["price_per_sqm"].mean()
        g_avg = float(train_df["price_per_sqm"].mean())
        tf_d = {k: float(v) for k,v in tf_avg.to_dict().items()}
        ty_d = {k: float(v) for k,v in ty_avg.to_dict().items()}
        df["town_flat_avg_price"] = df.apply(
            lambda r: tf_d.get((r["town"],r["flat_type"]), g_avg), axis=1)
        df["town_year_avg_price"] = df.apply(
            lambda r: ty_d.get((r["town"],r["year"]), g_avg), axis=1)
        if enc_maps is not None:
            enc_maps.update({"town_flat_avg": tf_d, "town_year_avg": ty_d, "global_avg": g_avg})
    elif enc_maps and enc_maps:
        tf_d = enc_maps.get("town_flat_avg", {})
        ty_d = enc_maps.get("town_year_avg", {})
        g_avg = enc_maps.get("global_avg", 0.0)
        df["town_flat_avg_price"] = df.apply(
            lambda r: tf_d.get((r["town"],r["flat_type"]), g_avg), axis=1)
        df["town_year_avg_price"] = df.apply(
            lambda r: ty_d.get((r["town"],r["year"]), g_avg), axis=1)
    else:
        df["town_flat_avg_price"] = 0.0
        df["town_year_avg_price"] = 0.0

    return df


# Feature groups
BASE_FEATURES = [
    "floor_area_sqm","remaining_lease","floor_age","storey_mid",
    "flat_type_ordinal","mrt_dist_km","school_count","year",
]
ENGINEERED_FEATURES = [
    "area_sq","lease_sq","storey_sq","age_sq",
    "area_x_lease","area_x_flat_type","lease_x_storey",
    "age_x_area","lease_x_flat_type","storey_x_flat_type",
    "area_per_room","lease_ratio","age_ratio",
    "year_since_2020","quarter","months_since_2020",
    "lease_bucket","area_bucket","storey_bucket",
]
TARGET_ENC_FEATURES = ["town_flat_avg_price","town_year_avg_price"]

# Safe features for Ridge (no target encoding — prevents leakage for linear models)
RIDGE_NUMERIC = BASE_FEATURES + ENGINEERED_FEATURES
# Tree features include target encoding (trees handle it better)
TREE_NUMERIC = BASE_FEATURES + ENGINEERED_FEATURES + TARGET_ENC_FEATURES
CAT_FEATURES = ["town"]


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    st.title("🤖 房价预测模型（增强版）")
    st.markdown(
        "丰富特征工程 → 时间切分训练 → RidgeCV · RF/ExtraTrees Hybrid → "
        "分房型评估 → 交互式预估"
    )

    # ---- Feature Engineering ----
    df = df.copy()
    df = _add_engineered_features(df)

    all_feature_cols = TREE_NUMERIC + CAT_FEATURES
    target_col = "price_per_sqm"
    model_df = df.dropna(subset=all_feature_cols + [target_col]).copy()

    with st.expander("📋 特征工程概览", expanded=False):
        st.markdown(f"""
        | 特征组 | 数量 | 说明 |
        |--------|------|------|
        | 基础数值 | {len(BASE_FEATURES)} | 面积、租约、房龄、楼层、房型编码等 |
        | 多项式/平方项 | 4 | 面积²、租约²、楼层²、房龄² |
        | 交互项 | 6 | 面积×租约、面积×房型等 |
        | 比率特征 | 3 | 每房面积、租约占比、房龄占比 |
        | 时序特征 | 3 | 年份编码、季度、月份累计 |
        | 分箱特征 | 3 | 租约/面积/楼层分桶 |
        | 目标编码 | 2 | 镇区×房型均价、镇区×年份均价（仅从训练集计算） |
        """)
        st.caption(
            f"预测目标: **单价 (新元/sqm)** | "
            f"总特征: Ridge {len(RIDGE_NUMERIC)+1}个 / Tree {len(TREE_NUMERIC)+1}个 | "
            f"有效样本: **{len(model_df):,}** 条"
        )

    # ---- Time split ----
    X = model_df[all_feature_cols + ["flat_type"]]
    y = model_df[target_col]
    X_train_raw = X[X["year"] <= 2023].copy()
    X_test_raw = X[X["year"] >= 2024].copy()
    y_train = y.loc[X_train_raw.index]
    y_test = y.loc[X_test_raw.index]

    if len(X_train_raw) < 100 or len(X_test_raw) < 30:
        st.error(f"数据不足: 训练集 {len(X_train_raw)} 条, 测试集 {len(X_test_raw)} 条")
        return

    st.markdown(
        f"**时间切分**: 训练集 (2020–2023) **{len(X_train_raw):,}** 条 | "
        f"测试集 (2024+) **{len(X_test_raw):,}** 条"
    )

    # Recompute target encoding from training data only
    enc_maps = {}
    train_enc_ref = model_df.loc[X_train_raw.index]
    X_train = _add_engineered_features(
        X_train_raw.drop(columns=TARGET_ENC_FEATURES, errors="ignore"),
        train_df=train_enc_ref, enc_maps=enc_maps)
    X_test = _add_engineered_features(
        X_test_raw.drop(columns=TARGET_ENC_FEATURES, errors="ignore"),
        train_df=train_enc_ref)

    # ---- Preprocessors ----
    # Ridge: uses safe features (no target encoding)
    prep_ridge = ColumnTransformer([
        ("num", StandardScaler(), RIDGE_NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
    ]).fit(X_train[RIDGE_NUMERIC + CAT_FEATURES])

    # Tree models: use full features including target encoding
    prep_tree = ColumnTransformer([
        ("num", StandardScaler(), TREE_NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
    ]).fit(X_train[TREE_NUMERIC + CAT_FEATURES])

    # ---- Sidebar ----
    st.sidebar.subheader("🤖 模型设置")
    use_ridge = st.sidebar.checkbox(
        "RidgeCV (增强)", value=True, key="use_ridge",
        help="RidgeCV在安全特征上（无目标编码），自动CV选择alpha")
    use_rf = st.sidebar.checkbox(
        "RF Hybrid", value=True, key="use_rf",
        help="Ridge趋势 + RandomForest/ExtraTrees学习残差")
    use_xgb = st.sidebar.checkbox(
        "XGBoost Hybrid", value=False, key="use_xgb",
        help="Ridge趋势 + XGBoost学习残差（需安装xgboost）")

    params = {}
    if use_rf or use_xgb:
        params["rf_n"] = st.sidebar.slider("RF/ET n_estimators", 200, 2000, 800, 100, key="rf_n")
        params["rf_depth"] = st.sidebar.slider("RF/ET max_depth", 10, 50, 30, key="rf_depth")

    # ---- Train models ----
    def _train_models():
        models = {}
        X_tr_tree = prep_tree.transform(X_train[TREE_NUMERIC + CAT_FEATURES])
        X_te_tree = prep_tree.transform(X_test[TREE_NUMERIC + CAT_FEATURES])

        # --- RidgeCV (safe features) ---
        if use_ridge or use_rf or use_xgb:
            ridge_pipe = Pipeline([
                ("prep", ColumnTransformer([
                    ("num", StandardScaler(), RIDGE_NUMERIC),
                    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
                     CAT_FEATURES),
                ])),
                ("ridge", RidgeCV(alphas=np.logspace(-1, 3, 20), cv=5)),
            ])
            ridge_pipe.fit(X_train[RIDGE_NUMERIC + CAT_FEATURES], y_train)
            ridge_pred = ridge_pipe.predict(X_test[RIDGE_NUMERIC + CAT_FEATURES])
            ridge_train_pred = ridge_pipe.predict(X_train[RIDGE_NUMERIC + CAT_FEATURES])
            residuals = y_train.values - ridge_train_pred

            if use_ridge:
                alpha_val = ridge_pipe.named_steps["ridge"].alpha_
                models["RidgeCV"] = {
                    "pipeline": ridge_pipe,
                    "pred": ridge_pred,
                    "ridge_input": RIDGE_NUMERIC + CAT_FEATURES,
                    "info": f"alpha={alpha_val:.4f}, {len(RIDGE_NUMERIC)+2}特征",
                }

            # --- RF Hybrid ---
            if use_rf:
                n_est = params.get("rf_n", 800)
                max_d = params.get("rf_depth", 30)

                # Train 2 RFs with different seeds for bagging
                rf1 = RandomForestRegressor(
                    n_estimators=n_est, max_depth=max_d, min_samples_leaf=3,
                    random_state=42, n_jobs=-1)
                rf1.fit(X_tr_tree, residuals)
                r1_pred = rf1.predict(X_te_tree)

                rf2 = RandomForestRegressor(
                    n_estimators=n_est, max_depth=max_d, min_samples_leaf=3,
                    random_state=123, n_jobs=-1)
                rf2.fit(X_tr_tree, residuals)
                r2_pred = rf2.predict(X_te_tree)

                # ExtraTrees for diversity
                et = ExtraTreesRegressor(
                    n_estimators=n_est, max_depth=max_d, min_samples_leaf=3,
                    random_state=42, n_jobs=-1)
                et.fit(X_tr_tree, residuals)
                et_pred = et.predict(X_te_tree)

                # Ensemble residual prediction
                resid_pred = (r1_pred + r2_pred + et_pred) / 3.0
                rf_hybrid_pred = ridge_pred + resid_pred

                models["RF Hybrid"] = {
                    "pipeline": {"ridge": ridge_pipe, "rf1": rf1, "rf2": rf2, "et": et},
                    "pred": rf_hybrid_pred,
                    "ridge_input": RIDGE_NUMERIC + CAT_FEATURES,
                    "info": f"2×RF+ET各{n_est}棵, depth={max_d}",
                }

                # Also single RF for comparison
                rf_single_pred = ridge_pred + r1_pred
                models["RF Single"] = {
                    "pipeline": {"ridge": ridge_pipe, "rf": rf1},
                    "pred": rf_single_pred,
                    "ridge_input": RIDGE_NUMERIC + CAT_FEATURES,
                    "info": f"单RF, {n_est}棵",
                    "_compare_only": True,
                }

            # --- XGBoost Hybrid ---
            if use_xgb:
                try:
                    from xgboost import XGBRegressor
                    xgb = XGBRegressor(
                        n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, n_jobs=-1)
                    xgb.fit(X_tr_tree, residuals)
                    xgb_pred = ridge_pred + xgb.predict(X_te_tree)
                    models["XGBoost Hybrid"] = {
                        "pipeline": {"ridge": ridge_pipe, "xgb": xgb},
                        "pred": xgb_pred,
                        "ridge_input": RIDGE_NUMERIC + CAT_FEATURES,
                        "info": "Ridge趋势 + XGB残差",
                    }
                except ImportError:
                    st.warning("XGBoost 未安装。运行 `pip install xgboost` 安装后可使用。")

        return models

    if not any([use_ridge, use_rf, use_xgb]):
        st.warning("请在侧边栏至少选择一个模型。")
        return

    with st.spinner("训练模型中…"):
        models = _train_models()

    main_models = {k: v for k, v in models.items() if not v.get("_compare_only")}
    compare_models = {k: v for k, v in models.items() if v.get("_compare_only")}

    # ==================== SEGMENT EVALUATION ====================
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
            if m.get("_compare_only"):
                continue
            pred_seg = m["pred"][seg_mask.values]
            mae = float(np.mean(np.abs(y_seg - pred_seg)))
            mape_val = _mape(y_seg.values, pred_seg)
            r2_val = float(
                1-np.sum((y_seg-pred_seg)**2)/np.sum((y_seg-y_seg.mean())**2)
            ) if y_seg.std() > 0 else 0
            all_seg_rows.append({
                "房源类型": seg_name, "模型": model_name,
                "样本数": n_seg, "MAE": mae, "MAPE(%)": mape_val, "R²": r2_val,
            })

    if all_seg_rows:
        seg_df = pd.DataFrame(all_seg_rows)
        st.dataframe(
            seg_df.style.format({
                "MAE": "S${:,.0f}", "MAPE(%)": "{:.1f}%", "R²": "{:.4f}",
            }), width='stretch', hide_index=True)

    # ==================== OVERALL EVALUATION ====================
    st.header("📐 整体评估")

    cols = st.columns(len(main_models))
    for i, (name, m) in enumerate(main_models.items()):
        y_p = m["pred"]
        mae = float(np.mean(np.abs(y_test - y_p)))
        rmse = float(np.sqrt(np.mean((y_test - y_p)**2)))
        r2 = float(1 - np.sum((y_test-y_p)**2)/np.sum((y_test-y_test.mean())**2))
        with cols[i]:
            delta_str = "✅ 达标" if r2 >= 0.80 else f"vs 旧版 +{r2-0.48 if 'Ridge' in name else r2-0.19:.2f}"
            st.metric(f"R² — {name}", f"{r2:.4f}", delta=delta_str)
            st.metric(f"MAE — {name}", f"S${mae:,.0f}/sqm")
            st.metric(f"RMSE — {name}", f"S${rmse:,.0f}/sqm")
            st.metric(f"MAPE — {name}", f"{_mape(y_test.values, y_p):.1f}%")
            if "info" in m:
                st.caption(m["info"])

    # ---- Cross-Validation ----
    with st.expander("🔍 交叉验证 (5-fold CV, 训练集)", expanded=False):
        X_tr_ridge = prep_ridge.transform(X_train[RIDGE_NUMERIC + CAT_FEATURES])
        X_tr_tree_cv = prep_tree.transform(X_train[TREE_NUMERIC + CAT_FEATURES])
        for name, m in {**main_models, **compare_models}.items():
            pipe = m["pipeline"]
            if isinstance(pipe, dict):
                st.caption(f"{name}: 混合模型，CV 省略（嵌套交叉验证）")
                continue
            try:
                cv_mae = -cross_val_score(
                    pipe, X_tr_ridge, y_train, cv=5, scoring="neg_mean_absolute_error")
                cv_r2 = cross_val_score(pipe, X_tr_ridge, y_train, cv=5, scoring="r2")
                st.caption(
                    f"{name}: CV MAE = S${cv_mae.mean():,.0f} ± "
                    f"S${cv_mae.std():,.0f}/sqm, CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
            except Exception as e:
                st.caption(f"{name}: CV 错误 — {e}")

    # ---- Predicted vs Actual ----
    st.subheader("预测值 vs 实际值")
    for name, m in {**main_models, **compare_models}.items():
        plot_df = pd.DataFrame({
            "实际单价 (新元/sqm)": y_test.values,
            "预测单价 (新元/sqm)": m["pred"],
            "房型": X_test["flat_type"].values,
        })
        fig = px.scatter(
            plot_df, x="实际单价 (新元/sqm)", y="预测单价 (新元/sqm)",
            color="房型", title=f"{name} — 预测 vs 实际", opacity=0.55)
        fig.add_scatter(
            x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
            mode="lines", name="完美预测", line=dict(dash="dash", color="gray"))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width='stretch')

    # ---- Residuals ----
    with st.expander("📉 残差诊断", expanded=False):
        for name, m in main_models.items():
            residuals = y_test.values - m["pred"]
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.histogram(
                    residuals, nbins=50, title=f"{name} 残差分布")
                fig1.update_layout(height=300)
                st.plotly_chart(fig1, width='stretch')
            with c2:
                fig2 = px.scatter(
                    x=m["pred"], y=residuals, opacity=0.4,
                    title=f"{name} 残差 vs 预测值")
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, width='stretch')

    # ---- Feature Importance (from RF) ----
    st.header("⭐ 特征重要性")
    cat_feature_names = list(
        prep_tree.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))
    all_tree_features = TREE_NUMERIC + cat_feature_names

    if "RF Single" in models:
        rf_pipe = models["RF Single"]["pipeline"]
        rf_model = rf_pipe["rf"]
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            agg_imp = {}
            for col in TREE_NUMERIC:
                agg_imp[col] = float(importances[all_tree_features.index(col)])
            for col in CAT_FEATURES:
                indices = [i for i, f in enumerate(cat_feature_names)
                          if f.startswith(f"{col}_")]
                if indices:
                    agg_imp[col] = float(sum(
                        importances[all_tree_features.index(cat_feature_names[i])]
                        for i in indices))
            imp_df = pd.DataFrame({
                "特征": list(agg_imp.keys()), "重要性": list(agg_imp.values()),
            }).sort_values("重要性", ascending=True).tail(20)
            fig_imp = px.bar(
                imp_df, x="重要性", y="特征", orientation="h",
                title="Random Forest 特征重要性 Top-20", text_auto=".4f")
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, width='stretch')

    # ==================== INTERACTIVE PREDICTION ====================
    st.divider()
    st.header("🎯 房价预估器")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            u_town = st.selectbox("镇区", sorted(df["town"].unique()), key="pred_town")
            u_type = st.selectbox("房型", sorted(df["flat_type"].unique()), key="pred_type")
            u_area = st.number_input("面积 (sqm)", 30.0, 250.0, 90.0, 1.0)
        with c2:
            u_lease = st.number_input("剩余租约年限", 0.0, 99.0, 70.0, 1.0)
            u_commence = st.number_input(
                "建成年份", 1960, 2030,
                max(1960, int(pd.Timestamp.now().year - u_lease)))
            u_age = pd.Timestamp.now().year - u_commence
            st.caption(f"→ 房龄约 {u_age} 年")
        with c3:
            u_storey = st.slider("楼层 (中位)", 1, 50, 10)
            u_year = st.number_input("交易年份", 2020, 2030, pd.Timestamp.now().year)
            u_month = st.selectbox("交易月份", list(range(1,13)), index=5)

        _, col_btn, _ = st.columns([2,1,2])
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
            "month": pd.Timestamp(f"{u_year}-{u_month:02d}-01"),
            "flat_type": u_type,
        }])
        input_data = _add_engineered_features(input_data, enc_maps=enc_maps)
        input_ridge = input_data[RIDGE_NUMERIC + CAT_FEATURES]
        input_tree = prep_tree.transform(input_data[TREE_NUMERIC + CAT_FEATURES])

        st.subheader("预测结果")
        results = {}
        cols = st.columns(len(main_models))
        for i, (name, m) in enumerate(main_models.items()):
            pipe = m["pipeline"]
            if isinstance(pipe, dict):
                # Hybrid model
                ridge_val = float(pipe["ridge"].predict(input_ridge)[0])
                # Average residual predictions from all sub-models
                resid_vals = []
                for key in pipe:
                    if key != "ridge":
                        resid_vals.append(float(pipe[key].predict(input_tree)[0]))
                resid_val = np.mean(resid_vals)
                psm = ridge_val + resid_val
            else:
                psm = float(pipe.predict(input_ridge)[0])
            total = psm * u_area
            results[name] = psm
            with cols[i]:
                st.metric(
                    label=name, value=f"S${total:,.0f}",
                    delta=f"S${psm:,.0f}/sqm")

        # Similar transactions
        st.subheader("📋 相似历史成交")
        def _find_similar(data, town, ft, area, lease):
            for ar, lr in [(10,5),(15,10),(20,15),(30,99)]:
                cand = data[
                    (data["flat_type"]==ft) & (data["town"]==town) &
                    (data["floor_area_sqm"].between(max(0,area-ar), area+ar))]
                if "remaining_lease" in cand.columns and lr < 99:
                    cand = cand[cand["remaining_lease"].between(
                        max(0,lease-lr), lease+lr)]
                if len(cand) >= 3:
                    return cand.nlargest(10, "month"), ar, lr
            fb = data[(data["flat_type"]==ft) & (data["town"]==town)]
            return fb.nlargest(10, "month"), None, None

        similar, a_rng, l_rng = _find_similar(df, u_town, u_type, u_area, u_lease)
        if len(similar) > 0:
            if a_rng:
                st.caption(f"匹配: 面积±{a_rng}sqm, 租约±{l_rng}年 → {len(similar)}条")
            else:
                st.caption(f"放宽至同镇区+同房型 → {len(similar)}条")
            sim_disp = similar[[
                "month","town","flat_type","floor_area_sqm",
                "remaining_lease","resale_price","price_per_sqm"]].copy()
            sim_disp["month"] = sim_disp["month"].dt.strftime("%Y-%m")
            for c in ["resale_price","price_per_sqm"]:
                sim_disp[c] = sim_disp[c].apply(lambda x: f"S${x:,.0f}")
            st.dataframe(sim_disp, width='stretch', hide_index=True)
        else:
            st.caption("未找到足够相似的历史交易记录。")
