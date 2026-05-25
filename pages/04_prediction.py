"""Page 4: Price Prediction — Linear Regression + Random Forest, cross-validation, residuals, feature importance, interactive prediction."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils.helpers import fmt_price, get_model_metrics


def run(df: pd.DataFrame):
    st.title("🤖 价格预测模型")
    st.markdown("Linear Regression + Random Forest — 交叉验证、残差诊断、特征重要性、交互式预测。")

    # ---- Prepare Data ----
    feature_cols = ["town", "flat_type", "floor_area_sqm", "remaining_lease", "storey_mid", "year"]
    model_df = df.dropna(subset=feature_cols + ["resale_price"]).copy()

    X = model_df[feature_cols]
    y = model_df["resale_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = ["floor_area_sqm", "remaining_lease", "storey_mid", "year"]
    categorical_cols = ["town", "flat_type"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ])

    # ---- Train Models ----
    @st.cache_resource
    def train_models(_X_train, _y_train, _X_test, _y_test):
        models = {}

        lr = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0))])
        lr.fit(_X_train, _y_train)
        models["Ridge Regression"] = {"pipeline": lr, "pred": lr.predict(_X_test)}

        rf = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=250, max_depth=18,
                                            min_samples_leaf=5, random_state=42, n_jobs=-1))
        ])
        rf.fit(_X_train, _y_train)
        models["Random Forest"] = {"pipeline": rf, "pred": rf.predict(_X_test)}

        return models

    with st.spinner("训练模型中…"):
        models = train_models(X_train, y_train, X_test, y_test)

    # ==================== MODEL EVALUATION ====================

    st.header("📐 模型评估")

    metrics_data = []
    for name, m in models.items():
        m_metrics = get_model_metrics(y_test, m["pred"])
        m_metrics["Model"] = name
        metrics_data.append(m_metrics)

    metrics_df = pd.DataFrame(metrics_data)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        for _, row in metrics_df.iterrows():
            st.metric(f"MAE — {row['Model']}", f"S${row['MAE']:,.0f}")
    with c2:
        for _, row in metrics_df.iterrows():
            st.metric(f"RMSE — {row['Model']}", f"S${row['RMSE']:,.0f}")
    with c3:
        for _, row in metrics_df.iterrows():
            st.metric(f"R² — {row['Model']}", f"{row['R²']:.4f}")

    # ---- Cross-Validation ----
    with st.expander("🔍 交叉验证 (5-fold CV)", expanded=False):
        cv_results = {}
        for name, m in models.items():
            # CV on training set
            pipe = m["pipeline"]
            cv_mae = -cross_val_score(pipe, X_train, y_train, cv=5,
                                       scoring="neg_mean_absolute_error")
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
            cv_results[name] = {"MAE_mean": cv_mae.mean(), "MAE_std": cv_mae.std(),
                                "R2_mean": cv_r2.mean(), "R2_std": cv_r2.std()}
            st.caption(
                f"{name}: CV MAE = S${cv_mae.mean():,.0f} ± S${cv_mae.std():,.0f}, "
                f"CV R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
            )

    # ---- Model Comparison Chart ----
    st.subheader("模型指标对比")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="MAE", x=metrics_df["Model"], y=metrics_df["MAE"],
                              text=[f"S${v:,.0f}" for v in metrics_df["MAE"]],
                              textposition="outside", marker_color="#FF6B6B"))
    fig_bar.add_trace(go.Bar(name="RMSE", x=metrics_df["Model"], y=metrics_df["RMSE"],
                              text=[f"S${v:,.0f}" for v in metrics_df["RMSE"]],
                              textposition="outside", marker_color="#4ECDC4"))
    fig_bar.update_layout(
        title="MAE & RMSE 对比 (越低越好)", height=380,
        margin=dict(l=0, r=0, t=30, b=0), barmode="group",
    )
    st.plotly_chart(fig_bar, width='stretch')

    # ---- Actual vs Predicted ----
    st.subheader("预测值 vs 实际值")
    c1, c2 = st.columns(2)
    for i, (name, m) in enumerate(models.items()):
        col = [c1, c2][i]
        with col:
            err = y_test - m["pred"]
            err_pct = (err / y_test * 100).abs()
            fig = px.scatter(
                x=y_test, y=m["pred"], opacity=0.45,
                color=err_pct, color_continuous_scale="Reds",
                title=f"{name} (误差率着色)",
                labels={"x": "实际价格 (新元)", "y": "预测价格 (新元)", "color": "|误差%|"},
            )
            fig.add_scatter(
                x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                mode="lines", name="完美", line=dict(dash="dash", color="gray"),
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, width='stretch')

    # ---- Residual Analysis ----
    with st.expander("📉 残差诊断", expanded=False):
        c1, c2 = st.columns(2)
        for i, (name, m) in enumerate(models.items()):
            col = [c1, c2][i]
            residuals = y_test - m["pred"]
            with col:
                # Residual histogram
                fig1 = px.histogram(
                    residuals, nbins=50, title=f"{name} 残差分布",
                    labels={"value": "残差 (新元)", "count": "频数"},
                )
                fig1.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig1, width='stretch')

                # Residual vs fitted
                fig2 = px.scatter(
                    x=m["pred"], y=residuals, opacity=0.4,
                    title=f"{name} 残差 vs 预测值",
                    labels={"x": "预测价格 (新元)", "y": "残差 (新元)"},
                )
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2, width='stretch')

    # ==================== FEATURE IMPORTANCE ====================

    st.header("⭐ 特征重要性")

    rf_pipeline = models["Random Forest"]["pipeline"]
    rf_model = rf_pipeline.named_steps["model"]
    prep = rf_pipeline.named_steps["prep"]

    cat_features = list(prep.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    all_features = numeric_cols + cat_features
    importances = rf_model.feature_importances_

    # Aggregate back to original categorical features
    agg_importance = {}
    agg_importance["floor_area_sqm"] = importances[all_features.index("floor_area_sqm")]
    agg_importance["remaining_lease"] = importances[all_features.index("remaining_lease")]
    agg_importance["storey_mid"] = importances[all_features.index("storey_mid")]
    agg_importance["year"] = importances[all_features.index("year")]
    for col in categorical_cols:
        col_idx = [i for i, f in enumerate(cat_features) if f.startswith(f"{col}_")]
        agg_importance[col] = importances[[all_features.index(cat_features[i]) for i in col_idx]].sum()

    imp_df = pd.DataFrame({
        "feature": list(agg_importance.keys()),
        "importance": list(agg_importance.values()),
    }).sort_values("importance", ascending=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Random Forest 特征重要性 (聚合)",
            labels={"importance": "重要性", "feature": "特征"},
            text=imp_df["importance"].apply(lambda x: f"{x:.4f}"),
        )
        fig_imp.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, width='stretch')

    with c2:
        # LR coefficients (standardized)
        lr_pipeline = models["Ridge Regression"]["pipeline"]
        lr_model = lr_pipeline.named_steps["model"]
        coefs = lr_model.coef_
        lr_importance = pd.DataFrame({
            "feature": ["floor_area_sqm", "remaining_lease", "storey_mid", "year"] + cat_features,
            "coefficient": np.abs(coefs),
        }).sort_values("coefficient", ascending=True).tail(15)
        fig_lr = px.bar(
            lr_importance, x="coefficient", y="feature", orientation="h",
            title="Ridge |系数| (标准化后)",
            labels={"coefficient": "|系数|", "feature": "特征"},
        )
        fig_lr.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_lr, width='stretch')

    # ==================== INTERACTIVE PREDICTION ====================

    st.divider()
    st.header("🎯 自定义价格预测")

    with st.form("prediction_form"):
        st.markdown("输入房产属性，模型实时预测转售价格。")

        c1, c2, c3 = st.columns(3)
        with c1:
            u_town = st.selectbox("镇区", sorted(df["town"].unique()), key="pred_town")
            u_type = st.selectbox("房型", sorted(df["flat_type"].unique()), key="pred_type")
        with c2:
            u_area = st.number_input("面积 (sqm)", 30.0, 250.0, 90.0, 1.0, key="pred_area")
            u_storey = st.slider("楼层 (中位)", 1, 50, 10, key="pred_storey")
        with c3:
            u_lease = st.number_input("剩余年限 (年)", 0.0, 99.0, 70.0, 1.0, key="pred_lease")
            u_year = st.number_input("交易年份", 2020, 2030, 2025, key="pred_year")

        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
            submitted = st.form_submit_button("💰 预测价格", width='stretch')

    if submitted:
        input_data = pd.DataFrame([{
            "town": u_town, "flat_type": u_type,
            "floor_area_sqm": u_area, "remaining_lease": u_lease,
            "storey_mid": u_storey, "year": u_year,
        }])

        st.subheader("预测结果")
        results = {}
        cols = st.columns(len(models))
        for i, (name, m) in enumerate(models.items()):
            pred = m["pipeline"].predict(input_data)[0]
            psm = pred / u_area if u_area > 0 else 0
            results[name] = pred
            with cols[i]:
                st.metric(
                    label=name,
                    value=f"S${pred:,.0f}",
                    delta=f"S${psm:,.0f}/sqm",
                )

        # RF prediction interval using individual trees
        rf_estimator = models["Random Forest"]["pipeline"].named_steps["model"]
        prep_data = models["Random Forest"]["pipeline"].named_steps["prep"].transform(input_data)
        tree_preds = np.array([tree.predict(prep_data)[0] for tree in rf_estimator.estimators_])
        lower, upper = np.percentile(tree_preds, [5, 95])
        st.caption(
            f"RF 90% 预测区间: S${lower:,.0f} – S${upper:,.0f} "
            f"(基于 {len(rf_estimator.estimators_)} 棵树的分布)"
        )

        # Find similar actual transactions
        st.subheader("📋 相似历史成交 (参考)")
        similar = df[
            (df["flat_type"] == u_type)
            & (df["floor_area_sqm"].between(u_area - 10, u_area + 10))
            & (df["remaining_lease"].between(u_lease - 5, u_lease + 5))
            & (df["town"] == u_town)
        ].nsmallest(10, "month")
        if len(similar) > 0:
            sim_display = similar[["month", "town", "flat_type", "floor_area_sqm",
                                    "remaining_lease", "resale_price"]].copy()
            sim_display["month"] = sim_display["month"].dt.strftime("%Y-%m")
            sim_display["resale_price"] = sim_display["resale_price"].apply(fmt_price)
            st.dataframe(sim_display, width='stretch', hide_index=True)
        else:
            st.caption("未找到足够相似的历史交易记录。")
