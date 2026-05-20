"""Shared utilities for HDB analysis app."""

import numpy as np
import pandas as pd
import streamlit as st

TOWN_COLORS = {"PUNGGOL": "#FF6B6B", "SENGKANG": "#4ECDC4", "HOUGANG": "#45B7D1"}

TOWN_COORDS = {
    "PUNGGOL": (1.4043, 103.9028),
    "SENGKANG": (1.3917, 103.8942),
    "HOUGANG": (1.3714, 103.8923),
}


def fmt_price(v: float) -> str:
    """Format price in SGD."""
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"S${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"S${v/1_000:,.0f}K"
    return f"S${v:,.0f}"


def fmt_pct(v: float) -> str:
    """Format as percentage string."""
    return f"{v*100:.1f}%" if pd.notna(v) else "—"


def metric_card(label: str, value: str, delta: str = None):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta)


def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate MAE, RMSE, R² for regression models."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R²": r2}
