"""Page 2: Map Visualization — pydeck-based town avg, heatmap, spatio-temporal, MRT/schools overlay."""

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

from utils.helpers import TOWN_COORDS, TOWN_COLORS, fmt_price

# ---- MRT / LRT stations in Punggol, Sengkang, Hougang area ----
MRT_STATIONS = pd.DataFrame([
    {"name": "Punggol MRT (NE17)", "lat": 1.4052, "lng": 103.9022, "type": "MRT"},
    {"name": "Sengkang MRT (NE16)", "lat": 1.3924, "lng": 103.8951, "type": "MRT"},
    {"name": "Buangkok MRT (NE15)", "lat": 1.3829, "lng": 103.8929, "type": "MRT"},
    {"name": "Hougang MRT (NE14)", "lat": 1.3716, "lng": 103.8923, "type": "MRT"},
    {"name": "Kovan MRT (NE13)",  "lat": 1.3603, "lng": 103.8853, "type": "MRT"},
    {"name": "Punggol LRT East Loop", "lat": 1.4071, "lng": 103.9056, "type": "LRT"},
    {"name": "Punggol LRT West Loop", "lat": 1.4035, "lng": 103.9000, "type": "LRT"},
    {"name": "Sengkang LRT East Loop", "lat": 1.3940, "lng": 103.8970, "type": "LRT"},
    {"name": "Sengkang LRT West Loop", "lat": 1.3905, "lng": 103.8925, "type": "LRT"},
])

# ---- Schools in Punggol, Sengkang, Hougang area ----
SCHOOLS = pd.DataFrame([
    {"name": "Punggol Green Primary",     "lat": 1.4075, "lng": 103.9040, "type": "小学"},
    {"name": "Mee Toh School",            "lat": 1.4035, "lng": 103.9080, "type": "小学"},
    {"name": "Sengkang Green Primary",    "lat": 1.3940, "lng": 103.8910, "type": "小学"},
    {"name": "Hougang Primary",           "lat": 1.3725, "lng": 103.8905, "type": "小学"},
    {"name": "Punggol Secondary",         "lat": 1.4060, "lng": 103.9070, "type": "中学"},
    {"name": "Sengkang Secondary",        "lat": 1.3920, "lng": 103.8990, "type": "中学"},
    {"name": "Hougang Secondary",         "lat": 1.3730, "lng": 103.8880, "type": "中学"},
    {"name": "Xinmin Secondary",          "lat": 1.3700, "lng": 103.8950, "type": "中学"},
    {"name": "Nan Chiau High",            "lat": 1.3915, "lng": 103.8880, "type": "中学"},
])

CENTER_LAT, CENTER_LNG = 1.3891, 103.8964
GLOBAL_VMIN, GLOBAL_VMAX = None, None  # set dynamically


# ====================== COLOR MAPPER ======================

def _price_color_hex(price: float, vmin: float, vmax: float) -> list:
    """Map price to an RGBA color array (yellow→orange→red gradient)."""
    if vmax <= vmin:
        return [240, 120, 60, 200]
    ratio = np.clip((price - vmin) / (vmax - vmin), 0, 1)
    r = int(255)
    g = int(220 * (1 - ratio))
    b = int(180 * (1 - ratio))
    return [r, g, b, 200]


# ====================== MAIN ======================

def run(df: pd.DataFrame):
    global GLOBAL_VMIN, GLOBAL_VMAX
    GLOBAL_VMIN = df["price_per_sqm"].quantile(0.02)
    GLOBAL_VMAX = df["price_per_sqm"].quantile(0.98)

    st.title("🗺️ 地图可视化")
    st.markdown("镇区均价、热力分布、时空变化 — 基于 pydeck，含 MRT/学校配套设施叠加。")

    # ---- Sidebar ----
    st.sidebar.subheader("🗺️ 地图设置")
    map_mode = st.sidebar.radio("视图模式", ["镇区均价", "热力分布", "时空变化"], key="map_mode")

    sel_types = st.sidebar.multiselect(
        "房型筛选", sorted(df["flat_type"].unique()),
        default=sorted(df["flat_type"].unique()), key="map_type_filter",
    )
    filtered = df[df["flat_type"].isin(sel_types)]

    # ---- Overlay: st.multiselect ----
    st.sidebar.subheader("🗂️ 配套设施叠加")
    overlay_options = st.sidebar.multiselect(
        "选择叠加图层", ["MRT/LRT 站点", "学校"],
        default=[], key="overlay_select",
    )

    # ---- Render ----
    if map_mode == "镇区均价":
        deck = _build_town_avg_map(filtered, overlay_options)
    elif map_mode == "热力分布":
        deck = _build_heatmap(filtered, overlay_options)
    else:
        deck, _ = _build_temporal_map(filtered, overlay_options)

    st.pydeck_chart(deck, height=550)

    # ---- Below-map ----
    if map_mode == "镇区均价":
        stats = filtered.groupby("town").agg(
            avg_price=("resale_price", "mean"), avg_psm=("price_per_sqm", "mean"),
            count=("resale_price", "count"),
        ).reset_index()
        _render_town_stats_table(stats)
    elif map_mode == "时空变化":
        _render_temporal_chart(filtered)


# ================== 1. TOWN AVG MAP ==================

def _build_town_avg_map(df: pd.DataFrame, overlays: list) -> pdk.Deck:
    stats = df.groupby("town").agg(
        avg_price=("resale_price", "mean"), avg_psm=("price_per_sqm", "mean"),
        count=("resale_price", "count"),
    ).reset_index()

    vmin, vmax = stats["avg_psm"].min(), stats["avg_psm"].max()
    radius_scale = 120

    chart_data = []
    for _, row in stats.iterrows():
        town = row["town"]
        if town not in TOWN_COORDS:
            continue
        lat, lng = TOWN_COORDS[town]
        chart_data.append({
            "lat": lat, "lng": lng,
            "radius": max(800, row["count"] * radius_scale),
            "color": _price_color_hex(row["avg_psm"], vmin, vmax),
            "town": town,
            "avg_psm": row["avg_psm"],
            "avg_price": row["avg_price"],
            "count": row["count"],
        })

    layers = [
        pdk.Layer("ScatterplotLayer", chart_data,
                  get_position=["lng", "lat"],
                  get_radius="radius",
                  get_fill_color="color",
                  pickable=True,
                  auto_highlight=True,
                  opacity=0.8,
                  radius_scale=1,
                  radius_min_pixels=30,
                  radius_max_pixels=80),
    ]
    _add_overlay_layers(layers, overlays)

    view = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LNG, zoom=12.5, pitch=0)
    tooltip = {"html": "<b>{town}</b><br>单价: S${avg_psm}/sqm<br>均价: S${avg_price}<br>交易量: {count}"}

    st.caption(f"圆圈大小 = 交易量 | 颜色深浅 = 均价 (浅→深: S${vmin:,.0f} → S${vmax:,.0f}/sqm)")
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style="light")


# ================== 2. HEATMAP ==================

def _build_heatmap(df: pd.DataFrame, overlays: list) -> pdk.Deck:
    st.sidebar.subheader("🔥 热力图参数")
    radius = st.sidebar.slider("热力半径 (像素)", 10, 100, 30, 5, key="heat_radius")
    intensity = st.sidebar.slider("强度", 0.1, 1.0, 0.5, 0.1, key="heat_intensity")

    rng = np.random.RandomState(42)
    heat_data = []
    for _, row in df.iterrows():
        town = row["town"]
        base_lat, base_lng = TOWN_COORDS.get(town, (CENTER_LAT, CENTER_LNG))
        lat = base_lat + rng.uniform(-0.008, 0.008)
        lng = base_lng + rng.uniform(-0.008, 0.008)
        weight = row["price_per_sqm"] if pd.notna(row.get("price_per_sqm")) else row["resale_price"]
        heat_data.append({"lat": lat, "lng": lng, "weight": float(weight)})

    layers = [
        pdk.Layer("HeatmapLayer", heat_data,
                  get_position=["lng", "lat"],
                  get_weight="weight",
                  radius_pixels=radius,
                  intensity=intensity,
                  threshold=0.05,
                  aggregation="MEAN"),
    ]
    _add_overlay_layers(layers, overlays)

    view = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LNG, zoom=13, pitch=0, bearing=0)
    return pdk.Deck(layers=layers, initial_view_state=view,
                    map_provider="carto", map_style="dark")


# ================== 3. TEMPORAL MAP ==================

def _build_temporal_map(df: pd.DataFrame, overlays: list) -> pdk.Deck:
    years = sorted(df["year"].unique())
    sel_year = st.sidebar.select_slider(
        "选择年份", options=years, value=years[-1], key="temp_year_slider",
    )

    year_df = df[df["year"] == sel_year]
    stats = year_df.groupby("town").agg(
        avg_price=("resale_price", "mean"), avg_psm=("price_per_sqm", "mean"),
        count=("resale_price", "count"),
    ).reset_index()

    # Use global color range for consistency across years
    vmin, vmax = GLOBAL_VMIN, GLOBAL_VMAX

    chart_data = []
    for _, row in stats.iterrows():
        town = row["town"]
        if town not in TOWN_COORDS:
            continue
        lat, lng = TOWN_COORDS[town]
        chart_data.append({
            "lat": lat, "lng": lng,
            "radius": max(600, row["count"] * 100),
            "color": _price_color_hex(row["avg_psm"], vmin, vmax),
            "town": town,
            "avg_psm": row["avg_psm"],
            "avg_price": row["avg_price"],
            "count": row["count"],
        })

    layers = [
        pdk.Layer("ScatterplotLayer", chart_data,
                  get_position=["lng", "lat"],
                  get_radius="radius",
                  get_fill_color="color",
                  pickable=True, auto_highlight=True,
                  opacity=0.85,
                  radius_scale=1,
                  radius_min_pixels=35,
                  radius_max_pixels=85),
    ]
    _add_overlay_layers(layers, overlays)

    view = pdk.ViewState(latitude=CENTER_LAT, longitude=CENTER_LNG, zoom=12.5, pitch=0)
    tooltip = {"html": f"<b>{{town}}</b> ({sel_year})<br>单价: S${{avg_psm}}/sqm<br>均价: S${{avg_price}}<br>交易量: {{count}}"}

    st.caption(f"({sel_year}) 颜色范围统一: S${vmin:,.0f} – S${vmax:,.0f}/sqm，便于跨年对比")
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip,
                    map_provider="carto", map_style="light"), sel_year



# ================== OVERLAY LAYERS ==================

def _add_overlay_layers(layers: list, overlays: list):
    """Add MRT and school overlay as separate ScatterplotLayers."""
    if "MRT/LRT 站点" in overlays:
        mrt_data = MRT_STATIONS.copy()
        mrt_data["color"] = mrt_data["type"].apply(
            lambda t: [220, 50, 50, 220] if t == "MRT" else [50, 180, 50, 200]
        )
        layers.append(
            pdk.Layer("ScatterplotLayer", mrt_data.to_dict("records"),
                      get_position=["lng", "lat"],
                      get_radius=180,
                      get_fill_color="color",
                      pickable=True,
                      radius_scale=1,
                      radius_min_pixels=8,
                      radius_max_pixels=14),
        )
    if "学校" in overlays:
        school_data = SCHOOLS.copy()
        school_data["color"] = school_data["type"].apply(
            lambda t: [50, 80, 200, 220] if t == "小学" else [180, 100, 50, 220]
        )
        layers.append(
            pdk.Layer("ScatterplotLayer", school_data.to_dict("records"),
                      get_position=["lng", "lat"],
                      get_radius=200,
                      get_fill_color="color",
                      pickable=True,
                      radius_scale=1,
                      radius_min_pixels=8,
                      radius_max_pixels=14),
        )


# ================== CHARTS ==================

def _render_temporal_chart(df: pd.DataFrame):
    st.subheader("📈 年度价格趋势")
    yearly = df.groupby(["year", "town"])["resale_price"].mean().reset_index()
    fig = px.line(
        yearly, x="year", y="resale_price", color="town",
        color_discrete_map=TOWN_COLORS, markers=True,
        labels={"year": "年份", "resale_price": "均价 (新元)", "town": "镇区"},
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    st.subheader("📊 年度涨跌幅 (%)")
    pivot = yearly.pivot(index="town", columns="year", values="resale_price")
    pct_change = pivot.pct_change(axis=1).dropna(axis=1, how="all") * 100
    fig2 = px.imshow(
        pct_change, text_auto=".1f", color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        labels=dict(x="年份", y="镇区", color="涨跌幅 (%)"),
    )
    fig2.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width='stretch')


def _render_town_stats_table(stats: pd.DataFrame):
    display = stats.copy()
    display["avg_price"] = display["avg_price"].apply(fmt_price)
    display["avg_psm"] = display["avg_psm"].apply(lambda x: f"{fmt_price(x)}/sqm")
    display["count"] = display["count"].apply(lambda x: f"{x:,}")
    st.dataframe(
        display.rename(columns={
            "town": "镇区", "avg_price": "均价", "avg_psm": "单价", "count": "交易量",
        }),
        width='stretch', hide_index=True,
    )
