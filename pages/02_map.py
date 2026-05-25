"""Page 2: Map Visualization — town avg, heatmap, spatio-temporal, MRT/schools/mall stubs."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import branca.colormap as cmp
import folium
from folium.plugins import HeatMap, Fullscreen
from streamlit_folium import st_folium

from utils.helpers import TOWN_COORDS, TOWN_COLORS, fmt_price

# Extended coordinates for realistic heatmap dispersion (HDB cluster centroids)
TOWN_BLOCK_CLUSTERS = {
    "PUNGGOL": [
        (1.4043, 103.9028), (1.4065, 103.9010), (1.4020, 103.9045),
        (1.4050, 103.9055), (1.4070, 103.8995), (1.4015, 103.9070),
    ],
    "SENGKANG": [
        (1.3917, 103.8942), (1.3935, 103.8920), (1.3895, 103.8960),
        (1.3925, 103.8975), (1.3945, 103.8950), (1.3900, 103.8915),
    ],
    "HOUGANG": [
        (1.3714, 103.8923), (1.3730, 103.8895), (1.3695, 103.8945),
        (1.3725, 103.8960), (1.3745, 103.8915), (1.3685, 103.8900),
    ],
}

CENTER = (1.3891, 103.8964)


def run(df: pd.DataFrame):
    st.title("🗺️ 地图可视化")
    st.markdown("镇区均价、热力分布、时空变化 — 含 MRT/学校/商场叠加图层接口。")

    # ---- Sidebar Controls ----
    st.sidebar.subheader("🗺️ 地图设置")
    map_mode = st.sidebar.radio(
        "视图模式",
        ["镇区均价", "热力分布", "时空变化"],
        key="map_mode",
    )

    # Common filters
    sel_types = st.sidebar.multiselect(
        "房型筛选 (地图)", sorted(df["flat_type"].unique()),
        default=sorted(df["flat_type"].unique()),
        key="map_type_filter",
    )
    filtered = df[df["flat_type"].isin(sel_types)]

    # Overlay toggles
    st.sidebar.subheader("🗂️ 叠加图层 (接口预留)")
    show_mrt = st.sidebar.checkbox("MRT 站点", value=False,
                                    help="LTA DataMall API — 待接入")
    show_schools = st.sidebar.checkbox("学校", value=False,
                                        help="data.gov.sg School API — 待接入")
    show_malls = st.sidebar.checkbox("商场", value=False,
                                      help="OSM Overpass / OneMap — 待接入")

    # ---- Render Map ----
    if map_mode == "镇区均价":
        m = _build_town_avg_map(filtered)
    elif map_mode == "热力分布":
        m = _build_heatmap(filtered)
    else:
        m, year = _build_temporal_map(filtered)

    # Attach overlay stubs
    towns = sorted(filtered["town"].unique())
    if show_mrt:
        add_mrt_layer(m, towns)
    if show_schools:
        add_school_layer(m, towns)
    if show_malls:
        add_mall_layer(m, towns)

    Fullscreen().add_to(m)
    st_folium(m, width=None, height=550, returned_objects=[])

    # Below-map chart for temporal mode
    if map_mode == "时空变化":
        _render_temporal_chart(filtered)


# ======================== 1. TOWN AVG MAP ========================

def _build_town_avg_map(df: pd.DataFrame) -> folium.Map:
    """CircleMarkers sized by volume, colored by avg price, with legend."""
    stats = df.groupby("town").agg(
        avg_price=("resale_price", "mean"),
        median_price=("resale_price", "median"),
        avg_psm=("price_per_sqm", "mean"),
        count=("resale_price", "count"),
    ).reset_index()

    vmin, vmax = stats["avg_price"].min(), stats["avg_price"].max()
    colormap = cmp.linear.YlOrRd_09.scale(vmin, vmax)

    m = folium.Map(location=CENTER, zoom_start=13, tiles="CartoDB Positron")

    for _, row in stats.iterrows():
        town = row["town"]
        if town not in TOWN_COORDS:
            continue
        lat, lng = TOWN_COORDS[town]
        color = colormap(row["avg_price"])
        radius = max(14, min(36, row["count"] / 60))

        folium.CircleMarker(
            location=[lat, lng], radius=radius, color=color,
            fill=True, fill_opacity=0.85, weight=2,
            popup=folium.Popup(
                f"<b>{town}</b><br>"
                f"均价: {fmt_price(row['avg_price'])}<br>"
                f"中位数: {fmt_price(row['median_price'])}<br>"
                f"单价: {fmt_price(row['avg_psm'])}/sqm<br>"
                f"交易量: {row['count']:,}",
                max_width=220,
            ),
        ).add_to(m)

    # Color legend via HTML
    _add_color_legend(m, colormap, vmin, vmax, "均价 (新元)")

    st.caption(f"圆圈大小 = 交易量 | 颜色深度 = 均价 (浅→深: {fmt_price(vmin)} → {fmt_price(vmax)})")
    _render_town_stats_table(stats)
    return m


# ======================== 2. HEATMAP ========================

def _build_heatmap(df: pd.DataFrame) -> folium.Map:
    """Heatmap weighted by price_per_sqm, using dispersed cluster coordinates."""
    rng = np.random.RandomState(42)
    heat_data = []

    for _, row in df.iterrows():
        town = row["town"]
        clusters = TOWN_BLOCK_CLUSTERS.get(town, [TOWN_COORDS.get(town, CENTER)])
        base_lat, base_lng = clusters[rng.randint(0, len(clusters))]
        # Jitter within ~300m
        lat = base_lat + rng.uniform(-0.003, 0.003)
        lng = base_lng + rng.uniform(-0.003, 0.003)
        weight = row["price_per_sqm"] if pd.notna(row["price_per_sqm"]) else row["resale_price"]
        heat_data.append([lat, lng, weight])

    m = folium.Map(location=CENTER, zoom_start=13, tiles="CartoDB Positron")

    HeatMap(
        heat_data, radius=18, blur=12, max_zoom=14, min_opacity=0.3,
        gradient={0.2: "#313695", 0.4: "#4575b4", 0.6: "#fdae61", 0.8: "#f46d43", 1.0: "#a50026"},
        name="价格热力",
    ).add_to(m)

    # Town labels
    for town, (lat, lng) in TOWN_COORDS.items():
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#fff;'
                     f'background:rgba(30,30,30,0.75);padding:2px 8px;border-radius:3px;">{town}</div>'
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ======================== 3. SPATIO-TEMPORAL ========================

def _build_temporal_map(df: pd.DataFrame) -> tuple:
    """Year-slider map: CircleMarkers update by year; returns (map, selected_year)."""
    years = sorted(df["year"].unique())
    sel_year = st.sidebar.selectbox("选择年份", years, index=len(years) - 1, key="temp_year")

    year_df = df[df["year"] == sel_year]
    stats = year_df.groupby("town").agg(
        avg_price=("resale_price", "mean"),
        avg_psm=("price_per_sqm", "mean"),
        count=("resale_price", "count"),
    ).reset_index()

    # Compute YoY change for color coding
    prev_df = df[df["year"] == (sel_year - 1)] if (sel_year - 1) in years else None
    town_changes = {}
    if prev_df is not None and not prev_df.empty:
        prev_stats = prev_df.groupby("town")["resale_price"].mean()
        for town in stats["town"]:
            if town in prev_stats.index and prev_stats[town] > 0:
                pct = (stats.loc[stats["town"] == town, "avg_price"].values[0] - prev_stats[town]) / prev_stats[town]
                town_changes[town] = pct

    vmin = stats["avg_price"].min()
    vmax = stats["avg_price"].max()
    colormap = cmp.linear.RdYlGn_11.scale(vmin, vmax)

    m = folium.Map(location=CENTER, zoom_start=13, tiles="CartoDB Positron")

    for _, row in stats.iterrows():
        town = row["town"]
        if town not in TOWN_COORDS:
            continue
        lat, lng = TOWN_COORDS[town]
        color = colormap(row["avg_price"])
        radius = max(16, min(36, row["count"] / 40))

        change_str = ""
        if town in town_changes:
            direction = "↑" if town_changes[town] > 0 else "↓"
            change_str = f"<br>同比: {direction}{abs(town_changes[town])*100:.1f}%"

        folium.CircleMarker(
            location=[lat, lng], radius=radius, color=color,
            fill=True, fill_opacity=0.85, weight=2,
            popup=folium.Popup(
                f"<b>{town}</b> ({sel_year})<br>"
                f"均价: {fmt_price(row['avg_price'])}<br>"
                f"单价: {fmt_price(row['avg_psm'])}/sqm<br>"
                f"交易量: {row['count']:,}"
                + change_str,
                max_width=220,
            ),
        ).add_to(m)

    _add_color_legend(m, colormap, vmin, vmax, f"{sel_year} 均价 (新元)")
    return m, sel_year


def _render_temporal_chart(df: pd.DataFrame):
    """Below-map trend chart for temporal mode context."""
    st.subheader("📈 年度价格趋势")
    yearly = df.groupby(["year", "town"])["resale_price"].mean().reset_index()
    fig = px.line(
        yearly, x="year", y="resale_price", color="town",
        color_discrete_map=TOWN_COLORS, markers=True,
        labels={"year": "年份", "resale_price": "均价 (新元)", "town": "镇区"},
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    # Year-over-year change heatmap
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


# ==================== OVERLAY STUBS ====================

def add_mrt_layer(m: folium.Map, towns: list):
    """叠加 MRT/LRT 站点图层 (接口预留).

    Args:
        m: folium.Map 实例，直接在其上添加图层。
        towns: 当前选中的镇区列表，用于过滤附近站点。

    数据源: LTA DataMall (https://datamall.lta.gov.sg/)
    接入方式:
      1. 注册 LTA DataMall 账号获取 API Key
      2. 调用 /ltaodataservice/TrainStation 获取 MRT/LRT 站点列表
      3. 按经纬度筛选 towns 范围内的站点 (bounding box ±0.02°)
      4. 以 folium.Marker(icon=folium.Icon(icon='train', prefix='fa')) 添加
    """
    pass


def add_school_layer(m: folium.Map, towns: list):
    """叠加学校图层 — 小学/中学/初院 (接口预留).

    Args:
        m: folium.Map 实例。
        towns: 当前选中的镇区列表。

    数据源: data.gov.sg — School Directory and Information
    API: https://data.gov.sg/api/action/datastore_search?resource_id=d_688b934f82c1059ed0a6993d2a829089
    接入方式:
      1. 调用 API 获取学校名称、地址、经纬度
      2. 按 towns 筛选 (通过地址或 planning_area 字段)
      3. 以 folium.Marker(icon=folium.Icon(icon='graduation-cap', prefix='fa', color='blue')) 添加
      4. popup 显示学校名称、类型、地址
    """
    pass


def add_mall_layer(m: folium.Map, towns: list):
    """叠加商场图层 (接口预留).

    Args:
        m: folium.Map 实例。
        towns: 当前选中的镇区列表。

    数据源: OpenStreetMap Overpass API / OneMap API
    接入方式 (OSM Overpass):
      1. 构造 Overpass QL: node[shop=mall] / way[shop=mall] / node[shop=shopping_centre]
      2. 按 towns bounding box 筛选
      3. 以 folium.Marker(icon=folium.Icon(icon='shopping-cart', prefix='fa', color='red')) 添加
      4. popup 显示商场名称

      OSM Overpass endpoint: https://overpass-api.de/api/interpreter
      OneMap API: https://www.onemap.gov.sg/docs/
    """
    pass


# ==================== HELPERS ====================

def _add_color_legend(m: folium.Map, colormap, vmin: float, vmax: float, title: str):
    """Add an HTML color legend bar to the folium map (bottom-right)."""
    steps = 5
    html = (
        f'<div style="position:fixed;bottom:30px;right:12px;z-index:9999;'
        f'background:white;padding:8px 10px;border-radius:6px;'
        f'box-shadow:0 1px 6px rgba(0,0,0,0.2);font-size:12px;font-family:sans-serif;">'
        f'<b>{title}</b><br>'
    )
    for i in range(steps):
        ratio = i / (steps - 1)
        val = vmin + (vmax - vmin) * ratio
        color = colormap(val)
        html += (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{color};border-radius:2px;margin-right:6px;"></span>'
            f'{fmt_price(val)}<br>'
        )
    html += "</div>"
    m.get_root().html.add_child(folium.Element(html))


def _render_town_stats_table(stats: pd.DataFrame):
    """Display a town statistics table below the map."""
    display = stats.copy()
    display["avg_price"] = display["avg_price"].apply(fmt_price)
    display["median_price"] = display["median_price"].apply(fmt_price)
    display["avg_psm"] = display["avg_psm"].apply(lambda x: f"{fmt_price(x)}/sqm")
    display["count"] = display["count"].apply(lambda x: f"{x:,}")
    st.dataframe(
        display.rename(columns={
            "town": "镇区", "avg_price": "均价", "median_price": "中位数",
            "avg_psm": "单价", "count": "交易量",
        }),
        width='stretch', hide_index=True,
    )
