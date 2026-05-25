"""HDB Resale Price Analysis & Prediction System
    Singapore HDB 组屋转售价格分析与预测系统
    Data: 2020–present, Punggol / Sengkang / Hougang
    网页：https://hdb-resale-analysis-mxz6kgxpljqn4ubkrqrvr5.streamlit.app/
"""

import importlib
import streamlit as st

from data.data_loader import load_data

st.set_page_config(
    page_title="HDB 转售价格分析系统",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import pages (filenames start with digits, use importlib)
_page_modules = {}
for _name, _file in [
    ("📊 数据概览", "01_overview"),
    ("🗺️ 地图可视化", "02_map"),
    ("📈 影响因素分析", "03_factors"),
    ("🤖 价格预测", "04_prediction"),
    ("💡 购房策略", "05_strategy"),
    ("💭 分析思考题", "06_questions"),
]:
    _page_modules[_name] = importlib.import_module(f"pages.{_file}")


def main():
    with st.sidebar:
        st.title("🏠 HDB 分析系统")
        st.markdown("---")
        page = st.radio("导航", list(_page_modules.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("数据来源: data.gov.sg")
        st.caption("镇区: Punggol / Sengkang / Hougang")
        st.caption("时间范围: 2020 年至今")

    with st.spinner("正在加载 HDB 转售数据…"):
        df = load_data()

    if df is None or df.empty:
        st.error("数据加载失败，请检查网络连接后刷新页面。")
        st.stop()

    _page_modules[page].run(df)


if __name__ == "__main__":
    main()
