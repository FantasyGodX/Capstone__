"""Capstone dashboard. Run with:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(page_title="Capstone: ML vs classical", page_icon="📈", layout="wide")

import pages_backtest, pages_compare, pages_overview, pages_performance, pages_robustness, pages_statistics  # noqa: E402,E401

PAGES = [
    st.Page(pages_overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(pages_performance.render, title="Performance", icon="🎯", url_path="performance"),
    st.Page(pages_backtest.render, title="Backtest", icon="💰", url_path="backtest"),
    st.Page(pages_statistics.render, title="Statistics", icon="📐", url_path="statistics"),
    st.Page(pages_robustness.render, title="Robustness", icon="🧪", url_path="robustness"),
    st.Page(pages_compare.render, title="Compare", icon="⚖️", url_path="compare"),
]

st.navigation(PAGES).run()
