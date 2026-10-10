"""Capstone dashboard. Run with:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(page_title="Capstone: ML vs classical", page_icon="📈", layout="wide")

import pages_backtest, pages_overview, pages_performance  # noqa: E402,E401

PAGES = [
    st.Page(pages_overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(pages_performance.render, title="Performance", icon="🎯", url_path="performance"),
    st.Page(pages_backtest.render, title="Backtest", icon="💰", url_path="backtest"),
]

st.navigation(PAGES).run()
