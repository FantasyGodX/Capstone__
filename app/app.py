"""Capstone dashboard. Run with:  streamlit run app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(page_title="Capstone: ML vs classical", page_icon="📈", layout="wide")

import pages_overview  # noqa: E402

PAGES = [
    st.Page(pages_overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
]

st.navigation(PAGES).run()
