"""Shared loaders, model styling and Plotly theme for the dashboard."""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capstone.config import DATA_PROCESSED, DATA_RAW, RESULTS, load_config  # noqa: E402

ORDER = ["AlwaysUp", "Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]
LABEL = {"AlwaysUp": "Buy & hold"}
CLASS = {"AlwaysUp": "Benchmark", "Logistic": "Classical", "ARIMA": "Classical",
         "RandomForest": "ML", "XGBoost": "ML", "LSTM": "Deep learning"}
COLOR = {"AlwaysUp": "#898781", "Logistic": "#4a95f0", "ARIMA": "#eb6834",
         "RandomForest": "#1baf7a", "XGBoost": "#e87ba4", "LSTM": "#8b7bff"}
DASH = {"AlwaysUp": "solid", "Logistic": "solid", "ARIMA": "dash", "RandomForest": "dashdot",
        "XGBoost": "dot", "LSTM": "solid"}
PALETTES = {
    "dark": dict(ink="#ececea", muted="#a9a8a3", grid="#2e3138", hover="#1a1d24"),
    "light": dict(ink="#0b0b0b", muted="#52514e", grid="#e1e0d9", hover="#ffffff"),
}


def palette() -> dict:
    """Plot colors for the theme currently selected in Streamlit's Settings menu."""
    try:
        kind = st.context.theme.type
    except Exception:
        kind = "dark"
    return PALETTES.get(kind, PALETTES["dark"])


TABLES = RESULTS / "tables"


GLOBAL_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
h1 {font-size: 2rem !important; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetricLabel"] p {font-size: 0.85rem;}
hr {margin: 1.2rem 0;}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def notes_path() -> Path:
    """Resolved on every call so the CAPSTONE_NOTES override (used by tests) always applies."""
    return Path(os.environ.get("CAPSTONE_NOTES", ROOT / "analysis" / "notes.md"))


def name(m: str) -> str:
    return LABEL.get(m, m)


def results_ready() -> bool:
    return (RESULTS / "predictions").exists() and any((RESULTS / "predictions").glob("*.parquet"))


def require_results():
    if not results_ready():
        st.warning("No results found yet. Open the **Pipeline** page and run the full pipeline first.")
        st.stop()


@st.cache_data(show_spinner=False)
def cfg() -> dict:
    return load_config()


@st.cache_data(show_spinner=False)
def predictions() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((RESULTS / "predictions").glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["correct"] = (df["pred"] == df["y_true"]).astype(float)
    return df


@st.cache_data(show_spinner=False)
def dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "dataset.parquet")


@st.cache_data(show_spinner=False)
def prices() -> dict:
    from capstone.data import load_prices
    return load_prices(DATA_RAW / "prices.db")


@st.cache_data(show_spinner=False)
def table(fname: str) -> pd.DataFrame:
    p = TABLES / fname
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def style_fig(fig: go.Figure, title: str | None = None, height: int = 430, yfmt: str | None = None) -> go.Figure:
    pal = palette()
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=pal["ink"])) if title else None,
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["muted"], size=12),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color=pal["ink"])),
        hoverlabel=dict(bgcolor=pal["hover"], font_color=pal["ink"]),
    )
    fig.update_xaxes(gridcolor=pal["grid"], linecolor=pal["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=pal["grid"], linecolor=pal["grid"], zerolinecolor=pal["muted"], tickformat=yfmt)
    return fig


def line_trace(x, y, m: str, **kw) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name(m), mode="lines",
                      line=dict(color=COLOR[m], dash=DASH[m], width=1.6 if m == "AlwaysUp" else 2.2), **kw)


def model_picker(key: str, default=None, label="Models") -> list[str]:
    return st.multiselect(label, ORDER, default=default or ORDER, format_func=name, key=key)
