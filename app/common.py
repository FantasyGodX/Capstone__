"""Shared loaders, model styling and Plotly theme for the dashboard."""
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
COLOR = {"AlwaysUp": "#898781", "Logistic": "#2a78d6", "ARIMA": "#eb6834",
         "RandomForest": "#1baf7a", "XGBoost": "#e87ba4", "LSTM": "#6250d6"}
DASH = {"AlwaysUp": "solid", "Logistic": "solid", "ARIMA": "dash", "RandomForest": "dashdot",
        "XGBoost": "dot", "LSTM": "solid"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
TABLES = RESULTS / "tables"


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
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=INK)) if title else None,
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, size=12),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color=INK)),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=MUTED, tickformat=yfmt)
    return fig


def line_trace(x, y, m: str, **kw) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name(m), mode="lines",
                      line=dict(color=COLOR[m], dash=DASH[m], width=1.6 if m == "AlwaysUp" else 2.2), **kw)


def model_picker(key: str, default=None, label="Models") -> list[str]:
    return st.multiselect(label, ORDER, default=default or ORDER, format_func=name, key=key)
