import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from capstone.backtest import backtest_all, performance
from common import ORDER, cfg, line_trace, model_picker, name, predictions, require_results, style_fig


@st.cache_data(show_spinner=False)
def _run(cost: float, tickers: tuple):
    p = predictions()
    p = p[p["ticker"].isin(tickers)]
    return backtest_all(p, cost, cfg()["backtest"]["trading_days"])


def render():
    st.title("Backtest")
    st.caption("Long if the model predicts up, cash otherwise. No leverage. Cost is charged per unit of position change.")
    require_results()
    p = predictions()

    with st.sidebar:
        st.header("Backtest settings")
        cost_bps = st.slider("Transaction cost (bps per trade)", 0.0, 50.0, cfg()["backtest"]["transaction_cost"] * 1e4, 0.5)
        tickers = st.multiselect("Tickers in portfolio", sorted(p["ticker"].unique()), default=sorted(p["ticker"].unique()))
        models = model_picker("bt_models")
        dmin, dmax = p["date"].min().date(), p["date"].max().date()
        start, end = st.slider("Date range", dmin, dmax, (dmin, dmax))
    if not tickers or not models:
        st.warning("Pick at least one ticker and one model.")
        return

    wide, port, per_ticker, _ = _run(cost_bps / 1e4, tuple(tickers))
    port = port.loc[pd.Timestamp(start):pd.Timestamp(end), [m for m in ORDER if m in models]]
    td = cfg()["backtest"]["trading_days"]

    stats = pd.DataFrame({name(m): performance(port[m], td) for m in port}).T
    st.subheader("Portfolio metrics")
    st.dataframe(stats.style.format({"cum_return": "{:.0%}", "ann_return": "{:.1%}", "sharpe": "{:.2f}",
                                     "max_drawdown": "{:.1%}", "calmar": "{:.2f}"}), width="stretch")

    eq = (1 + port).cumprod()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([line_trace(eq.index, eq[m] - 1, m) for m in eq])
        st.plotly_chart(style_fig(fig, "Cumulative return", yfmt=".0%"), width="stretch")
    with c2:
        fig = go.Figure([line_trace(eq.index, eq[m] / eq[m].cummax() - 1, m) for m in eq])
        st.plotly_chart(style_fig(fig, "Drawdown", yfmt=".0%"), width="stretch")

    st.subheader("Per-ticker results")
    metric = st.selectbox("Metric", ["sharpe", "cum_return", "max_drawdown", "calmar", "ann_return"])
    pt = per_ticker[metric].unstack(0).reindex(columns=[m for m in ORDER if m in models])
    pt.columns = [name(c) for c in pt.columns]
    fig = go.Figure(go.Heatmap(z=pt.values, x=pt.columns, y=pt.index, colorscale="RdBu", zmid=0 if metric != "calmar" else None,
                               text=pt.values.round(2), texttemplate="%{text}",
                               hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
    st.plotly_chart(style_fig(fig, f"{metric} by ticker (full test period)", height=380).update_layout(hovermode="closest"), width="stretch")
    with st.expander("Full per-ticker table"):
        st.dataframe(per_ticker.round(4), width="stretch")
    st.download_button("Download portfolio daily returns (CSV)", port.to_csv().encode(), "portfolio_daily_returns.csv")
