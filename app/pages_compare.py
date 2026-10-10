import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest

from capstone.stats import bootstrap_diff, diebold_mariano, sharpe
from common import COLOR, ORDER, cfg, line_trace, name, predictions, require_results, style_fig
from pages_backtest import _run


def render():
    st.title("Compare two models")
    st.caption("Head-to-head with live statistical tests. H1 for every one-sided test: model A is better than model B.")
    require_results()
    p = predictions()

    c = st.columns(4)
    a = c[0].selectbox("Model A", ORDER, index=ORDER.index("LSTM"), format_func=name)
    b = c[1].selectbox("Model B", ORDER, index=ORDER.index("Logistic"), format_func=name)
    tickers = c[2].multiselect("Tickers", sorted(p["ticker"].unique()), default=sorted(p["ticker"].unique()))
    years = c[3].multiselect("Test years", sorted(p["test_year"].unique()), default=sorted(p["test_year"].unique()))
    if a == b or not tickers or not years:
        st.warning("Choose two different models, and at least one ticker and year.")
        return

    f = p[p["ticker"].isin(tickers) & p["test_year"].isin(years)]
    wa, wb = (f[f["model"] == m].set_index(["ticker", "date"]) for m in (a, b))
    j = wa[["correct", "pred", "score", "test_year"]].join(wb[["correct", "pred", "score"]], lsuffix="_a", rsuffix="_b", how="inner")

    acc_a, acc_b = j["correct_a"].mean(), j["correct_b"].mean()
    only_a = int(((j["correct_a"] == 1) & (j["correct_b"] == 0)).sum())
    only_b = int(((j["correct_a"] == 0) & (j["correct_b"] == 1)).sum())
    m = st.columns(5)
    m[0].metric(f"{name(a)} accuracy", f"{acc_a:.2%}")
    m[1].metric(f"{name(b)} accuracy", f"{acc_b:.2%}")
    m[2].metric("Difference (A-B)", f"{(acc_a - acc_b) * 100:+.2f} pp")
    m[3].metric("Predictions agree", f"{(j['pred_a'] == j['pred_b']).mean():.1%}")
    m[4].metric("Score correlation", f"{j['score_a'].corr(j['score_b'], method='spearman'):.2f}" if j["score_a"].nunique() > 1 and j["score_b"].nunique() > 1 else "n/a")

    st.subheader("Statistical tests")
    dates = j.groupby(level="date")[["correct_a", "correct_b"]].mean()
    dm, p_dm = diebold_mariano(1 - dates["correct_b"].values, 1 - dates["correct_a"].values)
    mc = binomtest(only_a, only_a + only_b, 0.5, alternative="greater").pvalue if only_a + only_b else np.nan
    n_boot = st.select_slider("Bootstrap draws", [500, 1000, 2000, 5000], value=1000)
    rng = np.random.default_rng(cfg()["seed"])
    d, lo, hi, p_b = bootstrap_diff(dates["correct_a"].values, dates["correct_b"].values, lambda x: x.mean(-1), n_boot, 10, rng)

    cost = cfg()["backtest"]["transaction_cost"]
    _, port, _, _ = _run(cost, tuple(tickers))
    port = port.loc[dates.index.min():dates.index.max()]
    port = port[port.index.isin(dates.index)]
    sd, slo, shi, sp = bootstrap_diff(port[a].values, port[b].values, lambda x: sharpe(x, trading_days=cfg()["backtest"]["trading_days"]), n_boot, 10, rng)
    tests = pd.DataFrame([
        {"test": "Diebold-Mariano (0/1 loss)", "statistic": dm, "A - B": d, "95% CI low": np.nan, "95% CI high": np.nan, "p (one-sided)": p_dm},
        {"test": "Bootstrap, accuracy", "statistic": np.nan, "A - B": d, "95% CI low": lo, "95% CI high": hi, "p (one-sided)": p_b},
        {"test": "Exact sign test on discordant pairs", "statistic": np.nan, "A - B": (only_a - only_b) / len(j), "95% CI low": np.nan, "95% CI high": np.nan, "p (one-sided)": mc},
        {"test": "Bootstrap, Sharpe (portfolio)", "statistic": np.nan, "A - B": sd, "95% CI low": slo, "95% CI high": shi, "p (one-sided)": sp},
    ])
    st.dataframe(tests.style.format({c: "{:.4f}" for c in tests.columns if c != "test"}, na_rep="-"), hide_index=True, width="stretch")
    st.caption("These are single unadjusted tests. If you look at many pairs, correct for multiple comparisons (the Statistics page applies Holm).")

    st.subheader("Where do they differ?")
    c1, c2 = st.columns(2)
    with c1:
        cum = (dates["correct_a"] - dates["correct_b"]).cumsum()
        fig = go.Figure(go.Scatter(x=cum.index, y=cum, mode="lines", line=dict(color=COLOR[a], width=2), name=f"{name(a)} - {name(b)}"))
        fig.add_hline(y=0, line_color="#52514e", line_width=1)
        st.plotly_chart(style_fig(fig, "Cumulative extra correct calls per ticker-day (A - B)"), width="stretch")
    with c2:
        eq = (1 + port[[a, b]]).cumprod() - 1
        fig = go.Figure([line_trace(eq.index, eq[k], k) for k in (a, b)])
        st.plotly_chart(style_fig(fig, "Backtest cumulative return", yfmt=".0%"), width="stretch")

    c3, c4 = st.columns(2)
    for col, key, ttl in ((c3, "ticker", "Accuracy difference by ticker (pp)"), (c4, "test_year", "Accuracy difference by year (pp)")):
        g = (j["correct_a"] - j["correct_b"]).groupby(j.index.get_level_values("ticker") if key == "ticker" else j["test_year"]).mean() * 100
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=[COLOR[a] if v >= 0 else COLOR[b] for v in g.values],
                               hovertemplate="%{x}: %{y:+.2f} pp<extra></extra>"))
        fig.add_hline(y=0, line_color="#52514e", line_width=1)
        col.plotly_chart(style_fig(fig, ttl, height=360).update_layout(hovermode="closest"), width="stretch")
    st.caption(f"Bars in {name(a)}'s colour: A ahead. In {name(b)}'s colour: B ahead.")
    st.markdown(f"**Discordant calls:** only {name(a)} right on {only_a:,} ticker-days, only {name(b)} right on {only_b:,}.")
