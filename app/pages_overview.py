import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (CLASS, COLOR, ORDER, TABLES, cfg, line_trace, name, require_results, style_fig, table)


def render():
    st.title("ML vs classical models: next-day stock direction")
    st.caption("Walk-forward evaluation on 8 liquid US equities, test years 2020-2024")
    require_results()

    m, bt, sig = table("classification_metrics.csv"), table("backtest_portfolio.csv"), table("significance_vs_baselines.csv")
    m, bt = m.set_index(m.columns[0]).reindex(ORDER), bt.set_index(bt.columns[0]).reindex(ORDER)
    tested = m.drop(index="AlwaysUp")

    c = st.columns(4)
    c[0].metric("Best accuracy (non-benchmark)", f"{tested['accuracy'].max():.2%}", name(tested["accuracy"].idxmax()))
    c[1].metric("Always-up accuracy", f"{m.loc['AlwaysUp', 'accuracy']:.2%}")
    c[2].metric("Best ROC-AUC", f"{tested['roc_auc'].max():.3f}", name(tested["roc_auc"].idxmax()))
    c[3].metric("Best Sharpe", f"{bt.drop(index='AlwaysUp')['sharpe'].max():.2f}", name(bt.drop(index='AlwaysUp')["sharpe"].idxmax()))

    if not sig.empty:
        n_acc = int((sig["DM_p_holm"] < cfg()["stats"]["alpha"]).sum())
        n_sh = int((sig["boot_sharpe_p_holm"] < cfg()["stats"]["alpha"]).sum())
        st.info(f"**Verdict from the generated tests:** {n_acc} of {len(sig)} accuracy comparisons and {n_sh} of "
                f"{len(sig)} Sharpe comparisons are significant at {cfg()['stats']['alpha']:.0%} after Holm correction "
                f"(challenger vs each classical baseline).")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Classification metrics")
        t = m.rename_axis("model").reset_index()
        t.insert(1, "class", t["model"].map(CLASS))
        t["model"] = t["model"].map(name)
        st.dataframe(t.style.format({k: "{:.3f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]} | {"n": "{:,.0f}"}),
                     hide_index=True, width="stretch")
    with right:
        st.subheader("Backtest (equal-weight, after costs)")
        b = bt.rename_axis("model").reset_index()
        b["model"] = b["model"].map(name)
        st.dataframe(b.style.format({"cum_return": "{:.0%}", "ann_return": "{:.1%}", "sharpe": "{:.2f}",
                                     "max_drawdown": "{:.1%}", "calmar": "{:.2f}"}), hide_index=True, width="stretch")

    port = pd.read_parquet(TABLES.parent / "backtest" / "daily_returns_portfolio.parquet")
    fig = go.Figure([line_trace(port.index, (1 + port[k]).cumprod() - 1, k) for k in ORDER])
    st.plotly_chart(style_fig(fig, "Cumulative return of the equal-weight portfolio", yfmt=".0%"), width="stretch")
    st.caption("Use the other pages to change models, tickers, dates and transaction costs, and to run side-by-side comparisons.")
