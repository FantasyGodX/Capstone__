import plotly.graph_objects as go
import streamlit as st

from common import COLOR, ORDER, name, predictions, require_results, style_fig, table


def _regime_plot(df, kind: str, metric: str, title: str):
    fig = go.Figure()
    regs = sorted(df[kind].unique())
    for m in ORDER:
        g = df[df["model"] == m].set_index(kind).reindex(regs)
        fig.add_trace(go.Scatter(x=regs, y=g[metric], name=name(m), mode="markers+lines",
                                 marker=dict(size=11, color=COLOR[m], line=dict(color="white", width=2)),
                                 line=dict(color=COLOR[m], width=1.5)))
    return style_fig(fig, title, height=380, yfmt=".1%" if metric == "accuracy" else None).update_layout(hovermode="x unified")


def render():
    st.title("Robustness")
    st.caption("Regimes are defined ex-ante: bull/bear = SPY above/below its 200-day average; high/low vol = VIX above/below 20.")
    require_results()

    tabs = st.tabs(["Bull vs bear", "Volatility regimes", "By year", "By ticker"])
    for tab, kind, label in zip(tabs[:2], ["trend", "vol"], ["Bull vs bear", "Volatility regimes"]):
        with tab:
            df = table(f"regime_{kind}.csv")
            if df.empty:
                st.warning("Run `scripts/08_robustness.py` first.")
                continue
            metric = st.radio("Metric", ["accuracy", "sharpe", "mean_daily_ret"], horizontal=True, key=f"rm_{kind}")
            st.plotly_chart(_regime_plot(df, kind, metric, f"{metric} by regime"), width="stretch")
            st.dataframe(df.pivot(index="model", columns=kind, values=["n_days", "accuracy", "sharpe"]).reindex(ORDER).round(4), width="stretch")
            dm = table(f"regime_{kind}_dm_vs_logistic.csv")
            st.markdown("**Diebold-Mariano within regime (challenger vs Logistic, one-sided)**")
            st.dataframe(dm.style.format({c: "{:.4f}" for c in dm.columns if dm[c].dtype == float}), hide_index=True, width="stretch")

    with tabs[2]:
        p = predictions()
        a = p.groupby(["test_year", "model"])["correct"].mean().unstack("model")[ORDER]
        fig = go.Figure([go.Scatter(x=a.index, y=a[m], name=name(m), mode="markers+lines",
                                    marker=dict(size=9, color=COLOR[m]), line=dict(color=COLOR[m], width=1.8)) for m in ORDER])
        fig.add_hline(y=0.5, line_color="#898781", line_width=1)
        st.plotly_chart(style_fig(fig, "Accuracy by test year", yfmt=".1%").update_xaxes(dtick=1), width="stretch")
        st.dataframe(a.rename(columns=name).style.format("{:.3f}"), width="stretch")
    with tabs[3]:
        t = table("classification_by_ticker.csv")
        if not t.empty:
            metric = st.selectbox("Metric", ["accuracy", "f1", "roc_auc", "precision", "recall"], key="rt_metric")
            piv = t.pivot(index="ticker", columns="model", values=metric)[ORDER]
            piv.columns = [name(c) for c in piv.columns]
            fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Blues", text=piv.values.round(3),
                                       texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>"))
            st.plotly_chart(style_fig(fig, f"{metric} by ticker", height=400).update_layout(hovermode="closest"), width="stretch")
            best = piv.drop(columns="Buy & hold").idxmax(axis=1).value_counts()
            st.caption("Best non-benchmark model by ticker: " + ", ".join(f"{k} ({v})" for k, v in best.items()))
