import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

from capstone.evaluation import metrics_table
from common import COLOR, ORDER, line_trace, model_picker, name, predictions, require_results, style_fig


def render():
    st.title("Predictive performance")
    require_results()
    p = predictions()

    with st.sidebar:
        st.header("Filters")
        models = model_picker("perf_models")
        tickers = st.multiselect("Tickers", sorted(p["ticker"].unique()), default=sorted(p["ticker"].unique()))
        years = st.multiselect("Test years", sorted(p["test_year"].unique()), default=sorted(p["test_year"].unique()))
    f = p[p["model"].isin(models) & p["ticker"].isin(tickers) & p["test_year"].isin(years)]
    if f.empty:
        st.warning("Nothing selected.")
        return

    st.subheader("Metrics for the current selection")
    t = metrics_table(f).reindex([m for m in ORDER if m in models])
    t.index = [name(i) for i in t.index]
    st.dataframe(t.style.format({k: "{:.3f}" for k in t.columns if k != "n"} | {"n": "{:,.0f}"}), width="stretch")

    metric = st.radio("Compare on", ["accuracy", "precision", "recall", "f1", "roc_auc"], horizontal=True)
    tt = metrics_table(f)[metric].reindex([m for m in ORDER if m in models])
    fig = go.Figure(go.Scatter(x=tt.values, y=[name(m) for m in tt.index], mode="markers",
                               marker=dict(size=14, color=[COLOR[m] for m in tt.index], line=dict(color="white", width=2)),
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0.5, line_color="#52514e", line_width=1, annotation_text="0.5", annotation_position="top")
    st.plotly_chart(style_fig(fig, f"{metric} by model", height=320).update_layout(hovermode="closest"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m in [m for m in ORDER if m in models and m != "AlwaysUp"]:
            g = f[f["model"] == m]
            fpr, tpr, _ = roc_curve(g["y_true"], g["score"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name(m), mode="lines", line=dict(color=COLOR[m], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", mode="lines", line=dict(color="#898781", dash="dot", width=1)))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        st.plotly_chart(style_fig(fig, "ROC curves (pooled)").update_layout(hovermode="closest"), width="stretch")
    with c2:
        win = st.slider("Rolling window (trading days)", 20, 250, 60, 10)
        daily = f.groupby(["model", "date"])["correct"].mean().unstack("model")
        fig = go.Figure([line_trace(daily.index, daily[m].rolling(win).mean(), m) for m in ORDER if m in daily])
        fig.add_hline(y=0.5, line_color="#898781", line_width=1)
        st.plotly_chart(style_fig(fig, f"Rolling {win}-day accuracy", yfmt=".0%"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        piv = f.groupby(["ticker", "model"])["correct"].mean().unstack("model")[[m for m in ORDER if m in models]]
        piv.columns = [name(c) for c in piv.columns]
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Blues", zmid=0.5,
                                   text=np.round(piv.values, 3), texttemplate="%{text}",
                                   hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
        st.plotly_chart(style_fig(fig, "Accuracy by ticker and model", height=380).update_layout(hovermode="closest"), width="stretch")
    with c4:
        m = st.selectbox("Confusion matrix for", models, format_func=name)
        g = f[f["model"] == m]
        cm = confusion_matrix(g["y_true"], g["pred"], labels=[0, 1])
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred down", "Pred up"], y=["Actual down", "Actual up"], colorscale="Blues",
                                   text=cm, texttemplate="%{text:,}", showscale=False))
        st.plotly_chart(style_fig(fig, f"Confusion matrix: {name(m)}", height=380).update_layout(hovermode="closest"), width="stretch")
