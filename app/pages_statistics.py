import plotly.graph_objects as go
import streamlit as st

from common import COLOR, cfg, require_results, style_fig, table


def _forest(df, what: str, title: str):
    fig = go.Figure()
    for _, r in df.iterrows():
        label = f"{r.challenger} vs {r.baseline}"
        fig.add_trace(go.Scatter(x=[r[f"{what}_ci_low"], r[f"{what}_ci_high"]], y=[label, label], mode="lines",
                                 line=dict(color=COLOR[r.challenger], width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r[f"{what}_diff"]], y=[label], mode="markers", showlegend=False,
                                 marker=dict(size=12, color=COLOR[r.challenger], line=dict(color="white", width=2)),
                                 hovertemplate=f"{label}<br>diff %{{x:.4f}}<br>95% CI [{r[f'{what}_ci_low']:.4f}, {r[f'{what}_ci_high']:.4f}]<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_width=1)
    return style_fig(fig, title, height=380).update_layout(hovermode="closest")


def render():
    st.title("Statistical significance")
    st.caption("Challengers (RF, XGBoost, LSTM) vs classical baselines (Logistic, ARIMA). One-sided tests, H1: challenger is better. "
               "Holm correction is applied across the six comparisons.")
    require_results()
    sig = table("significance_vs_baselines.csv")
    if sig.empty:
        st.warning("Run `scripts/07_stat_tests.py` (or the Pipeline page) first.")
        return

    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, float(cfg()["stats"]["alpha"]), 0.01)

    def verdict(p):
        return "significant" if p < alpha else "not significant"

    view = sig[["challenger", "baseline", "acc_diff", "DM_p", "DM_p_holm", "boot_acc_p_holm", "ttest_p_holm",
                "sharpe_diff", "boot_sharpe_p", "boot_sharpe_p_holm"]].copy()
    view["accuracy verdict (DM, Holm)"] = view["DM_p_holm"].map(verdict)
    view["Sharpe verdict (Holm)"] = view["boot_sharpe_p_holm"].map(verdict)
    st.dataframe(view.style.format({c: "{:.4f}" for c in view.columns if view[c].dtype == float})
                 .map(lambda v: "background-color:#cfe9d9" if v == "significant" else "", subset=["accuracy verdict (DM, Holm)", "Sharpe verdict (Holm)"]),
                 hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    c1.plotly_chart(_forest(sig, "acc", "Accuracy difference (95% bootstrap CI)"), width="stretch")
    c2.plotly_chart(_forest(sig, "sharpe", "Sharpe difference (95% bootstrap CI)"), width="stretch")

    st.subheader("Sharpe vs buy-and-hold")
    sb = table("sharpe_vs_buy_and_hold.csv")
    if not sb.empty:
        st.dataframe(sb.style.format({c: "{:.4f}" for c in sb.columns if c != "model"}), hide_index=True, width="stretch")

    with st.expander("How to read this"):
        st.markdown(
            "- **DM** (Diebold-Mariano): compares 0/1 loss per day, averaged across tickers, with HAC variance.\n"
            "- **Bootstrap**: stationary block bootstrap over dates (mean block 10 days).\n"
            "- **t-test**: paired, on per-ticker per-year accuracies.\n"
            "- **Holm**: adjusts p-values for six simultaneous comparisons; use the `_holm` columns to decide.\n"
            "- A CI that includes 0 means the difference is not distinguishable from zero.\n"
            "- With ~1,250 test days, differences under about 1.5 percentage points of accuracy are undetectable."
        )
    st.download_button("Download significance table (CSV)", sig.to_csv(index=False).encode(), "significance_vs_baselines.csv")
