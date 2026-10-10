import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from capstone.features import feature_columns
from common import DATA_PROCESSED, DATA_RAW, INK, MUTED, dataset, prices, style_fig


def render():
    st.title("Data explorer")
    if not (DATA_RAW / "prices.db").exists() or not (DATA_PROCESSED / "dataset.parquet").exists():
        st.warning("No data yet. Open the **Pipeline** page and run steps 1 and 2.")
        return

    px, ds = prices(), dataset()
    tickers = sorted(ds.index.get_level_values("ticker").unique())
    tab_px, tab_feat, tab_tbl = st.tabs(["Prices", "Features", "Dataset table"])

    with tab_px:
        t = st.selectbox("Ticker", tickers + ["SPY", "^VIX"])
        df = px[t]
        lo, hi = df.index.min().date(), df.index.max().date()
        start, end = st.slider("Date range", lo, hi, (lo, hi), key="px_range")
        df = df.loc[str(start):str(end)]
        overlays = st.multiselect("Overlays", ["MA 20", "MA 50", "MA 200", "Bollinger (20, 2)"], default=["MA 50"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                     name=t, increasing_line_color="#1baf7a", decreasing_line_color="#e34948"), row=1, col=1)
        full = px[t]["Close"]
        for o in overlays:
            if o.startswith("MA"):
                w = int(o.split()[1])
                fig.add_trace(go.Scatter(x=df.index, y=full.rolling(w).mean().loc[df.index], name=o, mode="lines", line=dict(width=1.6)), row=1, col=1)
            else:
                m, s = full.rolling(20).mean(), full.rolling(20).std()
                for k, nm in ((m + 2 * s, "BB upper"), (m - 2 * s, "BB lower")):
                    fig.add_trace(go.Scatter(x=df.index, y=k.loc[df.index], name=nm, mode="lines", line=dict(width=1, dash="dot", color="#898781")), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#898781", showlegend=False), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(style_fig(fig, f"{t} (adjusted prices)", height=560), width="stretch")

    with tab_feat:
        feats = feature_columns(ds)
        c1, c2 = st.columns([1, 2])
        with c1:
            tk = st.multiselect("Tickers", tickers, default=tickers, key="feat_tk")
            feat = st.selectbox("Feature", feats)
        d = ds[ds.index.get_level_values("ticker").isin(tk)]
        with c2:
            fig = go.Figure()
            for lab, col, nm in ((0, "#e34948", "Next day down"), (1, "#1baf7a", "Next day up")):
                x = d.loc[d["y"] == lab, feat]
                lo, hi = x.quantile(0.005), x.quantile(0.995)
                fig.add_trace(go.Histogram(x=x.clip(lo, hi), name=nm, opacity=0.6, marker_color=col, histnorm="probability density", nbinsx=60))
            fig.update_layout(barmode="overlay")
            st.plotly_chart(style_fig(fig, f"{feat}: distribution by next-day direction (clipped at 0.5/99.5 pct)", height=360).update_layout(hovermode="closest"), width="stretch")
        corr = d[feats].corrwith(d["y"]).sort_values()
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color="#2a78d6",
                               hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        st.plotly_chart(style_fig(fig, "Correlation of each feature with next-day direction", height=620).update_layout(hovermode="closest"), width="stretch")
        st.caption("Correlations are tiny (|r| < 0.05 is typical): this is why next-day direction is hard to predict.")

    with tab_tbl:
        tk = st.selectbox("Ticker", tickers, key="tbl_tk")
        sub = ds.xs(tk, level="ticker")
        st.dataframe(sub.round(5), width="stretch", height=450)
        st.download_button("Download this ticker's dataset (CSV)", sub.to_csv().encode(), f"{tk}_dataset.csv")
        st.write(f"{len(sub):,} rows, {len(feature_columns(ds))} features. Up-day share: {sub['y'].mean():.1%}")
