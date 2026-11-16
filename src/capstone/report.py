"""Shared data loading, extra analyses and print-quality figures for the paper and poster.

Everything is computed from results/ and data/ so the written report never drifts from the pipeline output.
"""
import matplotlib

matplotlib.use("Agg")
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve

from .backtest import backtest_all, performance
from .config import DATA_PROCESSED, RESULTS, load_config
from .features import feature_columns

ORDER = ["AlwaysUp", "Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]
LABEL = {"AlwaysUp": "Buy & hold", "RandomForest": "Random forest"}
COLOR = {"AlwaysUp": "#7a7873", "Logistic": "#2a78d6", "ARIMA": "#eb6834", "RandomForest": "#1baf7a",
         "XGBoost": "#d9588a", "LSTM": "#6250d6"}
LINE = {"AlwaysUp": "-", "Logistic": "-", "ARIMA": "--", "RandomForest": "-.", "XGBoost": ":", "LSTM": "-"}
INK, MUTED, GRID, NAVY, ORANGE = "#111111", "#4a4a47", "#dcdbd3", "#14142b", "#e8742c"
FIG = RESULTS.parent / "paper" / "figures"
TABLES = RESULTS / "tables"


def lab(m: str) -> str:
    return LABEL.get(m, m)


def _style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- data
def load() -> dict:
    t = lambda f, **k: pd.read_csv(TABLES / f, **k)  # noqa: E731
    preds = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RESULTS / "predictions" / "*.parquet")))],
                      ignore_index=True)
    preds["correct"] = (preds["pred"] == preds["y_true"]).astype(float)
    ds = pd.read_parquet(DATA_PROCESSED / "dataset.parquet")
    return {
        "cfg": load_config(), "preds": preds, "ds": ds,
        "metrics": t("classification_metrics.csv", index_col=0).reindex(ORDER),
        "bt": t("backtest_portfolio.csv", index_col=0).reindex(ORDER),
        "sig": t("significance_vs_baselines.csv"),
        "sbh": t("sharpe_vs_buy_and_hold.csv", index_col=0),
        "trend": t("regime_trend.csv"), "vol": t("regime_vol.csv"),
        "trend_dm": t("regime_trend_dm_vs_logistic.csv"), "vol_dm": t("regime_vol_dm_vs_logistic.csv"),
        "by_year": t("classification_by_year.csv"), "by_ticker": t("classification_by_ticker.csv"),
        "bt_ticker": t("backtest_by_ticker.csv"),
    }


# ---------------------------------------------------------------- extra analyses
def exposure_turnover(preds: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Share of ticker-days spent long, and position changes per ticker per year."""
    rows = {}
    for m, g in preds.groupby("model"):
        g = g.sort_values(["ticker", "date"])
        flips = g.groupby("ticker")["pred"].apply(lambda s: s.diff().abs().fillna(s.iloc[0]).sum())
        yrs = g.groupby("ticker").size().mean() / trading_days
        rows[m] = {"pct_long": g["pred"].mean(), "trades_per_year": flips.mean() / yrs}
    return pd.DataFrame(rows).T.reindex(ORDER)


def cost_sensitivity(preds: pd.DataFrame, costs_bps=(0, 5, 10, 25, 50), trading_days: int = 252) -> pd.DataFrame:
    out = {}
    for c in costs_bps:
        _, _, _, port = backtest_all(preds, c / 1e4, trading_days)
        out[c] = port["sharpe"]
    return pd.DataFrame(out).reindex(ORDER)


def exposure_matched_buyhold(preds: pd.DataFrame, port_daily: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Sharpe of a naive strategy with the same average market exposure as each model but no timing:
    hold buy-and-hold on a fixed fraction of the capital, the rest in cash. Sharpe is scale-invariant, so this
    equals buy-and-hold's Sharpe; return/drawdown are what scale. Returns annualised return of that naive mix."""
    expo = preds.groupby("model")["pred"].mean()
    bh = port_daily["AlwaysUp"]
    return pd.Series({m: performance(bh * expo[m], trading_days)["ann_return"] for m in ORDER})


def feature_target_corr(ds: pd.DataFrame) -> pd.Series:
    f = feature_columns(ds)
    return ds[f].corrwith(ds["y"]).sort_values()


def portfolio_daily() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "backtest" / "daily_returns_portfolio.parquet")[ORDER]


def check_claims(d):
    """The prose below asserts these results. If a re-run breaks one, revise the text before publishing."""
    m, sig, bt = d["metrics"], d["sig"], d["bt"]
    tested = [k for k in ORDER if k != "AlwaysUp"]
    assert all(m.loc[k, "accuracy"] < m.loc["AlwaysUp", "accuracy"] for k in tested), "a model now beats always-up on accuracy"
    assert (sig["DM_p_holm"] >= 0.05).all() and (sig["boot_acc_p_holm"] >= 0.05).all(), "an accuracy comparison is now significant"
    assert max(abs(m.loc[k, "roc_auc"] - 0.5) for k in tested) < 0.01, "ROC-AUC is no longer within 0.01 of 0.5"
    s = sig[sig["boot_sharpe_p_holm"] < 0.05]
    assert len(s) == 1 and s.iloc[0]["challenger"] == "LSTM" and s.iloc[0]["baseline"] == "ARIMA", "Sharpe significance pattern changed"
    assert bt.loc[tested, "sharpe"].idxmax() == "LSTM", "LSTM is no longer the best Sharpe among tested models"
    assert bt.loc["AlwaysUp", "cum_return"] == bt["cum_return"].max(), "buy & hold no longer has the highest cumulative return"
    assert bt.loc[tested, "max_drawdown"].idxmax() == "LSTM", "LSTM no longer has the smallest drawdown"
    et = exposure_turnover(d["preds"])
    assert et.loc["LSTM", "pct_long"] == et.loc[tested, "pct_long"].max(), "LSTM is no longer the most-invested model"
    assert et.loc["LSTM", "trades_per_year"] < et.drop(["AlwaysUp", "LSTM"])["trades_per_year"].min(), "LSTM is no longer the lowest-turnover model"
    cs = cost_sensitivity(d["preds"])
    assert (cs.drop(["AlwaysUp", "LSTM"])[50] < 0.5).all() and cs.loc["LSTM", 50] > 1.0, "cost-sensitivity story changed"
    corr = feature_target_corr(d["ds"]); big = corr[corr.abs() > 2 / len(d["ds"]) ** 0.5]
    assert len(big) > 0 and (big < 0).all(), "the significant indicator correlations are no longer all negative"


# ---------------------------------------------------------------- figures
POSTER_FIG = RESULTS.parent / "poster" / "figures"  # private: git-ignored, never published


def _save(fig, name, outdir=None):
    outdir = outdir or FIG
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, tc="white", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))


def fig_pipeline(d):
    _style()
    c = d["cfg"]
    steps = [(f"Data\n{len(c['tickers'])} stocks + SPY\n+ VIX (Yahoo)", "#2a78d6"), ("Features\n26 indicators\nknown at close", "#2a78d6"),
             ("Walk-forward\n5-yr train,\n1-yr test", "#1baf7a"), ("Models\nclassical, ML,\nLSTM, benchmark", "#1baf7a"),
             ("Predictions\nup / down per\nstock-day", "#6250d6"), ("Backtest\nlong / cash,\n0.05% cost", "#eb6834"),
             ("Evaluation\nmetrics and\nsignificance tests", "#eb6834")]
    fig, ax = plt.subplots(figsize=(13, 1.9))
    ax.set_xlim(0, 7 * 1.8); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    for i, (t, col) in enumerate(steps):
        x = i * 1.8 + 0.05
        _box(ax, x, 0.12, 1.4, 0.76, t, col, fs=9)
        if i < len(steps) - 1:
            _arrow(ax, x + 1.46, x + 1.76, 0.5)
    _save(fig, "fig_pipeline.png")


def fig_walkforward(d):
    _style()
    wf = d["cfg"]["walk_forward"]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    years = range(wf["first_test_year"], wf["last_test_year"] + 1)
    first = wf["first_test_year"] - wf["train_years"]
    ax.barh(len(years) + 0.0, 1, left=first - 1, height=0.55, color="#c9c8c2")
    ax.text(first - 0.5, len(years), "warm-up", ha="center", va="center", fontsize=8, color=INK)
    for i, y in enumerate(years):
        r = len(years) - 1 - i
        ax.barh(r, wf["train_years"], left=y - wf["train_years"], height=0.55, color="#2a78d6")
        ax.barh(r, 1, left=y, height=0.55, color=ORANGE)
        ax.text(y - wf["train_years"] / 2, r, "train", ha="center", va="center", color="white", fontsize=9)
        ax.text(y + 0.5, r, "test", ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks(range(len(years)), [f"Fold {y}" for y in reversed(list(years))])
    ax.set_xlim(first - 1, wf["last_test_year"] + 1)
    ax.set_xticks(range(first - 1, wf["last_test_year"] + 1))
    ax.set_xlabel("Calendar year (each bar spans the year starting at its left edge)")
    ax.grid(axis="y", visible=False); ax.set_title("Rolling walk-forward validation: a fresh model per stock and fold")
    _save(fig, "fig_walkforward.png")


def fig_lstm(d):
    _style()
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off"); ax.grid(False)
    parts = [("Input window\n20 days x 26\nstandardised features", "#2a78d6", 0.1), ("LSTM layer\n32 hidden units\n(memory + 3 gates)", "#6250d6", 2.4),
             ("Dropout\n30% of units off\nwhile training", "#8b7bff", 4.7), ("Linear layer\n32 -> 1 neuron", "#1baf7a", 7.0),
             ("Sigmoid\nP(up tomorrow)\nup if >= 0.5", "#eb6834", 9.2)]
    for t, col, x in parts:
        _box(ax, x, 1.2, 1.7, 1.3, t, col, fs=9)
    for x in (1.85, 4.15, 6.45, 8.75):
        _arrow(ax, x, x + 0.5, 1.85)
    ax.text(5.5, 0.55, "Training: Adam (lr 0.001, weight decay 1e-4) on binary cross-entropy, batches of 64, up to 40 epochs;\n"
            "early stopping (patience 5) on the last 15% of the training years, best weights restored.",
            ha="center", va="center", fontsize=9.2, color=MUTED, linespacing=1.4)
    _save(fig, "fig_lstm.png")


def fig_accuracy(d):
    _style()
    m = d["metrics"]; n = m["n"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    acc = m["accuracy"]; se = np.sqrt(acc * (1 - acc) / n)
    ax.bar([lab(k) for k in ORDER], acc, color=[COLOR[k] for k in ORDER], yerr=1.96 * se, capsize=4, width=0.62,
           error_kw=dict(ecolor=INK, lw=1.2))
    ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.text(-0.45, 0.5008, "coin flip (50%)", ha="left", fontsize=8.5, color=MUTED)
    ax.set_ylim(0.48, 0.545); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    for i, k in enumerate(ORDER):
        ax.text(i, acc[k] + 1.96 * se[k] + 0.0012, f"{acc[k]:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Directional accuracy"); ax.set_title("Out-of-sample accuracy, 10,048 predictions (95% binomial interval)")
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_accuracy.png")


def fig_roc(d):
    _style()
    p = d["preds"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for m in ORDER[1:]:
        g = p[p["model"] == m]
        fpr, tpr, _ = roc_curve(g["y_true"], g["score"])
        ax.plot(fpr, tpr, color=COLOR[m], ls=LINE[m], lw=2, label=f"{lab(m)} (AUC {roc_auc_score(g['y_true'], g['score']):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance (AUC 0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("Pooled ROC curves")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, "fig_roc.png")


def fig_sharpe(d):
    _style()
    bt = d["bt"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar([lab(k) for k in ORDER], bt["sharpe"], color=[COLOR[k] for k in ORDER], width=0.62)
    for i, k in enumerate(ORDER):
        axes[0].text(i, bt["sharpe"][k] + 0.03, f"{bt['sharpe'][k]:.2f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Sharpe ratio (annualised, after costs)"); axes[0].grid(axis="x", visible=False)
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[1].bar([lab(k) for k in ORDER], bt["max_drawdown"], color=[COLOR[k] for k in ORDER], width=0.62)
    for i, k in enumerate(ORDER):
        axes[1].text(i, bt["max_drawdown"][k] - 0.02, f"{bt['max_drawdown'][k]:.0%}", ha="center", va="top", fontsize=9, fontweight="bold")
    axes[1].set_title("Maximum drawdown"); axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_ylim(min(bt["max_drawdown"]) - 0.08, 0); axes[1].grid(axis="x", visible=False)
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    _save(fig, "fig_sharpe.png")


def fig_cost(d):
    _style()
    cs = cost_sensitivity(d["preds"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ORDER:
        ax.plot(cs.columns, cs.loc[m], color=COLOR[m], ls=LINE[m], lw=2, marker="o", ms=4, label=lab(m))
    ax.set_xlabel("Transaction cost (basis points per unit of turnover)"); ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sensitivity of the backtest to trading costs"); ax.legend(frameon=False, fontsize=8.5, ncol=2)
    _save(fig, "fig_cost.png")
    return cs


def fig_by_year(d):
    _style()
    t = d["by_year"].pivot(index="test_year", columns="model", values="accuracy")[ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in ORDER:
        ax.plot(t.index, t[m], color=COLOR[m], ls=LINE[m], lw=2, marker="o", ms=4, label=lab(m))
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_xticks(list(t.index))
    ax.set_ylabel("Directional accuracy"); ax.set_title("Accuracy by test year"); ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, "fig_by_year.png")


def fig_ticker_heatmap(d):
    _style()
    t = d["by_ticker"].pivot(index="ticker", columns="model", values="accuracy")[ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(t.values, cmap="Blues", vmin=0.47, vmax=0.56, aspect="auto")
    ax.set_xticks(range(len(ORDER)), [lab(m) for m in ORDER], rotation=25, ha="right"); ax.set_yticks(range(len(t)), t.index)
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.525 else INK)
    ax.set_title("Accuracy by stock and model")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _save(fig, "fig_ticker_heatmap.png")


def fig_feature_corr(d):
    _style()
    c = feature_target_corr(d["ds"])
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    colors = ["#d9588a" if v < 0 else "#2a78d6" for v in c.values]
    ax.barh(c.index, c.values, color=colors)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Pearson correlation with next-day direction"); ax.set_title("Each indicator vs. tomorrow's direction")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
    _save(fig, "fig_feature_corr.png")
    return c


def make_all(d=None) -> dict:
    d = d or load()
    for f in (fig_pipeline, fig_walkforward, fig_lstm, fig_accuracy, fig_roc, fig_sharpe, fig_by_year, fig_ticker_heatmap):
        f(d)
    return {"cost": fig_cost(d), "corr": fig_feature_corr(d)}


# ---------------------------------------------------------------- poster figures (wide, large text, no titles)
def make_poster_figs(d=None):
    d = d or load()
    _style()
    plt.rcParams.update({"font.size": 11.5})
    m, bt = d["metrics"], d["bt"]
    n = m["n"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.4, 1.95))
    acc = m["accuracy"]; se = np.sqrt(acc * (1 - acc) / n)
    ax.bar([("Buy &" + chr(10) + "hold") if k == "AlwaysUp" else lab(k).replace(" ", chr(10)) for k in ORDER], acc, color=[COLOR[k] for k in ORDER], yerr=1.96 * se, capsize=4, width=0.62, error_kw=dict(ecolor=INK, lw=1.2))
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_ylim(0.48, 0.545); ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.grid(axis="x", visible=False)
    for i, k in enumerate(ORDER):
        ax.text(i, acc[k] + 1.96 * se[k] + 0.0015, f"{acc[k]:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy")
    _save(fig, "poster_accuracy.png", POSTER_FIG)

    port = portfolio_daily(); eq = (1 + port).cumprod() - 1
    fig, ax = plt.subplots(figsize=(3.7, 2.05))
    for k in ORDER:
        ax.plot(eq.index, eq[k], color=COLOR[k], ls=LINE[k], lw=2, label=lab(k))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); ax.set_ylabel("Cumulative return")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2); ax.tick_params(axis="x", labelsize=9)
    _save(fig, "poster_equity.png", POSTER_FIG)

    cs = cost_sensitivity(d["preds"])
    fig, ax = plt.subplots(figsize=(3.7, 2.05))
    for k in ORDER:
        ax.plot(cs.columns, cs.loc[k], color=COLOR[k], ls=LINE[k], lw=2, marker="o", ms=3.5, label=lab(k))
    ax.set_xlabel("Cost per trade (basis points)"); ax.set_ylabel("Sharpe ratio")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=1, labelspacing=0.25)
    _save(fig, "poster_cost.png", POSTER_FIG)
