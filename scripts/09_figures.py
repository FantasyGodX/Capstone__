"""Step 9: publication figures -> results/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from capstone.config import RESULTS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
# fixed colour per model (never re-assigned by rank); linestyle is the secondary encoding
STYLE = {"AlwaysUp": ("#898781", "-"), "Logistic": ("#2a78d6", "-"), "ARIMA": ("#eb6834", "--"),
         "RandomForest": ("#1baf7a", "-."), "XGBoost": ("#e87ba4", ":"), "LSTM": ("#6250d6", "-")}
LABEL = {"AlwaysUp": "Buy & hold"}
FIG = RESULTS / "figures"
T = RESULTS / "tables"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def end_labels(ax, series: dict, fmt=lambda v: v):
    """Direct labels at line ends, nudged apart to avoid collisions."""
    items = sorted(series.items(), key=lambda kv: kv[1])
    lo, hi = ax.get_ylim()
    gap, prev = (hi - lo) * 0.045, None
    for m, y in items:
        ypos = y if prev is None or y - prev >= gap else prev + gap
        ax.annotate(LABEL.get(m, m), (1.0, ypos), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
        prev = ypos


def line_fig(df, title, ylabel, fname, pct=True):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for m in STYLE:
        c, ls = STYLE[m]
        ax.plot(df.index, df[m], color=c, ls=ls, lw=2 if m != "AlwaysUp" else 1.6, label=LABEL.get(m, m))
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    fig.canvas.draw()
    end_labels(ax, {m: df[m].iloc[-1] for m in STYLE})
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)
    fig.subplots_adjust(right=0.86)
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    port = pd.read_parquet(RESULTS / "backtest" / "daily_returns_portfolio.parquet")[list(STYLE)]
    eq = (1 + port).cumprod() - 1
    line_fig(eq, "Equal-weight portfolio: cumulative return, 2020-2024 (0.05% costs)", "Cumulative return", "equity_curves.png")
    eq1 = 1 + eq
    line_fig(eq1 / eq1.cummax() - 1, "Drawdown from running peak", "Drawdown", "drawdowns.png")

    # forest plot: Sharpe difference vs Logistic with 95% block-bootstrap CI
    s = pd.read_csv(T / "significance_vs_baselines.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (what, ttl) in zip(axes, [("acc", "Accuracy difference"), ("sharpe", "Sharpe difference")]):
        for i, (_, r) in enumerate(s[s.baseline == "Logistic"].iterrows()):
            c = STYLE[r.challenger][0]
            ax.plot([r[f"{what}_ci_low"], r[f"{what}_ci_high"]], [i, i], color=c, lw=2)
            ax.plot(r[f"{what}_diff"], i, "o", color=c, ms=8, mec=SURFACE, mew=2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(3), ["RandomForest", "XGBoost", "LSTM"])
        ax.set_title(f"{ttl} vs Logistic (95% CI)", fontsize=10); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "difference_vs_logistic.png", dpi=200); plt.close(fig)

    # accuracy by regime
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, kind, ttl in zip(axes, ["trend", "vol"], ["Bull vs bear (SPY vs 200d MA)", "Low vs high vol (VIX 20)"]):
        t = pd.read_csv(T / f"regime_{kind}.csv").pivot(index="model", columns=kind, values="accuracy")
        for m in STYLE:
            for j, reg in enumerate(t.columns):
                ax.plot(j + (list(STYLE).index(m) - 2.5) * 0.06, t.loc[m, reg], "o", color=STYLE[m][0], ms=8,
                        mec=SURFACE, mew=2, label=LABEL.get(m, m) if j == 0 else None)
        ax.set_xticks(range(2), list(t.columns)); ax.set_xlim(-0.5, 1.5); ax.set_title(ttl, fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].set_ylabel("Directional accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "accuracy_by_regime.png", dpi=200); plt.close(fig)
    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
