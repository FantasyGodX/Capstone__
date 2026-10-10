"""Step 7: significance tests of ML/DL models vs classical baselines."""
import numpy as np
import pandas as pd

from capstone.config import RESULTS, load_config
from capstone.stats import bootstrap_diff, diebold_mariano, holm, paired_t, sharpe

CHALLENGERS = ["RandomForest", "XGBoost", "LSTM"]
BASELINES = ["Logistic", "ARIMA"]


def load_predictions() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((RESULTS / "predictions").glob("*.parquet"))],
                     ignore_index=True)


def main():
    cfg = load_config()
    B, alpha = cfg["stats"]["bootstrap_iterations"], cfg["stats"]["alpha"]
    td = cfg["backtest"]["trading_days"]
    rng = np.random.default_rng(cfg["seed"])
    preds = load_predictions()
    preds["correct"] = (preds["pred"] == preds["y_true"]).astype(float)

    # date-level series: share of tickers predicted correctly (cross-sectional average)
    correct = preds.pivot_table(index="date", columns="model", values="correct", aggfunc="mean")
    port = pd.read_parquet(RESULTS / "backtest" / "daily_returns_portfolio.parquet")
    acc_te = preds.groupby(["model", "ticker", "test_year"])["correct"].mean().unstack("model")

    rows = []
    for c in CHALLENGERS:
        for b in BASELINES:
            dm, p_dm = diebold_mariano(1 - correct[b].values, 1 - correct[c].values)
            acc_d, lo, hi, p_boot = bootstrap_diff(correct[c].values, correct[b].values, lambda x: x.mean(-1), B, 10, rng, alpha)
            sh_d, slo, shi, p_sh = bootstrap_diff(port[c].values, port[b].values, lambda x: sharpe(x, trading_days=td), B, 10, rng, alpha)
            t, p_t = paired_t(acc_te[c].values, acc_te[b].values)
            rows.append({"challenger": c, "baseline": b,
                         "acc_diff": acc_d, "acc_ci_low": lo, "acc_ci_high": hi,
                         "DM_stat": dm, "DM_p": p_dm, "boot_acc_p": p_boot,
                         "ttest_t": t, "ttest_p": p_t,
                         "sharpe_diff": sh_d, "sharpe_ci_low": slo, "sharpe_ci_high": shi, "boot_sharpe_p": p_sh})
    res = pd.DataFrame(rows)
    for col in ["DM_p", "boot_acc_p", "ttest_p", "boot_sharpe_p"]:
        res[col + "_holm"] = holm(res[col])

    # context: every model vs buy-and-hold (AlwaysUp) on Sharpe
    ctx = []
    for m in ["Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]:
        d, lo, hi, p = bootstrap_diff(port[m].values, port["AlwaysUp"].values,
                                      lambda x: sharpe(x, trading_days=td), B, 10, rng, alpha)
        ctx.append({"model": m, "sharpe_diff_vs_buy_hold": d, "ci_low": lo, "ci_high": hi, "p_one_sided": p})
    ctx = pd.DataFrame(ctx)

    # each model's accuracy / AUC-free CI (bootstrap on the date-level hit rate)
    out = RESULTS / "tables"
    res.round(4).to_csv(out / "significance_vs_baselines.csv", index=False)
    ctx.round(4).to_csv(out / "sharpe_vs_buy_and_hold.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(res.round(4).to_string(index=False), "\n")
    print(ctx.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
