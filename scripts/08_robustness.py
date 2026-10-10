"""Step 8: bull/bear and volatility-regime robustness checks."""
import pandas as pd

from capstone.config import RESULTS, load_config
from capstone.data import load_prices
from capstone.robustness import regime_dm, regime_labels, regime_table
from capstone.stats import holm

ORDER = ["AlwaysUp", "Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]


def main():
    cfg = load_config()
    prices = load_prices()
    reg = regime_labels(prices, cfg["market_ticker"], cfg["vix_ticker"])
    preds = pd.concat([pd.read_parquet(p) for p in sorted((RESULTS / "predictions").glob("*.parquet"))], ignore_index=True)
    port = pd.read_parquet(RESULTS / "backtest" / "daily_returns_portfolio.parquet")
    out = RESULTS / "tables"
    pd.set_option("display.width", 160)
    for kind in ["trend", "vol"]:
        t = regime_table(preds, port, reg, kind, cfg["backtest"]["trading_days"])
        t = t.reindex(ORDER, level="model").round(4)
        t.to_csv(out / f"regime_{kind}.csv")
        d = regime_dm(preds, reg, kind, "Logistic", ["RandomForest", "XGBoost", "LSTM"])
        d["DM_p_holm"] = holm(d["DM_p"])
        d.round(4).to_csv(out / f"regime_{kind}_dm_vs_logistic.csv", index=False)
        print(f"== {kind} ==\n{t.unstack(kind).round(3).to_string()}\n{d.round(4).to_string(index=False)}\n")
    print("regime day share (test period):")
    test = reg[reg.index.year >= cfg["walk_forward"]["first_test_year"]]
    print(test["trend"].value_counts(normalize=True).round(3).to_dict(), test["vol"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
