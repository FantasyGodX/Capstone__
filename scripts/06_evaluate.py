"""Step 6: statistical metrics + backtest -> results/tables."""
import pandas as pd

from capstone.backtest import backtest_all
from capstone.config import RESULTS, load_config
from capstone.evaluation import metrics_table

ORDER = ["AlwaysUp", "Logistic", "ARIMA", "RandomForest", "XGBoost", "LSTM"]


def load_predictions() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted((RESULTS / "predictions").glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True)


def main():
    cfg = load_config()
    preds = load_predictions()
    tables = RESULTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    m = metrics_table(preds).reindex(ORDER).round(4)
    m.to_csv(tables / "classification_metrics.csv")
    metrics_table(preds, ["ticker"]).round(4).to_csv(tables / "classification_by_ticker.csv")
    metrics_table(preds, ["test_year"]).round(4).to_csv(tables / "classification_by_year.csv")

    wide, port, per_ticker, portfolio = backtest_all(preds, cfg["backtest"]["transaction_cost"],
                                                     cfg["backtest"]["trading_days"])
    portfolio = portfolio.reindex(ORDER).round(4)
    portfolio.to_csv(tables / "backtest_portfolio.csv")
    per_ticker.round(4).to_csv(tables / "backtest_by_ticker.csv")
    bt_dir = RESULTS / "backtest"
    bt_dir.mkdir(exist_ok=True)
    wide.to_parquet(bt_dir / "daily_returns_by_ticker.parquet")  # reused by the stats step
    port.to_parquet(bt_dir / "daily_returns_portfolio.parquet")
    pd.set_option("display.width", 140)
    print("Classification (pooled OOS 2020-2024)\n", m, "\n")
    print("Equal-weight portfolio backtest (0.05% cost)\n", portfolio)


if __name__ == "__main__":
    main()
