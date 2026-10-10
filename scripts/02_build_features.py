"""Step 2: build the feature/target dataset -> data/processed/dataset.parquet."""
from capstone.config import DATA_PROCESSED, load_config
from capstone.data import load_prices
from capstone.features import build_dataset, feature_columns


def main():
    cfg = load_config()
    ds = build_dataset(load_prices(), cfg["tickers"], cfg["market_ticker"], cfg["vix_ticker"])
    ds = ds[ds.index.get_level_values("date") >= "2015-01-01"]  # 2014 was warm-up only
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(DATA_PROCESSED / "dataset.parquet")
    print(f"{len(ds)} rows, {len(feature_columns(ds))} features")
    print("Up-day share by ticker:\n", ds.groupby("ticker")["y"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
