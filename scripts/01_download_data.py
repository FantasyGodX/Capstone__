"""Step 1: download raw OHLCV for all tickers into data/raw/prices.db."""
from capstone.config import load_config
from capstone.data import DB_PATH, download_ticker, save_prices, validate


def main():
    cfg = load_config()
    symbols = cfg["tickers"] + [cfg["market_ticker"], cfg["vix_ticker"]]
    frames = {}
    for s in symbols:
        df = validate(download_ticker(s, cfg["start_date"], cfg["end_date"]), s)
        frames[s] = df
        print(f"{s:6s} {len(df):5d} rows  {df.index.min().date()} -> {df.index.max().date()}")
    save_prices(frames)
    print(f"Saved to {DB_PATH}")


if __name__ == "__main__":
    main()
