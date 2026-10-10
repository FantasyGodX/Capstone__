"""Download daily OHLCV from Yahoo Finance and persist to SQLite."""
import sqlite3

import pandas as pd
import yfinance as yf

from .config import DATA_RAW

DB_PATH = DATA_RAW / "prices.db"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV with split/dividend-adjusted prices (auto_adjust=True)."""
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, multi_level_index=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df[COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Basic cleaning: sort, drop duplicate dates and rows with missing/non-positive prices."""
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    bad = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if bad.any():
        raise ValueError(f"{ticker}: {bad.sum()} rows with non-positive prices")
    return df


def save_prices(frames: dict[str, pd.DataFrame], db_path=DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    long = pd.concat(
        [f.reset_index().assign(ticker=t) for t, f in frames.items()], ignore_index=True
    )
    long.columns = [c.lower() for c in long.columns]
    with sqlite3.connect(db_path) as con:
        long.to_sql("prices", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_prices ON prices (ticker, date)")


def load_prices(db_path=DB_PATH) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as con:
        long = pd.read_sql("SELECT * FROM prices", con, parse_dates=["date"])
    out = {}
    for t, g in long.groupby("ticker"):
        g = g.drop(columns="ticker").set_index("date").sort_index()
        g.columns = [c.capitalize() for c in g.columns]
        out[t] = g
    return out
