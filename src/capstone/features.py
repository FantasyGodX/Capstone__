"""Feature engineering and target construction.

Leakage rule: every feature in row t uses only information available at the close of
day t. The target at row t is the direction of the return from close t to close t+1.
Normalisation (z-scores) is NOT applied here; it is fitted on the training window inside
each model pipeline so no test-period statistics leak into training.
"""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def build_features(df: pd.DataFrame, market: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)

    # Technical indicators
    f["rsi14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = macd / c, sig / c, (macd - sig) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    f["bb_width"] = 4 * sd20 / ma20
    for w in (5, 20, 50):
        f[f"ma{w}_ratio"] = c / c.rolling(w).mean() - 1
    f["atr14_pct"] = atr(df) / c

    # Statistical features
    for w in (5, 10, 20):
        f[f"vol{w}"] = ret.rolling(w).std()
    for k in range(1, 6):
        f[f"ret_lag{k}"] = ret.shift(k - 1)  # lag1 = today's return (known at close t)
    f["skew20"] = ret.rolling(20).skew()
    for w in (5, 10, 20):
        f[f"mom{w}"] = c / c.shift(w) - 1

    # Market context
    mret = market["Close"].pct_change().reindex(df.index)
    f["mkt_ret"] = mret
    f["corr_mkt20"] = ret.rolling(20).corr(mret)
    v = vix["Close"].reindex(df.index).ffill()
    f["vix"] = v
    f["vix_chg"] = v.pct_change()
    return f


def add_target(feats: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    out = feats.copy()
    out["ret_next"] = close.shift(-1) / close - 1
    out["y"] = (out["ret_next"] > 0).astype(int)
    out.loc[out["ret_next"].isna(), "y"] = np.nan
    return out


def build_dataset(prices: dict, tickers: list[str], market: str, vix: str) -> pd.DataFrame:
    """Long-format dataset (MultiIndex: ticker, date) with features, ret_next and y."""
    parts = []
    for t in tickers:
        d = add_target(build_features(prices[t], prices[market], prices[vix]), prices[t]["Close"])
        d = d.dropna()  # drops warm-up rows and the final (unlabelled) day
        d.insert(0, "ticker", t)
        parts.append(d)
    ds = pd.concat(parts)
    ds["y"] = ds["y"].astype(int)
    return ds.reset_index().set_index(["ticker", "date"]).sort_index()


FEATURE_EXCLUDE = {"ret_next", "y"}


def feature_columns(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c not in FEATURE_EXCLUDE]
