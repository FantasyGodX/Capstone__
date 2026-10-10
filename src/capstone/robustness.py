"""Regime analysis. Regimes use only information known at the close of day t."""
import numpy as np
import pandas as pd

from .stats import diebold_mariano


def regime_labels(prices: dict, market: str, vix: str, ma_window: int = 200, vix_cut: float = 20.0) -> pd.DataFrame:
    spy = prices[market]["Close"]
    bull = spy > spy.rolling(ma_window).mean()
    v = prices[vix]["Close"]
    out = pd.DataFrame({"trend": np.where(bull, "bull", "bear"),
                        "vol": np.where(v > vix_cut, "high_vol", "low_vol")}, index=spy.index)
    out.loc[spy.rolling(ma_window).mean().isna(), "trend"] = np.nan
    return out


def regime_table(preds: pd.DataFrame, port: pd.DataFrame, regimes: pd.DataFrame, kind: str,
                 trading_days: int = 252) -> pd.DataFrame:
    """Per model x regime: n days, accuracy (pooled across tickers), annualised portfolio Sharpe."""
    p = preds.merge(regimes[[kind]], left_on="date", right_index=True)
    p["correct"] = (p["pred"] == p["y_true"]).astype(float)
    acc = p.groupby(["model", kind])["correct"].mean().rename("accuracy")
    n = p.groupby(["model", kind])["date"].nunique().rename("n_days")
    lab = regimes[kind].reindex(port.index)
    rows = []
    for (m, reg), _ in acc.items():
        r = port.loc[lab == reg, m]
        sd = r.std(ddof=1)
        rows.append({"model": m, kind: reg, "sharpe": r.mean() / sd * np.sqrt(trading_days) if sd > 0 else 0.0,
                     "mean_daily_ret": r.mean()})
    sh = pd.DataFrame(rows).set_index(["model", kind])
    return pd.concat([n, acc, sh], axis=1)


def regime_dm(preds: pd.DataFrame, regimes: pd.DataFrame, kind: str, baseline: str, challengers: list[str]) -> pd.DataFrame:
    """One-sided DM (0/1 loss, cross-ticker average per date) within each regime."""
    p = preds.merge(regimes[[kind]], left_on="date", right_index=True)
    p["loss"] = (p["pred"] != p["y_true"]).astype(float)
    wide = p.pivot_table(index=["date", kind], columns="model", values="loss", aggfunc="mean")
    rows = []
    for reg, g in wide.groupby(level=kind):
        for c in challengers:
            dm, pv = diebold_mariano(g[baseline].values, g[c].values)
            rows.append({kind: reg, "challenger": c, "baseline": baseline,
                         "acc_diff": g[baseline].mean() - g[c].mean(), "DM_stat": dm, "DM_p": pv})
    return pd.DataFrame(rows)
