"""Long/cash backtest with proportional transaction costs, no leverage.

Rule: predict up (1) at close t -> hold the stock over t->t+1; predict 0 -> cash (0 return).
Cost: `cost` per unit of position change (entering or exiting costs `cost`).
"""
import numpy as np
import pandas as pd


def strategy_returns(pred: pd.Series, ret_next: pd.Series, cost: float) -> pd.Series:
    pos = pred.astype(float)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0]  # initial entry from cash
    return pos * ret_next - cost * turnover


def performance(r: pd.Series, trading_days: int = 252) -> pd.Series:
    r = r.dropna()
    equity = (1 + r).cumprod()
    cum = equity.iloc[-1] - 1
    years = len(r) / trading_days
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(trading_days) if sd > 0 else 0.0
    mdd = (equity / equity.cummax() - 1).min()
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    return pd.Series({"cum_return": cum, "ann_return": ann_ret, "sharpe": sharpe,
                      "max_drawdown": mdd, "calmar": calmar})


def backtest_all(preds: pd.DataFrame, cost: float, trading_days: int = 252):
    """Returns (daily returns wide frame: index=date, cols=(model, ticker); per-ticker table;
    equal-weight portfolio table)."""
    daily = {}
    for (m, t), g in preds.sort_values("date").groupby(["model", "ticker"]):
        g = g.set_index("date")
        daily[(m, t)] = strategy_returns(g["pred"], g["ret_next"], cost)
    wide = pd.DataFrame(daily)
    wide.columns.names = ["model", "ticker"]
    per_ticker = pd.DataFrame({k: performance(v, trading_days) for k, v in wide.items()}).T
    port = wide.T.groupby(level="model").mean().T  # equal-weight across tickers, daily rebalanced
    portfolio = pd.DataFrame({m: performance(port[m], trading_days) for m in port}).T
    return wide, port, per_ticker, portfolio
