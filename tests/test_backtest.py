import numpy as np
import pandas as pd

from capstone.backtest import performance, strategy_returns


def test_strategy_returns_costs_and_cash():
    pred = pd.Series([1, 1, 0, 1])
    ret = pd.Series([0.01, 0.02, 0.05, -0.01])
    r = strategy_returns(pred, ret, cost=0.001)
    # day0: enter (cost), day1: hold (no cost), day2: exit -> cash, pays cost, day3: re-enter
    np.testing.assert_allclose(r, [0.01 - 0.001, 0.02, -0.001, -0.01 - 0.001])


def test_performance_known_values():
    r = pd.Series([0.1, -0.1, 0.1])
    p = performance(r)
    assert np.isclose(p["cum_return"], 1.1 * 0.9 * 1.1 - 1)
    # equity: 1.1, 0.99, 1.089 -> max drawdown = 0.99/1.1 - 1
    assert np.isclose(p["max_drawdown"], 0.99 / 1.1 - 1)
    assert np.isclose(p["sharpe"], r.mean() / r.std() * np.sqrt(252))
