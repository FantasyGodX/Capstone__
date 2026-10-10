"""Significance tests: Diebold-Mariano, stationary block bootstrap, paired t-test, Holm."""
import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(loss_base: np.ndarray, loss_chal: np.ndarray, h: int = 1):
    """One-sided DM test, H1: challenger has LOWER expected loss than baseline.

    d_t = L_base - L_chal. Newey-West (Bartlett) long-run variance, Harvey-Leybourne-Newbold
    small-sample correction, Student-t reference distribution. Returns (statistic, p_one_sided).
    """
    d = np.asarray(loss_base, float) - np.asarray(loss_chal, float)
    n = len(d)
    dbar = d.mean()
    lags = int(np.floor(n ** (1 / 3)))
    dc = d - dbar
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= hln
    return dm, 1 - stats.t.cdf(dm, df=n - 1)


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: (n_boot, n) index matrix."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def sharpe(r: np.ndarray, axis=-1, trading_days: int = 252):
    sd = r.std(axis=axis, ddof=1)
    return np.where(sd > 0, r.mean(axis=axis) / np.where(sd > 0, sd, 1), 0.0) * np.sqrt(trading_days)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, stat, n_boot: int, mean_block: float, rng, alpha=0.05):
    """Paired bootstrap of stat(a) - stat(b), resampling dates jointly.

    Returns (observed diff, ci_low, ci_high, one-sided p for H1: diff > 0)."""
    idx = stationary_bootstrap_indices(len(a), n_boot, mean_block, rng)
    diffs = stat(a[idx]) - stat(b[idx])
    obs = float(stat(a) - stat(b))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # centred bootstrap p-value: how often the null-centred statistic exceeds the observed one
    p = (np.sum(diffs - diffs.mean() >= obs) + 1) / (n_boot + 1)
    return obs, float(lo), float(hi), float(p)


def paired_t(acc_chal: np.ndarray, acc_base: np.ndarray):
    """Paired t-test on matched accuracies, one-sided (challenger > baseline)."""
    t, p_two = stats.ttest_rel(acc_chal, acc_base)
    p = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(t), float(p)


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni step-down adjusted p-values."""
    order = pvals.sort_values().index
    m = len(pvals)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return pd.Series(adj).reindex(pvals.index)
