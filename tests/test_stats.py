import numpy as np
import pandas as pd

from capstone.stats import bootstrap_diff, diebold_mariano, holm, paired_t


def test_dm_detects_better_forecaster_and_not_equal_ones():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.5, 1000)
    chal = base - 0.2 + rng.normal(0, 0.3, 1000)  # clearly lower loss
    _, p = diebold_mariano(base, chal)
    assert p < 0.01
    _, p_same = diebold_mariano(base, base + rng.normal(0, 0.3, 1000))
    assert p_same > 0.05


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.3, 1, 800)
    b = rng.normal(0.0, 1, 800)
    obs, lo, hi, p = bootstrap_diff(a, b, lambda x: x.mean(-1), 2000, 10, rng)
    assert lo < 0.3 < hi and lo > 0 and p < 0.05


def test_paired_t_one_sided_direction():
    _, p_better = paired_t(np.array([.6, .62, .58, .61]), np.array([.5, .52, .5, .51]))
    _, p_worse = paired_t(np.array([.5, .52, .5, .51]), np.array([.6, .62, .58, .61]))
    assert p_better < 0.05 < p_worse


def test_holm_matches_hand_calc():
    adj = holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03}))
    assert np.allclose(adj[["a", "c", "b"]], [0.03, 0.06, 0.06])
