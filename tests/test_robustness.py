import numpy as np
import pandas as pd

from capstone.robustness import regime_labels


def test_regime_labels_are_causal():
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(0)
    spy = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0, .01, 500)))}, index=idx)
    vix = pd.DataFrame({"Close": rng.uniform(10, 30, 500)}, index=idx)
    full = regime_labels({"SPY": spy, "V": vix}, "SPY", "V")
    spy2 = spy.copy(); spy2.iloc[300:] *= 5
    part = regime_labels({"SPY": spy2, "V": vix}, "SPY", "V")
    pd.testing.assert_frame_equal(full.iloc[:300], part.iloc[:300])
    assert full["trend"].iloc[:199].isna().all()  # no label until the 200-day MA exists
