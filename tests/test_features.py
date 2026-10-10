import numpy as np
import pandas as pd

from capstone.features import add_target, build_features


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 1e6}, index=idx)


def test_features_do_not_use_future_data():
    px, mkt, vix = _prices(seed=0), _prices(seed=1), _prices(seed=2)
    full = build_features(px, mkt, vix)
    cut = 200
    # corrupt everything after `cut`: features up to `cut` must be unchanged
    px2, mkt2, vix2 = px.copy(), mkt.copy(), vix.copy()
    for d in (px2, mkt2, vix2):
        d.iloc[cut + 1:] *= 3.7
    part = build_features(px2, mkt2, vix2)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], part.iloc[: cut + 1])


def test_target_is_next_day_direction():
    px = _prices()
    d = add_target(build_features(px, px, px), px["Close"])
    t = 150
    expected = int(px["Close"].iloc[t + 1] > px["Close"].iloc[t])
    assert d["y"].iloc[t] == expected
    assert np.isnan(d["y"].iloc[-1])
