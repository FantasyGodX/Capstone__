import pandas as pd
import pytest

from capstone.data import load_prices, save_prices, validate


def _frame():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-06"])
    return pd.DataFrame({"Open": [1, 1, 1, 2.0], "High": [2, 2, 2, 3.0], "Low": [1, 1, 1, 1.5],
                         "Close": [1.5, 1.4, 1.4, 2.5], "Volume": [10, 10, 10, 20]}, index=idx)


def test_validate_sorts_and_dedups():
    out = validate(_frame(), "X")
    assert out.index.is_monotonic_increasing and out.index.is_unique and len(out) == 3


def test_validate_rejects_nonpositive():
    df = _frame()
    df.iloc[0, 3] = 0
    with pytest.raises(ValueError):
        validate(df, "X")


def test_sqlite_roundtrip(tmp_path):
    df = validate(_frame(), "X")
    df.index.name = "date"
    save_prices({"X": df}, tmp_path / "p.db")
    back = load_prices(tmp_path / "p.db")["X"]
    pd.testing.assert_frame_equal(back, df, check_freq=False, check_dtype=False)
