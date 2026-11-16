"""The paper and poster read every number from results/; their analysis helpers must be internally consistent."""
import glob
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not glob.glob(str(ROOT / "results" / "predictions" / "*.parquet")), reason="run the pipeline first")


@pytest.fixture(scope="module")
def data():
    from capstone import report
    return report.load()


def test_exposure_and_turnover(data):
    from capstone import report
    et = report.exposure_turnover(data["preds"])
    assert list(et.index) == report.ORDER
    assert et.loc["AlwaysUp", "pct_long"] == 1.0           # buy & hold is always invested
    assert et.loc["AlwaysUp", "trades_per_year"] < 1.0      # one entry over five years
    assert ((et["pct_long"] >= 0) & (et["pct_long"] <= 1)).all()


def test_cost_sensitivity_matches_saved_backtest(data):
    from capstone import report
    cs = report.cost_sensitivity(data["preds"], costs_bps=(5,))
    cfg_cost = data["cfg"]["backtest"]["transaction_cost"] * 1e4
    if cfg_cost == 5:  # the 5 bp column must reproduce the saved backtest table
        assert (cs[5] - data["bt"]["sharpe"]).abs().max() < 1e-3


def test_cost_never_helps(data):
    from capstone import report
    cs = report.cost_sensitivity(data["preds"], costs_bps=(0, 25))
    assert (cs[25] <= cs[0] + 1e-9).all()


def test_claims_hold_for_current_results(data):
    from capstone import report
    report.check_claims(data)  # raises if the prose in the paper/poster is no longer supported
