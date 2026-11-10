"""The website reads JSON written by scripts/10_export_site.py; it must mirror the pipeline output exactly."""
import glob
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRED_FILES = glob.glob(str(ROOT / "results" / "predictions" / "*.parquet"))

pytestmark = pytest.mark.skipif(not PRED_FILES, reason="run the pipeline first")


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("export_site", ROOT / "scripts" / "10_export_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = tmp_path_factory.mktemp("site_data")
    mod.export_predictions({})
    return json.loads((mod.OUT / "predictions.json").read_text(encoding="utf-8"))


def test_grid_shape(export):
    n = len(export["tickers"]) * len(export["dates"])
    assert len(export["y"]) == len(export["ret"]) == n
    for m in export["models"].values():
        assert len(m["pred"]) == len(m["score"]) == n


def test_predictions_match_parquet(export):
    p = pd.concat([pd.read_parquet(f) for f in PRED_FILES]).sort_values(["model", "ticker", "date"])
    for name, m in export["models"].items():
        g = p[p["model"] == name]
        assert g["pred"].tolist() == m["pred"], name
        assert max(abs(a - b) for a, b in zip(g["score"], m["score"])) < 1e-5, name


def test_dates_are_sorted_and_years_align(export):
    assert export["dates"] == sorted(export["dates"])
    assert [int(d[:4]) for d in export["dates"]] == export["year"]
