"""Headless smoke tests: every dashboard page must render without raising."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "performance", "backtest", "statistics", "robustness", "compare", "data", "notes", "pipeline"]

SCRIPT = """
import sys
sys.path.insert(0, r"{app}")
import pages_{page}
pages_{page}.render()
"""


@pytest.mark.skipif(not any((ROOT / "results" / "predictions").glob("*.parquet")), reason="run the pipeline first")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_string(SCRIPT.format(app=ROOT / "app", page=page), default_timeout=90)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.markdown) + len(at.dataframe) > 0


@pytest.mark.skipif(not any((ROOT / "results" / "predictions").glob("*.parquet")), reason="run the pipeline first")
def test_compare_interaction_and_backtest_cost_slider():
    at = AppTest.from_string(SCRIPT.format(app=ROOT / "app", page="compare"), default_timeout=90).run()
    at.selectbox[0].select("XGBoost")  # model A
    at.selectbox[1].select("ARIMA")    # model B
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    at = AppTest.from_string(SCRIPT.format(app=ROOT / "app", page="backtest"), default_timeout=90).run()
    at.slider[0].set_value(25.0)  # 25 bps cost
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_notes_edit_insert_and_save(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTONE_NOTES", str(tmp_path / "notes.md"))
    at = AppTest.from_string(SCRIPT.format(app=ROOT / "app", page="notes"), default_timeout=60).run()
    at.text_area[0].set_value("# My analysis\n\nLSTM is not significantly better.")
    at.run()
    [b for b in at.button if b.label == "Save"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "not significantly better" in (tmp_path / "notes.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not any((ROOT / "results" / "predictions").glob("*.parquet")), reason="run the pipeline first")
def test_main_entrypoint_runs():
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
