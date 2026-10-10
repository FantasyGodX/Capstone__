"""Headless smoke tests: every dashboard page must render without raising."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "performance", "backtest"]

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
