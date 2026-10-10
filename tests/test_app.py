"""Headless smoke tests: every dashboard page must render without raising."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
PAGES = ["overview"]


def _results_exist():
    return any((Path(APP).parents[1] / "results" / "predictions").glob("*.parquet"))


@pytest.mark.skipif(not _results_exist(), reason="run the pipeline first")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
