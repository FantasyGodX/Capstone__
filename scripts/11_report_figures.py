"""Step 11 (optional): print-quality figures for the paper and poster -> paper/figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capstone import report  # noqa: E402

if __name__ == "__main__":
    extra = report.make_all()
    print("figures written to", report.FIG)
    print(sorted(p.name for p in report.FIG.glob("*.png")))
