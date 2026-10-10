"""Run the whole pipeline end-to-end, in order (several minutes on CPU)."""
import subprocess
import sys
from pathlib import Path

STEPS = ["01_download_data", "02_build_features", "03_run_classical", "04_run_ml", "05_run_lstm",
         "06_evaluate", "07_stat_tests", "08_robustness", "09_figures"]

for s in STEPS:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / f"{s}.py")], check=True)
