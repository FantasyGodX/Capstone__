import subprocess
import sys
import time
from datetime import datetime

import streamlit as st

from common import DATA_PROCESSED, DATA_RAW, RESULTS, ROOT

# (script, description, expected output, note)
STEPS = [
    ("01_download_data", "Download prices from Yahoo Finance", DATA_RAW / "prices.db", "needs internet"),
    ("02_build_features", "Build features and target", DATA_PROCESSED / "dataset.parquet", ""),
    ("03_run_classical", "Logistic, ARIMA, always-up", RESULTS / "predictions" / "classical.parquet", "~1-2 min"),
    ("04_run_ml", "Random Forest, XGBoost", RESULTS / "predictions" / "ml.parquet", "~1 min"),
    ("05_run_lstm", "LSTM", RESULTS / "predictions" / "lstm.parquet", "a few minutes"),
    ("06_evaluate", "Metrics and backtest", RESULTS / "tables" / "classification_metrics.csv", ""),
    ("07_stat_tests", "Significance tests", RESULTS / "tables" / "significance_vs_baselines.csv", ""),
    ("08_robustness", "Regime analysis", RESULTS / "tables" / "regime_trend.csv", ""),
    ("09_figures", "Static figures", RESULTS / "figures" / "equity_curves.png", ""),
]


def _run(script: str, log):
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / f"{script}.py")], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        log.code("\n".join(lines[-25:]))
    return proc.wait() == 0


def render():
    st.title("Pipeline")
    st.caption("Run the pipeline from the browser. Steps read the previous step's output, so run them in order. "
               "Re-running everything with the same data and seed reproduces the same results.")

    rows = []
    for s, desc, out, note in STEPS:
        ok = out.exists()
        rows.append({"step": s, "what": desc, "status": "done" if ok else "missing",
                     "last updated": datetime.fromtimestamp(out.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if ok else "-",
                     "note": note})
    st.dataframe(rows, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Steps to run", [s[0] for s in STEPS], default=[])
    run_all = c2.checkbox("Run all steps")
    c2.caption("Step 01 re-downloads data; Yahoo may revise history slightly, so numbers can shift a little.")
    if st.button("Run", type="primary", disabled=not (chosen or run_all)):
        todo = [s[0] for s in STEPS] if run_all else [s[0] for s in STEPS if s[0] in chosen]
        log = st.empty()
        t0 = time.time()
        for s in todo:
            with st.status(f"Running {s}...", expanded=True) as status:
                ok = _run(s, log)
                status.update(label=f"{s}: {'done' if ok else 'FAILED'}", state="complete" if ok else "error")
            if not ok:
                st.error(f"{s} failed. Later steps were skipped.")
                break
        else:
            st.success(f"Finished {len(todo)} step(s) in {time.time() - t0:.0f}s.")
        st.cache_data.clear()
