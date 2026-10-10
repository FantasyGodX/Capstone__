# Machine Learning vs. Classical Time-Series Models for Short-Term Stock Return Prediction

Capstone research project. **Question:** do ML/deep-learning models give statistically
significant improvements over a classical baseline in predicting next-day return
*direction* for liquid US equities, under strict walk-forward validation and realistic
trading costs?

- **H0:** no significant difference in predictive performance between classical and ML models.
- **H1:** ML and/or deep-learning models significantly outperform the classical baseline.

**Headline result:** we fail to reject H0. All models reach about 50.6-51.7% next-day accuracy
(ROC-AUC about 0.50), below the always-up benchmark (52.5%). No accuracy improvement from Random
Forest, XGBoost or LSTM over the logistic baseline is significant after Holm correction. The LSTM has
the best backtest Sharpe ratio (1.78 vs 1.31 for logistic regression), but that difference is
significant only against ARIMA. Full write-up: [paper/paper.md](paper/paper.md).

---

## Contents

1. [Quick start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Using the dashboard](#5-using-the-dashboard)
6. [Configuration](#6-configuration)
7. [Tests](#7-tests)
8. [Project layout](#8-project-layout)
9. [How the method works](#9-how-the-method-works)
10. [Reproducibility](#10-reproducibility)
11. [Troubleshooting](#11-troubleshooting)
12. [Scope and limitations](#12-scope-and-limitations)
13. [Typical workflows](#13-typical-workflows)

---

## 1. Quick start

Windows (PowerShell):

```powershell
git clone https://github.com/FantasyGodX/Capstone__.git
cd Capstone__
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/run_all.py          # generates all data, predictions, tables and figures
streamlit run app/app.py           # opens the dashboard at http://localhost:8501
```

macOS / Linux: identical, except activate with `source .venv/bin/activate`.

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Developed and tested on 3.14 |
| git | To clone and commit |
| Internet | Needed for step 1 only (downloads prices from Yahoo Finance) |
| Disk | About 2 GB, mostly PyTorch |
| GPU | Not needed; everything runs on CPU |

## 3. Installation

```powershell
python -m venv .venv                 # create an isolated environment
.venv\Scripts\activate               # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt      # pandas, scikit-learn, statsmodels, xgboost, torch, yfinance, streamlit, plotly, ...
pip install -e .                     # makes the `capstone` package importable from scripts and the app
```

Check the install with `pytest` (see [Tests](#7-tests)). Dashboard tests are skipped until the pipeline
has produced results.

## 4. Running the pipeline

The pipeline is nine numbered scripts. Each reads the previous step's output and writes its own.

### Option A: one command

```powershell
python scripts/run_all.py
```

Runs all nine steps in order and stops at the first failure. Total time is a few minutes on a laptop
CPU (data download and model fitting dominate).

### Option B: one step at a time

```powershell
python scripts/01_download_data.py
python scripts/02_build_features.py
...
python scripts/09_figures.py
```

You can re-run any single step; later steps then need re-running to pick up the change.

### Option C: from the dashboard

Open the **Pipeline** page, tick **Run all steps** (or choose individual steps) and press **Run**. Logs
stream into the page and cached results are refreshed afterwards.

### What each step does

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_download_data.py` | Yahoo Finance | `data/raw/prices.db` (SQLite) |
| 2 | `02_build_features.py` | prices | `data/processed/dataset.parquet` (26 features + target) |
| 3 | `03_run_classical.py` | dataset | `results/predictions/classical.parquet` (Logistic, ARIMA, always-up) |
| 4 | `04_run_ml.py` | dataset | `results/predictions/ml.parquet` (Random Forest, XGBoost) |
| 5 | `05_run_lstm.py` | dataset | `results/predictions/lstm.parquet` |
| 6 | `06_evaluate.py` | predictions | `results/tables/classification_*.csv`, `backtest_*.csv`, `results/backtest/*.parquet` |
| 7 | `07_stat_tests.py` | predictions, backtest | `results/tables/significance_vs_baselines.csv`, `sharpe_vs_buy_and_hold.csv` |
| 8 | `08_robustness.py` | predictions, prices | `results/tables/regime_*.csv` |
| 9 | `09_figures.py` | tables, backtest | `results/figures/*.png` |

`data/`, `results/predictions/` and `results/backtest/` are git-ignored because they are regenerated.
The tables and figures in `results/tables/` and `results/figures/` are committed.

## 5. Using the dashboard

```powershell
streamlit run app/app.py
```

It opens at http://localhost:8501 (use `--server.port 8600` to change the port). The dashboard needs the
pipeline results to exist; otherwise pages show a message pointing to the Pipeline page.

| Page | What you can do |
|---|---|
| **Overview** | Headline metrics, an automatic summary of the significance tests, and the cumulative-return chart |
| **Performance** | Filter by model, ticker and test year. Metric table, ROC curves, rolling accuracy, confusion matrix, ticker x model heatmap |
| **Backtest** | Transaction-cost slider (re-runs the backtest live), ticker and date selection, cumulative return, drawdown, per-ticker results, CSV download |
| **Statistics** | Significance tables with an adjustable alpha, confidence-interval plots for accuracy and Sharpe differences, Sharpe vs buy-and-hold |
| **Robustness** | Bull vs bear, high vs low volatility, results by year and by ticker |
| **Compare** | Pick any two models for a head-to-head: accuracy difference, agreement, live Diebold-Mariano, bootstrap and sign tests, per-ticker and per-year differences |
| **Data explorer** | Candlestick prices with moving-average and Bollinger overlays, feature distributions by next-day direction, feature correlations, dataset table |
| **Analysis notes** | Markdown editor with live preview. Insert any results table or figure from the sidebar, then **Save** to `analysis/notes.md` or download the `.md` |
| **Pipeline** | Status of each step (done/missing, last updated) and buttons to run steps |

Notes on the dashboard:

- The Compare page's tests are single, unadjusted tests. If you look at many pairs, correct for multiple
  comparisons (the Statistics page applies Holm correction).
- `analysis/notes.md` does not exist until you press Save. Commit it with git to keep your notes in the repo.
- Changing the transaction cost or tickers on the Backtest page only affects that page; it does not rewrite
  the saved tables.

## 6. Configuration

All experiment settings live in [config.yaml](config.yaml); every script reads it.

| Setting | Meaning | Default |
|---|---|---|
| `tickers` | Equities to model | AAPL, MSFT, NVDA, JPM, TSLA, XOM, JNJ, WMT |
| `market_ticker`, `vix_ticker` | Market context series | SPY, ^VIX |
| `start_date`, `end_date` | Download range (2014 is indicator warm-up) | 2014-01-01 to 2024-12-31 |
| `walk_forward.first_test_year`, `last_test_year` | Out-of-sample years | 2020 to 2024 |
| `walk_forward.train_years` | Rolling training window | 5 |
| `backtest.transaction_cost` | Cost per unit of turnover | 0.0005 (0.05%) |
| `stats.bootstrap_iterations`, `stats.alpha` | Bootstrap draws, significance level | 5000, 0.05 |
| `seed` | Random seed | 42 |

After editing `config.yaml`, re-run the affected steps (a ticker or date change means re-running from step
1 or 2). **The paper's numbers are typed in by hand**, so update `paper/paper.md` after any re-run.

Model hyperparameters are set in [src/capstone/models.py](src/capstone/models.py). They were fixed in
advance and not tuned on test years.

## 7. Tests

```powershell
pytest
```

28 tests, about 10 seconds. They cover:

- Data validation and the SQLite round trip.
- **Look-ahead leakage:** perturbing future prices must not change any earlier feature or regime label.
- Walk-forward folds are chronological and purged; ARIMA forecasts use only past returns.
- Backtest arithmetic, statistical tests (Diebold-Mariano, bootstrap, paired t, Holm) against known values.
- Every dashboard page renders headlessly, plus interaction tests (changing models, the cost slider, saving notes).

The dashboard tests are skipped automatically if `results/predictions/` is empty.

## 8. Project layout

```
config.yaml          experiment configuration (single source of truth)
requirements.txt     dependencies
src/capstone/        pipeline logic
  data.py            Yahoo Finance download, validation, SQLite
  features.py        26 features and the next-day direction target
  models.py          Logistic, ARIMA, Random Forest, XGBoost, LSTM, always-up
  walk_forward.py    rolling train/test engine (per ticker, purged)
  evaluation.py      accuracy, precision, recall, F1, ROC-AUC
  backtest.py        long/cash backtest with transaction costs
  stats.py           Diebold-Mariano, block bootstrap, paired t-test, Holm
  robustness.py      bull/bear and volatility regimes
scripts/             numbered entry points (01-09) and run_all.py
app/                 Streamlit dashboard (app.py + one file per page)
tests/               pytest suite
data/                raw and processed data (git-ignored)
results/             predictions (ignored), tables and figures (committed)
paper/paper.md       the write-up
analysis/notes.md    your notes from the dashboard (created on first Save)
```

## 9. How the method works

1. **Target.** For each stock and day *t*: `y_t = 1` if the close on day *t+1* is above the close on day *t*.
2. **Features (26).** Technical (RSI, MACD, Bollinger, moving-average ratios, ATR), statistical (rolling
   volatility, lagged returns, skewness, momentum) and market context (SPY return, rolling correlation, VIX).
   Every feature on day *t* uses information only up to the close of day *t*.
3. **Walk-forward validation.** Train on the previous five calendar years, test on the next year
   (2015-19 to 2020, 2016-20 to 2021, ..., 2019-23 to 2024). A fresh model is fitted per ticker and fold.
   The last training row of each fold is dropped because its label depends on the first test-year price.
   Scalers are fitted on training data only.
4. **Models.** Logistic regression (L2) and ARIMA (order chosen by AIC) as the classical baselines; Random
   Forest and XGBoost; an LSTM over a 20-day window with early stopping on a chronological tail; plus an
   always-up benchmark equivalent to buy-and-hold.
5. **Statistical metrics.** Accuracy, precision, recall, F1 and ROC-AUC on the pooled out-of-sample predictions.
6. **Backtest.** If the model predicts up, hold the stock for the next day; otherwise hold cash. No leverage,
   0.05% cost per unit of turnover, equal-weight portfolio across tickers. Reports cumulative return, Sharpe,
   maximum drawdown and Calmar ratio.
7. **Significance tests.** Each challenger (RF, XGBoost, LSTM) versus each classical baseline (Logistic, ARIMA),
   one-sided as in H1: Diebold-Mariano on 0/1 loss, stationary block-bootstrap confidence intervals for accuracy
   and Sharpe differences, and a paired t-test. Holm correction across the six comparisons.
8. **Robustness.** Results split by bull/bear (SPY vs its 200-day average) and by volatility (VIX above or below
   20), both defined using only information known at the close of each day, plus by year and by ticker.

## 10. Reproducibility

- The seed is fixed in `config.yaml`, and the models are deterministic given the data (refitting gives the
  same predictions).
- **Yahoo Finance revises historical adjusted prices.** Re-running step 1 on a different day can change the
  inputs slightly. In one re-run, ARIMA's backtest cumulative return moved from 135% to 145% and XGBoost's from
  219% to 209%, while the conclusions did not change. The paper's tables correspond to the data downloaded on
  2026-10-10.
- To reproduce the paper's numbers exactly, keep the existing `data/raw/prices.db` and skip step 1, starting
  from `02_build_features.py`.
- `data/raw/prices.db` is not stored in git. If you want a frozen copy, back it up before re-downloading.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| PowerShell: "running scripts is disabled" when activating the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError: capstone` | Activate the venv and run `pip install -e .` from the project root |
| `ModuleNotFoundError` for another package | You are probably using the system Python; activate `.venv` first |
| Dashboard says "No results found yet" | Run the pipeline (`python scripts/run_all.py` or the Pipeline page) |
| Step 1 fails or returns no data | Check your internet connection. Yahoo may rate-limit; wait a minute and retry |
| `pip install torch` fails | Your Python version may be too new or old for the available wheels; use Python 3.10-3.13 or see pytorch.org for install instructions |
| Port 8501 already in use | `streamlit run app/app.py --server.port 8600` |
| Streamlit asks for an email on first run | Press Enter to skip, or start with `--server.headless true` |
| Dashboard shows old numbers after a re-run | Press `C` in the app, choose "Clear cache", or restart it |
| Dashboard tests are skipped | No results yet; run the pipeline first |

## 12. Scope and limitations

Scope choices (the plan lists more models than its abstract commits to):

- Core comparison: Logistic Regression and ARIMA (classical), Random Forest and XGBoost (ML), LSTM (deep learning).
  SVM, GRU and GARCH were left out as optional extensions.
- SQLite is used instead of PostgreSQL/MySQL so no database server is needed.
- The "different time windows" robustness check is covered by the per-year results, not by varying the window length.

Main limitations (full list in the paper):

- **Statistical power:** with about 1,250 test days, accuracy differences under roughly 1.5 percentage points
  cannot be detected. "Not significant" is not proof of equality.
- **Selection bias:** the tickers are well-known long-run winners, which flatters long-biased strategies.
- **Single seed and fixed hyperparameters** for the stochastic models.
- Zero risk-free rate, one transaction-cost level, close-to-close execution, no slippage or taxes.

## 13. Typical workflows

**Just look at the results.** Install, run `python scripts/run_all.py`, then `streamlit run app/app.py`. Read the
write-up in [paper/paper.md](paper/paper.md).

**Write your analysis.** Open **Analysis notes**, insert tables and figures from the sidebar, write your text, press
Save, then `git add analysis/notes.md && git commit && git push`.

**Test a different question.** Change the tickers, dates or cost in `config.yaml`, re-run the pipeline, then compare
results in the dashboard. Remember to update the paper's numbers.

**Try another model.** Add a class with `fit(X, y)`, `predict_score(X)` and a `threshold` attribute to
`src/capstone/models.py`, register it in one of the `scripts/0[3-5]_*.py` files, and add it to the `ORDER`, `COLOR`
and `DASH` lists in `app/common.py` and `scripts/06_evaluate.py` / `09_figures.py` so it shows up in tables and charts.
