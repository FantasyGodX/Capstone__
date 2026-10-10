# Machine Learning vs. Classical Time-Series Models for Short-Term Stock Return Prediction

Capstone research project. **Question:** do ML/deep-learning models give statistically
significant improvements over a classical baseline in predicting next-day return
*direction* for liquid US equities, under strict walk-forward validation and realistic
trading costs?

- H0: no significant difference in predictive performance between classical and ML models.
- H1: ML and/or deep-learning models significantly outperform the classical baseline.

## Design decisions (scope)

The research plan lists more models than its abstract commits to. To finish in depth
rather than breadth, the core comparison is:

| Class     | Models                         |
|-----------|--------------------------------|
| Classical | Logistic Regression, ARIMA     |
| ML        | Random Forest, XGBoost         |
| Deep      | LSTM                           |

Plus naive benchmarks (always-up, buy-and-hold). SVM, GRU and GARCH are listed in the
plan as optional extensions. Other deliberate choices:

- **Data:** Yahoo Finance via `yfinance`, 8 tickers, 2014-2024 (2014 is indicator warm-up).
- **Storage:** SQLite instead of PostgreSQL/MySQL (zero-setup, file based, fully reproducible).
- **Validation:** rolling walk-forward, 5-year train window, test years 2020-2024.
- **Target:** `y_t = 1 if r_{t+1} > 0`, using features known at close of day `t`.

## Layout

```
config.yaml        experiment configuration (single source of truth)
src/capstone/      pipeline package (data, features, models, backtest, stats)
scripts/           numbered entry points, run in order
tests/             unit tests (incl. look-ahead-leakage checks)
data/              raw + processed data (git-ignored, regenerated)
results/           tables and figures
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Run

```
python scripts/run_all.py     # download -> features -> models -> evaluation -> tests -> figures
```

Scripts `01`-`09` can also be run individually, in order. Each step reads the previous step's output.

## Headline result

We fail to reject H0. All models reach ~50.6-51.7% next-day accuracy (ROC-AUC ~0.50), below the
always-up benchmark (52.5%). No accuracy improvement from RF, XGBoost or LSTM over the
logistic baseline is significant after Holm correction. The LSTM has the best backtest Sharpe
(1.78 vs 1.31) but that is significant only against ARIMA. Full write-up: [paper/paper.md](paper/paper.md);
tables in `results/tables/`, figures in `results/figures/`.

## Progress

Complete. Each pipeline stage is one commit in the git history.
