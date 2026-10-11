# How Everything Works: Logic, Code and Decisions



Code excerpts are copied from the repository and trimmed. File paths are relative to the repo root.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Data layer](#2-data-layer)
3. [Target and features](#3-target-and-features)
4. [Walk-forward validation](#4-walk-forward-validation)
5. [The models](#5-the-models)
6. [Statistical metrics](#6-statistical-metrics)
7. [The backtest](#7-the-backtest)
8. [Significance testing](#8-significance-testing)
9. [Robustness (regimes)](#9-robustness-regimes)
10. [The dashboard](#10-the-dashboard)
11. [How the tests protect the method](#11-how-the-tests-protect-the-method)
12. [Decision log](#12-decision-log)
13. [Known quirks and limitations](#13-known-quirks-and-limitations)
14. [How to extend](#14-how-to-extend)

---

## 1. The big picture

**The question:** can complex models (Random Forest, XGBoost, LSTM) predict whether a stock rises
tomorrow *significantly better* than classical models (logistic regression, ARIMA)?

**The core difficulty:** next-day stock direction is almost unpredictable. Any honest study has to
(a) never let the model see the future, (b) compare against sensible baselines, and (c) test whether a
small difference is real or luck. Almost every design choice below serves one of those three goals.

### Data flow

```
Yahoo Finance
     |  01_download_data.py        data.py
     v
data/raw/prices.db  (SQLite: OHLCV for 8 stocks + SPY + VIX)
     |  02_build_features.py       features.py
     v
data/processed/dataset.parquet     (26 features + ret_next + y, per ticker per day)
     |  03/04/05_run_*.py          walk_forward.py + models.py
     v
results/predictions/*.parquet      (one row per model, ticker, day: score, prediction, truth)
     |
     +--> 06_evaluate.py           evaluation.py + backtest.py   -> classification + backtest tables
     +--> 07_stat_tests.py         stats.py                      -> significance tables
     +--> 08_robustness.py         robustness.py                 -> regime tables
     +--> 09_figures.py                                          -> static PNG charts
                                   |
                                   v
                      app/ (Streamlit dashboard)  and  paper/paper.md
```

The key design idea: **all models produce the same kind of output** (a score per day and a 0/1
prediction), stored in one common table. Everything downstream (metrics, backtest, tests, dashboard)
works on that table and does not care which model made the predictions. That is why adding or removing a
model changes almost nothing else.

### The common model interface

Every model is a small class with the same three things (see `src/capstone/models.py`):

```python
model.fit(X_train, y_train)          # learn from the training window
model.predict_score(X_test)          # higher score = more likely "up"
model.threshold                      # score >= threshold means predict "up"
```

Probability models (logistic, forests, LSTM) use threshold 0.5. ARIMA's score is a forecast return, so its
threshold is 0.0 ("predict up if the forecast return is positive").

---

## 2. Data layer

**Files:** `src/capstone/data.py`, `scripts/01_download_data.py`

### What it does

Downloads daily open/high/low/close/volume for the 8 stocks plus SPY (market proxy) and ^VIX (volatility),
cleans them, and stores them in one SQLite table called `prices`.

```python
df = yf.download(ticker, start=start, end=end, interval="1d",
                 auto_adjust=True, progress=False, multi_level_index=False)
```

`validate()` then sorts by date, drops duplicate dates, drops rows with missing prices, and raises an error
if any price is zero or negative.

### Decisions

| Decision | Why | Consequence |
|---|---|---|
| `auto_adjust=True` | Splits and dividends would otherwise create fake price jumps that look like huge returns | Yahoo **retroactively rescales** history when new dividends are paid, so re-downloading later can change old prices slightly. This really happened in this project (ARIMA's backtest return moved from 135% to 145%) |
| Start in 2014 though modelling starts in 2015 | The 50-day moving average and 20-day windows need history, so 2014 is "warm-up" | Step 2 drops 2014 rows after computing features |
| SQLite, not PostgreSQL/MySQL | The plan mentioned PostgreSQL or MySQL, but a server adds setup friction for no analytical benefit here | One file, zero setup, easy to delete and rebuild |
| Raise an error on bad data | Silent bad data would corrupt every result downstream | The pipeline stops loudly instead |
| Validation lives in `validate()`, separate from download | Makes it unit-testable without internet | `tests/test_data.py` tests it offline |

---

## 3. Target and features

**File:** `src/capstone/features.py`

### The target

For stock price `P` on day `t`:

```
ret_next(t) = P(t+1) / P(t) - 1         the return from today's close to tomorrow's close
y(t)        = 1 if ret_next(t) > 0 else 0
```

```python
out["ret_next"] = close.shift(-1) / close - 1
out["y"] = (out["ret_next"] > 0).astype(int)
out.loc[out["ret_next"].isna(), "y"] = np.nan      # the final day has no tomorrow
```

Why direction instead of price? Predicting the price level would be dominated by the trend, and any model
can look good just by saying "tomorrow's price is about today's price". Direction is a clean yes/no
question with a roughly 50/50 baseline (up-days are 50.4% to 54.6% depending on the stock), so accuracy
near 50% means no skill.

### The leakage rule (the most important rule in the project)

> **Every feature in row `t` may only use information available at the close of day `t`.**

If this is violated, the model "predicts" tomorrow using tomorrow's data and the results are fake.
Everything in `build_features` therefore uses only past or current values (`rolling`, `ewm`, `shift(k)`
with positive `k`, `pct_change`). The only thing that looks forward is the target, which is stored in
separate columns (`ret_next`, `y`) and excluded from the model inputs:

```python
FEATURE_EXCLUDE = {"ret_next", "y"}
def feature_columns(ds): return [c for c in ds.columns if c not in FEATURE_EXCLUDE]
```

A unit test enforces this by corrupting every price after some date and checking that every feature before
that date is unchanged (`tests/test_features.py`).

### The 26 features

| Group | Features | Idea |
|---|---|---|
| Technical | `rsi14` | Relative Strength Index: recent gains vs losses (overbought/oversold) |
| | `macd`, `macd_signal`, `macd_hist` | Trend momentum: difference of 12 and 26-day exponential averages, divided by price |
| | `bb_pctb`, `bb_width` | Bollinger Bands: where price sits inside its 20-day volatility band, and how wide the band is |
| | `ma5_ratio`, `ma20_ratio`, `ma50_ratio` | Price relative to its 5, 20, 50-day average |
| | `atr14_pct` | Average True Range (typical daily range) as a fraction of price |
| Statistical | `vol5`, `vol10`, `vol20` | Rolling standard deviation of returns |
| | `ret_lag1` ... `ret_lag5` | Returns of the last 5 days (`ret_lag1` is **today's** return, which is known at the close) |
| | `skew20` | 20-day skewness of returns |
| | `mom5`, `mom10`, `mom20` | Price change over 5, 10, 20 days |
| Market context | `mkt_ret` | SPY's return today |
| | `corr_mkt20` | 20-day rolling correlation of the stock with SPY |
| | `vix`, `vix_chg` | Volatility index level and its daily change |

### Decisions

| Decision | Why |
|---|---|
| Features are **ratios and returns**, not raw price levels (`macd / c`, `c / ma - 1`) | Raw levels trend upward over time and differ wildly between stocks, which breaks learning. Ratios are comparable across time and tickers |
| **No normalisation here** (the plan lists z-scores) | Computing a mean/standard deviation over the whole dataset would leak test-period information into training. Scaling is done *inside each model*, fitted on the training window only (see sections 4 and 5) |
| `vix` is forward-filled onto each stock's dates | VIX and stock calendars occasionally differ by a day; forward-fill uses only past values |
| Rows with any missing feature are dropped (`dropna`) | Removes the warm-up period and the unlabelled final day |
| One shared feature set for all models | The comparison should isolate the *model*, not the information it receives |

---

## 4. Walk-forward validation

**File:** `src/capstone/walk_forward.py`

### Why not ordinary cross-validation?

Standard k-fold shuffles the data, so a model trained on 2023 can be tested on 2021, which lets it use the
future. Financial data is also autocorrelated, so neighbouring days are nearly duplicates. Walk-forward
validation copies how a real forecaster works: **train only on the past, test on the next unseen period,
then move forward.**

### The folds

With the default config (5-year window, test years 2020 to 2024):

```
Fold 1: train 2015-2019 -> test 2020
Fold 2: train 2016-2020 -> test 2021
Fold 3: train 2017-2021 -> test 2022
Fold 4: train 2018-2022 -> test 2023
Fold 5: train 2019-2023 -> test 2024
```

```python
def make_folds(dates, first_test, last_test, train_years):
    yr = dates.year
    for y in range(first_test, last_test + 1):
        train = (yr >= y - train_years) & (yr < y)
        test = yr == y
        idx = train.nonzero()[0]
        train[idx[-1]] = False   # purge: last train label uses the first test-year close
        yield y, train, test
```

### The purge

The label on the last training day (31 December) is "did the price rise *tomorrow*?", and tomorrow is the
first trading day of the test year. Keeping that row would let the training labels touch the test period. So
the last training row of each fold is removed.

### One fresh model per ticker per fold

```python
for t in tickers:
    d = ds.xs(t, level="ticker")
    for year, tr, te in make_folds(...):
        for name, factory in model_factories.items():
            m = factory()                       # brand-new model every time
            m.fit(d.loc[tr, feats], d.loc[tr, "y"])
            score = m.predict_score(d.loc[te, feats])
```

Models are passed as **factories** (functions that build a new model), so nothing learned in one fold can
leak into another. The output row for each test day stores `score`, `pred` (score vs threshold), `y_true`
and `ret_next`.

### Decisions

| Decision | Why | Cost |
|---|---|---|
| **Rolling** 5-year window (not expanding) | Keeps the training amount constant across folds, so differences between years are not caused by growing data. Also lets the model forget old regimes | Throws away older data |
| **Per-ticker** models | Each stock has its own dynamics and scale; simple to reason about; ARIMA is naturally per-series | Only about 1,250 training rows per model, which is small, especially for the LSTM |
| Test period 2020-2024 | Contains a crash (2020), a boom, a bear market (2022) and a rally, so models face different conditions | Only five folds, and 2020 has few earlier-crisis examples |
| Yearly steps | Simple and mirrors the plan; refitting daily would be far slower | The model gets stale during the year |

---

## 5. The models

**File:** `src/capstone/models.py`

Principle behind every hyperparameter: **financial data is extremely noisy, so models should be
heavily constrained.** An unconstrained flexible model will memorise noise and look brilliant in training
and useless in testing. No hyperparameters were tuned on the test years; all were fixed in advance (tuning
on the test set would inflate results and invalidate the significance tests).

### 5.1 Always-up (benchmark)

```python
class AlwaysUp:
    threshold = 0.5
    def predict_score(self, X): return np.ones(len(X))
```

Predicts "up" every day, which is identical to buying and holding. It answers: *"is the model better
than doing nothing smart?"* Because up-days are about 52.5% of test days, this trivial rule scores 52.5%
accuracy, which is higher than every real model. It is the most important number to compare against.

### 5.2 Logistic regression (classical baseline)

```python
make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=seed))
```

- A linear model: it weights the 26 features and outputs a probability of "up".
- `StandardScaler` is **inside the pipeline**, so it is fitted on training data only for each fold.
- `C=0.1` means fairly strong L2 regularisation (shrinks weights toward zero), appropriate when almost
  all the signal is noise.
- **Why this is the primary baseline:** it sees exactly the same 26 features as the ML models, so any gain
  from RF/XGBoost/LSTM must come from *nonlinearity or sequence modelling*, not from extra information.

### 5.3 ARIMA (classical econometric baseline)

ARIMA models a series using only its **own past values** (AR terms) and past forecast errors (MA terms). It
ignores all 26 features, which is what makes it the classical time-series reference point.

```python
self.orders = [(p, 0, q) for p in range(max_p + 1) for q in range(max_q + 1)]   # p,q in {0,1,2}
...
res = ARIMA(r, order=order, trend="c").fit()          # r = daily returns in percent
if best is None or res.aic < best[0]: best = (res.aic, order, res)
```

How it works here:

1. **Fit:** try the 9 orders `(p, 0, q)` on the training returns and keep the one with the lowest AIC
   (a score that rewards fit but penalises complexity).
2. **Forecast one day ahead through the test year *without refitting*:** the fitted parameters are reused and the
   model is rolled forward over observed test returns.

```python
ext = pd.concat([r, pd.Series([0.0])], ignore_index=True)       # placeholder for the last forecast
applied = self.res.apply(ext, refit=False)
pred = applied.predict(start=1, end=len(ext) - 1)               # forecast at k+1 made from data up to k
```

3. **Score** = forecasted next-day return; predict "up" if it is above 0 (threshold 0.0).

Decisions: `d=0` because returns are already stationary (differencing is for price levels); returns are
multiplied by 100 because the optimiser behaves better on percent-sized numbers; a small order grid keeps
the search cheap and limits overfitting; no refit during the test year keeps it comparable to the other
models. A test (`tests/test_walk_forward.py`) checks that perturbing future returns does not change earlier
forecasts.

### 5.4 Random Forest

```python
RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                       max_features="sqrt", n_jobs=-1, random_state=seed)
```

An average of 300 decision trees, each trained on a random sample of rows and a random subset of features.
Averaging many noisy trees reduces variance. The constraints are deliberate:

- `max_depth=5`: shallow trees cannot memorise individual days.
- `min_samples_leaf=20`: every leaf must rest on at least 20 days, so no rule is built on a handful of events.
- Trees need no feature scaling, so no scaler is used.

### 5.5 XGBoost

```python
XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, ...)
```

Gradient boosting builds trees one after another, each correcting the previous errors. It is more
powerful than a forest and **overfits more easily**, so it is constrained harder:

- very shallow trees (depth 3), a slow learning rate (0.03), and 200 rounds;
- `subsample` and `colsample_bytree` of 0.8 (random rows and features per tree);
- `min_child_weight=10` and `reg_lambda=5` (L2 penalty) to resist fitting noise.

### 5.6 LSTM (deep learning)

An LSTM is a neural network for sequences. Instead of seeing one day's 26 features, it sees the **last 20
days** in order, and can learn patterns across time.

**Architecture:**

```python
s.lstm = nn.LSTM(n_in, hidden, batch_first=True)    # n_in = 26, hidden = 32, one layer
s.drop = nn.Dropout(dropout)                        # 0.3
s.out  = nn.Linear(hidden, 1)
def forward(s, x):
    h, _ = s.lstm(x)
    return s.out(s.drop(h[:, -1])).squeeze(-1)      # only the last time step -> one logit
```

**Training procedure:**

1. Standardise with a scaler fitted on the training window, then clip to [-5, 5] so extreme days cannot
   dominate.
2. Build overlapping 20-day windows; the label for a window is `y` of its last day.
3. Hold out the **last 15% of windows (chronologically)** as a validation set for early stopping.
4. Train with Adam (`lr=1e-3`, `weight_decay=1e-4`), batch size 64, up to 40 epochs, using binary
   cross-entropy.
5. After each epoch measure validation loss. If it fails to improve by at least `1e-4` for 5 epochs in a
   row, stop, and **restore the best weights seen**.

```python
if v < best - 1e-4:
    best, bad = v, 0
    best_state = {k: p.clone() for k, p in self.net.state_dict().items()}
else:
    bad += 1
    if bad >= self.patience: break
self.net.load_state_dict(best_state)
```

6. At prediction time, the **last 19 training rows** are prepended so that the first test days also have a
   full 20-day window. This uses only past features, so it is not leakage.

**Why it is small (32 units, dropout, weight decay, early stopping):** with about 1,050 training windows per
fold, a large network would instantly overfit. All the regularisers are there to keep it honest. The seed
is fixed for reproducibility, but only one seed is used (a limitation, since neural network results vary
from seed to seed).

**Shuffling:** training windows are shuffled *within the training set only*. That is acceptable because each
window is internally ordered and the test set is strictly later in time.

### 5.7 Why these models and not the others in the plan?

The plan's abstract commits to "one carefully chosen model per class" while the body lists more. The study
keeps two classical models, two ML models and one deep model. **SVM, GRU and GARCH were dropped** to finish
in depth rather than breadth: GRU is very similar to LSTM, SVM is another nonlinear classifier that would add
little to the conclusion, and GARCH models volatility, not direction.

---

## 6. Statistical metrics

**File:** `src/capstone/evaluation.py`

Computed on **all out-of-sample predictions pooled** (8 stocks x about 1,256 days = 10,048 predictions per
model). Definitions, with "positive" meaning "predicted up":

| Metric | Meaning |
|---|---|
| Accuracy | Share of days where the predicted direction was right |
| Precision | Of the days predicted up, how many really were up |
| Recall | Of the days that really were up, how many were predicted up |
| F1 | Combines precision and recall |
| ROC-AUC | How well the **score** ranks up-days above down-days; 0.5 means no ranking skill |

```python
auc = roc_auc_score(y, s) if s.nunique() > 1 else 0.5      # a constant score carries no ranking information
```

### How to read them (important)

- **Accuracy and F1 are dominated by how often a model says "up".** Because up-days are the majority,
  predicting "up" a lot raises recall and F1 even with no skill. That is why AlwaysUp has the highest F1
  (0.689), and why the LSTM (79% recall) beats logistic on F1 without being better at ranking.
- **ROC-AUC is the cleanest signal measure** because it ignores the threshold. All models are between about
  0.495 and 0.506, i.e. essentially no ranking power.
- AlwaysUp is given AUC 0.5 explicitly, because a constant score has no ordering.

---

## 7. The backtest

**File:** `src/capstone/backtest.py`

Statistical accuracy does not tell you whether a model would make money, so every prediction is also turned
into a simple trading strategy.

### The trading rule

```
At the close of day t:   predict up (1)  -> hold the stock from t to t+1
                          predict down (0) -> hold cash (earn 0)
```

```python
def strategy_returns(pred, ret_next, cost):
    pos = pred.astype(float)
    turnover = pos.diff().abs()
    turnover.iloc[0] = pos.iloc[0]          # the first entry from cash also costs
    return pos * ret_next - cost * turnover
```

- `pos * ret_next`: the return earned while holding. In cash (`pos = 0`) the return is zero.
- `cost * turnover`: 0.05% charged whenever the position changes (buying or selling), so a model that
  flips back and forth pays more.
- **No leverage and no short selling:** the position is only 0 or 1.

### Portfolio

Each model gets a daily return series per stock. The **portfolio** is the equal-weight average across the
8 stocks each day:

```python
port = wide.T.groupby(level="model").mean().T     # equal-weight, daily rebalanced
```

### Performance measures (`performance`)

| Measure | Formula |
|---|---|
| Cumulative return | `final equity - 1`, where equity = product of `(1 + daily return)` |
| Annual return | `equity^(1 / years) - 1`, with `years = days / 252` |
| Sharpe ratio | `mean(daily) / std(daily) * sqrt(252)` (risk-free rate assumed 0) |
| Max drawdown | Worst fall from a running peak, `min(equity / running max - 1)` |
| Calmar ratio | `annual return / |max drawdown|` |

### Decisions and caveats

| Decision | Why | Caveat |
|---|---|---|
| Long/cash only | Simplest realistic rule, matches the plan | No shorting means models can only profit from up-calls |
| 0.05% cost | The plan's example value | Real costs vary; the dashboard lets you change it |
| Execution at the close that produced the signal | Standard simplification | Assumes you can trade at the close price with no slippage, which is optimistic |
| Risk-free rate = 0 | Keeps Sharpe simple | Slightly flatters all strategies equally |
| Buy-and-hold (AlwaysUp) shown alongside | It is the honest benchmark | Strongly bullish sample (NVDA, TSLA, AAPL) makes it hard to beat |

An important reading rule: **a high backtest return can come from simply being invested.** The LSTM is long
about 79% of days in a rising market, so some of its performance is market exposure, not timing skill. That
is why the backtest is read together with AUC and the significance tests.

---

## 8. Significance testing

**Files:** `src/capstone/stats.py`, `scripts/07_stat_tests.py`

A difference of half a percentage point in accuracy could easily be luck. These tests ask how likely a
difference that large is **if the models were really equally good**.

### What is compared

Three **challengers** (Random Forest, XGBoost, LSTM) against two **baselines** (Logistic, ARIMA) gives six
comparisons. Every test is **one-sided**, because H1 says the challenger is *better*:
`H0: no better`, `H1: challenger better`.

### A crucial step: averaging across tickers per day

On a given day all 8 stocks tend to move together, so their prediction errors are correlated. Treating the
10,048 ticker-days as independent would overstate the evidence. So the date-level series is built first:

```python
correct = preds.pivot_table(index="date", columns="model", values="correct", aggfunc="mean")
```

This gives one number per date (the share of tickers predicted correctly), about 1,256 values per model.

### 8.1 Diebold-Mariano test

Tests whether two forecasts have different average loss. The loss is 0/1 (1 for a wrong call), the only loss
every model shares (ARIMA has no probabilities).

```python
d = loss_base - loss_chal                       # positive when the challenger is better
lags = int(np.floor(n ** (1 / 3)))              # Newey-West lag
...
dm = dbar / np.sqrt(lrv / n)                    # lrv = long-run variance (autocorrelation-robust)
dm *= hln                                       # Harvey-Leybourne-Newbold small-sample correction
return dm, 1 - stats.t.cdf(dm, df=n - 1)
```

- **Newey-West variance** corrects for the daily loss differences being autocorrelated.
- **HLN correction** and the t distribution give more accurate p-values for finite samples.

### 8.2 Stationary block bootstrap

Resamples the data many times to see how much the statistic wobbles by chance. Because returns are
autocorrelated, it resamples **blocks of consecutive days** (average length 10) instead of single days:

```python
p = 1.0 / mean_block
new = rng.random((n_boot, n)) < p                      # with probability 1/10 start a new block
idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)   # else continue the block
```

`bootstrap_diff` resamples the *same dates for both models* (a **paired** bootstrap, which keeps their
correlation), 5,000 times, and returns the observed difference, a 95% confidence interval (2.5th to 97.5th
percentiles of the resampled differences) and a one-sided p-value (computed by centring the resampled
distribution at zero). It is used for:

- the **accuracy difference** (the mean of the date-level hit rate), and
- the **Sharpe difference** on the portfolio's daily returns.

A confidence interval containing 0 means the difference is not distinguishable from zero.

### 8.3 Paired t-test

Compares accuracies of the two models across matched units, here the 40 (ticker, year) pairs
(8 stocks x 5 years). It is the test named in the plan, but it is the **weakest** here, since the pairs are
not independent (tickers move together within a year), which makes it somewhat too optimistic. Treat it as
supporting, not primary, evidence.

### 8.4 Holm correction

Running six comparisons gives six chances for a false positive. Holm's step-down method adjusts the p-values:

```python
running = max(running, min(1.0, (m - i) * pvals[k]))    # i-th smallest p-value is multiplied by (m - i)
```

It is applied separately to each family of six p-values (DM, bootstrap accuracy, t-test, bootstrap Sharpe).
**Decisions should use the `_holm` columns.** The dashboard's Statistics page shows both raw and adjusted values.

### 8.5 Sharpe versus buy-and-hold

A separate table tests each model's Sharpe against buy-and-hold (AlwaysUp), because the backtest suggested the
LSTM might beat it. The difference is positive but not significant.

### Decisions

| Decision | Why |
|---|---|
| One-sided tests | Matches the plan's H1 (challenger better) |
| 0/1 loss for DM | Only loss common to all five models |
| Block bootstrap, mean block 10 | Handles autocorrelation; 10 days is a common compromise |
| Holm, not Bonferroni | Equally valid control of false positives but less conservative |
| Two baselines | Logistic isolates nonlinearity; ARIMA is the traditional econometric benchmark |

---

## 9. Robustness (regimes)

**File:** `src/capstone/robustness.py`

A model might be useless on average but good in specific conditions, or an apparent edge might exist only in
one kind of market. Predictions are split by market regime **defined using only information known at the
close of day `t`**:

```python
bull = spy > spy.rolling(200).mean()          # SPY above its 200-day average -> bull, else bear
vol  = vix > 20                               # VIX above 20 -> high volatility
```

For each regime the code reports accuracy (pooled across tickers), the portfolio Sharpe on those days, and a
one-sided Diebold-Mariano test of each challenger against Logistic, Holm-adjusted across the six tests.

Caveats:
- Bull/bear splits are lopsided (about 78% bull, 22% bear of test days), so bear-market numbers are noisy.
- Regime Sharpe ratios use non-contiguous days, so they are rough descriptions, not clean estimates.
- Thresholds (200 days, VIX 20) are conventional choices, not optimised.
- The by-year and by-ticker tables stand in for the plan's "different time windows" check; the *window length*
  itself was not varied.

---

## 10. The dashboard

**Folder:** `app/`

Built with Streamlit (Python web apps) and Plotly (interactive charts). It is a view over the same code and
tables the pipeline produces; it contains no separate modelling logic.

### Structure

```
app/app.py              registers the pages with st.navigation
app/common.py           loaders, model colours, chart theme, path helpers
app/pages_<name>.py     each defines render(), the function that draws the page
```

`app.py` adds `app/` to the import path and lists the pages:

```python
st.Page(pages_overview.render, title="Overview", icon="🏠", url_path="overview", default=True)
```

### Key mechanisms

| Mechanism | How it works |
|---|---|
| **Shared code** | `common.py` puts `src/` on `sys.path`, so pages call `capstone.backtest`, `capstone.stats`, etc. directly. The Backtest page's cost slider really re-runs `backtest_all`, and the Compare page really runs the same DM and bootstrap functions as the pipeline |
| **Caching** | Loaders and the backtest runner use `@st.cache_data`, so changing a filter does not reload files or recompute everything. The Pipeline page calls `st.cache_data.clear()` after a run |
| **Consistent colours** | `COLOR` and `DASH` in `common.py` fix one colour and line style per model, so a model never changes colour between charts. Dash styles are a second cue for colour-blind readers |
| **Pipeline page** | Starts each script with `subprocess.Popen([sys.executable, "-u", script])` and streams its output into the page |
| **Notes page** | Keeps the text in `st.session_state`; the Save button writes `analysis/notes.md`. The path comes from the function `notes_path()`, evaluated on every call so tests can redirect it with an environment variable |
| **Missing results** | `require_results()` shows a friendly message instead of crashing when the pipeline has not been run |

### Which page uses what

| Page | Reads | Computes live |
|---|---|---|
| Overview | tables | none |
| Performance | predictions | metrics, ROC curves, rolling accuracy, confusion matrix |
| Backtest | predictions | the whole backtest (cost, tickers, dates) |
| Statistics | significance tables | none |
| Robustness | regime tables, predictions | by-year accuracy |
| Compare | predictions | DM, bootstrap, sign test, backtest for two models |
| Data explorer | prices database, dataset | moving averages, feature histograms |

---

## 11. How the tests protect the method

28 tests in `tests/`. Each one guards a specific assumption, so a future change that breaks the method fails
loudly:

| Test file | Protects against |
|---|---|
| `test_features.py` | **Look-ahead leakage in features**, and a wrong target |
| `test_walk_forward.py` | Folds out of order, a missing purge, **ARIMA using future data** |
| `test_robustness.py` | Regime labels that peek at the future |
| `test_models.py` | Models that cannot learn a simple signal, or output scores outside [0, 1]; LSTM misaligned with the test rows |
| `test_backtest.py` | Wrong cost handling, Sharpe or drawdown arithmetic (checked against hand-calculated values) |
| `test_stats.py` | Broken DM, bootstrap, t-test or Holm (checked on cases with known answers) |
| `test_data.py` | Bad data slipping through, database round-trip errors |
| `test_app.py` | Any dashboard page crashing; the main entry point; selecting models; the cost slider; saving notes |

Writing the dashboard tests caught one real bug (saved CSV files lose their index name, which broke the Overview
page). An invalid import in `app.py` was found by inspection, after which the entrypoint test was added so that
`app.py` itself is exercised. One test originally wrote to the real notes file, which led to the notes path being
resolved on every call.

---

## 12. Decision log

| # | Decision | Alternative considered | Reason | Trade-off |
|---|---|---|---|---|
| 1 | Predict direction (up/down) | Predict price or return | Clean 50/50 problem; avoids trend-driven fake accuracy | Ignores the size of moves |
| 2 | Yahoo Finance via `yfinance` | Alpha Vantage | Free, no key, adjusted prices | Prices get revised retroactively |
| 3 | SQLite | PostgreSQL / MySQL | Zero setup, one file | Not a multi-user database |
| 4 | 8 tickers across sectors | 10-12 | Enough diversity, manageable run time | Includes known winners, so selection bias |
| 5 | No scaling in feature step | Z-score whole dataset | Avoids leaking test statistics | Each model must scale itself |
| 6 | Rolling 5-year window, yearly step | Expanding window, daily refit | Constant training size; fast | Stale models within a year |
| 7 | Purge last training row | None | Its label touches the test period | Loses one row per fold |
| 8 | Separate model per ticker | One pooled model | Simplicity, ticker-specific dynamics | Small samples per model |
| 9 | Logistic regression as primary baseline | Only ARIMA | Same features as ML models, so only the model differs | The "classical" baseline is multivariate |
| 10 | ARIMA grid p,q <= 2, d = 0, AIC | Auto-ARIMA, larger grid | Returns are stationary; low risk of overfitting | May miss better orders |
| 11 | Heavily regularised RF/XGBoost | Tuned deep trees | Noise dominates; no test-set tuning | May underfit any real signal |
| 12 | Small LSTM, early stopping | Large or stacked network | About 1,050 windows per fold would overfit anything bigger | Limited capacity to find structure |
| 13 | Fixed 0.5 threshold | Tune threshold per model | Tuning needs extra validation data and could overfit | Predictions follow the model's own calibration |
| 14 | No hyperparameter search | Grid or Bayesian search | Would inflate results unless done in nested validation | Models are not individually optimised |
| 15 | One seed (42) | Average over many seeds | Time | LSTM results could vary with the seed |
| 16 | Long/cash backtest | Long/short | Matches the plan; no leverage | Cannot profit from down-calls |
| 17 | 0/1 loss for DM | Brier or log loss | Only loss all models share | Ignores confidence |
| 18 | Average across tickers before testing | Pool 10,048 ticker-days | Respects cross-sectional correlation | Fewer effective observations (more honest) |
| 19 | One-sided tests + Holm | Two-sided, Bonferroni | Matches H1; Holm less conservative | Needs correct family definition |
| 20 | Block bootstrap | Plain bootstrap | Returns are autocorrelated | Block length is a tuning choice |
| 21 | Dropped SVM, GRU, GARCH | Include all | Depth over breadth in one semester | Fewer model families |
| 22 | Streamlit dashboard reusing pipeline code | Separate web app | One source of truth; live recomputation | Local use, not a public website |

---

## 13. Known quirks and limitations

These are real, small imperfections found while re-reading the code. None changes the conclusions, but you
should know about them in case you are asked.

1. **ARIMA restarts its filter at the start of each test year.** `predict_score` runs the fitted model over
   the test returns only, so the model state is re-initialised rather than carried over from the end of
   training. The first few forecasts of each year are probably close to the model's constant term. This is
   conservative (it never uses future data) but slightly weakens ARIMA.
2. **LSTM windows skip one day at each year boundary.** The purge removes 31 December from training, but the
   test windows are completed using the last 19 rows of the *training* set. So the first 19 test windows of
   each year contain a one-day gap. It uses only past data (no leakage), but the sequence is not perfectly
   contiguous there.
3. **The paired t-test treats the 40 (ticker, year) pairs as independent**, though stocks move together. It is
   therefore the least reliable of the three tests, and conclusions should lean on DM and the bootstrap.
4. **The sign test on the Compare page** counts ticker-days as independent, for the same reason. It is a
   convenience view; use DM and the bootstrap for conclusions.
5. **ROC-AUC pools scores across tickers and years.** Score scales can differ between folds, so pooled AUC is
   an approximation. The metrics broken down by ticker and year are in the tables.
6. **Execution is assumed at the signal close with no slippage**, and each stock's cost is charged on its own
   turnover; daily rebalancing of the equal-weight portfolio is not charged separately.
7. **The LSTM validation set (last 15% of the training window) is used to pick the stopping epoch but is not
   added back for final training.** That wastes a little data, and the validation windows share some input rows
   with the training windows (their labels do not overlap).
8. **Survivorship and selection bias:** the tickers are familiar long-run winners, which flatters long-only
   strategies and makes buy-and-hold hard to beat.
9. **Power:** with about 1,250 test days, differences smaller than about 1.5 percentage points of accuracy cannot be
   detected. "Not significant" does not mean "equal".
10. **Data revisions:** re-downloading can shift results (observed: ARIMA's backtest return 135% to 145%).
    The models themselves are deterministic given the data.
11. **Minor code note:** the line `torch.set_num_threads(max(1, torch.get_num_threads()))` in the LSTM does not
    change anything (it sets the thread count to its current value). It is harmless.

---

## 14. How to extend

### Add a new model

1. In `src/capstone/models.py`, write a class with `threshold`, `fit(X, y)` and `predict_score(X)` (higher score
   means more likely up). Use a scaler inside the model if it needs one, fitted in `fit`.
2. Register it in a runner such as `scripts/04_run_ml.py`:
   ```python
   preds = run_walk_forward(ds, {"RandomForest": ..., "MyModel": lambda: MyModel(seed)}, cfg)
   ```
3. Add the name to `ORDER` in `scripts/06_evaluate.py`, and to `STYLE` in `scripts/09_figures.py`.
4. Add it to `ORDER`, `COLOR`, `DASH` and `CLASS` in `app/common.py`.
5. To include it in the significance tests, add it to `CHALLENGERS` or `BASELINES` in
   `scripts/07_stat_tests.py` (this changes the number of comparisons and so the Holm adjustment).
6. Re-run the pipeline, then update the numbers in the paper by hand.

### Change the experiment

Edit `config.yaml` (tickers, years, window length, cost, seed) and re-run from the earliest affected step.
Changing tickers or dates needs steps 1 or 2 onward; changing only the cost needs step 6 onward (or just use the
Backtest page's slider).

### Change a feature

Edit `build_features` in `features.py`. Keep the leakage rule: use only data up to day `t`. Run `pytest`; the
leakage test will fail if you accidentally look forward. Then re-run from step 2.
