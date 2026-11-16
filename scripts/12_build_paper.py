"""Step 12 (optional): build the research paper -> paper/paper.md, paper/paper.html and paper/paper.pdf.

All numbers are read from results/ at build time, so the paper cannot drift from the pipeline output.
The prose states conclusions; `check_claims` stops the build if the data no longer supports them.
Re-run steps 11 and 12 after any pipeline re-run.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capstone import report as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TITLE = "Do Machine Learning and Neural Networks Beat Classical Models at Predicting Next-Day Stock Direction?"
SUBTITLE = "A Walk-Forward Comparison on Eight U.S. Equities, 2020-2024"
AUTHOR = "Mubarak Odufade"
AFFIL = "Department of Computer Science, College of Staten Island, City University of New York"
MENTOR = "Faculty Mentor: Dr. Yumei Huo"
DATE = "November 2026"


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def table(headers, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_md(d):
    c, m, bt, sig, sbh = d["cfg"], d["metrics"], d["bt"], d["sig"], d["sbh"]
    wf, tickers = c["walk_forward"], c["tickers"]
    n = int(m["n"].iloc[0])
    ds, preds = d["ds"], d["preds"]
    et = R.exposure_turnover(preds)
    cs = R.cost_sensitivity(preds)
    pd_daily = R.portfolio_daily()
    naive = R.exposure_matched_buyhold(preds, pd_daily)
    corr = R.feature_target_corr(ds)
    up_share = ds["y"].mean()
    n_obs = len(ds)
    se_r = 1 / n_obs ** 0.5
    big = corr[corr.abs() > 2 * se_r]
    cum = lambda k: pct(bt.loc[k, "cum_return"], 0)  # noqa: E731
    row = lambda a, b: sig[(sig.challenger == a) & (sig.baseline == b)].iloc[0]  # noqa: E731
    lstm_log, lstm_arima = row("LSTM", "Logistic"), row("LSTM", "ARIMA")
    n_sig_acc = int((sig["DM_p_holm"] < c["stats"]["alpha"]).sum())
    n_sig_sh = int((sig["boot_sharpe_p_holm"] < c["stats"]["alpha"]).sum())
    best_auc = m.drop("AlwaysUp")["roc_auc"].max()
    worst_auc = m.drop("AlwaysUp")["roc_auc"].min()
    tr = d["trend"].set_index(["model", "trend"]); vo = d["vol"].set_index(["model", "vol"])
    bull_days, bear_days = int(tr.loc[("Logistic", "bull"), "n_days"]), int(tr.loc[("Logistic", "bear"), "n_days"])
    hv_days, lv_days = int(vo.loc[("Logistic", "high_vol"), "n_days"]), int(vo.loc[("Logistic", "low_vol"), "n_days"])
    n_test_days = bull_days + bear_days
    regime_p_min = min(d["trend_dm"]["DM_p_holm"].min(), d["vol_dm"]["DM_p_holm"].min())
    bt_t = d["by_ticker"]
    winners = bt_t[bt_t.model != "AlwaysUp"].loc[lambda x: x.groupby("ticker")["accuracy"].idxmax()]["model"].nunique()
    cells = d["by_ticker"][d["by_ticker"].model != "AlwaysUp"]
    lo_cell, hi_cell = cells.loc[cells["accuracy"].idxmin()], cells.loc[cells["accuracy"].idxmax()]
    up_by_ticker = ds.groupby(level="ticker")["y"].mean()

    perf_rows = [[R.lab(k), f"{m.loc[k, 'accuracy']:.4f}", f"{m.loc[k, 'precision']:.4f}", f"{m.loc[k, 'recall']:.4f}",
                  f"{m.loc[k, 'f1']:.4f}", f"{m.loc[k, 'roc_auc']:.4f}"] for k in R.ORDER]
    bt_rows = [[R.lab(k), cum(k), pct(bt.loc[k, "ann_return"]), f"{bt.loc[k, 'sharpe']:.2f}", pct(bt.loc[k, "max_drawdown"]),
                f"{bt.loc[k, 'calmar']:.2f}"] for k in R.ORDER]
    sig_rows = [[f"{r.challenger} vs {r.baseline}", f"{r.acc_diff * 100:+.2f} pp", f"{r.DM_p:.3f}", f"{r.DM_p_holm:.3f}",
                 f"{r.sharpe_diff:+.2f}&nbsp;[{r.sharpe_ci_low:.2f},&nbsp;{r.sharpe_ci_high:.2f}]", f"{r.boot_sharpe_p:.4f}",
                 f"{r.boot_sharpe_p_holm:.3f}"] for r in sig.itertuples()]
    et_rows = [[R.lab(k), pct(et.loc[k, "pct_long"], 0), f"{et.loc[k, 'trades_per_year']:.0f}", pct(naive[k], 1), pct(bt.loc[k, "ann_return"], 1)]
               for k in R.ORDER]
    cs_rows = [[R.lab(k)] + [f"{cs.loc[k, c_]:.2f}" for c_ in cs.columns] for k in R.ORDER]
    reg_rows = []
    for k in R.ORDER:
        reg_rows.append([R.lab(k), pct(tr.loc[(k, "bull"), "accuracy"]), pct(tr.loc[(k, "bear"), "accuracy"]),
                         pct(vo.loc[(k, "low_vol"), "accuracy"]), pct(vo.loc[(k, "high_vol"), "accuracy"])])
    by_year = d["by_year"].pivot(index="test_year", columns="model", values="accuracy")[R.ORDER]
    year_rows = [[str(y)] + [pct(by_year.loc[y, k]) for k in R.ORDER] for y in by_year.index]
    corr_rows = [[k, f"{corr[k]:+.4f}"] for k in list(corr.index[:4]) + list(corr.index[-4:])[::-1]]

    feat_rows = [
        ["Technical", "rsi14", "Relative Strength Index, 14 days (Wilder's smoothing)"],
        ["Technical", "macd, macd_signal, macd_hist", "12/26-day EMA difference, its 9-day EMA, and their gap; each divided by price"],
        ["Technical", "bb_pctb, bb_width", "Position within, and width of, the 20-day Bollinger band (2 standard deviations)"],
        ["Technical", "ma5_ratio, ma20_ratio, ma50_ratio", "Close divided by its 5/20/50-day mean, minus 1"],
        ["Technical", "atr14_pct", "14-day Average True Range as a fraction of the close"],
        ["Statistical", "vol5, vol10, vol20", "Rolling standard deviation of daily returns"],
        ["Statistical", "ret_lag1 ... ret_lag5", "The last five daily returns (lag 1 is the day's own return, known at the close)"],
        ["Statistical", "skew20, mom5, mom10, mom20", "20-day return skewness; 5/10/20-day price momentum"],
        ["Market context", "mkt_ret, corr_mkt20", "SPY daily return; 20-day rolling correlation of the stock with SPY"],
        ["Market context", "vix, vix_chg", "VIX level and its one-day change"],
    ]

    return f"""# {TITLE}

**{SUBTITLE}**

<b>{AUTHOR}</b><br>
{AFFIL}<br>
{MENTOR}<br>
{DATE}

---

## Abstract

Whether machine-learning and deep-learning models can predict short-term stock movements better than simple
statistical models is a central and contested question in quantitative finance. We study it under conditions designed to
avoid the usual sources of false optimism. Two classical models (logistic regression and ARIMA), two machine-learning
models (random forest and XGBoost) and one neural network (an LSTM) forecast the next-day direction of {len(tickers)}
liquid U.S. equities, and are compared with an always-"up" benchmark. Every model is retrained each year on a rolling
{wf['train_years']}-year window and judged only on the following unseen year ({wf['first_test_year']}-{wf['last_test_year']};
{n:,} out-of-sample predictions). Predictions are then traded in a long/cash backtest with transaction costs. All
models reach {pct(m.drop('AlwaysUp')['accuracy'].min())}-{pct(m.drop('AlwaysUp')['accuracy'].max())} accuracy with ROC-AUC between
{worst_auc:.3f} and {best_auc:.3f}, below the {pct(m.loc['AlwaysUp', 'accuracy'])} of always predicting "up". After Holm
correction, none of the {len(sig)} accuracy comparisons between a challenger and a classical baseline is statistically
significant ({n_sig_acc} of {len(sig)}). The LSTM has the best backtest Sharpe ratio ({bt.loc['LSTM', 'sharpe']:.2f} versus
{bt.loc['Logistic', 'sharpe']:.2f} for logistic regression), but that advantage is significant only against ARIMA
({n_sig_sh} of {len(sig)} Sharpe comparisons), and it rests on low turnover rather than on better predictions. We fail to
reject the null hypothesis that added model complexity gives no significant improvement.

**Keywords:** stock direction prediction, walk-forward validation, LSTM, XGBoost, random forest, ARIMA, backtesting,
Diebold-Mariano test, market efficiency

---

## 1. Introduction

Each trading day, a model that could tell in advance whether a stock will close higher or lower tomorrow would be
valuable. The efficient-market hypothesis (Fama, 1970) argues that, in its weak form, past prices contain no such
information. Empirical work has found that return predictability is weak and unstable (Welch & Goyal, 2008), yet recent
studies report that machine-learning methods can extract signal that linear models miss (Gu, Kelly & Xiu, 2020; Fischer &
Krauss, 2018). At the same time, financial machine learning is prone to over-optimistic results, caused by look-ahead
leakage, repeated tuning on test data and backtest overfitting (Bailey, Borwein, López de Prado & Zhu, 2014; López de
Prado, 2018).

This project asks a narrow, testable version of the question:

> Do machine-learning and deep-learning models significantly outperform classical models at predicting next-day return
> direction, under strict walk-forward validation and realistic trading costs?

- **H0:** there is no significant difference in predictive performance between classical and ML/deep-learning models.
- **H1:** ML and/or deep-learning models significantly outperform the classical baseline.

**Contributions.** (i) A leakage-controlled walk-forward comparison of six strategies on one common set of predictions;
(ii) formal significance testing of both accuracy and risk-adjusted return, with a multiple-comparison correction;
(iii) analysis of *why* any backtest advantage appears, covering market exposure, turnover and sensitivity to trading
costs; and (iv) a fully reproducible pipeline and interactive website that regenerate every number in this paper.

![Overview of the study pipeline](figures/fig_pipeline.png)
*Figure 1. The study pipeline, from raw prices to significance tests.*

---

## 2. Background and related work

**Market efficiency and predictability.** Under weak-form efficiency, prices already reflect all past price
information, so technical indicators should not predict returns (Fama, 1970). Welch and Goyal (2008) showed that many
variables proposed to predict returns fail out of sample. This sets a high bar and motivates our emphasis on
out-of-sample testing.

**Machine learning for returns.** Gu, Kelly and Xiu (2020) found that tree-based models and neural networks improve
return forecasts for cross-sections of stocks using a large panel of characteristics. Fischer and Krauss (2018) reported
profitable LSTM signals for S&P 500 constituents before costs and in an earlier period. Our setting is narrower (eight
stocks, price-based indicators, one-day horizon, recent data) and deliberately conservative.

**Methodological cautions.** Backtests that are tuned repeatedly overfit (Bailey et al., 2014), and
standard cross-validation leaks information in time series (López de Prado, 2018). We therefore use chronological
splits with a purge, fix every hyperparameter in advance, and test differences formally rather than by ranking.

**Models.** ARIMA (Box & Jenkins, 1970) and logistic regression are the classical baselines. Random forests (Breiman,
2001) and gradient-boosted trees (Chen & Guestrin, 2016) are the machine-learning models. The LSTM (Hochreiter &
Schmidhuber, 1997) is the deep-learning model, trained with Adam (Kingma & Ba, 2015) and dropout (Srivastava et al.,
2014).

---

## 3. Data

Daily split- and dividend-adjusted Open, High, Low, Close and Volume were downloaded from Yahoo Finance for
{', '.join(tickers)} (a deliberately mixed set of technology, financial, energy, healthcare and retail firms), plus
{c['market_ticker']} (an S&P 500 fund, used as the market) and the VIX index (used as a volatility measure). The
download range is {c['start_date']} to {c['end_date']}; 2014 only warms up indicators that need up to 50 days of history,
so the modelled data begin in 2015 ({n_obs:,} stock-day observations). Prices are stored in a local SQLite database.

**Target.** For stock *i* on day *t*, `y = 1` if the close on day *t+1* exceeds the close on day *t*, and `0` otherwise.
Up-days make up {pct(up_share)} of observations overall (between {pct(up_by_ticker.min())} and {pct(up_by_ticker.max())} depending
on the stock), so a model that always predicts "up" is right about {pct(up_share, 0)} of the time. Beating that is the real
test, not beating 50%.

---

## 4. Methodology

### 4.1 Features: 26 indicators known at the close

Each feature on day *t* uses only information available at the close of day *t*. A unit test verifies this by
perturbing future prices and checking that no earlier feature changes.

{table(["Family", "Features", "Definition"], feat_rows, ["---", "---", "---"])}

Features are standardised (z-scores) using the training window only, inside each model's pipeline, so no test-period
statistics leak into training.

### 4.2 Walk-forward validation

Time series cannot be shuffled: a model must never learn from the future of the period it is tested on. We use a
rolling walk-forward scheme. For each test year *Y* from {wf['first_test_year']} to {wf['last_test_year']}, a new model
is fitted for each stock on the preceding {wf['train_years']} calendar years, then predicts every day of *Y* one day at a
time. The window then rolls forward one year. The last training row of each fold is dropped (purged) because its label
uses the first test-year close. No state is shared across folds or stocks.

![Walk-forward validation](figures/fig_walkforward.png)
*Figure 2. Rolling walk-forward folds. Each stock and fold gets a freshly trained model.*

### 4.3 Models

Every model outputs a score for "up tomorrow" and a 0/1 prediction. Hyperparameters were fixed in advance, set
conservatively for noisy data, and **not tuned on any test year**.

{table(["Class", "Model", "Specification"], [
    ["Benchmark", "Buy & hold (always-up)", "Predicts up every day; equivalent to staying invested"],
    ["Classical", "Logistic regression", "L2 penalty (C = 0.1) on standardised features; up if P(up) >= 0.5"],
    ["Classical", "ARIMA(p,0,q)", "Order chosen by AIC (p, q <= 2) on the training window; one-step-ahead return forecast; up if forecast > 0; uses only the stock's own returns"],
    ["Machine learning", "Random forest", "300 trees, maximum depth 5, minimum 20 samples per leaf, sqrt(features) per split"],
    ["Machine learning", "XGBoost", "200 trees, depth 3, learning rate 0.03, 80% row and column subsampling, L2 penalty 5, minimum child weight 10"],
    ["Deep learning", "LSTM", "20-day input window, 32 hidden units, dropout 0.3, Adam (lr 0.001), early stopping on a chronological 15% validation tail"],
], ["---", "---", "---"])}

**The LSTM in plain terms.** Unlike the other models, which see one day's 26 indicators, the LSTM reads the last
20 days in order. It keeps a small memory vector (32 numbers) and, for each new day, uses three learned gates to decide
what to forget, what to store and what to output. After the 20th day, the memory feeds one output neuron whose sigmoid
gives the probability of "up". Training minimises binary cross-entropy; early stopping on a chronological validation
tail (never shuffled into the future) and dropout limit memorisation of noise.

![LSTM architecture](figures/fig_lstm.png)
*Figure 3. LSTM architecture and training procedure.*

### 4.4 Backtest

The backtest converts predictions into returns using deliberately simple rules. At each close, if the model predicts
"up" the strategy holds the stock through the next day, otherwise it holds cash (0% return). There is no leverage and
no short selling. Each unit of position change (entering or exiting) costs {pct(c['backtest']['transaction_cost'], 2)}
(5 basis points). The portfolio is an equal-weight, daily-rebalanced average across the {len(tickers)} stocks.
Reported measures are cumulative and annualised return, the Sharpe ratio (mean over standard deviation of daily
returns, times the square root of {c['backtest']['trading_days']}; zero risk-free rate; Sharpe, 1966), maximum
drawdown, and the Calmar ratio (annualised return over absolute maximum drawdown).

### 4.5 Evaluation metrics and statistical tests

**Classification:** accuracy, precision, recall, F1 and ROC-AUC on the pooled out-of-sample predictions. AUC measures
ranking skill independent of the threshold (0.5 means no skill).

**Significance.** Each challenger (random forest, XGBoost, LSTM) is compared with each classical baseline (logistic
regression, ARIMA), giving {len(sig)} one-sided comparisons (H1: challenger better). Accuracy: Diebold-Mariano test
(Diebold & Mariano, 1995) on daily 0/1 loss averaged across stocks, with a Newey-West (1987) long-run variance and the
Harvey-Leybourne-Newbold (1997) small-sample correction; a paired t-test on per-stock, per-year accuracies; and a
stationary block bootstrap (Politis & Romano, 1994; {c['stats']['bootstrap_iterations']:,} draws, mean block 10 days)
for confidence intervals. Sharpe differences use the same bootstrap on paired daily returns. P-values are adjusted
across the {len(sig)} comparisons with Holm's (1979) step-down method; a result counts as significant at
{pct(c['stats']['alpha'], 0)} only if its Holm-adjusted p-value is below {c['stats']['alpha']}.

---

## 5. Assumptions

The conclusions hold under the following explicit assumptions. Several are simplifications that a real trading
operation would not share.

1. **Information timing.** Features are computed from the close of day *t*, and a trade at that close earns the *t* to
   *t+1* return. This assumes execution at the closing price, which is optimistic for large orders.
2. **Transaction costs.** A single proportional cost of {pct(c['backtest']['transaction_cost'], 2)} per unit of turnover,
   with no market impact, bid-ask spread dynamics, slippage or taxes. Section 7.3 varies this assumption.
3. **Zero risk-free rate** in the Sharpe ratio, and cash earns 0% when out of the market.
4. **Long/cash only,** with no leverage or short selling, so a "down" call can only avoid a loss, not profit.
5. **Equal weighting and daily rebalancing** across the {len(tickers)} stocks.
6. **Stationarity within a window.** Relationships learned over {wf['train_years']} years are assumed to persist for the
   next year. Retraining each year is the only adaptation.
7. **Fixed hyperparameters.** No search was run; the settings are conservative defaults chosen before looking at test
   results, and each stochastic model uses a single random seed ({c['seed']}).
8. **Data quality.** Yahoo Finance adjusted prices are treated as correct, although they are revised retroactively
   (Section 9).
9. **Test-statistic assumptions.** The Diebold-Mariano test and block bootstrap assume the daily loss differences are
   weakly dependent and stationary. Averaging across stocks per day reduces, but does not remove, cross-sectional
   dependence. The accuracy error bars in Figure 4 assume independent predictions and are therefore too narrow.
10. **Universe.** The eight stocks are large, liquid and survived to 2024, so the study is subject to survivorship and
    selection bias that favours long positions.

---

## 6. Results

### 6.1 Predictive performance

Table 1 and Figure 4 summarise the {n:,} pooled out-of-sample predictions.

{table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], perf_rows)}

*Table 1. Pooled out-of-sample classification metrics, {wf['first_test_year']}-{wf['last_test_year']}.*

![Accuracy by model](figures/fig_accuracy.png)
*Figure 4. Directional accuracy with 95% binomial intervals (which assume independent predictions and so understate
the real uncertainty).*

No model beats the always-up benchmark on accuracy. All ROC-AUC values lie between {worst_auc:.3f} and {best_auc:.3f}, so
there is essentially no ranking signal (Figure 5). Accuracy and F1 differ across models mainly because models differ in
how often they say "up" (recall), not because they separate up-days from down-days better.

![ROC curves](figures/fig_roc.png)
*Figure 5. Pooled ROC curves. All models track the diagonal (no skill).*

### 6.2 Backtest

{table(["Strategy", "Cumulative return", "Annualised", "Sharpe", "Max drawdown", "Calmar"], bt_rows)}

*Table 2. Equal-weight portfolio backtest, {pct(c['backtest']['transaction_cost'], 2)} cost per unit of turnover.*

![Equity curves](../results/figures/equity_curves.png)
*Figure 6. Cumulative return of each strategy.*

![Drawdowns](../results/figures/drawdowns.png)
*Figure 7. Drawdown from the running peak.*

![Sharpe and drawdown](figures/fig_sharpe.png)
*Figure 8. Sharpe ratio and maximum drawdown by strategy.*

Buy-and-hold earns the highest cumulative return ({cum('AlwaysUp')}), reflecting the strong 2020-2024 market for these
stocks. The LSTM comes closest ({cum('LSTM')}) and has the highest Sharpe ({bt.loc['LSTM', 'sharpe']:.2f}) and Calmar
({bt.loc['LSTM', 'calmar']:.2f}) ratios and the smallest drawdown ({pct(bt.loc['LSTM', 'max_drawdown'])}). The classical and
tree-based strategies earn less than buy-and-hold.

### 6.3 Statistical significance

{table(["Challenger vs baseline", "Accuracy diff.", "DM p", "DM p (Holm)", "Sharpe diff. [95% CI]", "Bootstrap p", "Bootstrap p (Holm)"], sig_rows)}

*Table 3. One-sided tests of whether the challenger beats the baseline. Accuracy differences are challenger minus baseline
in percentage points.*

![Differences vs logistic regression](../results/figures/difference_vs_logistic.png)
*Figure 9. Accuracy and Sharpe differences versus logistic regression with 95% block-bootstrap intervals. All accuracy
intervals contain zero.*

- **Accuracy:** none of the {len(sig)} comparisons is significant under any of the three tests after Holm correction
  (smallest Holm-adjusted DM p-value: {sig['DM_p_holm'].min():.2f}; smallest raw: {sig['DM_p'].min():.3f}, LSTM vs ARIMA).
- **Sharpe:** only one of the {len(sig)} comparisons is significant: LSTM over ARIMA (Holm-adjusted p = {lstm_arima.boot_sharpe_p_holm:.3f}).
  Against logistic regression, the LSTM's raw bootstrap p-value is {lstm_log.boot_sharpe_p:.3f} but the Holm-adjusted value is
  {lstm_log.boot_sharpe_p_holm:.3f}. Against buy-and-hold its Sharpe advantage is {sbh.loc['LSTM', 'sharpe_diff_vs_buy_hold']:+.2f}
  (95% CI [{sbh.loc['LSTM', 'ci_low']:.2f}, {sbh.loc['LSTM', 'ci_high']:.2f}], unadjusted p = {sbh.loc['LSTM', 'p_one_sided']:.3f}), also
  not significant.

### 6.4 Robustness

Regimes are defined using only information known at each close: bull/bear by SPY above or below its 200-day average
({bull_days} bull and {bear_days} bear test days), and low/high volatility by VIX below or above 20 ({lv_days} and {hv_days} days).

{table(["Model", "Bull", "Bear", "Low volatility", "High volatility"], reg_rows)}

*Table 4. Accuracy by market regime.*

![Accuracy by regime](../results/figures/accuracy_by_regime.png)
*Figure 10. Accuracy by regime. Differences between models are small and inconsistent in sign.*

Within every regime, no challenger's accuracy advantage over logistic regression is significant (smallest Holm-adjusted
p-value {regime_p_min:.2f}).

{table(["Test year"] + [R.lab(k) for k in R.ORDER], year_rows)}

*Table 5. Accuracy by test year.*

![Accuracy by year](figures/fig_by_year.png)
*Figure 11. Accuracy by test year.*

All models, and the benchmark, fall in 2022 (when {pct(by_year.loc[2022, 'AlwaysUp'])} of days were up) and recover in 2023, so year-to-year
swings are driven by how many days were up, not by model quality. By stock (Figure 12), the best non-benchmark model differs
across tickers ({winners} different winners among the eight stocks) and accuracy ranges from
{pct(lo_cell.accuracy)} ({lo_cell.ticker}, {R.lab(lo_cell.model)}) to {pct(hi_cell.accuracy)} ({hi_cell.ticker}, {R.lab(hi_cell.model)}), the pattern expected when differences are mostly noise.

![Accuracy by stock and model](figures/fig_ticker_heatmap.png)
*Figure 12. Accuracy by stock and model.*

---

## 7. Analysis

### 7.1 Why the indicators carry so little information

Figure 13 shows the correlation of each of the 26 indicators with the next-day direction label, pooled over all
{n_obs:,} stock-days. The strongest correlation has magnitude {corr.abs().max():.3f}, which explains well under 0.1% of the variance. Under an
independence assumption the standard error of a correlation here is about {se_r:.4f}, so only {len(big)} indicators
({', '.join(big.index)}) exceed two standard errors, and the independence assumption is generous because stocks move together.
The significant ones are negative, which is a faint short-term reversal effect: a stock that rose today or this week is
slightly less likely to rise tomorrow. Many indicators are also strongly related to each other (the moving-average
ratios, momentum and RSI all describe recent trend), so adding more of them adds little new information.

![Indicator correlations](figures/fig_feature_corr.png)
*Figure 13. Correlation of each indicator with next-day direction.*

{table(["Indicator", "Correlation with next-day direction"], corr_rows)}

*Table 6. The four most negative and four most positive correlations.*

### 7.2 Market exposure and turnover

Accuracy and AUC say nothing about how a strategy earns money. Table 7 shows two properties that determine its backtest.

{table(["Strategy", "Share of days long", "Position changes per stock per year", "Same exposure without timing (annualised)", "Actual annualised return"], et_rows)}

*Table 7. Exposure and turnover. "Same exposure without timing" holds buy-and-hold on a fixed fraction of capital equal to
the model's average exposure, with the rest in cash.*

The LSTM is invested on {pct(et.loc['LSTM', 'pct_long'], 0)} of days, more than any other predictive model, in a rising market,
so some of its return is simply market exposure: a naive mix with the same exposure and no timing earns {pct(naive['LSTM'])}
a year, versus the LSTM's actual {pct(bt.loc['LSTM', 'ann_return'])}. The more striking difference is turnover. The LSTM changes
position about {et.loc['LSTM', 'trades_per_year']:.0f} times per stock per year; the other predictive models change position
{et.drop(['AlwaysUp', 'LSTM'])['trades_per_year'].min():.0f} to {et.drop(['AlwaysUp', 'LSTM'])['trades_per_year'].max():.0f} times.

### 7.3 Sensitivity to transaction costs

{table(["Strategy"] + [f"{int(c_)} bp" for c_ in cs.columns], cs_rows)}

*Table 8. Sharpe ratio at different transaction costs (basis points per unit of turnover).*

![Cost sensitivity](figures/fig_cost.png)
*Figure 14. Sharpe ratio as trading costs rise.*

Without costs, logistic regression ({cs.loc['Logistic', 0]:.2f}) and XGBoost ({cs.loc['XGBoost', 0]:.2f}) are comparable to
buy-and-hold ({cs.loc['AlwaysUp', 0]:.2f}), so their weak backtests at 5 bp are largely the price of frequent trading,
not of bad calls. At 25 bp the classical and tree strategies fall to between {cs.drop(['AlwaysUp', 'LSTM']).loc[:, 25].min():.2f} and
{cs.drop(['AlwaysUp', 'LSTM']).loc[:, 25].max():.2f}, and at 50 bp their Sharpe ratios range from {cs.drop(['AlwaysUp', 'LSTM']).loc[:, 50].min():.2f} to
{cs.drop(['AlwaysUp', 'LSTM']).loc[:, 50].max():.2f}. The LSTM's Sharpe falls only from {cs.loc['LSTM', 0]:.2f} to {cs.loc['LSTM', 50]:.2f} because it trades
rarely. Buy-and-hold is nearly unaffected.

### 7.4 What explains the LSTM's backtest advantage?

The evidence points to **persistence of positions, not superior forecasting**. Its AUC ({m.loc['LSTM', 'roc_auc']:.3f}) is no better than the simpler models', its accuracy
({pct(m.loc['LSTM', 'accuracy'])}) is only slightly higher and not significantly so, and it remains below buy-and-hold on accuracy, but
its positions change rarely and it is mostly invested, which keeps costs low and captures much of the market's upward drift
while avoiding some drawdowns (maximum drawdown {pct(bt.loc['LSTM', 'max_drawdown'])} versus {pct(bt.loc['AlwaysUp', 'max_drawdown'])}
for buy-and-hold). A plausible but untested explanation is that its 20-day input windows overlap almost entirely from one day to the next,
so its output probability changes slowly and crosses the 0.5 threshold rarely. Testing this (for example by smoothing the other
models' scores, or by comparing against a model with a longer holding rule) is future work.
Even so, the advantage is statistically fragile: it is significant against ARIMA only, and its confidence interval against
buy-and-hold includes zero.

### 7.5 Run-to-run sensitivity

The models are deterministic given the data, yet results moved between successive complete runs of the pipeline during
development, because Yahoo Finance retroactively revises adjusted prices. Across those runs ARIMA's cumulative backtest return
ranged from about 135% to 154% and XGBoost's from about 197% to 219%, while the always-up and logistic results barely moved and
the conclusions never changed. That small input changes shift some strategies by tens of percentage points is itself evidence of how
little signal there is to anchor the models.

---

## 8. Summary of findings

1. **Complexity did not buy predictive accuracy.** All models score {pct(m.drop('AlwaysUp')['accuracy'].min())}-{pct(m.drop('AlwaysUp')['accuracy'].max())}, and none beats
   always-up ({pct(m.loc['AlwaysUp', 'accuracy'])}). None of the {len(sig)} challenger-vs-baseline accuracy comparisons is significant
   after Holm correction.
2. **There is almost no ranking signal.** ROC-AUC lies between {worst_auc:.3f} and {best_auc:.3f} for every model, and no single indicator
   correlates with next-day direction above {corr.abs().max():.3f}.
3. **The LSTM has the best backtest, but for reasons other than forecasting skill.** Sharpe {bt.loc['LSTM', 'sharpe']:.2f}, driven by high
   market exposure ({pct(et.loc['LSTM', 'pct_long'], 0)} of days) and low turnover ({et.loc['LSTM', 'trades_per_year']:.0f} position changes per stock per year).
   Its advantage is significant against ARIMA only.
4. **Costs decide the ranking.** The classical and tree-based strategies roughly match buy-and-hold before costs but lose to it
   as costs rise; at 50 bp their Sharpe ratios are between {cs.drop(['AlwaysUp', 'LSTM']).loc[:, 50].min():.2f} and {cs.drop(['AlwaysUp', 'LSTM']).loc[:, 50].max():.2f}.
5. **Results are consistent across regimes, years and stocks.** No challenger advantage is significant in any regime, and year-to-year
   changes track the share of up-days rather than model quality.
6. **We fail to reject H0.** "No significant difference" is not proof of equality: with about {n_test_days:,} test days, differences smaller
   than roughly 1.5 percentage points of accuracy cannot be detected.

---

## 9. Limitations

- **Statistical power.** With about {n_test_days:,} test days, the standard error of an accuracy difference is around 0.6 percentage points.
- **Survivorship and selection bias.** The stocks are well-known long-run winners, which flatters long-biased strategies.
- **Single seed and fixed hyperparameters** for stochastic models; no tuning, so stronger configurations may exist.
- **Data revisions.** Section 7.5 shows results move with the vendor's revisions.
- **ARIMA** forecasts a return, not a probability, so its AUC uses the forecast as the score.
- **Loss function.** Diebold-Mariano tests use 0/1 loss, the only loss common to all models, which ignores confidence and
  the size of moves.
- **Simplified trading.** Close-to-close execution, one cost level, zero risk-free rate, no slippage, taxes or capacity limits.
- **One horizon, one window length.** Only next-day direction and a {wf['train_years']}-year window were studied; the per-year tables stand in for a
  window-length sensitivity analysis.
- **Untested explanation.** The persistence account of the LSTM result (Section 7.4) is consistent with the data but not
  directly tested.

## 10. Future work

1. **Wider, survivorship-free universe:** hundreds of stocks including delisted names, with cross-sectional (ranking) targets and a
   long/short portfolio, which also raises statistical power.
2. **More horizons and targets:** multi-day and weekly direction, return magnitude, and volatility forecasting, where predictability
   is generally stronger than for direction.
3. **Test the persistence hypothesis:** smooth or threshold-band the other models' signals, or add holding-period rules, to see whether
   they close the gap with the LSTM.
4. **Richer information:** volume and order-flow features, fundamentals, news and sentiment text, and cross-asset signals.
5. **More careful tuning and ensembling:** nested time-series cross-validation, many random seeds, and ensembles of networks, with a
   correction for the number of configurations tried.
6. **Other architectures:** GRU, temporal convolution and Transformer models, plus the originally planned SVM and GARCH baselines.
7. **Probability-aware trading:** calibrated probabilities, confidence-based position sizing, and a no-trade band to reduce turnover.
8. **More realistic execution:** bid-ask spreads, market impact, a risk-free rate, and eventually paper trading on live data.
9. **Stronger inference:** deflated Sharpe ratios and other multiple-testing corrections across strategies, and longer test periods that
   include several market cycles.

---

## 11. Conclusion

We compared classical, machine-learning and deep-learning models for next-day stock direction under leakage-controlled walk-forward
validation. None of the more complex models improved accuracy significantly over a logistic-regression baseline, and all fell short of
always predicting "up". The LSTM's attractive risk-adjusted backtest is real in this sample, but it comes from staying invested and
trading rarely rather than from better predictions, and it is statistically significant only against the weakest baseline. For
short-horizon direction from daily price-based indicators, the evidence supports market efficiency over model sophistication, and shows why
costs, exposure and formal testing matter as much as headline accuracy.

## 12. Reproducibility

All code, configuration and data-handling steps are in the project repository. `python scripts/run_all.py` regenerates the data,
predictions, tables and website data (seed {c['seed']}; settings in `config.yaml`); `python scripts/11_report_figures.py` and
`python scripts/12_build_paper.py` rebuild the figures and this paper from the saved results, so every number here is read from the
result files at build time. An interactive version of the results is provided by the accompanying website. Because Yahoo Finance revises
history, a fresh download can change some numbers slightly (Section 7.5).

---

## References

Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematical financial charlatanism: The effects of
backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society, 61*(5), 458-471.

Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253-263.

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance, 25*(2), 383-417.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions.
*European Journal of Operational Research, 270*(2), 654-669.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies, 33*(5), 2223-2273.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of
Forecasting, 13*(2), 281-291.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65-70.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR).*

López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance
matrix. *Econometrica, 55*(3), 703-708.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association, 89*(428), 1303-1313.

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business, 39*(1), 119-138.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural
networks from overfitting. *Journal of Machine Learning Research, 15*, 1929-1958.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *The Review of
Financial Studies, 21*(4), 1455-1508.
"""


CSS = """
@page { size: Letter; margin: 0.85in 0.8in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.45; color: #111; max-width: 7in; margin: 0 auto; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 6pt; text-align: center; }
h1 + p { text-align: center; font-size: 12.5pt; }
h2 { font-size: 14pt; margin: 22pt 0 6pt; border-bottom: 1px solid #999; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 8pt 0 2pt; break-inside: avoid; }
th, td { border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 3pt 5pt; }
th { background: #eee; }
td, th { text-align: right; } td:first-child, th:first-child { text-align: left; }
img { display: block; max-width: 100%; max-height: 3.5in; width: auto; margin: 10pt auto 2pt; break-inside: avoid; }
p.cap { font-size: 9.5pt; text-align: center; color: #333; margin: 2pt 0 12pt; }
table.dense { font-size: 8pt; } table.dense td, table.dense th { padding: 3pt 3pt; }
table.textual td { white-space: normal; text-align: left; vertical-align: top; }
table.textual th { text-align: left; }
#refs p { text-align: left; padding-left: 0.3in; text-indent: -0.3in; margin: 0 0 5pt; }
blockquote { margin: 8pt 24pt; font-style: italic; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f1f1f1; padding: 0 2pt; }
hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
"""


def find_chrome():
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")


def render_html(md_text: str) -> Path:
    marked = (ROOT / "site" / "vendor" / "marked.min.js").read_text(encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>{CSS}</style><script>{marked}</script></head><body><div id="paper"></div>
<script>
const el = document.getElementById("paper");
el.innerHTML = marked.parse({json.dumps(md_text)}, {{gfm: true}});
el.querySelectorAll("p").forEach(p => {{
  const t = p.textContent.trim();
  if (p.children.length === 1 && p.firstElementChild.tagName === "EM" && p.childNodes.length === 1 && /^(Figure|Table) [0-9]+/.test(t)) p.classList.add("cap");
}});
el.querySelectorAll("table").forEach(tb => {{
  if (tb.querySelectorAll("th").length >= 7) tb.classList.add("dense");
  const h = tb.querySelector("th:last-child"); if (h && /Definition|Specification/.test(h.textContent)) tb.classList.add("textual");
}});
const refs = [...el.querySelectorAll("h2")].find(h => h.textContent.trim() === "References");
if (refs) {{ const box = document.createElement("div"); box.id = "refs"; let n = refs.nextElementSibling; refs.after(box); while (n) {{ const nx = n.nextElementSibling; box.append(n); n = nx; }} }}
</script></body></html>"""
    out = PAPER / "paper.html"
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html: Path) -> bool:
    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found: skipped PDF (open paper/paper.html and print to PDF)")
        return False
    pdf = PAPER / "paper.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    print("wrote", pdf)
    return True


def publish_to_site(md_text: str) -> None:
    """Copy the paper, its figures and the PDF into site/paper/ so the Netlify website can show and offer them."""
    dest = ROOT / "site" / "paper"
    figs = dest / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for f in (PAPER / "figures").glob("*.png"):
        if not f.name.startswith("poster_"):  # poster artwork is private
            shutil.copyfile(f, figs / f.name)
    site_md = md_text
    for f in (ROOT / "results" / "figures").glob("*.png"):  # figures the paper links from results/
        shutil.copyfile(f, figs / f.name)
        site_md = site_md.replace(f"../results/figures/{f.name}", f"figures/{f.name}")
    # The site page lives at the site root, so image links must point into site/paper/.
    site_md = site_md.replace("](figures/", "](paper/figures/")
    (dest / "paper.md").write_text(site_md, encoding="utf-8")
    if (PAPER / "paper.pdf").exists():
        shutil.copyfile(PAPER / "paper.pdf", dest / "paper.pdf")
    print("published to", dest)


def main():
    d = R.load()
    R.check_claims(d)
    md_text = build_md(d)
    (PAPER / "paper.md").write_text(md_text, encoding="utf-8")
    print("wrote", PAPER / "paper.md")
    to_pdf(render_html(md_text))
    publish_to_site(md_text)


if __name__ == "__main__":
    main()
