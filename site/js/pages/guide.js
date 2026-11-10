// "How it works": a plain-English walkthrough of the whole project. Values in the text come from the live config.
import { callout, flow, h, md, page, tabs } from "../ui.js";

const wrap = (...kids) => h("div", { class: "prose" }, ...kids);

function bigPicture() {
  return wrap(
    md(`## The big question

Every evening, after the stock market closes, we ask one simple yes/no question about each stock:

> **"Will this stock close higher tomorrow than it did today?"**

That is the entire prediction task. The project then asks a second, more interesting question:

> **Do fancy models (machine learning and neural networks) answer that question better than simple, old-fashioned ones?**

To find out we pit six "players" against each other on the same stocks, the same days and the same rules:

| Player | Type | One-line idea |
|---|---|---|
| **Buy & hold** (always-up) | Benchmark | Always guess "up". Stocks rise slightly more often than they fall, so this is hard to beat. |
| **Logistic regression** | Classical | Weigh each clue (indicator) up or down, add them, squash into a probability. |
| **ARIMA** | Classical | Look only at the stock's own recent returns and extrapolate the pattern. |
| **Random forest** | Machine learning | 300 small decision trees vote. |
| **XGBoost** | Machine learning | Many small trees, each one correcting the mistakes of the previous ones. |
| **LSTM** | Neural network | A network with *memory* that reads the last 20 days in order. |

The whole project is one assembly line:`),
    flow(["Download prices", "Build indicators", "Walk-forward split", "Train each model", "Daily up/down predictions", "Backtest trades", "Score & significance tests"]),
    md(`The next tabs walk through each stage in order. No finance or maths background is needed.

**A spoiler worth knowing up front:** predicting tomorrow's stock move is *extremely* hard. Markets are noisy and
thousands of smart people are already competing. A model that is right 52% of the time is doing about as well as
anyone realistically can, and the real scientific question is whether a small edge is **genuinely better than luck**.`));
}

function dataAndTarget(s) {
  const m = s.meta;
  return wrap(
    md(`## Step 1: The data and what we predict

### Which stocks and which dates
- **Stocks (${s.T}):** ${s.tickers.join(", ")}. Big, heavily traded US companies from different industries (tech, banks, energy,
  healthcare, retail) so the result is not about one sector.
- **Extra context series:** **${m.market}** (an S&P 500 fund, a stand-in for "the market") and **VIX** (the market's "fear gauge":
  how nervous investors are).
- **Dates:** ${m.start_date} to ${m.end_date}. The first year is only used to "warm up" indicators that need 50 days of history.
- **Prices:** daily Open, High, Low, Close and Volume, downloaded from Yahoo Finance and stored in a local database.

### The target: a single 1 or 0 per day
For each stock and each day *t* we compute tomorrow's return:

<code>next-day return = close(t+1) / close(t) - 1</code>

Then we label the row **1 if it is positive ("up")** and **0 otherwise ("down")**. That label is the *answer key*.
The model never sees it when predicting; it only uses it when learning from the past.`),
    callout("<b>Golden rule: no peeking at the future.</b> Every clue for day <i>t</i> uses only information that was known at the close of day <i>t</i>. " +
      "The tomorrow-return is used only as the answer key. Breaking this rule (&ldquo;data leakage&rdquo;) makes any model look like a genius and is " +
      "the most common mistake in this kind of project. Everything below is designed around avoiding it."));
}

function indicators() {
  const body = wrap(md(`## Step 2: The clues (indicators / features)

Raw prices alone are not very informative to a model. So for every stock-day we compute **26 numbers** that summarise what the
stock has been doing lately. Think of them as 26 clues a detective looks at. They come in three families.
(A *feature* is just one such number.)`));
  const fam = tabs([
    { label: "Technical indicators (10)", render: () => md(`These are the classic chart-reading tools traders use.

| Feature | Plain-English meaning | How it is read |
|---|---|---|
| **rsi14** (Relative Strength Index) | Compares the average size of up-days with down-days over the last 14 days, scaled 0-100. | Above ~70 = "overbought" (rose a lot, maybe due a pause). Below ~30 = "oversold". |
| **macd**, **macd_signal**, **macd_hist** | Difference between a fast (12-day) and slow (26-day) moving average of price, a smoothed version of it (9-day), and the gap between the two. | When the fast average pulls away from the slow one, momentum is building. Divided by price so different stocks are comparable. |
| **bb_pctb** (Bollinger %B) | Where today's price sits inside a band drawn 2 standard deviations around the 20-day average. | 0 = at the bottom of the band, 1 = at the top. |
| **bb_width** | How wide that band is relative to price. | Wide = choppy and volatile, narrow = calm. |
| **ma5_ratio, ma20_ratio, ma50_ratio** | Today's price divided by its 5/20/50-day average, minus 1. | +3% means the price is 3% above its recent average (short/medium/long trend). |
| **atr14_pct** (Average True Range) | The typical size of a day's full high-to-low swing over 14 days, as a % of price. | A volatility measure that also notices overnight gaps. |`) },
    { label: "Statistical features (12)", render: () => md(`Simple statistics describing the recent behaviour of returns.

| Feature | Plain-English meaning |
|---|---|
| **vol5, vol10, vol20** | Volatility: how much daily returns have been bouncing around over the last 5/10/20 days (standard deviation). |
| **ret_lag1 ... ret_lag5** | The actual returns of the last 5 days. <code>ret_lag1</code> is *today's* return (known at the close), <code>ret_lag5</code> is four days ago. |
| **skew20** | Whether the last 20 days had more big drops or big jumps (asymmetry). |
| **mom5, mom10, mom20** | Momentum: total price change over the last 5/10/20 days. |`) },
    { label: "Market context (4)", render: () => md(`Stocks do not move in isolation, so we give the models a feel for the overall mood.

| Feature | Plain-English meaning |
|---|---|
| **mkt_ret** | Today's return of the S&P 500 fund. |
| **corr_mkt20** | How closely this stock has been moving with the market over 20 days. |
| **vix** | The fear-gauge level today. |
| **vix_chg** | How much the fear gauge changed today. |`) },
  ]);
  body.append(fam, md(`### Why scaling matters
RSI lives between 0 and 100, while a daily return is something like 0.01. Models (especially neural networks) behave badly when
inputs have wildly different sizes, so every feature is **standardised** (a z-score: subtract the average, divide by the typical
spread) so they all live on a similar scale.

Crucially, the average and spread are measured **only on the training data** and then reused unchanged on the test data.
Measuring them on the test period would quietly leak future information.`));
  return body;
}

function walkForward(s) {
  const wf = s.meta.walk_forward, ty = wf.train_years;
  const rows = [];
  for (let y = wf.first_test_year; y <= wf.last_test_year; y++) rows.push(`| ${y} | ${y - ty} - ${y - 1} |`);
  return wrap(
    md(`## Step 3: Fair testing with walk-forward evaluation

A model must be judged on data it has **never seen**, and that data must come *after* the data it learned from, just as in real life.
A random shuffle of days would let the model learn from 2023 to "predict" 2021, which is cheating. Instead we use a **rolling walk-forward**:

| Test year | Model learns from (last ${ty} years) |
|---|---|
${rows.join("\n")}

For each stock and each test year, a **brand-new model** is trained on the preceding years, then asked to predict every day of the
test year, **one day at a time, without ever peeking at the answer**. Then the window rolls forward a year and we repeat.
Every prediction in the results is therefore *out-of-sample*.

Two extra safety measures:
- **Purge:** the answer key for the last training day depends on the first test-year price, so that one day is dropped from training to avoid overlap.
- **Fresh model every time:** nothing carries over between years or stocks.`),
    flow([`Train on ${ty} years`, "Freeze the model", "Predict next test year", "Roll window forward 1 year", "Repeat"]));
}

function models(s) {
  const det = (title, text) => h("details", {}, h("summary", {}, title), md(text));
  return wrap(
    md(`## Step 4: The players

Every model has the same job: look at today's 26 clues and output a **score**: how likely is *up* tomorrow? If the score is above a
threshold (50% for probability models) the prediction is **UP (1)**, otherwise **DOWN (0)**.`),
    det("Buy & hold (always-up): the benchmark", "Predicts UP every day. It needs no skill, and since stocks tend to drift upward, it is right a bit more than half the time. Any model that cannot beat it has learned nothing useful."),
    det("Logistic regression: the simple classical baseline", "Gives each of the 26 clues a **weight** (positive means \"pushes toward up\", negative means \"toward down\"), adds the weighted clues together, and squashes the sum into a probability between 0 and 1 using an S-shaped curve. It can only learn straight-line relationships, which is exactly why it is a useful yardstick: if a complex model cannot beat it, the complexity bought nothing. A strong regularisation setting (C = 0.1) keeps the weights modest so it does not memorise noise."),
    det("ARIMA: the time-series classic", "Ignores all indicators and looks only at the stock's own history of daily returns: \"yesterday was +1% and the day before was -0.5%, so what is tomorrow likely to be?\" It tries a handful of simple recipes (looking back up to 2 days and 2 forecast errors), picks the best by a standard fit score (AIC), and forecasts tomorrow's return. Forecast above zero means UP."),
    det("Random forest: a vote of many decision trees", "A **decision tree** is a flowchart of yes/no questions (\"is RSI above 60? is volatility below 1.5%?\") that ends in a prediction. One tree overfits easily, so a random forest grows **300 trees**, each on a random slice of the data and clues, and lets them vote. We keep the trees shallow (depth 5, at least 20 days per leaf) because financial data is mostly noise and deep trees just memorise it."),
    det("XGBoost: trees that learn from each other's mistakes", "Also uses trees, but **builds them one after another**: each new small tree focuses on the days the previous trees got wrong. Small steps (learning rate 0.03), 200 shallow trees, and penalties for complexity keep it from overfitting. Often the strongest model on spreadsheet-style data."),
    md(`### How the models use the indicators

Every model receives the same thing: **one row of 26 numbers per stock per day**, all known at that day's close (standardised using
training data only). A model never sees an indicator "by name". It sees a position in a list of numbers and learns, by comparing its
guesses to the known answers in the training years, which positions matter and how.

| Model | What it does with the 26 numbers |
|---|---|
| **Logistic regression** | Learns **one weight per indicator**. It multiplies each indicator by its weight and adds everything up. A positive weight pushes toward "up", a negative one toward "down". The sum becomes a probability. Each indicator contributes independently and in a straight line, so a rule like "RSI only matters when volatility is high" is out of reach. |
| **Random forest** | Each tree asks a chain of **threshold questions** such as "is <code>ma20_ratio</code> above 2%?" then "is <code>vix</code> below 18?", ending in a leaf that leans up or down. Each tree sees only a random subset of the indicators, and the 300 trees vote. Because questions are chained, it can learn **interactions** between indicators. |
| **XGBoost** | The same kind of threshold questions, but trees are built **one after another**, each concentrating on the days the earlier trees got wrong. The final score is the sum of all trees' contributions. |
| **LSTM** | Reads the indicator rows for the **last 20 days in order**, updating a 32-number memory with each new row. The final memory becomes one probability. Unlike the single-day models, it can notice **how indicators changed** over time (for example, RSI climbing for a week). |
| **ARIMA** | **Ignores the indicators.** It uses only the stock's own recent daily returns (<code>ret_lag1</code>). |

**What "learning" means.** Training adjusts the weights, tree splits or network weights so that, on the past years, the indicators lead
to the correct up/down answer more often. Prediction then applies those *frozen* rules to a new day's indicators. The model cannot see
the true answer, and a fresh model is trained for every stock and test year.

**Why the indicators help so little here**
- **They overlap.** The moving-average ratios, momentum and RSI all describe recent price trend, so they carry much the same information. Adding more of them adds little that is new.
- **The target is mostly noise.** Models find patterns in the training years that largely fail to repeat afterwards. That is why ROC-AUC scores sit close to 0.5.
- **The safeguards limit memorising.** Shallow trees, small weights, dropout and early stopping stop the models from chasing noise, at the cost of a weaker fit to the past.

### The neural network: LSTM

The other models look at **one day's snapshot** of 26 clues. The LSTM looks at a **movie**: the last **20 days** of clues, in order.
That is why it is the "deep learning" entrant.

#### What is a neural network?
Picture a chain of tiny calculators ("neurons"). Each takes numbers in, multiplies each by a weight, adds them up, and passes the
result on. Stack layers of them and the network can learn very flexible patterns. Training means nudging millions of weights until
the network's guesses match the answer key.

#### What is an LSTM?
*Long Short-Term Memory* is a neural network built for sequences. Reading day by day, it keeps a small **memory** (here 32 numbers)
and at each new day uses three "gates" to decide:
1. **Forget gate:** which parts of the old memory are no longer relevant?
2. **Input gate:** which parts of today's information are worth storing?
3. **Output gate:** what should I report right now, based on my memory?

After reading all 20 days, the final memory goes through one last calculator that outputs a single number, which is turned into a
probability of "up".`),
    flow(["Last 20 days × 26 indicators", "LSTM layer (32 memory cells)", "Dropout (30% off while training)", "1 output neuron", "Probability of UP"]),
    md(`#### How it is trained
1. **Cut the history into 20-day windows.** Each window is one training example, labelled with whether the *next* day was up.
2. **Hold back the last 15%** of the training period as a private *validation set* (still chronological: the future relative to what it trains on).
3. **Repeat in rounds ("epochs"):** show the network batches of 64 windows, compare its guess to the truth using *cross-entropy loss*
   (a penalty that is large when it is confidently wrong), and nudge every weight to reduce the penalty (the **Adam** optimiser, learning rate 0.001).
4. **Early stopping:** after each round, check the loss on the validation set. If it has not improved for 5 rounds in a row, stop
   (maximum 40 rounds) and **restore the best version seen**. This stops the network from memorising noise.
5. **Anti-overfitting extras:** *dropout* randomly switches off 30% of the network while training so it cannot lean on any one
   pathway, and *weight decay* gently shrinks the weights.
6. **Predict:** for each test day, feed in the previous 20 days (the first test windows reach back into the end of training: past data only)
   and read off the probability. Above 50% means UP.

The network is seeded (seed = ${s.meta.seed}) so re-running reproduces the same numbers.

#### Why can't a bigger network just "solve" the market?
Because the signal-to-noise ratio is tiny. A flexible model is *better* at fitting the past, but much of what it fits is random
noise that will not repeat. That is the whole reason this project compares against simple baselines and runs significance tests
instead of just trusting a good-looking accuracy.`));
}

function trades(s) {
  const c = s.meta.backtest.transaction_cost, pct = (c * 100).toFixed(2);
  return wrap(md(`## Step 5: From predictions to money

Accuracy is not profit, so we simulate what would have happened if someone had **acted on the predictions**. This is the
**backtest**, and the rules are deliberately simple.

### The trading rule (long / cash)
At the close of each day, for each stock:
- Model says **UP** → **hold the stock** through the next day and earn (or lose) its return.
- Model says **DOWN** → **sit in cash** and earn 0 that day.

No short-selling and no borrowed money (no leverage).

### Transaction costs
Trading is not free. Each time the position changes (cash → stock or stock → cash) we pay a cost of **${pct}% of the amount traded**.
A model that flip-flops daily pays a lot; one that holds steady pays little. Costs are what quietly kill many "profitable"
strategies, so we always include them. (The Backtest page lets you change this.)

### A tiny worked example
| Day | Model says | Position | Stock's next-day move | Cost | Strategy return |
|---|---|---|---|---|---|
| Mon | UP | stock | +1.0% | ${pct}% (buy) | +${(1 - c * 100).toFixed(2)}% |
| Tue | UP | stock | -0.5% | none (no change) | -0.50% |
| Wed | DOWN | cash | +2.0% | ${pct}% (sell) | -${(c * 100).toFixed(2)}% (missed the gain!) |
| Thu | DOWN | cash | -1.0% | none | 0.00% (dodged the loss) |

### Combining stocks
Each model's daily results for the ${s.T} stocks are **averaged with equal weights**, like putting an equal share of your money in
each. That is the *portfolio* return, which is what the Overview page shows.

### Compounding into the equity curve
Daily returns are chained together: **(1 + r₁) × (1 + r₂) × …**. The result is the **equity curve**: how $1 would have grown.
The "cumulative return" is just where that curve ends, minus 1.`));
}

function scoring() {
  return wrap(md(`## Step 6: Scoring the results

### Classification scores (did it call the direction right?)
| Metric | Plain English |
|---|---|
| **Accuracy** | Share of days it called correctly. Coin-flip is 50%; always-up scores whatever share of days were up (about 52-53% here). |
| **Precision** | When it said UP, how often was it right? |
| **Recall** | Of all the days that really were up, what share did it catch? (Always-up has 100% recall: it never misses an up-day, but is wrong on every down-day.) |
| **F1** | A single number blending precision and recall. |
| **ROC-AUC** | Ranking quality ignoring the threshold: pick a random up-day and a random down-day, how often does the model give the up-day the higher score? **0.5 = no skill**, 1.0 = perfect. |

### Trading scores (did acting on it make money?)
| Metric | Plain English |
|---|---|
| **Cumulative return** | Total gain/loss over the test period. |
| **Annualised return** | The same, expressed as a steady yearly rate. |
| **Sharpe ratio** | Return per unit of risk (average daily return ÷ its ups-and-downs, scaled to a year). Higher = smoother, better-compensated ride. Around 1 is good; negative is bad. |
| **Max drawdown** | The worst peak-to-trough fall: "at the worst point, how much of my money would I have lost?" |
| **Calmar** | Annual return ÷ max drawdown: reward relative to the worst pain. |

### Is it skill or luck? The significance tests
Say a model scores 51.7% and another 51.2%. Is that a real difference or just random wobble? We test it formally, comparing each
ML/neural challenger with each classical baseline:

- **Diebold-Mariano test** compares the *errors* of two models day by day and asks whether one is consistently lower.
- **Block bootstrap** reshuffles the history thousands of times (in chunks, to preserve the way market days cluster) to build a
  **confidence interval** for the difference in accuracy or Sharpe. If the interval comfortably includes zero, the difference could easily be luck.
- **Holm correction:** when you run many tests, a few will look "significant" by pure chance. Holm tightens the threshold to
  compensate. A result is only called **significant** if it survives this at the 5% level.
- **Robustness checks:** the Robustness page splits results by bull/bear markets, calm/volatile periods, year and stock to see if any
  edge is consistent or only shows up in one corner.`));
}

function findings(s) {
  const cm = s.table("classification_metrics.csv"), bt = s.table("backtest_portfolio.csv"), sig = s.table("significance_vs_baselines.csv");
  const nm = (m) => (m === "AlwaysUp" ? "Buy & hold" : m);
  const body = wrap(h("h2", {}, "Step 7: What did we find?"));
  const rows = s.order.map((m) => ({ m, c: cm.find((r) => r.model === m), b: bt.find((r) => r.model === m) })).filter((r) => r.c && r.b);
  const tr = rows.map((r) => `| ${nm(r.m)} | ${(r.c.accuracy * 100).toFixed(2)}% | ${r.c.roc_auc.toFixed(3)} | ${(r.b.cum_return * 100).toFixed(0)}% | ${r.b.sharpe.toFixed(2)} | ${(r.b.max_drawdown * 100).toFixed(1)}% |`).join("\n");
  const tested = rows.filter((r) => r.m !== "AlwaysUp"), best = tested.reduce((a, b) => (b.c.roc_auc > a.c.roc_auc ? b : a));
  let sigLine = "";
  if (sig.length) {
    const a = s.meta.stats.alpha, nAcc = sig.filter((r) => r.DM_p_holm < a).length, nSh = sig.filter((r) => r.boot_sharpe_p_holm < a).length;
    sigLine = `**Significance:** ${nAcc} of ${sig.length} accuracy comparisons and ${nSh} of ${sig.length} Sharpe comparisons are significant at ${(a * 100).toFixed(0)}% after Holm correction.`;
  }
  body.append(md(`| Model | Accuracy | ROC-AUC | Cumulative return | Sharpe | Max drawdown |
|---|---|---|---|---|---|
${tr}

${sigLine}

**How to read this honestly**
- The best ROC-AUC among the tested models is **${best.c.roc_auc.toFixed(3)}** (${nm(best.m)}), and 0.5 means no skill. Values this close to 0.5 say that next-day direction is almost unpredictable from these clues.
- Some models post a higher *Sharpe* or *cumulative return* than buy & hold, but look at whether that edge is statistically significant above and whether it holds across market regimes before believing it.
- The takeaway is not that neural networks are useless. It is that **extra complexity does not automatically buy predictive power when the data is mostly noise**, and that careful evaluation (walk-forward, costs, significance tests) is what separates a real result from a lucky one.

*Limitations: only ${s.T} large US stocks, daily data, one 5-year window, simple long/cash rule, and buy & hold is a tough bar in a rising market. Results may differ for other assets or periods.*`));
  return body;
}

export default async function (s) {
  const root = page("How it all works", "A start-to-finish, plain-English guide: from raw prices, through indicators and neural networks, to trading returns.");
  root.append(tabs([
    { label: "1. Big picture", render: bigPicture },
    { label: "2. Data & target", render: () => dataAndTarget(s) },
    { label: "3. Indicators", render: indicators },
    { label: "4. Fair testing", render: () => walkForward(s) },
    { label: "5. Models & LSTM", render: () => models(s) },
    { label: "6. Trades & returns", render: () => trades(s) },
    { label: "7. Scoring", render: scoring },
    { label: "8. Findings", render: () => findings(s) },
  ]));
  return root;
}
