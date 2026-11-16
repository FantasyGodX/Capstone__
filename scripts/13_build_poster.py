"""Step 13 (optional): build the research poster -> poster/poster.html, poster.pdf (48 x 36 in) and poster.png.

Numbers are read from results/ at build time. Run after scripts/11_report_figures.py.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capstone import report as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster"
TITLE = "Do Machine Learning and Neural Networks Beat Classical Models at Predicting Next-Day Stock Direction?"
AUTHOR, DEPT, MENTOR = "Mubarak Odufade", "Department of Computer Science", "Faculty Mentor: Dr. Yumei Huo"
SCHOOL, SCHOOL_SUB = "College of Staten Island", "The City University of New York"
W_PX, H_PX = 4608, 3456  # 48 x 36 inches at 96 dpi


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def build_html(d):
    c, m, bt, sig = d["cfg"], d["metrics"], d["bt"], d["sig"]
    wf, preds = c["walk_forward"], d["preds"]
    et, cs = R.exposure_turnover(preds), R.cost_sensitivity(preds)
    n = int(m["n"].iloc[0])
    tested = [k for k in R.ORDER if k != "AlwaysUp"]
    acc_lo, acc_hi = m.loc[tested, "accuracy"].min(), m.loc[tested, "accuracy"].max()
    auc_lo, auc_hi = m.loc[tested, "roc_auc"].min(), m.loc[tested, "roc_auc"].max()
    n_sig_acc = int((sig["DM_p_holm"] < c["stats"]["alpha"]).sum())
    n_sig_sh = int((sig["boot_sharpe_p_holm"] < c["stats"]["alpha"]).sum())
    others = [k for k in tested if k != "LSTM"]
    c50_lo, c50_hi = cs.loc[others, 50].min(), cs.loc[others, 50].max()

    sigrows = ""
    for r_ in sig.itertuples():
        a_hi = "background:#d9f0e0;font-weight:700" if r_.DM_p_holm < c["stats"]["alpha"] else ""
        s_hi = "background:#d9f0e0;font-weight:700" if r_.boot_sharpe_p_holm < c["stats"]["alpha"] else ""
        sigrows += (f"<tr><td>{r_.challenger} vs {r_.baseline}</td><td>{r_.acc_diff * 100:+.2f} pp</td><td style='{a_hi}'>{r_.DM_p_holm:.2f}</td>"
                    f"<td>{r_.sharpe_diff:+.2f}</td><td style='{s_hi}'>{r_.boot_sharpe_p_holm:.2f}</td></tr>")
    glance = ""
    for k in R.ORDER:
        glance += (f"<tr><td>{R.lab(k)}</td><td>{pct(m.loc[k, 'accuracy'])}</td><td>{m.loc[k, 'roc_auc']:.3f}</td>"
                   f"<td>{pct(bt.loc[k, 'cum_return'], 0)}</td><td>{bt.loc[k, 'sharpe']:.2f}</td>"
                   f"<td>{pct(bt.loc[k, 'max_drawdown'], 0)}</td><td>{pct(et.loc[k, 'pct_long'], 0)}</td>"
                   f"<td>{et.loc[k, 'trades_per_year']:.0f}</td></tr>")
    models_rows = [("Buy &amp; hold", "Benchmark", "Always predicts up"),
                   ("Logistic regression", "Classical", "Weights each indicator, adds them up"),
                   ("ARIMA", "Classical", "Extrapolates the stock's own recent returns"),
                   ("Random forest", "Machine learning", "300 shallow decision trees vote"),
                   ("XGBoost", "Machine learning", "Trees that correct each other's errors"),
                   ("LSTM", "Neural network", "Reads the last 20 days in order, with memory")]
    models_html = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c_}</td></tr>" for a, b, c_ in models_rows)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
@page {{ size: 48in 36in; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
body {{ width: {W_PX}px; height: {H_PX}px; overflow: hidden; background: #f4f5f7; color: #16161d;
  font-family: "Segoe UI", Arial, sans-serif; font-size: 36px; line-height: 1.38; display: flex; flex-direction: column; }}
header {{ background: #14142b; color: #fff; height: 400px; display: flex; align-items: center; padding: 0 70px; gap: 70px; flex: none; }}
.school {{ background: #fff; color: #14142b; width: 760px; height: 280px; border-radius: 10px; padding: 30px 38px; flex: none;
  display: flex; flex-direction: column; justify-content: center; font-family: Georgia, serif; }}
.school b {{ font-size: 70px; line-height: 1.05; letter-spacing: -1px; }}
.school span {{ font-size: 32px; margin-top: 14px; color: #44445a; font-family: "Segoe UI", Arial, sans-serif; }}
.titlebox {{ flex: 1; text-align: center; }}
h1 {{ font-family: Georgia, "Times New Roman", serif; font-size: 100px; line-height: 1.1; margin: 0 0 24px; font-weight: 700; }}
.byline {{ font-size: 40px; color: #cfd3e6; }} .byline b {{ color: #fff; }} .byline i {{ font-style: normal; color: #7f86a8; margin: 0 22px; }}
.tag {{ width: 520px; flex: none; text-align: right; font-size: 34px; color: #cfd3e6; line-height: 1.4; }}
.tag b {{ display: block; font-family: Georgia, serif; font-size: 52px; color: #fff; margin-bottom: 10px; }}
main {{ flex: 1; display: flex; gap: 52px; padding: 48px 70px 40px; min-height: 0; }}
.col {{ flex: 1; display: flex; flex-direction: column; justify-content: space-between; min-width: 0; }}
section {{ margin-bottom: 0; }}
h2 {{ font-family: Georgia, "Times New Roman", serif; font-size: 62px; margin: 0 0 18px; color: #14142b; border-bottom: 5px solid #14142b; padding-bottom: 6px; line-height: 1.1; }}
p {{ margin: 0 0 16px; }}
.card {{ background: #e8ebf1; border-radius: 12px; padding: 26px 32px; }}
.phase {{ display: flex; gap: 26px; align-items: flex-start; background: #e8ebf1; border-radius: 12px; padding: 22px 30px; margin-bottom: 16px; }}
.phase .n {{ flex: none; width: 66px; height: 66px; background: #14142b; color: #fff; border-radius: 10px; font-size: 40px; font-weight: 700; display: flex; align-items: center; justify-content: center; }}
.phase b {{ font-size: 42px; display: block; margin-bottom: 4px; color: #14142b; }}
.strip {{ background: #14142b; color: #fff; border-radius: 12px; display: flex; justify-content: space-around; padding: 24px 10px; text-align: center; }}
.strip div b {{ display: block; font-family: Georgia, serif; font-size: 84px; color: #f08a3c; line-height: 1; }}
.strip div span {{ font-size: 30px; color: #cfd3e6; }}
.plain {{ background: #14142b; color: #fff; border-radius: 12px; padding: 18px 32px; margin-top: 14px; font-size: 33px; }}
.plain .lab {{ color: #f08a3c; font-weight: 700; display: block; margin-bottom: 4px; }}
.plain em {{ color: #f08a3c; font-style: normal; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; font-size: 31px; background: #fff; border-radius: 8px; overflow: hidden; }}
th {{ background: #14142b; color: #fff; padding: 10px 12px; text-align: right; font-weight: 600; }}
td {{ padding: 4px 12px; text-align: right; border-bottom: 1px solid #dde0e8; }}
th:first-child, td:first-child {{ text-align: left; }}
table.txt td, table.txt th {{ text-align: left; }}
tr:last-child td {{ border-bottom: 0; }} tr.hi td {{ background: #fff1e3; font-weight: 700; }}
.fig {{ background: #fff; border-radius: 12px; padding: 14px; text-align: center; }}
.fig img {{ width: 100%; display: block; }}
.fig img.short {{ width: auto; max-width: 100%; max-height: 370px; margin: 0 auto; }}
.fig img.short2 {{ width: auto; max-width: 100%; max-height: 250px; margin: 0 auto; }}
.cap {{ font-size: 28px; color: #4a4a58; margin-top: 8px; line-height: 1.25; }}
ul {{ margin: 0 0 14px; padding-left: 40px; }} li {{ margin-bottom: 8px; }}
.refs {{ font-size: 25px; line-height: 1.28; color: #33333f; }} .refs p {{ margin: 0 0 6px; padding-left: 28px; text-indent: -28px; }}
.flowrow {{ display: flex; align-items: center; gap: 14px; }} .flowrow span {{ font-size: 44px; color: #55556a; }}
.fb {{ flex: 1; color: #fff; border-radius: 12px; padding: 18px 12px; text-align: center; font-weight: 700; font-size: 40px; line-height: 1.15; }}
.fb small {{ display: block; font-weight: 400; font-size: 25px; margin-top: 6px; opacity: .95; line-height: 1.2; }}
.two {{ display: flex; gap: 20px; }} .two > * {{ flex: 1; min-width: 0; }}
</style></head><body>
<header>
  <div class="school"><b>{SCHOOL}</b><span>{SCHOOL_SUB}</span></div>
  <div class="titlebox"><h1>{TITLE}</h1>
    <div class="byline"><b>{AUTHOR}</b><i>|</i>{DEPT}<i>|</i>{MENTOR}</div></div>
  <div class="tag"><b>Capstone Research</b>Walk-forward evaluation<br>on {len(c['tickers'])} U.S. equities, {wf['first_test_year']}&ndash;{wf['last_test_year']}</div>
</header>
<main>

<div class="col">
  <section><h2>Abstract</h2>
    <p>We test whether machine learning and a neural network predict <b>next-day stock direction</b> better than simple statistical
    models. Six strategies (logistic regression, ARIMA, random forest, XGBoost, an LSTM, and an always-up benchmark) are retrained
    every year on a rolling {wf['train_years']}-year window and judged only on the following unseen year, then traded with
    transaction costs.</p>
    <p>All models reach <b>{pct(acc_lo)}&ndash;{pct(acc_hi)}</b> accuracy (ROC-AUC {auc_lo:.3f}&ndash;{auc_hi:.3f}), below the
    {pct(m.loc['AlwaysUp', 'accuracy'])} of always guessing &ldquo;up&rdquo;. <b>{n_sig_acc} of {len(sig)}</b> accuracy comparisons are significant
    after correction. The LSTM has the best backtest, but through low turnover and high market exposure, not better predictions.</p></section>

  <section><h2>Introduction</h2>
    <p>If a model could call tomorrow&rsquo;s direction reliably it would be valuable, yet markets are noisy and heavily competed.
    Financial machine learning is also prone to over-optimistic results from data leakage and repeated tuning.</p>
    <p><b>Research question:</b> <i>Do ML and deep-learning models significantly outperform classical models at next-day direction,
    under strict walk-forward validation and realistic costs?</i></p>
    <ul><li><b>H0:</b> no significant difference between classical and ML models.</li><li><b>H1:</b> ML and/or deep learning is significantly better.</li></ul>
    <div class="strip"><div><b>{n:,}</b><span>out-of-sample predictions</span></div><div><b>{len(c['tickers'])}</b><span>stocks</span></div>
      <div><b>{wf['last_test_year'] - wf['first_test_year'] + 1}</b><span>test years</span></div><div><b>26</b><span>indicators</span></div></div></section>

  <section><h2>Data &amp; Indicators</h2>
    <p>Daily prices (Yahoo Finance, 2014&ndash;2024) for {', '.join(c['tickers'])}, plus the S&amp;P 500 fund (SPY) and the VIX.
    Target: <b>1 if tomorrow&rsquo;s close is higher</b>, else 0. Every indicator uses only information known at the close.</p>
    <table class="txt"><tr><th>Family</th><th>Indicators (26)</th></tr>
      <tr><td><b>Technical (10)</b></td><td>RSI, MACD (3), Bollinger %B and width, price / moving-average ratios (3), ATR</td></tr>
      <tr><td><b>Statistical (12)</b></td><td>Volatility (3), last 5 daily returns, skewness, momentum (3)</td></tr>
      <tr><td><b>Market (4)</b></td><td>SPY return, correlation with SPY, VIX level and change</td></tr></table>
    <p style="margin-top:14px">Up-days are {pct(d['ds']['y'].mean())} of the data, so <b>always-up is right {pct(d['ds']['y'].mean(), 0)} of the time</b>: the real bar to beat.</p></section>

  <section><h2>Related Work</h2>
    <table class="txt"><tr><th>Study</th><th>Finding</th></tr>
      <tr><td>Fama (1970)</td><td>Weak-form efficiency: past prices should not predict returns</td></tr>
      <tr><td>Welch &amp; Goyal (2008)</td><td>Many return predictors fail out of sample</td></tr>
      <tr><td>Gu, Kelly &amp; Xiu (2020)</td><td>Trees and neural nets improve forecasts on large stock panels</td></tr>
      <tr><td>Fischer &amp; Krauss (2018)</td><td>LSTM signals profitable before costs, fading in later years</td></tr>
      <tr><td><b>This study</b></td><td><b>Few stocks, daily indicators, strict walk-forward, costs, formal tests</b></td></tr></table></section>
  <section><h2>Backtest Rules</h2>
    <ul><li>Model says <b>up</b> &rarr; hold the stock until tomorrow&rsquo;s close; <b>down</b> &rarr; hold cash.</li>
      <li>No leverage, no shorting; <b>{pct(c['backtest']['transaction_cost'], 2)}</b> cost on every position change.</li>
      <li>Equal-weight portfolio of the {len(c['tickers'])} stocks, rebalanced daily.</li>
      <li>Reported: return, Sharpe ratio, maximum drawdown, Calmar ratio.</li></ul></section>
  <section class="refs"><h2 style="font-size:46px">References</h2>
    <p>Fama (1970), <i>J. Finance</i> 25(2). Welch &amp; Goyal (2008), <i>Rev. Financial Studies</i> 21(4). Gu, Kelly &amp; Xiu (2020), <i>Rev. Financial Studies</i> 33(5).
    Fischer &amp; Krauss (2018), <i>Eur. J. Oper. Res.</i> 270(2). Hochreiter &amp; Schmidhuber (1997), <i>Neural Comput.</i> 9(8). Chen &amp; Guestrin (2016), <i>KDD</i>.
    Diebold &amp; Mariano (1995), <i>J. Bus. Econ. Stat.</i> 13(3). Politis &amp; Romano (1994), <i>JASA</i> 89(428). Holm (1979), <i>Scand. J. Stat.</i> 6(2).
    Bailey et al. (2014), <i>Notices AMS</i> 61(5). Full list in the paper.</p></section>
</div>

<div class="col">
  <section><h2>Methodology: 4-Phase Pipeline</h2>
    <div class="phase"><div class="n">1</div><div><b>Data &amp; indicators</b>26 indicators per stock-day, standardised using training years only.</div></div>
    <div class="phase"><div class="n">2</div><div><b>Walk-forward validation</b>Train on {wf['train_years']} years, test on the next; a fresh model per stock and fold; last training day purged.</div></div>
    <div class="phase"><div class="n">3</div><div><b>Six models, one prediction table</b>Each outputs a score and an up/down call for every stock-day.</div></div>
    <div class="phase"><div class="n">4</div><div><b>Backtest &amp; significance tests</b>Long when &ldquo;up&rdquo;, cash otherwise, {pct(c['backtest']['transaction_cost'], 2)} cost per trade; Diebold-Mariano, block bootstrap, Holm correction.</div></div>
    <div class="fig"><img src="../paper/figures/fig_walkforward.png"><div class="cap">Rolling walk-forward folds: every test year is unseen by its model.</div></div></section>

  <section><h2>The Models</h2>
    <table class="txt"><tr><th>Model</th><th>Type</th><th>Idea</th></tr>{models_html}</table></section>

  <section><h2>How the LSTM Works</h2>
    <div class="flowrow"><div class="fb" style="background:#2a78d6">Input<small>last 20 days &times; 26 indicators</small></div><span>&rarr;</span>
      <div class="fb" style="background:#6250d6">LSTM<small>32 memory cells, 3 gates</small></div><span>&rarr;</span>
      <div class="fb" style="background:#1baf7a">Dropout<small>30% off in training</small></div><span>&rarr;</span>
      <div class="fb" style="background:#e8742c">P(up)<small>sigmoid; up if &ge; 0.5</small></div></div>
    <div class="cap" style="margin:8px 0 14px">Trained with Adam on cross-entropy; early stopping on the last 15% of the training years.</div>
    <div class="plain"><span class="lab">In Plain English:</span>Most models see one day of indicators. The LSTM reads the <em>last 20 days in order</em>,
    keeping a small memory (32 numbers) and deciding at each day what to forget, store and report. The final memory becomes the probability
    that tomorrow is up. <em>Dropout and early stopping</em> stop it memorising noise.</div></section>

</div>

<div class="col">
  <section><h2>Results: Prediction Accuracy</h2>
    <div class="fig"><img src="figures/poster_accuracy.png"><div class="cap">95% intervals; dashed line = coin flip (50%).</div></div>
    <div class="plain"><span class="lab">In Plain English:</span>Next-day direction is close to a coin flip. Every model lands near 51&ndash;52%, and simply staying invested was <em>right more often ({pct(m.loc['AlwaysUp', 'accuracy'])})</em>.</div></section>

  <section><h2>Results at a Glance</h2>
    <table><tr><th>Model</th><th>Acc.</th><th>AUC</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>% long</th><th>Trades/yr</th></tr>{glance}</table></section>

  <section><h2>Backtest &amp; Significance</h2>
    <div class="two"><div class="fig"><img src="figures/poster_equity.png"><div class="cap">Cumulative return, {pct(c['backtest']['transaction_cost'], 2)} cost.</div></div>
      <div class="fig"><img src="figures/poster_cost.png"><div class="cap">Sharpe ratio as trading costs rise.</div></div></div>
    <div class="plain"><span class="lab">In Plain English:</span>The LSTM looks best because it <em>trades about {et.loc['LSTM', 'trades_per_year']:.0f} times a year</em> (others
    {et.loc[others, 'trades_per_year'].min():.0f}&ndash;{et.loc[others, 'trades_per_year'].max():.0f}) and stays invested {pct(et.loc['LSTM', 'pct_long'], 0)} of the time, not because it predicts better.
    At 50 bp of cost the busy models&rsquo; Sharpe falls to <em>{c50_lo:.2f} to {c50_hi:.2f}</em>; the LSTM keeps {cs.loc['LSTM', 50]:.2f}.</div>
    <div style="margin-top:14px"><table><tr><th>Challenger vs baseline</th><th>Acc. diff.</th><th>p (Holm)</th><th>Sharpe diff.</th><th>p (Holm)</th></tr>{sigrows}</table>
      <div class="cap">One-sided, Holm-adjusted. Shaded = significant at 5%.</div></div></section>

  <section><h2>Conclusion &amp; Future Work</h2>
    <p><b>We fail to reject H0.</b> None of the {len(sig)} accuracy comparisons is significant after Holm correction; only the LSTM&rsquo;s Sharpe advantage
    over ARIMA ({n_sig_sh} of {len(sig)}) is, and that rests on low turnover.</p>
    <p><b>Next:</b> more stocks (including delisted) &bull; longer horizons &bull; test why the LSTM trades less &bull; news and fundamentals &bull; tuned ensembles &bull; realistic costs.</p></section>

</div>

</main>
<script>
window.addEventListener("load", () => {{
  const h = [...document.querySelectorAll(".col")].map(c => Math.round(c.scrollHeight - c.clientHeight));
  document.title = "overflow:" + h.join(",");
}});
</script>
</body></html>"""


def find_chrome():
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return None


def main():
    d = R.load()
    R.check_claims(d)
    R.make_poster_figs(d)
    OUT.mkdir(exist_ok=True)
    html = OUT / "poster.html"
    html.write_text(build_html(d), encoding="utf-8")
    print("wrote", html)
    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found: open poster/poster.html and print to a 48 x 36 in PDF")
        return
    base = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000", "--hide-scrollbars"]
    r = subprocess.run(base + ["--dump-dom", html.as_uri()], capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout or ""
    title = out.split("<title>")[1].split("</title>")[0] if "<title>" in out else "?"
    print("column overflow in px (<= 0 is good):", title)
    subprocess.run(base + [f"--print-to-pdf={OUT / 'poster.pdf'}", html.as_uri()], check=True, capture_output=True)
    subprocess.run(base + [f"--window-size={W_PX},{H_PX}", f"--screenshot={OUT / 'poster.png'}", html.as_uri()], check=True, capture_output=True)
    print("wrote poster.pdf and poster.png in", OUT)


if __name__ == "__main__":
    main()
