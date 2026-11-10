import { name, COLOR } from "../data.js";
import { binomUpperP, bootstrapDiff, dieboldMariano, mean, sharpe, spearman } from "../stats.js";
import { chips, controls, ctl, cumulative, f, h, lineTrace, metric, page, plot, select, table } from "../ui.js";

export default async function (s) {
  const root = page("Compare two models", "Head-to-head with live statistical tests. H1 for every one-sided test: model A is better than model B.");
  const st = { a: "LSTM", b: "Logistic", tickers: [...s.tickers], years: [...s.years], nBoot: 1000 };
  const out = h("div");
  let token = 0;
  root.append(controls(
    ctl("Model A", select(s.order, st.a, (v) => { st.a = v; run(); }, name)),
    ctl("Model B", select(s.order, st.b, (v) => { st.b = v; run(); }, name)),
    ctl("Tickers", chips(s.tickers, st.tickers, (v) => { st.tickers = v; run(); })),
    ctl("Test years", chips(s.years, st.years, (v) => { st.years = v; run(); }, String)),
    ctl("Bootstrap draws", select([500, 1000, 2000, 5000], st.nBoot, (v) => { st.nBoot = v; run(); }))), out);

  function run() {
    out.replaceChildren(h("p", { class: "muted" }, "Computing…"));
    const mine = ++token;
    setTimeout(() => { if (mine === token) draw(); }, 20);
  }

  function draw() {
    out.replaceChildren();
    const { a, b } = st;
    if (a === b || !st.tickers.length || !st.years.length) { out.append(h("p", { class: "callout warn" }, "Choose two different models, and at least one ticker and year.")); return; }
    const tix = st.tickers.map((t) => s.tickers.indexOf(t)), ys = new Set(st.years), rows = s.select(tix, ys);
    const A = s.models[a], B = s.models[b], n = rows.length;
    let onlyA = 0, onlyB = 0, agree = 0, sa = 0, sb = 0;
    for (const r of rows) {
      const ca = A.correct[r], cb = B.correct[r];
      if (ca && !cb) onlyA++; else if (!ca && cb) onlyB++;
      if (A.pred[r] === B.pred[r]) agree++;
      sa += ca; sb += cb;
    }
    const accA = sa / n, accB = sb / n;
    const scA = rows.map((r) => A.score[r]), scB = rows.map((r) => B.score[r]);
    const uniq = (x) => new Set(x).size > 1;
    const corr = uniq(scA) && uniq(scB) ? spearman(scA, scB).toFixed(2) : "n/a";

    out.append(h("div", { class: "grid c5" },
      metric(`${name(a)} accuracy`, f.pct(2)(accA)), metric(`${name(b)} accuracy`, f.pct(2)(accB)),
      metric("Difference (A-B)", `${((accA - accB) * 100 >= 0 ? "+" : "") + ((accA - accB) * 100).toFixed(2)} pp`),
      metric("Predictions agree", f.pct(1)(agree / n)), metric("Score correlation", corr)));

    // per-date mean correctness across selected tickers (the unit used by the tests)
    const dIdx = [...Array(s.D).keys()].filter((d) => ys.has(s.year[d]));
    const dA = dIdx.map((d) => mean(tix.map((t) => A.correct[t * s.D + d]))), dB = dIdx.map((d) => mean(tix.map((t) => B.correct[t * s.D + d])));
    const dm = dieboldMariano(dB.map((v) => 1 - v), dA.map((v) => 1 - v));
    const pSign = onlyA + onlyB ? binomUpperP(onlyA, onlyA + onlyB) : NaN;
    const boot = bootstrapDiff(Float64Array.from(dA), Float64Array.from(dB), (x) => mean(x), st.nBoot, 10, s.meta.seed);
    const port = s.portfolio(s.meta.backtest.transaction_cost, tix);
    const pa = Float64Array.from(dIdx.map((d) => port[a][d])), pb = Float64Array.from(dIdx.map((d) => port[b][d]));
    const td = s.meta.backtest.trading_days;
    const sh = bootstrapDiff(pa, pb, (x) => sharpe(x, td), st.nBoot, 10, s.meta.seed + 1);

    const tests = [
      { test: "Diebold-Mariano (0/1 loss)", stat: dm.stat, diff: boot.obs, lo: NaN, hi: NaN, p: dm.p },
      { test: "Bootstrap, accuracy", stat: NaN, diff: boot.obs, lo: boot.lo, hi: boot.hi, p: boot.p },
      { test: "Exact sign test on discordant pairs", stat: NaN, diff: (onlyA - onlyB) / n, lo: NaN, hi: NaN, p: pSign },
      { test: "Bootstrap, Sharpe (portfolio)", stat: NaN, diff: sh.obs, lo: sh.lo, hi: sh.hi, p: sh.p },
    ];
    const d4 = f.d(4);
    out.append(h("h3", {}, "Statistical tests"),
      table([{ key: "test", label: "test", text: true }, { key: "stat", label: "statistic", fmt: d4 }, { key: "diff", label: "A - B", fmt: d4 },
        { key: "lo", label: "95% CI low", fmt: d4 }, { key: "hi", label: "95% CI high", fmt: d4 }, { key: "p", label: "p (one-sided)", fmt: d4 }], tests),
      h("p", { class: "caption" }, "These are single unadjusted tests. If you look at many pairs, correct for multiple comparisons (the Statistics page applies Holm). Bootstrap results use a fixed seed, so they are reproducible but differ slightly from the Python pipeline's draws."));

    out.append(h("h3", {}, "Where do they differ?"));
    let cum = 0;
    const extra = dA.map((v, i) => (cum += v - dB[i]));
    const dates = dIdx.map((d) => s.dates[d]), eqA = cumulative(pa), eqB = cumulative(pb);
    out.append(h("div", { class: "grid c2" },
      plot("Cumulative extra correct calls per ticker-day (A - B)", [{ type: "scatter", mode: "lines", x: dates, y: extra, name: `${name(a)} - ${name(b)}`, line: { color: COLOR[a], width: 2 } }],
        {}, { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8984", width: 1 } }] }),
      plot("Backtest cumulative return", [lineTrace(dates, eqA.map((v) => v - 1), a), lineTrace(dates, eqB.map((v) => v - 1), b)], { yfmt: ".0%" })));

    const diffBy = (keyFn, keys) => keys.map((k) => {
      const rs = rows.filter((r) => keyFn(r) === k); let t = 0; for (const r of rs) t += A.correct[r] - B.correct[r];
      return (t / rs.length) * 100;
    });
    const bar = (title, labels, vals) => plot(title, [{ type: "bar", x: labels.map(String), y: vals, marker: { color: vals.map((v) => (v >= 0 ? COLOR[a] : COLOR[b])) },
      hovertemplate: "%{x}: %{y:+.2f} pp<extra></extra>" }], { height: 360, hover: "closest" },
    { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#8a8984", width: 1 } }] });
    out.append(h("div", { class: "grid c2" },
      bar("Accuracy difference by ticker (pp)", st.tickers, diffBy((r) => Math.floor(r / s.D), tix)),
      bar("Accuracy difference by year (pp)", st.years, diffBy((r) => s.year[r % s.D], st.years))));
    out.append(h("p", { class: "caption" }, `Bars in ${name(a)}'s colour: A ahead. In ${name(b)}'s colour: B ahead.`),
      h("p", {}, h("b", {}, "Discordant calls: "), `only ${name(a)} right on ${onlyA.toLocaleString("en-US")} ticker-days, only ${name(b)} right on ${onlyB.toLocaleString("en-US")}.`));
  }
  run();
  return root;
}
