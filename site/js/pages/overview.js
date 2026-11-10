import { name, CLASS } from "../data.js";
import { callout, cumulative, f, h, lineTrace, metric, page, plot, table } from "../ui.js";

export default async function (s) {
  const root = page("ML vs classical models: next-day stock direction",
    `Walk-forward evaluation on ${s.T} liquid US equities, test years ${s.years[0]}-${s.years.at(-1)}`);
  const cm = s.table("classification_metrics.csv"), bt = s.table("backtest_portfolio.csv"), sig = s.table("significance_vs_baselines.csv");
  const byM = (rows) => s.order.map((m) => rows.find((r) => r.model === m)).filter(Boolean);
  const mRows = byM(cm), bRows = byM(bt);
  const tested = (rows) => rows.filter((r) => r.model !== "AlwaysUp");
  const best = (rows, k) => tested(rows).reduce((a, b) => (b[k] > a[k] ? b : a));
  const up = cm.find((r) => r.model === "AlwaysUp");
  const ba = best(cm, "accuracy"), br = best(cm, "roc_auc"), bs = best(bt, "sharpe");

  root.append(h("div", { class: "grid c4" },
    metric("Best accuracy (non-benchmark)", f.pct(2)(ba.accuracy), name(ba.model)),
    metric("Always-up accuracy", f.pct(2)(up.accuracy), "benchmark"),
    metric("Best ROC-AUC", f.d(3)(br.roc_auc), name(br.model)),
    metric("Best Sharpe", f.d(2)(bs.sharpe), name(bs.model))));

  if (sig.length) {
    const a = s.meta.stats.alpha, nAcc = sig.filter((r) => r.DM_p_holm < a).length, nSh = sig.filter((r) => r.boot_sharpe_p_holm < a).length;
    root.append(callout(`<b>Verdict from the generated tests:</b> ${nAcc} of ${sig.length} accuracy comparisons and ${nSh} of ${sig.length} Sharpe comparisons are significant at ${(a * 100).toFixed(0)}% after Holm correction (challenger vs each classical baseline).`));
  }

  const num = (k, label) => ({ key: k, label: label || k, fmt: f.d(3) });
  root.append(h("div", { class: "grid c2" },
    h("div", {}, h("h3", {}, "Classification"), h("p", { class: "caption" }, "Out-of-sample next-day direction metrics"),
      table([{ key: (r) => name(r.model), label: "model", text: true }, { key: (r) => CLASS[r.model], label: "class", text: true },
        num("accuracy"), num("precision"), num("recall"), num("f1"), num("roc_auc", "AUC"), { key: "n", label: "n", fmt: f.int }], mRows, { compact: true })),
    h("div", {}, h("h3", {}, "Backtest"), h("p", { class: "caption" }, "Equal-weight portfolio, after costs"),
      table([{ key: (r) => name(r.model), label: "model", text: true }, { key: "cum_return", label: "cum ret", fmt: f.pct(0) },
        { key: "ann_return", label: "ann ret", fmt: f.pct(1) }, { key: "sharpe", label: "sharpe", fmt: f.d(2) },
        { key: "max_drawdown", label: "max DD", fmt: f.pct(1) }, { key: "calmar", label: "calmar", fmt: f.d(2) }], bRows, { compact: true }))));

  const port = s.portfolio(s.meta.backtest.transaction_cost, [...Array(s.T).keys()]);
  root.append(plot("Cumulative return of the equal-weight portfolio",
    s.order.map((m) => lineTrace(s.dates, cumulative(port[m]).map((v) => v - 1), m)), { yfmt: ".0%" }));
  root.append(h("p", { class: "caption" }, "Use the other pages to change models, tickers, dates and transaction costs, and to run side-by-side comparisons."));
  return root;
}
