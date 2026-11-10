import { COLOR } from "../data.js";
import { callout, controls, csv, ctl, dlButton, f, h, md, page, plot, slider, table } from "../ui.js";

function forest(sig, what, title) {
  const traces = [];
  for (const r of sig) {
    const label = `${r.challenger} vs ${r.baseline}`, c = COLOR[r.challenger], lo = r[`${what}_ci_low`], hi = r[`${what}_ci_high`], d = r[`${what}_diff`];
    traces.push({ type: "scatter", x: [lo, hi], y: [label, label], mode: "lines", line: { color: c, width: 3 }, showlegend: false, hoverinfo: "skip" });
    traces.push({ type: "scatter", x: [d], y: [label], mode: "markers", showlegend: false, marker: { size: 12, color: c, line: { color: "white", width: 2 } },
      hovertemplate: `${label}<br>diff %{x:.4f}<br>95% CI [${lo.toFixed(4)}, ${hi.toFixed(4)}]<extra></extra>` });
  }
  return plot(title, traces, { height: 380, hover: "closest" },
    { shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: 0, x1: 0, line: { color: "#8a8984", width: 1 } }], yaxis: { autorange: "reversed" } });
}

export default async function (s) {
  const root = page("Statistical significance",
    "Challengers (RF, XGBoost, LSTM) vs classical baselines (Logistic, ARIMA). One-sided tests, H1: challenger is better. Holm correction is applied across the six comparisons.");
  const sig = s.table("significance_vs_baselines.csv");
  if (!sig.length) { root.append(callout("Run <code>scripts/07_stat_tests.py</code> first.", true)); return root; }

  let alpha = s.meta.stats.alpha;
  const out = h("div");
  root.append(controls(ctl("Significance level (alpha)", slider(0.01, 0.10, 0.01, alpha, (v) => v.toFixed(2), (v) => { alpha = v; draw(); }))), out);

  function draw() {
    out.replaceChildren();
    const verdict = (p) => (p < alpha ? "significant" : "not significant");
    const sc = (key) => ({ key, label: key, text: true, cls: (v) => (v === "significant" ? "sig" : "") });
    const num = (k, label) => ({ key: k, label: label || k, fmt: f.d(4) });
    const rows = sig.map((r) => ({ ...r, accV: verdict(r.DM_p_holm), shV: verdict(r.boot_sharpe_p_holm) }));
    out.append(table([
      { key: "challenger", label: "challenger", text: true }, { key: "baseline", label: "baseline", text: true },
      num("acc_diff"), num("DM_p"), num("DM_p_holm"), num("boot_acc_p_holm"), num("ttest_p_holm"), num("sharpe_diff"), num("boot_sharpe_p"), num("boot_sharpe_p_holm"),
      { ...sc("accV"), label: "accuracy verdict" }, { ...sc("shV"), label: "Sharpe verdict" }], rows, { compact: true }));
    out.append(h("div", { class: "grid c2" },
      forest(sig, "acc", "Accuracy difference (95% bootstrap CI)"), forest(sig, "sharpe", "Sharpe difference (95% bootstrap CI)")));

    const sb = s.table("sharpe_vs_buy_and_hold.csv");
    if (sb.length) {
      out.append(h("h3", {}, "Sharpe vs buy-and-hold"),
        table([{ key: "model", label: "model", text: true }, ...Object.keys(sb[0]).filter((k) => k !== "model").map((k) => num(k))], sb));
    }
    out.append(h("details", { class: "card" }, h("summary", {}, "How to read this"), md(
      "- **DM** (Diebold-Mariano): compares 0/1 loss per day, averaged across tickers, with HAC variance.\n" +
      "- **Bootstrap**: stationary block bootstrap over dates (mean block 10 days).\n" +
      "- **t-test**: paired, on per-ticker per-year accuracies.\n" +
      "- **Holm**: adjusts p-values for six simultaneous comparisons; use the `_holm` columns to decide.\n" +
      "- A CI that includes 0 means the difference is not distinguishable from zero.\n" +
      "- With ~1,250 test days, differences under about 1.5 percentage points of accuracy are undetectable.")));
    const cols = Object.keys(sig[0]);
    out.append(h("p", {}, dlButton("Download significance table (CSV)", "significance_vs_baselines.csv", () => csv(cols, sig.map((r) => cols.map((c) => r[c]))))));
  }
  draw();
  return root;
}
