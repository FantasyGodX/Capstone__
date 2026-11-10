import { name, COLOR } from "../data.js";
import { callout, controls, ctl, f, h, page, plot, radio, select, table, tabs } from "../ui.js";

function regimeTab(s, kind) {
  const df = s.table(`regime_${kind}.csv`);
  if (!df.length) return callout("Run <code>scripts/08_robustness.py</code> first.", true);
  const regs = [...new Set(df.map((r) => r[kind]))].sort();
  let metric = "accuracy";
  const box = h("div"), wrap = h("div");
  const draw = () => {
    box.replaceChildren(plot(`${metric} by regime`, s.order.map((m) => ({ type: "scatter", mode: "markers+lines", name: name(m), x: regs,
      y: regs.map((rg) => (df.find((r) => r.model === m && r[kind] === rg) || {})[metric]),
      marker: { size: 11, color: COLOR[m], line: { color: "white", width: 2 } }, line: { color: COLOR[m], width: 1.5 } })),
    { height: 380, yfmt: metric === "accuracy" ? ".1%" : undefined }));
  };
  wrap.append(controls(ctl("Metric", radio(["accuracy", "sharpe", "mean_daily_ret"], metric, (v) => { metric = v; draw(); }))), box);
  draw();
  const cols = [{ key: (m) => name(m), label: "model", text: true }];
  for (const k of ["n_days", "accuracy", "sharpe"]) for (const rg of regs) {
    cols.push({ key: (m) => (df.find((r) => r.model === m && r[kind] === rg) || {})[k], label: `${k} · ${rg}`, fmt: k === "n_days" ? f.int : f.d(4) });
  }
  wrap.append(table(cols, s.order));
  const dm = s.table(`regime_${kind}_dm_vs_logistic.csv`);
  wrap.append(h("p", {}, h("b", {}, "Diebold-Mariano within regime (challenger vs Logistic, one-sided)")));
  if (dm.length) {
    wrap.append(table(Object.keys(dm[0]).map((k) => ({ key: k, label: k, text: typeof dm[0][k] === "string", fmt: typeof dm[0][k] === "number" ? f.d(4) : f.txt })), dm));
  }
  return wrap;
}

export default async function (s) {
  const root = page("Robustness", "Regimes are defined ex-ante: bull/bear = SPY above/below its 200-day average; high/low vol = VIX above/below 20.");
  root.append(tabs([
    { label: "Bull vs bear", render: () => regimeTab(s, "trend") },
    { label: "Volatility regimes", render: () => regimeTab(s, "vol") },
    { label: "By year", render: () => {
      const acc = s.years.map((y) => s.order.map((m) => {
        let a = 0, n = 0; for (let t = 0; t < s.T; t++) for (let d = 0; d < s.D; d++) if (s.year[d] === y) { a += s.models[m].correct[t * s.D + d]; n++; }
        return a / n;
      }));
      const w = h("div");
      w.append(plot("Accuracy by test year", s.order.map((m, j) => ({ type: "scatter", mode: "markers+lines", name: name(m), x: s.years, y: acc.map((r) => r[j]),
        marker: { size: 9, color: COLOR[m] }, line: { color: COLOR[m], width: 1.8 } })), { yfmt: ".1%" },
      { xaxis: { dtick: 1 }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: "#898781", width: 1 } }] }));
      w.append(table([{ key: "year", label: "test year", text: true }, ...s.order.map((m, j) => ({ key: (r) => r.v[j], label: name(m), fmt: f.d(3) }))],
        s.years.map((y, i) => ({ year: y, v: acc[i] }))));
      return w;
    } },
    { label: "By ticker", render: () => {
      const t = s.table("classification_by_ticker.csv");
      if (!t.length) return null;
      let metric = "accuracy";
      const box = h("div"), cap = h("p", { class: "caption" }), w = h("div");
      const draw = () => {
        const z = s.tickers.map((tk) => s.order.map((m) => (t.find((r) => r.ticker === tk && r.model === m) || {})[metric]));
        box.replaceChildren(plot(`${metric} by ticker`, [{ type: "heatmap", z, x: s.order.map(name), y: s.tickers, colorscale: "Blues",
          text: z.map((r) => r.map((v) => (v == null ? "" : v.toFixed(3)))), texttemplate: "%{text}", hovertemplate: "%{y} / %{x}: %{z:.4f}<extra></extra>" }],
        { height: 400, hover: "closest" }, { yaxis: { autorange: "reversed" } }));
        const counts = {};
        z.forEach((r) => { const cand = r.map((v, j) => [v, j]).filter(([, j]) => s.order[j] !== "AlwaysUp").sort((a, b) => b[0] - a[0])[0]; const k = name(s.order[cand[1]]); counts[k] = (counts[k] || 0) + 1; });
        cap.textContent = "Best non-benchmark model by ticker: " + Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} (${v})`).join(", ");
      };
      w.append(controls(ctl("Metric", select(["accuracy", "f1", "roc_auc", "precision", "recall"], metric, (v) => { metric = v; draw(); }))), box, cap);
      draw();
      return w;
    } },
  ]));
  return root;
}
