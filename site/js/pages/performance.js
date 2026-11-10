import { name, COLOR } from "../data.js";
import { rocCurve } from "../stats.js";
import { chips, controls, ctl, f, h, lineTrace, modelChips, page, plot, radio, select, slider, table } from "../ui.js";

export default async function (s) {
  const root = page("Predictive performance");
  const st = { models: [...s.order], tickers: [...s.tickers], years: [...s.years], metric: "accuracy", win: 60, cm: null };
  const out = h("div");
  root.append(controls(
    ctl("Models", modelChips(s, st.models, (v) => { st.models = v; draw(); })),
    ctl("Tickers", chips(s.tickers, st.tickers, (v) => { st.tickers = v; draw(); })),
    ctl("Test years", chips(s.years, st.years, (v) => { st.years = v; draw(); }, String))), out);

  function draw() {
    out.replaceChildren();
    const ms = s.order.filter((m) => st.models.includes(m));
    const tix = st.tickers.map((t) => s.tickers.indexOf(t)), ys = new Set(st.years);
    const rows = s.select(tix, ys);
    if (!rows.length || !ms.length) { out.append(h("p", { class: "callout warn" }, "Nothing selected.")); return; }
    const M = Object.fromEntries(ms.map((m) => [m, s.metrics(m, rows)]));

    out.append(h("h3", {}, "Metrics for the current selection"),
      table([{ key: (m) => name(m), label: "model", text: true },
        ...["accuracy", "precision", "recall", "f1", "roc_auc"].map((k) => ({ key: (m) => M[m][k], label: k, fmt: f.d(3) })),
        { key: (m) => M[m].n, label: "n", fmt: f.int }], ms));

    const dot = h("div");
    const drawDot = () => {
      dot.replaceChildren(plot(`${st.metric} by model`, [{ type: "scatter", mode: "markers", x: ms.map((m) => M[m][st.metric]), y: ms.map(name),
        marker: { size: 14, color: ms.map((m) => COLOR[m]), line: { color: "white", width: 2 } }, hovertemplate: "%{y}: %{x:.4f}<extra></extra>" }],
      { height: 320, hover: "closest" },
      { shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: 0.5, x1: 0.5, line: { color: "#8a8984", width: 1 } }], yaxis: { autorange: "reversed" } }));
    };
    out.append(h("div", { class: "controls" }, ctl("Compare on", radio(["accuracy", "precision", "recall", "f1", "roc_auc"], st.metric, (v) => { st.metric = v; drawDot(); }))), dot);
    drawDot();

    // ROC + rolling accuracy
    const roc = ms.filter((m) => m !== "AlwaysUp").map((m) => {
      const c = rocCurve(M[m].y, M[m].score);
      return { x: c.fpr, y: c.tpr, name: name(m), mode: "lines", type: "scatter", line: { color: COLOR[m], width: 2 } };
    });
    roc.push({ x: [0, 1], y: [0, 1], name: "Chance", mode: "lines", type: "scatter", line: { color: "#898781", dash: "dot", width: 1 } });
    const rollBox = h("div");
    const dates = [...Array(s.D).keys()].filter((d) => ys.has(s.year[d]));
    const daily = Object.fromEntries(ms.map((m) => [m, dates.map((d) => {
      let a = 0; for (const t of tix) a += s.models[m].correct[t * s.D + d]; return a / tix.length;
    })]));
    const drawRoll = () => {
      const W = st.win;
      const roll = (a) => {
        const o = new Array(a.length).fill(null); let sum = 0;
        a.forEach((v, i) => { sum += v; if (i >= W) sum -= a[i - W]; if (i >= W - 1) o[i] = sum / W; });
        return o;
      };
      rollBox.replaceChildren(plot(`Rolling ${W}-day accuracy`, ms.map((m) => lineTrace(dates.map((d) => s.dates[d]), roll(daily[m]), m)), { yfmt: ".0%" },
        { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: { color: "#898781", width: 1 } }] }));
    };
    out.append(h("div", { class: "grid c2" },
      plot("ROC curves (pooled)", roc, { hover: "closest" }, { xaxis: { title: "False positive rate" }, yaxis: { title: "True positive rate" } }),
      h("div", {}, h("div", { class: "controls" }, ctl("Rolling window (trading days)", slider(20, 250, 10, st.win, (v) => v, (v) => { st.win = v; drawRoll(); }))), rollBox)));
    drawRoll();

    // heatmap + confusion matrix
    const acc = st.tickers.map((t) => ms.map((m) => {
      const ti = s.tickers.indexOf(t); let a = 0, n = 0;
      for (let d = 0; d < s.D; d++) if (ys.has(s.year[d])) { a += s.models[m].correct[ti * s.D + d]; n++; }
      return a / n;
    }));
    const cmBox = h("div");
    st.cm = ms.includes(st.cm) ? st.cm : ms[0];
    const drawCm = () => {
      const c = M[st.cm], z = [[c.tn, c.fp], [c.fn, c.tp]];
      cmBox.replaceChildren(plot(`Confusion matrix: ${name(st.cm)}`, [{ type: "heatmap", z, x: ["Pred down", "Pred up"], y: ["Actual down", "Actual up"],
        colorscale: "Blues", showscale: false, text: z.map((r) => r.map((v) => v.toLocaleString("en-US"))), texttemplate: "%{text}" }],
      { height: 380, hover: "closest" }, { yaxis: { autorange: "reversed" } }));
    };
    out.append(h("div", { class: "grid c2" },
      plot("Accuracy by ticker and model", [{ type: "heatmap", z: acc, x: ms.map(name), y: st.tickers, colorscale: "Blues", zmid: 0.5,
        text: acc.map((r) => r.map((v) => v.toFixed(3))), texttemplate: "%{text}", hovertemplate: "%{y} / %{x}: %{z:.3f}<extra></extra>" }],
      { height: 380, hover: "closest" }, { yaxis: { autorange: "reversed" } }),
      h("div", {}, h("div", { class: "controls" }, ctl("Confusion matrix for", select(ms, st.cm, (v) => { st.cm = v; drawCm(); }, name))), cmBox)));
    drawCm();
  }
  draw();
  return root;
}
