import { loadFeatures, loadPrices } from "../data.js";
import { mean, pearson, quantile } from "../stats.js";
import { callout, chips, controls, csv, ctl, dlButton, f, h, page, plot, select, slider, table, tabs } from "../ui.js";

const rolling = (a, w, fn) => a.map((_, i) => (i < w - 1 ? null : fn(a.slice(i - w + 1, i + 1))));
const sd = (x) => { const m = mean(x); return Math.sqrt(x.reduce((p, v) => p + (v - m) ** 2, 0) / (x.length - 1)); };

function pricesTab(s, px) {
  const keys = [...s.tickers, s.meta.market, s.meta.vix];
  const st = { t: keys[0], a: 0, b: null, over: ["MA 50"] };
  const box = h("div"), wrap = h("div");
  const draw = () => {
    const d = px[st.t], n = d.d.length, b = st.b == null ? n - 1 : st.b, a = Math.min(st.a, b);
    const sl = (arr) => arr.slice(a, b + 1), x = sl(d.d), traces = [{ type: "candlestick", x, open: sl(d.o), high: sl(d.h), low: sl(d.l), close: sl(d.c), name: st.t,
      increasing: { line: { color: "#1baf7a" } }, decreasing: { line: { color: "#e34948" } } }];
    for (const o of st.over) {
      if (o.startsWith("MA")) traces.push({ type: "scatter", mode: "lines", x, y: sl(rolling(d.c, +o.split(" ")[1], mean)), name: o, line: { width: 1.6 } });
      else {
        const m = rolling(d.c, 20, mean), v = rolling(d.c, 20, sd);
        [["BB upper", 2], ["BB lower", -2]].forEach(([nm, k]) => traces.push({ type: "scatter", mode: "lines", x, y: sl(m.map((mm, i) => (mm == null ? null : mm + k * v[i]))),
          name: nm, line: { width: 1, dash: "dot", color: "#898781" } }));
      }
    }
    traces.push({ type: "bar", x, y: sl(d.v), name: "Volume", marker: { color: "#898781" }, yaxis: "y2", showlegend: false });
    box.replaceChildren(plot(`${st.t} (adjusted prices)`, traces, { height: 560 },
      { xaxis: { rangeslider: { visible: false } }, yaxis: { domain: [0.27, 1] }, yaxis2: { domain: [0, 0.22], gridcolor: "rgba(128,128,128,.2)" } }));
  };
  // Rebuilt whenever the ticker changes so the date sliders match that series.
  const build = () => {
    const n = px[st.t].d.length, lbl = (i) => px[st.t].d[Math.min(i, n - 1)];
    wrap.replaceChildren(controls(
      ctl("Ticker", select(keys, st.t, (v) => { st.t = v; st.a = 0; st.b = null; build(); })),
      ctl("Overlays", chips(["MA 20", "MA 50", "MA 200", "Bollinger (20, 2)"], st.over, (v) => { st.over = v; draw(); })),
      ctl("From", slider(0, n - 1, 1, st.a, lbl, (v) => { st.a = v; draw(); })),
      ctl("To", slider(0, n - 1, 1, st.b == null ? n - 1 : st.b, lbl, (v) => { st.b = v; draw(); }))), box);
    draw();
  };
  build();
  return wrap;
}

function featuresTab(s, fe) {
  const feats = fe.columns.filter((c) => c !== "ret_next" && c !== "y"), yi = fe.columns.indexOf("y");
  const st = { tk: [...s.tickers], feat: feats[0] };
  const histBox = h("div"), corrBox = h("div"), wrap = h("div");
  const rowsFor = () => st.tk.flatMap((t) => fe.tickers[t].v);
  const draw = () => {
    const rows = rowsFor();
    if (!rows.length) { histBox.replaceChildren(callout("Pick at least one ticker.", true)); corrBox.replaceChildren(); return; }
    const ci = fe.columns.indexOf(st.feat), traces = [];
    for (const [lab, col, nm] of [[0, "#e34948", "Next day down"], [1, "#1baf7a", "Next day up"]]) {
      const x = rows.filter((r) => r[yi] === lab).map((r) => r[ci]), lo = quantile(x, 0.005), hi = quantile(x, 0.995);
      traces.push({ type: "histogram", x: x.map((v) => Math.min(hi, Math.max(lo, v))), name: nm, opacity: 0.6, marker: { color: col }, histnorm: "probability density", nbinsx: 60 });
    }
    histBox.replaceChildren(plot(`${st.feat}: distribution by next-day direction (clipped at 0.5/99.5 pct)`, traces, { height: 360, hover: "closest" }, { barmode: "overlay" }));
    const y = rows.map((r) => r[yi]);
    const corr = feats.map((c) => [c, pearson(rows.map((r) => r[fe.columns.indexOf(c)]), y)]).sort((p, q) => p[1] - q[1]);
    corrBox.replaceChildren(plot("Correlation of each feature with next-day direction", [{ type: "bar", orientation: "h", x: corr.map((c) => c[1]), y: corr.map((c) => c[0]),
      marker: { color: "#4a95f0" }, hovertemplate: "%{y}: %{x:.4f}<extra></extra>" }], { height: 620, hover: "closest" }), h("p", { class: "caption" },
    "Correlations are tiny (|r| < 0.05 is typical): this is why next-day direction is hard to predict."));
  };
  wrap.append(controls(ctl("Tickers", chips(s.tickers, st.tk, (v) => { st.tk = v; draw(); })), ctl("Feature", select(feats, st.feat, (v) => { st.feat = v; draw(); }))), histBox, corrBox);
  draw();
  return wrap;
}

function tableTab(s, fe) {
  let tk = s.tickers[0], shown = 250;
  const box = h("div"), wrap = h("div");
  const draw = () => {
    const d = fe.tickers[tk], rows = d.d.map((dt, i) => ({ date: dt, ...Object.fromEntries(fe.columns.map((c, j) => [c, d.v[i][j]])) }));
    const upShare = d.v.reduce((a, r) => a + r[fe.columns.indexOf("y")], 0) / d.v.length;
    box.replaceChildren(
      table([{ key: "date", label: "date", text: true }, ...fe.columns.map((c) => ({ key: c, label: c, fmt: c === "y" ? f.int : f.d(5) }))], rows.slice(0, shown), { scroll: true }),
      h("p", { class: "caption" }, `Showing ${Math.min(shown, rows.length).toLocaleString("en-US")} of ${rows.length.toLocaleString("en-US")} rows, ${fe.columns.length - 2} features. Up-day share: ${(upShare * 100).toFixed(1)}%.`),
      h("p", {}, shown < rows.length ? h("button", { class: "btn", type: "button", onclick: () => { shown += 500; draw(); } }, "Show more rows") : null, " ",
        dlButton("Download this ticker's dataset (CSV)", `${tk}_dataset.csv`, () => csv(["date", ...fe.columns], rows.map((r) => [r.date, ...fe.columns.map((c) => r[c])])))));
  };
  wrap.append(controls(ctl("Ticker", select(s.tickers, tk, (v) => { tk = v; shown = 250; draw(); }))), box);
  draw();
  return wrap;
}

export default async function (s) {
  const root = page("Data explorer");
  root.append(h("p", { class: "muted" }, "Loading data…"));
  const [px, fe] = await Promise.all([loadPrices(), loadFeatures()]);
  root.querySelector("p.muted").remove();
  root.append(tabs([
    { label: "Prices", render: () => pricesTab(s, px) },
    { label: "Features", render: () => featuresTab(s, fe) },
    { label: "Dataset table", render: () => tableTab(s, fe) },
  ]));
  return root;
}
