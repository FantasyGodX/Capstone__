import { name, COLOR } from "../data.js";
import { performance } from "../stats.js";
import { chips, controls, csv, ctl, cumulative, dlButton, f, h, lineTrace, modelChips, page, plot, select, slider, table } from "../ui.js";

const PM = ["cum_return", "ann_return", "sharpe", "max_drawdown", "calmar"];
const FMT = { cum_return: f.pct(0), ann_return: f.pct(1), sharpe: f.d(2), max_drawdown: f.pct(1), calmar: f.d(2) };

export default async function (s) {
  const root = page("Backtest", "Long if the model predicts up, cash otherwise. No leverage. Cost is charged per unit of position change.");
  const td = s.meta.backtest.trading_days;
  const st = { cost: s.meta.backtest.transaction_cost * 1e4, tickers: [...s.tickers], models: [...s.order], start: 0, end: s.D - 1, metric: "sharpe" };
  const out = h("div");
  const dateLbl = (i) => s.dates[i];
  root.append(controls(
    ctl("Transaction cost (bps per trade)", slider(0, 50, 0.5, st.cost, (v) => v.toFixed(1), (v) => { st.cost = v; draw(); })),
    ctl("Models", modelChips(s, st.models, (v) => { st.models = v; draw(); })),
    ctl("Tickers in portfolio", chips(s.tickers, st.tickers, (v) => { st.tickers = v; draw(); })),
    ctl("Start date", slider(0, s.D - 1, 1, 0, dateLbl, (v) => { st.start = Math.min(v, st.end); draw(); })),
    ctl("End date", slider(0, s.D - 1, 1, s.D - 1, dateLbl, (v) => { st.end = Math.max(v, st.start); draw(); }))), out);

  function draw() {
    out.replaceChildren();
    const ms = s.order.filter((m) => st.models.includes(m)), tix = st.tickers.map((t) => s.tickers.indexOf(t));
    if (!tix.length || !ms.length) { out.append(h("p", { class: "callout warn" }, "Pick at least one ticker and one model.")); return; }
    const cost = st.cost / 1e4, port = s.portfolio(cost, tix), a = st.start, b = st.end + 1;
    const seg = Object.fromEntries(ms.map((m) => [m, port[m].slice(a, b)]));
    const dates = s.dates.slice(a, b), perf = Object.fromEntries(ms.map((m) => [m, performance(seg[m], td)]));

    out.append(h("h3", {}, "Portfolio metrics"),
      table([{ key: (m) => name(m), label: "model", text: true }, ...PM.map((k) => ({ key: (m) => perf[m][k], label: k, fmt: FMT[k] }))], ms));
    const eq = Object.fromEntries(ms.map((m) => [m, cumulative(seg[m])]));
    out.append(h("div", { class: "grid c2" },
      plot("Cumulative return", ms.map((m) => lineTrace(dates, eq[m].map((v) => v - 1), m)), { yfmt: ".0%" }),
      plot("Drawdown", ms.map((m) => { let pk = 0; return lineTrace(dates, eq[m].map((v) => { pk = Math.max(pk, v); return v / pk - 1; }), m); }), { yfmt: ".0%" })));

    // per-ticker results over the full test period (selected tickers, selected models)
    const st2 = s.strategy(cost);
    const pt = tix.map((t) => ms.map((m) => performance(st2[m][t], td)));
    const box = h("div");
    const drawHeat = () => {
      const k = st.metric, z = pt.map((r) => r.map((p) => p[k]));
      box.replaceChildren(plot(`${k} by ticker (full test period)`, [{ type: "heatmap", z, x: ms.map(name), y: st.tickers,
        colorscale: "RdBu", ...(k !== "calmar" ? { zmid: 0 } : {}), text: z.map((r) => r.map((v) => (Number.isFinite(v) ? v.toFixed(2) : ""))),
        texttemplate: "%{text}", hovertemplate: "%{y} / %{x}: %{z:.3f}<extra></extra>" }], { height: 380, hover: "closest" }, { yaxis: { autorange: "reversed" } }));
    };
    out.append(h("h3", {}, "Per-ticker results"),
      h("div", { class: "controls" }, ctl("Metric", select(["sharpe", "cum_return", "max_drawdown", "calmar", "ann_return"], st.metric, (v) => { st.metric = v; drawHeat(); }))), box);
    drawHeat();

    const rowsFull = [];
    ms.forEach((m, j) => tix.forEach((t, i) => rowsFull.push({ model: name(m), ticker: s.tickers[t], ...pt[i][j] })));
    out.append(h("details", { class: "card" }, h("summary", {}, "Full per-ticker table"),
      table([{ key: "model", label: "model", text: true }, { key: "ticker", label: "ticker", text: true }, ...PM.map((k) => ({ key: k, label: k, fmt: f.d(4) }))], rowsFull, { scroll: true })));
    out.append(h("p", {}, dlButton("Download portfolio daily returns (CSV)", "portfolio_daily_returns.csv",
      () => csv(["date", ...ms.map(name)], dates.map((d, i) => [d, ...ms.map((m) => seg[m][i].toFixed(8))])))));
  }
  draw();
  return root;
}
