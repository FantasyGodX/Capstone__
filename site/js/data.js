// Data loading + derived structures. `buildStore(raw)` is pure so it can be tested in Node.
import { classificationMetrics, strategyReturns, mean } from "./stats.js";

export const LABEL = { AlwaysUp: "Buy & hold" };
export const name = (m) => LABEL[m] || m;
export const CLASS = { AlwaysUp: "Benchmark", Logistic: "Classical", ARIMA: "Classical", RandomForest: "ML", XGBoost: "ML", LSTM: "Deep learning" };
export const COLOR = { AlwaysUp: "#898781", Logistic: "#4a95f0", ARIMA: "#eb6834", RandomForest: "#1baf7a", XGBoost: "#e87ba4", LSTM: "#8b7bff" };
export const DASH = { AlwaysUp: "solid", Logistic: "solid", ARIMA: "dash", RandomForest: "dashdot", XGBoost: "dot", LSTM: "solid" };

export function buildStore(meta, pred, tables) {
  const T = pred.tickers.length, D = pred.dates.length, N = T * D;
  const s = { meta, tickers: pred.tickers, dates: pred.dates, year: pred.year, years: [...new Set(pred.year)].sort(),
    T, D, N, order: meta.order, tables, models: {}, y: Int8Array.from(pred.y), ret: Float64Array.from(pred.ret), _bt: new Map() };
  for (const m of meta.order) {
    const score = Float64Array.from(pred.models[m].score), p = Int8Array.from(pred.models[m].pred);
    const correct = new Uint8Array(N); for (let i = 0; i < N; i++) correct[i] = p[i] === s.y[i] ? 1 : 0;
    s.models[m] = { score, pred: p, correct };
  }
  // table helper: records from a loaded CSV
  s.table = (file) => {
    const t = tables[file]; if (!t) return [];
    return t.rows.map((r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i]])));
  };
  // Row indices for a ticker/year selection (rows are laid out as ticker * D + date).
  s.select = (tickerIdx, yearSet) => {
    const out = [];
    for (const t of tickerIdx) for (let d = 0; d < D; d++) if (!yearSet || yearSet.has(s.year[d])) out.push(t * D + d);
    return out;
  };
  s.metrics = (model, rows) => {
    const m = s.models[model], y = new Array(rows.length), p = new Array(rows.length), sc = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) { const r = rows[i]; y[i] = s.y[r]; p[i] = m.pred[r]; sc[i] = m.score[r]; }
    return { ...classificationMetrics(y, p, sc), y, score: sc };
  };
  // Per-ticker daily strategy returns for every model at a given cost: Map(model -> Float64Array[T][D]).
  s.strategy = (cost) => {
    const key = cost.toFixed(8);
    if (!s._bt.has(key)) {
      const out = {};
      for (const m of s.order) {
        out[m] = [];
        for (let t = 0; t < T; t++) {
          out[m].push(strategyReturns(s.models[m].pred.subarray(t * D, (t + 1) * D), s.ret.subarray(t * D, (t + 1) * D), cost));
        }
      }
      s._bt.set(key, out);
    }
    return s._bt.get(key);
  };
  // Equal-weight portfolio daily returns over chosen tickers, per model.
  s.portfolio = (cost, tickerIdx) => {
    const st = s.strategy(cost), out = {};
    for (const m of s.order) {
      const r = new Float64Array(D);
      for (const t of tickerIdx) for (let d = 0; d < D; d++) r[d] += st[m][t][d];
      for (let d = 0; d < D; d++) r[d] /= tickerIdx.length;
      out[m] = r;
    }
    return out;
  };
  return s;
}

const cache = {};
const get = (url, type = "json") => (cache[url] ||= fetch(url).then((r) => {
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
  return r[type]();
}));

export async function loadStore() {
  if (!cache.store) {
    cache.store = Promise.all([get("data/meta.json"), get("data/predictions.json"), get("data/tables.json")])
      .then(([m, p, t]) => buildStore(m, p, t));
  }
  return cache.store;
}
export const loadPrices = () => get("data/prices.json");
export const loadFeatures = () => get("data/features.json");
export const loadLogic = () => get("data/logic.md", "text");
export { mean };
