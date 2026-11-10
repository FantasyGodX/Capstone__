// Pure maths (no DOM). Mirrors src/capstone/{stats,backtest,evaluation}.py so the site can
// recompute metrics, backtests and significance tests live in the browser.

export const sum = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
export const mean = (a) => (a.length ? sum(a) / a.length : NaN);
export function std(a) { // sample std (ddof = 1)
  const n = a.length; if (n < 2) return NaN;
  const m = mean(a); let s = 0;
  for (let i = 0; i < n; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / (n - 1));
}
export function quantile(a, q) { // linear interpolation, same as numpy default
  const s = Float64Array.from(a).sort();
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// ---------- special functions ----------
function lgamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (let j = 0; j < 6; j++) s += c[j] / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function betacf(a, b, x) {
  const FPMIN = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN; c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-15) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
export function tCdf(t, df) { // Student-t CDF
  const x = df / (df + t * t), p = 0.5 * betai(df / 2, 0.5, x);
  return t > 0 ? 1 - p : p;
}
export function binomUpperP(k, n, p = 0.5) { // exact one-sided P(X >= k)
  if (n === 0) return NaN; if (k <= 0) return 1;
  let s = 0; const lp = Math.log(p), lq = Math.log(1 - p);
  for (let i = k; i <= n; i++) s += Math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * lp + (n - i) * lq);
  return Math.min(1, s);
}

// ---------- ranks / correlation ----------
export function ranks(a) { // average ranks for ties
  const n = a.length, idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i] - a[j]), r = new Float64Array(n);
  for (let i = 0; i < n;) {
    let j = i; while (j + 1 < n && a[idx[j + 1]] === a[idx[i]]) j++;
    const avg = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k]] = avg; i = j + 1;
  }
  return r;
}
export function pearson(a, b) {
  const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
}
export const spearman = (a, b) => pearson(ranks(a), ranks(b));

// ---------- classification ----------
export function rocAuc(y, s) {
  let pos = 0; for (let i = 0; i < y.length; i++) pos += y[i];
  const neg = y.length - pos; if (!pos || !neg) return 0.5;
  const r = ranks(s); if (r.every((v) => v === r[0])) return 0.5;
  let rp = 0; for (let i = 0; i < y.length; i++) if (y[i] === 1) rp += r[i];
  return (rp - pos * (pos + 1) / 2) / (pos * neg);
}
export function rocCurve(y, s) {
  const n = y.length, idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => s[j] - s[i]);
  let P = 0; for (let i = 0; i < n; i++) P += y[i]; const N = n - P;
  const fpr = [0], tpr = [0]; let tp = 0, fp = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k]; if (y[i] === 1) tp++; else fp++;
    if (k === n - 1 || s[idx[k + 1]] !== s[i]) { fpr.push(N ? fp / N : 0); tpr.push(P ? tp / P : 0); }
  }
  const stride = Math.ceil(fpr.length / 1200); // thin for plotting, keep the end points
  return { fpr: fpr.filter((_, i) => i % stride === 0 || i === fpr.length - 1), tpr: tpr.filter((_, i) => i % stride === 0 || i === tpr.length - 1) };
}
export function classificationMetrics(y, pred, score) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < y.length; i++) {
    if (pred[i] === 1) { if (y[i] === 1) tp++; else fp++; } else if (y[i] === 1) fn++; else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
  return { accuracy: (tp + tn) / y.length, precision, recall,
    f1: precision + recall ? 2 * precision * recall / (precision + recall) : 0,
    roc_auc: rocAuc(y, score), n: y.length, tp, fp, fn, tn };
}

// ---------- backtest ----------
export function strategyReturns(pred, ret, cost) { // long if pred = 1, else cash
  const out = new Float64Array(pred.length);
  for (let t = 0; t < pred.length; t++) {
    const turnover = t === 0 ? pred[0] : Math.abs(pred[t] - pred[t - 1]);
    out[t] = pred[t] * ret[t] - cost * turnover;
  }
  return out;
}
export function performance(r, td = 252) {
  const n = r.length; if (!n) return null;
  let eq = 1, peak = 1, mdd = 0;
  for (let i = 0; i < n; i++) { eq *= 1 + r[i]; peak = Math.max(peak, eq); mdd = Math.min(mdd, eq / peak - 1); }
  const years = n / td, ann = Math.pow(eq, 1 / years) - 1, sd = std(r);
  return { cum_return: eq - 1, ann_return: ann, sharpe: sd > 0 ? mean(r) / sd * Math.sqrt(td) : 0,
    max_drawdown: mdd, calmar: mdd < 0 ? ann / Math.abs(mdd) : NaN };
}
export function sharpe(r, td = 252) { const sd = std(r); return sd > 0 ? mean(r) / sd * Math.sqrt(td) : 0; }

// ---------- significance ----------
export function dieboldMariano(lossBase, lossChal, h = 1) {
  const n = lossBase.length, d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = lossBase[i] - lossChal[i];
  const dbar = mean(d), lags = Math.floor(Math.cbrt(n)), dc = d.map((v) => v - dbar);
  const dot = (k) => { let s = 0; for (let i = k; i < n; i++) s += dc[i] * dc[i - k]; return s; };
  let lrv = dot(0) / n;
  for (let k = 1; k <= lags; k++) lrv += 2 * (1 - k / (lags + 1)) * dot(k) / n;
  if (!(lrv > 0)) return { stat: NaN, p: NaN };
  let dm = dbar / Math.sqrt(lrv / n);
  dm *= Math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n);
  return { stat: dm, p: 1 - tCdf(dm, n - 1) };
}
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// Paired stationary block bootstrap of stat(a) - stat(b) (Politis-Romano), one-sided p for H1: diff > 0.
export function bootstrapDiff(a, b, stat, nBoot, meanBlock, seed, alpha = 0.05) {
  const n = a.length, p = 1 / meanBlock, rng = mulberry32(seed), diffs = new Float64Array(nBoot);
  const ra = new Float64Array(n), rb = new Float64Array(n);
  for (let k = 0; k < nBoot; k++) {
    let idx = Math.floor(rng() * n);
    for (let t = 0; t < n; t++) {
      if (t > 0) idx = rng() < p ? Math.floor(rng() * n) : (idx + 1) % n;
      ra[t] = a[idx]; rb[t] = b[idx];
    }
    diffs[k] = stat(ra) - stat(rb);
  }
  const obs = stat(a) - stat(b), dm = mean(diffs);
  let cnt = 0; for (let k = 0; k < nBoot; k++) if (diffs[k] - dm >= obs) cnt++;
  return { obs, lo: quantile(diffs, alpha / 2), hi: quantile(diffs, 1 - alpha / 2), p: (cnt + 1) / (nBoot + 1) };
}
