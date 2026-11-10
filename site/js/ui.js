// Small DOM + Plotly helpers shared by all pages.
import { COLOR, DASH, name } from "./data.js";

export function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
export const md = (src) => h("div", { class: "prose", html: window.marked.parse(src, { gfm: true }) });
export const page = (title, caption) => {
  const root = h("div", {}, h("h1", {}, title), caption ? h("p", { class: "caption" }, caption) : null);
  return root;
};
export const callout = (html, warn = false) => h("div", { class: "callout" + (warn ? " warn" : ""), html });
export const clear = (e) => { e.replaceChildren(); return e; };

// ---------- formatting ----------
export const f = {
  d: (nd) => (v) => (v == null || Number.isNaN(v) ? "-" : Number(v).toFixed(nd)),
  pct: (nd) => (v) => (v == null || Number.isNaN(v) ? "-" : (v * 100).toFixed(nd) + "%"),
  int: (v) => (v == null ? "-" : Math.round(v).toLocaleString("en-US")),
  txt: (v) => (v == null ? "-" : String(v)),
};

// ---------- tables ----------
// cols: [{key, label, fmt, cls(value,row)}]; rows: array of objects.
export function table(cols, rows, { scroll = false, compact = false } = {}) {
  const head = h("tr", {}, cols.map((c) => h("th", { class: c.text ? "txt" : "" }, c.label ?? c.key)));
  const body = rows.map((r) => h("tr", {}, cols.map((c) => {
    const v = typeof c.key === "function" ? c.key(r) : r[c.key];
    return h("td", { class: [c.text ? "txt" : "", c.cls ? c.cls(v, r) : ""].join(" ").trim() }, (c.fmt || f.txt)(v));
  })));
  return h("div", { class: "tablewrap" + (scroll ? " scroll" : "") + (compact ? " compact" : "") }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

// ---------- controls ----------
export const ctl = (label, node, style) => h("div", { class: "ctl", style }, h("span", { class: "lbl" }, label), node);
export const controls = (...items) => h("div", { class: "controls" }, items);

export function chips(options, selected, onChange, label = (x) => x) {
  const sel = new Set(selected), wrap = h("div", { class: "chips" });
  const draw = () => { clear(wrap); options.forEach((o) => wrap.append(h("span", { class: "chip" + (sel.has(o) ? " on" : ""), role: "button", tabindex: 0,
    onclick: () => { sel.has(o) ? sel.delete(o) : sel.add(o); draw(); onChange(options.filter((x) => sel.has(x))); },
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.target.click(); } } }, label(o)))); };
  draw();
  return wrap;
}
export function radio(options, value, onChange, label = (x) => x) {
  const wrap = h("div", { class: "radio" });
  const draw = () => { clear(wrap); options.forEach((o) => wrap.append(h("button", { class: o === value ? "on" : "", type: "button",
    onclick: () => { value = o; draw(); onChange(o); } }, label(o)))); };
  draw();
  return wrap;
}
export function select(options, value, onChange, label = (x) => x) {
  const s = h("select", { onchange: (e) => onChange(options[e.target.selectedIndex]) },
    options.map((o) => h("option", { value: String(o), ...(o === value ? { selected: true } : {}) }, label(o))));
  return s;
}
export function slider(min, max, step, value, fmt, onInput) {
  const out = h("span", { class: "rangeval" }, fmt(value));
  const inp = h("input", { type: "range", min, max, step, value, oninput: (e) => { out.textContent = fmt(+e.target.value); },
    onchange: (e) => onInput(+e.target.value) });
  return h("div", {}, inp, out);
}
export const modelChips = (store, selected, onChange) => chips(store.order, selected, onChange, name);

export function metric(label, value, sub) {
  return h("div", { class: "card metric" }, h("div", { class: "lbl" }, label), h("div", { class: "val" }, value), h("div", { class: "sub" }, sub || " "));
}

export function tabs(items, initial = 0) {
  const bar = h("div", { class: "tabs", role: "tablist" }), body = h("div");
  const show = (i) => {
    [...bar.children].forEach((b, j) => b.classList.toggle("on", i === j));
    clear(body);
    const out = items[i].render();
    if (out) body.append(out);
  };
  items.forEach((it, i) => bar.append(h("button", { type: "button", role: "tab", onclick: () => show(i) }, it.label)));
  show(initial);
  return h("div", {}, bar, body);
}

export function flow(steps) {
  const w = h("div", { class: "flow" });
  steps.forEach((s, i) => { if (i) w.append(h("span", { class: "arrow" }, "→")); w.append(h("span", { class: "step" }, s)); });
  return w;
}

export function download(filename, text) {
  const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: filename });
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
export const csv = (cols, rows) => [cols.join(","), ...rows.map((r) => r.join(","))].join("\n");
export const dlButton = (label, filename, getText) => h("button", { class: "btn", type: "button", onclick: () => download(filename, getText()) }, label);

// ---------- Plotly ----------
const cssv = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
export function baseLayout(title, { height = 430, yfmt, hover = "x unified" } = {}) {
  const ink = cssv("--text"), muted = cssv("--muted"), grid = cssv("--border");
  return {
    title: title ? { text: title, x: 0, xanchor: "left", font: { size: 15, color: ink } } : undefined,
    height, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: muted, size: 12 },
    margin: { l: 55, r: 15, t: title ? 50 : 20, b: 45 }, hovermode: hover,
    legend: { orientation: "h", y: -0.18, x: 0, font: { color: ink } },
    hoverlabel: { bgcolor: cssv("--surface2"), font: { color: ink } },
    xaxis: { gridcolor: grid, linecolor: grid, zeroline: false, automargin: true },
    yaxis: { gridcolor: grid, linecolor: grid, zerolinecolor: muted, tickformat: yfmt, automargin: true },
  };
}
export function plot(title, traces, opts = {}, extra = {}) {
  const el = h("div", { class: "plot" });
  const layout = { ...baseLayout(title, opts), ...extra };
  if (extra.xaxis) layout.xaxis = { ...baseLayout(title, opts).xaxis, ...extra.xaxis };
  if (extra.yaxis) layout.yaxis = { ...baseLayout(title, opts).yaxis, ...extra.yaxis };
  // Wait until the element is attached to the page so Plotly can measure its width.
  let tries = 0;
  const go = () => {
    if (el.isConnected || tries++ > 120) window.Plotly.newPlot(el, traces, layout, { responsive: true, displaylogo: false });
    else requestAnimationFrame(go);
  };
  requestAnimationFrame(go);
  return el;
}
export function replot(el, title, traces, opts = {}, extra = {}) { // refresh an existing plot container in place
  const layout = { ...baseLayout(title, opts), ...extra };
  if (extra.xaxis) layout.xaxis = { ...baseLayout(title, opts).xaxis, ...extra.xaxis };
  if (extra.yaxis) layout.yaxis = { ...baseLayout(title, opts).yaxis, ...extra.yaxis };
  window.Plotly.react(el, traces, layout, { responsive: true, displaylogo: false });
}
export const lineTrace = (x, y, m, extra = {}) => ({ x, y, name: name(m), mode: "lines", type: "scatter",
  line: { color: COLOR[m], dash: DASH[m], width: m === "AlwaysUp" ? 1.6 : 2.2 }, ...extra });
export const refLine = (y, color = "#8a8984") => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 1 } });
export const vLine = (x, color = "#8a8984") => ({ type: "line", yref: "paper", y0: 0, y1: 1, x0: x, x1: x, line: { color, width: 1 } });
export const cumulative = (r) => { let e = 1; return Array.from(r, (v) => (e *= 1 + v)); };
