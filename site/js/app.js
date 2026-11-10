// Hash router + theme toggle. Each page module default-exports async (store) => HTMLElement.
import { loadStore } from "./data.js";
import { h } from "./ui.js";

const PAGES = [
  { id: "overview", title: "Overview", icon: "🏠", load: () => import("./pages/overview.js") },
  { id: "how-it-works", title: "How it works", icon: "📖", load: () => import("./pages/guide.js") },
  { id: "performance", title: "Performance", icon: "🎯", load: () => import("./pages/performance.js") },
  { id: "backtest", title: "Backtest", icon: "💰", load: () => import("./pages/backtest.js") },
  { id: "statistics", title: "Statistics", icon: "📐", load: () => import("./pages/statistics.js") },
  { id: "robustness", title: "Robustness", icon: "🧪", load: () => import("./pages/robustness.js") },
  { id: "compare", title: "Compare", icon: "⚖️", load: () => import("./pages/compare.js") },
  { id: "data", title: "Data explorer", icon: "🗂️", load: () => import("./pages/explorer.js") },
  { id: "write-up", title: "Full write-up", icon: "📝", load: () => import("./pages/logic.js") },
];

const app = document.getElementById("app"), nav = document.getElementById("nav");
let renderToken = 0;

const current = () => PAGES.find((p) => p.id === location.hash.replace(/^#\//, "")) || PAGES[0];

function buildNav() {
  nav.replaceChildren(...PAGES.map((p) => h("a", { href: `#/${p.id}`, "data-id": p.id }, h("span", {}, p.icon), p.title)));
}

async function route() {
  const page = current(), token = ++renderToken;
  nav.querySelectorAll("a").forEach((a) => a.classList.toggle("active", a.dataset.id === page.id));
  nav.classList.remove("open");
  document.title = `${page.title} | ML vs Classical`;
  app.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  try {
    const [store, mod] = await Promise.all([loadStore(), page.load()]);
    const el = await mod.default(store);
    if (token !== renderToken) return; // a newer navigation superseded this one
    app.replaceChildren(el);
  } catch (err) {
    console.error(err);
    if (token === renderToken) {
      app.replaceChildren(h("div", { class: "callout warn" }, h("b", {}, "Could not load this page. "), String(err.message || err),
        h("p", { class: "muted" }, "If you are running locally, serve the site over HTTP (for example: python -m http.server --directory site) instead of opening the file directly.")));
    }
  }
}

function setTheme(t, rerender = true) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch (e) { /* storage unavailable */ }
  document.getElementById("theme-btn").textContent = t === "dark" ? "☀️" : "🌙";
  if (rerender) route(); // charts read theme colours when they are drawn
}

buildNav();
setTheme(document.documentElement.dataset.theme || "dark", false);
document.getElementById("theme-btn").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
document.getElementById("menu-btn").addEventListener("click", () => nav.classList.toggle("open"));
window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });
route();
