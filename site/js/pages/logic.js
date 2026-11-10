// Renders the long-form project write-up (Logic.md) with working in-page links.
import { loadLogic } from "../data.js";
import { h, md, page } from "../ui.js";

const slug = (t) => t.toLowerCase().trim().replace(/[^\w\s-]/g, "").replace(/\s/g, "-");

export default async function () {
  const root = page("Full write-up", "The complete written explanation of the project: design decisions, code walk-through and results.");
  const body = md(await loadLogic());
  const used = new Set();
  body.querySelectorAll("h1,h2,h3,h4").forEach((el) => {
    let id = slug(el.textContent), i = 1;
    while (used.has(id)) id = `${slug(el.textContent)}-${i++}`;
    used.add(id); el.id = id;
  });
  // Table-of-contents links like "#1-introduction" must scroll, not change the page route.
  body.addEventListener("click", (e) => {
    const a = e.target.closest("a[href^='#']");
    if (!a || a.getAttribute("href").startsWith("#/")) return;
    e.preventDefault();
    const t = body.querySelector("#" + CSS.escape(a.getAttribute("href").slice(1)));
    if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  root.append(body);
  return h("div", {}, root);
}
