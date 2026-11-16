// The research paper: rendered from site/paper/paper.md (written by scripts/12_build_paper.py), with a PDF download.
import { h, md, page } from "../ui.js";

const slug = (t) => t.toLowerCase().trim().replace(/[^\w\s-]/g, "").replace(/\s/g, "-");

export default async function () {
  const root = page("Research paper", "The full write-up: methods, results, analysis, limitations, future work and references.");
  const res = await fetch("paper/paper.md");
  if (!res.ok) throw new Error(`Could not load paper/paper.md (${res.status})`);
  const body = md(await res.text());
  body.classList.add("paper");
  const used = new Set();
  body.querySelectorAll("h2,h3").forEach((el) => {
    let id = slug(el.textContent), i = 1;
    while (used.has(id)) id = `${slug(el.textContent)}-${i++}`;
    used.add(id); el.id = id;
  });
  root.append(h("p", {}, h("a", { class: "btn", href: "paper/paper.pdf", download: "paper.pdf", style: "text-decoration:none;display:inline-block" }, "Download the PDF")), body);
  return root;
}
