import datetime as dt

import streamlit as st

from common import ROOT, TABLES, notes_path, table

TEMPLATE = """# Analysis notes

## Research question
Do ML/DL models significantly outperform classical models at predicting next-day return direction?

## Key findings
-

## Interpretation
-

## Limitations
-
"""


def _load() -> str:
    path = notes_path()
    return path.read_text(encoding="utf-8") if path.exists() else TEMPLATE


def _append(text: str):
    st.session_state["notes_text"] = st.session_state["notes_text"].rstrip() + "\n\n" + text + "\n"


def _table_md(fname: str) -> str:
    df = table(fname)
    if df.empty:
        return f"_(table {fname} not found)_"
    df = df.copy()
    df.columns = [("model" if str(c).startswith("Unnamed") else c) for c in df.columns]
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(4)
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return f"**{fname}**\n\n" + "\n".join(rows)


def render():
    st.title("Analysis notes")
    path = notes_path()
    st.caption(f"Write your analysis here. It is saved to `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}` "
               "(commit it with git to keep it in the repo).")
    if "notes_text" not in st.session_state:
        st.session_state["notes_text"] = _load()

    with st.sidebar:
        st.header("Insert")
        files = sorted(p.name for p in TABLES.glob("*.csv")) if TABLES.exists() else []
        tf = st.selectbox("Results table", files) if files else None
        if tf:
            st.button("Insert table", on_click=lambda: _append(_table_md(tf)), width="stretch")
        figs = sorted(p.name for p in (ROOT / "results" / "figures").glob("*.png"))
        fg = st.selectbox("Figure", figs) if figs else None
        if fg:
            st.button("Insert figure link", on_click=lambda: _append(f"![{fg}](../results/figures/{fg})"), width="stretch")
        st.button("Insert timestamp", on_click=lambda: _append(f"_Updated {dt.datetime.now():%Y-%m-%d %H:%M}_"), width="stretch")

    edit, preview = st.tabs(["Edit", "Preview"])
    with edit:
        st.text_area("Markdown", key="notes_text", height=520, label_visibility="collapsed")
    with preview:
        st.markdown(st.session_state["notes_text"])
        for fg_name in sorted(p.name for p in (ROOT / "results" / "figures").glob("*.png")):
            if fg_name in st.session_state["notes_text"]:
                st.image(str(ROOT / "results" / "figures" / fg_name), caption=fg_name)

    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("Save", type="primary"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(st.session_state["notes_text"], encoding="utf-8")
        st.session_state["saved_msg"] = f"Saved at {dt.datetime.now():%H:%M:%S}"
    c2.download_button("Download .md", st.session_state["notes_text"].encode(), "analysis_notes.md")
    if st.session_state.get("saved_msg"):
        c3.success(st.session_state["saved_msg"])
