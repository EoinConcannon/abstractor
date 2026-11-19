"""Streamlit front end for Abstractor.

Run with:
    streamlit run app.py

It is a thin layer over the same functions the command-line tools use:
    pipeline.py (extract -> sections -> summary -> verification), rag.py / ask.py (follow-up questions).
"""
import re
import shutil
from pathlib import Path

import streamlit as st

import ask
import pipeline
import rag
from sections import section_text

PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Abstractor", layout="wide")


# ---------- cached resources (created once, reused on every rerun) ----------

@st.cache_resource(show_spinner=False)
def get_llm():
    from llm import LLMClient
    return LLMClient()


@st.cache_resource(show_spinner=False)
def get_embedder():
    return rag.load_embedder()


@st.cache_resource(show_spinner=False)
def get_index(pdf_path: str):
    sections = pipeline.get_sections(pdf_path)
    return rag.build_index(pipeline.paper_dir(pdf_path), sections, get_embedder(), progress=lambda m: None)


def try_llm():
    try:
        return get_llm(), None
    except Exception as e:  # missing .env values, bad configuration
        return None, str(e)


# ---------- helpers ----------

def safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "paper"
    return f"{stem}.pdf"


def save_upload(uploaded) -> Path:
    path = PAPERS_DIR / safe_filename(uploaded.name)
    data = uploaded.getvalue()
    if path.exists() and path.read_bytes() != data:
        # A different file under the same name: discard everything saved for the old one.
        shutil.rmtree(pipeline.paper_dir(str(path)), ignore_errors=True)
        get_index.clear()
    path.write_bytes(data)
    return path


def page_range(section) -> str:
    nums = [p["page"] for p in section["paragraphs"]]
    return str(nums[0]) if min(nums) == max(nums) else f"{min(nums)}-{max(nums)}"


def show_assistant(msg):
    st.markdown(msg["content"].replace("$", "\\$"))   # stop "$" being read as maths
    if msg.get("missing"):
        st.warning("These numbers are not in the retrieved text, so check them: " + ", ".join(msg["missing"]))
    if msg.get("sources"):
        with st.expander("Sources used"):
            for i, s in enumerate(msg["sources"], 1):
                st.markdown(f"**[{i}] {s['label']}**")
                st.text(s["text"])


# ---------- sidebar ----------

with st.sidebar:
    st.title("Abstractor")
    st.caption("Summarise research papers and ask questions about them, using free tools.")

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None:
        upload_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
        if st.session_state.get("last_upload") != upload_id:
            saved_path = save_upload(uploaded)
            st.session_state["last_upload"] = upload_id
            st.session_state["paper_choice"] = saved_path.name

    pdfs = sorted(p.name for p in PAPERS_DIR.glob("*.pdf"))
    if st.session_state.get("paper_choice") not in pdfs:
        st.session_state.pop("paper_choice", None)
    if not pdfs:
        st.info("Upload a PDF to begin.")
        st.stop()

    choice = st.selectbox("Paper", pdfs, key="paper_choice", format_func=lambda n: Path(n).stem)
    method = st.radio(
        "Summary method",
        ["mapreduce", "single"],
        format_func={"mapreduce": "Section by section (better coverage)",
                     "single": "One prompt (fewer API calls)"}.get,
        help="Section by section makes about 8 API calls. One prompt makes 1 but tends to skim the results.",
    )

    llm, llm_error = try_llm()
    if llm:
        st.caption(f"Model: {llm.model}")
    else:
        st.warning(f"No language model configured. {llm_error}. Copy .env.example to .env and add your key.")


# ---------- load the paper (saved after the first time) ----------

pdf_path = str(PAPERS_DIR / choice)
try:
    with st.spinner("Reading the PDF..."):
        pages = pipeline.get_pages(pdf_path)
        sections = pipeline.get_sections(pdf_path)
except Exception as e:
    st.error(f"Could not read this PDF: {e}")
    st.stop()

total_words = sum(len(section_text(s).split()) for s in sections)

st.header(Path(choice).stem.replace("_", " "))
c1, c2, c3 = st.columns(3)
c1.metric("Pages", len(pages))
c2.metric("Sections found", len(sections))
c3.metric("Words (excluding references)", f"{total_words:,}")
if total_words < 300:
    st.warning("Very little text was found. This may be a scanned PDF, which isn't supported (no OCR).")

tab_sections, tab_summary, tab_ask = st.tabs(["Sections", "Summary", "Ask"])


# ---------- tab 1: sections ----------

with tab_sections:
    st.caption("Check this list against the paper before trusting the summary. "
               "References are cut off automatically.")
    st.dataframe(
        [{"Section": s["title"], "Pages": page_range(s), "Words": len(section_text(s).split())}
         for s in sections],
        hide_index=True,
    )
    for s in sections:
        with st.expander(s["title"]):
            st.text(section_text(s))


# ---------- tab 2: summary + verification ----------

with tab_summary:
    if llm is None:
        st.warning("Add your API key to .env to generate summaries.")
    else:
        saved = pipeline.cached_summary(pdf_path, llm.model, method)

        if saved is None:
            st.write("No summary yet for this paper with these settings.")
            if st.button("Generate summary", type="primary"):
                ok = False
                with st.status("Summarising...", expanded=True) as status:
                    try:
                        pipeline.get_summary(pdf_path, llm, method, progress=status.write)
                        status.update(label="Done", state="complete")
                        ok = True
                    except Exception as e:
                        status.update(label="Failed", state="error")
                        st.error(f"{type(e).__name__}: {e}")
                        st.info("Free APIs sometimes fail. Anything already finished is saved, "
                                "so clicking again resumes where it stopped.")
                if ok:
                    st.rerun()
        else:
            path, text = saved
            st.caption(f"Saved summary: {path.name}")
            st.markdown(text.replace("$", "\\$"))
            st.download_button("Download summary (.md)", text,
                               file_name=f"{Path(choice).stem}_summary.md", mime="text/markdown")

            st.subheader("Verification")
            report = pipeline.verify_summary(text, pages)
            counts = re.search(r"(\d+) OK, (\d+) on a different page than cited, (\d+) missing", report)
            if counts:
                ok_n, elsewhere_n, missing_n = counts.groups()
                v1, v2, v3 = st.columns(3)
                v1.metric("Numbers found on the cited page", ok_n)
                v2.metric("Found on a different page", elsewhere_n)
                v3.metric("Not found in the paper", missing_n)
            st.caption("Each number in the summary is looked up in the paper. This catches invented figures "
                       "and wrong page citations, but it cannot tell whether a number is attached to the "
                       "right claim, so still read the summary against the paper.")
            with st.expander("Details"):
                st.text(report)


# ---------- tab 3: ask ----------

with tab_ask:
    with st.expander("Exact word search (for questions like 'does the paper mention X?')"):
        term = st.text_input("Word or phrase", key="find_term")
        if term.strip():
            total, shown = ask.find_hits(term.strip(), pages)
            if total:
                st.write(f"**{total}** match(es) in the extracted text.")
                for page_no, snippet in shown:
                    st.text(f"p. {page_no}: {snippet}")
            else:
                st.write("Not found in the extracted text (tables, equations and images aren't searched).")

    if llm is None:
        st.warning("Add your API key to .env to ask questions.")
    else:
        if get_embedder() is None:
            st.info("fastembed isn't installed, so search uses keywords only.")
        with st.spinner("Preparing the search index (the first time takes a minute or two)..."):
            index = get_index(pdf_path)
        st.caption(f"Search mode: {index.mode}. Answers only use passages retrieved from this paper.")

        history = st.session_state.setdefault(f"chat::{choice}", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    show_assistant(msg)

        question = st.chat_input("Ask a question about this paper")
        if question:
            user_msg = {"role": "user", "content": question}
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching and answering..."):
                    try:
                        result = ask.get_answer(question, index, llm)
                    except Exception as e:
                        result = None
                        st.error(f"{type(e).__name__}: {e}")
                if result:
                    msg = {
                        "role": "assistant",
                        "content": result["answer"],
                        "missing": result["missing"],
                        "sources": [
                            {"label": f"{c['section']}, {rag.format_pages(c['pages'])}", "text": c["text"]}
                            for c in result["chunks"]
                        ] if result["answered"] else [],
                    }
                    show_assistant(msg)
                    history.extend([user_msg, msg])
