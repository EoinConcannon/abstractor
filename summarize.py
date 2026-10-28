"""Step 5: summarise a research paper into structured, page-cited bullet points.

Usage:
    python summarize.py papers/ca_rag.pdf
Writes summaries/<name>.md
"""
import argparse
import sys
from pathlib import Path

from extract import extract_pages
from llm import LLMClient
from sections import split_sections

SYSTEM_PROMPT = """You are a careful research assistant who summarises academic papers.

Rules:
- Use ONLY the paper text provided. If something is not stated, write "Not stated in the paper".
- Every bullet must end with the page(s) it comes from, like (p. 5) or (pp. 9-10).
- Cite only the one or two pages where the specific claim appears, not every page that mentions the topic.
- Quote numbers and results exactly as the paper gives them. Do not round or invent figures.
- When a number comes from a table or experiment, say which one it belongs to.
- If two numbers for the same metric differ, report both with their context rather than choosing one.
- Write symbols and numbers as plain text (theta = 0.3, not LaTeX). Do not use backticks.
- Be concise: short bullets, no filler.
- The text was extracted from a PDF, so tables and equations may be garbled. If a result
  depends on a garbled table, say so rather than guessing."""

USER_TEMPLATE = """Summarise the paper below in exactly this Markdown structure:

## Problem
(2-3 bullets: what are the authors trying to solve, and why does it matter?)

## Approach
(3-5 bullets: what did they do, and what is new compared with prior work?)

## Key results
(3-6 bullets: the main findings with specific numbers where given)

## Limitations
(2-4 bullets: weaknesses or untested cases the authors mention)

## Takeaway
(1-2 sentences)

Page numbers appear as [p.N] at the start of each paragraph.

--- PAPER START ---
{paper}
--- PAPER END ---"""


def build_paper_text(sections) -> str:
    """Flatten sections into one string, tagging each paragraph with its page."""
    parts = []
    for s in sections:
        parts.append(f"### {s['title']}")
        parts.extend(f"[p.{p['page']}] {p['text']}" for p in s["paragraphs"])
    return "\n\n".join(parts)


def summarize_paper(pdf_path: str, llm: LLMClient) -> str:
    sections = split_sections(extract_pages(pdf_path))
    paper = build_paper_text(sections)
    print(f"Sending ~{len(paper.split())} words to {llm.model}...")
    return llm.complete(SYSTEM_PROMPT, USER_TEMPLATE.format(paper=paper))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # avoid Windows console encoding errors
    parser = argparse.ArgumentParser(description="Summarise a research paper PDF.")
    parser.add_argument("pdf", help="path to the PDF")
    args = parser.parse_args()

    summary = summarize_paper(args.pdf, LLMClient())

    out_dir = Path("summaries")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(args.pdf).stem}.md"
    out_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\n" + summary)
