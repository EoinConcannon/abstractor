"""Ask follow-up questions about a paper.

Usage:
    python ask.py papers/ca_rag.pdf "What embedding models did they use?"
    python ask.py papers/ca_rag.pdf                 # interactive: keep asking, blank line to quit
    python ask.py papers/ca_rag.pdf "..." --chunks  # show retrieved chunks only (no LLM call)
    python ask.py papers/ca_rag.pdf --find blockchain   # exact word search over the whole text
"""
import argparse
import re
import sys

import pipeline
import rag

ASK_SYSTEM = """You answer questions about ONE research paper using only the excerpts provided.

Rules:
- Use only the excerpts. If they do not contain the answer, say "The retrieved excerpts don't answer this" and suggest what to look for. Do not guess.
- Cite the page(s) for every claim like (p. 5), using the page labels on the excerpts.
- Cite pages only. Never refer to excerpt numbers (say "p. 5", not "Excerpt 3").
- Quote numbers exactly as written. When a number comes from a table or experiment, say which one.
- If excerpts disagree, report both with their context.
- Plain text only: no LaTeX, no backticks.
- Be concise: a short paragraph or a few bullets.
- The text came from a PDF, so tables and equations may be garbled. Say so rather than guess."""

ASK_USER = """Excerpts from the paper:

{excerpts}

Question: {question}"""


def format_excerpts(chunks) -> str:
    return "\n\n".join(
        f"[Excerpt {n} | {c['section']} | {rag.format_pages(c['pages'])}]\n{c['text']}"
        for n, c in enumerate(chunks, 1)
    )


def unsupported_numbers(answer: str, chunks):
    """Numbers in the answer that appear in none of the retrieved excerpts (likely invented)."""
    body = pipeline.CITATION.sub("", answer)
    haystack = " ".join(c["text"] for c in chunks)
    missing = []
    for num in pipeline.NUMBER.findall(body):
        if "." not in num and "%" not in num and len(num) < 2:
            continue
        if not pipeline.page_has_number(haystack, num) and num not in missing:
            missing.append(num)
    return missing


def show_chunks(chunks):
    for n, c in enumerate(chunks, 1):
        print(f"\n[{n}] {c['section']} | {rag.format_pages(c['pages'])} | "
              f"score {c['score']} | found by {', '.join(c['found_by'])}")
        print("    " + c["text"][:300].replace("\n", " ") + ("..." if len(c["text"]) > 300 else ""))


def get_answer(question, index, llm, k=6):
    """Retrieve, ask the LLM, and return everything the caller needs (no printing)."""
    chunks = index.search(question, k=k)
    if not chunks:
        return {"answer": "Nothing in the paper matched that question.", "chunks": [],
                "missing": [], "answered": False}
    answer = llm.complete(ASK_SYSTEM, ASK_USER.format(excerpts=format_excerpts(chunks), question=question)).strip()
    declined = answer.lower().replace("\u2019", "'").startswith("the retrieved excerpts don't answer")
    missing = [] if declined else unsupported_numbers(answer, chunks)
    return {"answer": answer, "chunks": chunks, "missing": missing, "answered": not declined}


def answer_question(question, index, llm, k=6, chunks_only=False):
    if chunks_only:
        show_chunks(index.search(question, k=k))
        return

    result = get_answer(question, index, llm, k=k)
    print("\n" + result["answer"])
    if not result["answered"]:
        return  # no point listing sources for an answer that says there isn't one

    print("\nSources used:")
    for n, c in enumerate(result["chunks"], 1):
        print(f"  [{n}] {c['section']}, {rag.format_pages(c['pages'])}")
    if result["missing"]:
        print(f"\nWarning: these numbers are not in the retrieved excerpts, so check them: "
              f"{', '.join(result['missing'])}")


def find_hits(term: str, pages, width: int = 60, max_per_page: int = 3):
    """Literal, case-insensitive whole-word search over every page. Unlike retrieval, this is exhaustive,
    so it is the right tool for 'does the paper mention X?' and 'how often?' questions.
    Returns (total_count, [(page, snippet), ...])."""
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
    total, shown = 0, []
    for page in pages:
        text = page["text"]
        hits = list(pattern.finditer(text))
        total += len(hits)
        for m in hits[:max_per_page]:
            start, end = max(0, m.start() - width), min(len(text), m.end() + width)
            shown.append((page["page"], f"...{text[start:end]}...".replace("\n", " ")))
        if len(hits) > max_per_page:
            shown.append((page["page"], f"(+{len(hits) - max_per_page} more on this page)"))
    return total, shown


def find_term(term: str, pages, width: int = 60, max_per_page: int = 3):
    total, shown = find_hits(term, pages, width, max_per_page)
    for page, snippet in shown:
        print(f"  p. {page}: {snippet}")
    if total:
        print(f"\n'{term}' appears {total} time(s) in the extracted text.")
    else:
        print(f"'{term}' does not appear in the extracted text (tables, equations and images aren't searched).")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask questions about a research paper PDF.")
    parser.add_argument("pdf")
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("-k", type=int, default=6, help="how many chunks to retrieve (default 6)")
    parser.add_argument("--chunks", action="store_true", help="show retrieved chunks only, no LLM call")
    parser.add_argument("--find", metavar="TERM", help="exact word search over the whole paper, then exit")
    args = parser.parse_args()

    if args.find:
        find_term(args.find, pipeline.get_pages(args.pdf))
        return

    sections = pipeline.get_sections(args.pdf)
    embedder = rag.load_embedder()
    if embedder is None:
        print("Note: fastembed is not installed, so using keyword search (BM25) only.")
    index = rag.build_index(pipeline.paper_dir(args.pdf), sections, embedder)
    print(f"Index ready: {len(index.chunks)} chunks, search mode: {index.mode}")

    llm = None
    if not args.chunks:
        from llm import LLMClient
        llm = LLMClient()

    if args.question:
        answer_question(args.question, index, llm, k=args.k, chunks_only=args.chunks)
        return

    print("Ask a question (blank line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer_question(question, index, llm, k=args.k, chunks_only=args.chunks)


if __name__ == "__main__":
    main()
