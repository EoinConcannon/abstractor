"""Ask follow-up questions about a paper.

Usage:
    python ask.py papers/ca_rag.pdf "What embedding models did they use?"
    python ask.py papers/ca_rag.pdf                 # interactive: keep asking, blank line to quit
    python ask.py papers/ca_rag.pdf "..." --chunks  # show retrieved chunks only (no LLM call)
"""
import argparse
import sys

import pipeline
import rag

ASK_SYSTEM = """You answer questions about ONE research paper using only the excerpts provided.

Rules:
- Use only the excerpts. If they do not contain the answer, say "The retrieved excerpts don't answer this" and suggest what to look for. Do not guess.
- Cite the page(s) for every claim like (p. 5), using the page labels on the excerpts.
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


def answer_question(question, index, llm, k=6, chunks_only=False):
    chunks = index.search(question, k=k)
    if chunks_only:
        show_chunks(chunks)
        return
    if not chunks:
        print("Nothing in the paper matched that question.")
        return

    answer = llm.complete(ASK_SYSTEM, ASK_USER.format(excerpts=format_excerpts(chunks), question=question))
    print("\n" + answer.strip())
    if answer.strip().lower().replace("\u2019", "'").startswith("the retrieved excerpts don't answer"):
        return  # no point listing sources for an answer that says there isn't one

    print("\nSources used:")
    for n, c in enumerate(chunks, 1):
        print(f"  [{n}] {c['section']}, {rag.format_pages(c['pages'])}")

    missing = unsupported_numbers(answer, chunks)
    if missing:
        print(f"\nWarning: these numbers are not in the retrieved excerpts, so check them: {', '.join(missing)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask questions about a research paper PDF.")
    parser.add_argument("pdf")
    parser.add_argument("question", nargs="?", help="leave out for interactive mode")
    parser.add_argument("-k", type=int, default=6, help="how many chunks to retrieve (default 6)")
    parser.add_argument("--chunks", action="store_true", help="show retrieved chunks only, no LLM call")
    args = parser.parse_args()

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
