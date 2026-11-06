"""Score retrieval quality against questions whose answer pages you already know.

A retrieved chunk counts as a HIT when it sits on one of the question's answer pages
AND contains all of the question's "terms" (so a chunk that merely overlaps the right
page, without the actual fact, does not count).

Usage:
    python eval.py                      # run eval/questions.json
    python eval.py --max-words 120      # try a different chunk size
Results are also saved to eval/results_<time>.json so you can compare runs.

This measures RETRIEVAL only (did the right passage come back?), not answer quality.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pipeline
import rag

QUESTIONS_FILE = Path("eval/questions.json")
DEPTH = 10  # how far down the ranking to look


def is_hit(chunk, gold_pages, terms) -> bool:
    on_page = bool(set(chunk["pages"]) & set(gold_pages))
    text = chunk["text"].lower()
    return on_page and all(t.lower() in text for t in terms)


def first_hit_rank(chunks, gold_pages, terms):
    for rank, chunk in enumerate(chunks, 1):
        if is_hit(chunk, gold_pages, terms):
            return rank
    return None


def summarise(ranks):
    n = len(ranks)

    def hit_at(k):
        return sum(1 for r in ranks if r is not None and r <= k) / n

    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@6": hit_at(6),
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate retrieval on eval/questions.json")
    parser.add_argument("--max-words", type=int, default=200, help="chunk size in words (default 200)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    embedder = rag.load_embedder()
    modes = ["bm25", "vector", "hybrid"] if embedder else ["bm25"]
    if not embedder:
        print("Note: fastembed is not installed, so only BM25 is evaluated.")

    indexes, results = {}, []
    for q in questions:
        paper = q["paper"]
        if paper not in indexes:
            pdf = Path("papers") / f"{paper}.pdf"
            if not pdf.exists():
                print(f"Skipping '{paper}': {pdf} not found.")
                indexes[paper] = None
                continue
            sections = pipeline.get_sections(str(pdf))
            indexes[paper] = rag.build_index(pipeline.paper_dir(str(pdf)), sections, embedder, args.max_words)
        index = indexes[paper]
        if index is None:
            continue

        ranks = {m: first_hit_rank(index.search(q["question"], k=DEPTH, mode=m), q["pages"], q["terms"])
                 for m in modes}
        results.append({**q, "ranks": ranks})

    if not results:
        print("No questions could be run.")
        return

    cols = "".join(f"{m:>8}" for m in modes)
    print(f"\n{'#':>3}{cols}   question  (rank of first correct chunk; - = not in top {DEPTH})")
    for n, r in enumerate(results, 1):
        cells = "".join(f"{(r['ranks'][m] or '-')!s:>8}" for m in modes)
        print(f"{n:>3}{cells}   {r['paper'][:6]}: {r['question'][:62]}")

    summary = {m: summarise([r["ranks"][m] for r in results]) for m in modes}
    print(f"\n{len(results)} questions, chunk size {args.max_words} words")
    print(f"{'mode':<8}{'hit@1':>8}{'hit@3':>8}{'hit@6':>8}{'MRR':>8}")
    for m in modes:
        s = summary[m]
        print(f"{m:<8}{s['hit@1']:>8.0%}{s['hit@3']:>8.0%}{s['hit@6']:>8.0%}{s['mrr']:>8.2f}")

    out = Path("eval") / f"results_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"max_words": args.max_words, "summary": summary, "questions": results},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
