"""Step 3: extract clean, page-numbered text from a (possibly two-column) research paper.

Usage:
    python extract.py papers/my_paper.pdf
    python extract.py papers/my_paper.pdf --debug     # also shows what was filtered out
"""
import argparse
import re
from collections import Counter

import pymupdf  # installed as "pymupdf"; older versions import as "fitz"

MARGIN_FRACTION = 0.08   # top/bottom 8% of the page counts as header/footer zone
REPEAT_THRESHOLD = 0.4   # a margin line on >=40% of pages is treated as a running header/footer


def normalise(text: str) -> str:
    """Make 'IEEE Access, vol. 12, p. 4' and '... p. 5' compare equal."""
    return re.sub(r"\d+", "#", " ".join(text.split())).strip().lower()


def clean_text(text: str) -> str:
    """Join hyphenated line breaks and collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "meth-\nod" -> "method"
    return " ".join(text.split())


def classify_line(x0, x1, page_width) -> str:
    """Is a line in the left column, right column, or spanning the full width?"""
    mid, tol = page_width / 2, page_width * 0.05
    if x0 < mid - tol and x1 > mid + tol:
        return "full"
    return "left" if (x0 + x1) / 2 < mid else "right"


def get_text_blocks(page):
    """Return text blocks with a column label ("left" / "right" / "full").

    PyMuPDF sometimes merges both columns into one block, so we work line by
    line and split blocks wherever the column changes.
    """
    width = page.rect.width
    blocks = []
    for raw in page.get_text("dict")["blocks"]:
        if raw["type"] != 0:
            continue
        current = None
        for line in raw["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            col = classify_line(x0, x1, width)
            if current and current["col"] == col:
                current["text"] += "\n" + text
                current["x0"], current["x1"] = min(current["x0"], x0), max(current["x1"], x1)
                current["y1"] = y1
            else:
                current = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text, "col": col}
                blocks.append(current)
    return blocks


def in_margin_zone(block, page_height) -> bool:
    return block["y1"] < page_height * MARGIN_FRACTION or block["y0"] > page_height * (1 - MARGIN_FRACTION)


def find_repeated_margin_text(doc) -> set:
    """Find text that appears in the header/footer zone on many pages."""
    counts = Counter()
    for page in doc:
        seen = set()
        for b in get_text_blocks(page):
            if in_margin_zone(b, page.rect.height):
                seen.add(normalise(b["text"]))
        counts.update(seen)  # count each string once per page
    min_pages = max(2, int(len(doc) * REPEAT_THRESHOLD))
    return {text for text, n in counts.items() if n >= min_pages}


def reading_order(blocks):
    """Sort blocks into reading order for a two-column layout.

    Full-width blocks (title, abstract, wide figures) act as separators:
    everything above one is read first (left column, then right column),
    then the full-width block, then we continue below it.
    """
    ordered, left, right = [], [], []

    def flush():
        ordered.extend(sorted(left, key=lambda b: b["y0"]))
        ordered.extend(sorted(right, key=lambda b: b["y0"]))
        left.clear()
        right.clear()

    for b in sorted(blocks, key=lambda b: b["y0"]):
        if b["col"] == "full":
            flush()
            ordered.append(b)
        elif b["col"] == "left":
            left.append(b)
        else:
            right.append(b)
    flush()
    return ordered


def extract_pages(pdf_path: str, debug: bool = False):
    """Return a list of {"page": int, "text": str} with headers/footers removed."""
    doc = pymupdf.open(pdf_path)
    repeated = find_repeated_margin_text(doc)
    pages = []

    for page in doc:
        kept = []
        for b in get_text_blocks(page):
            norm = normalise(b["text"])
            is_margin = in_margin_zone(b, page.rect.height)
            is_bare_page_number = is_margin and re.fullmatch(r"[#\s]*", norm) is not None
            if is_margin and (norm in repeated or is_bare_page_number):
                if debug:
                    print(f"  [filtered p.{page.number + 1}] {clean_text(b['text'])[:70]}")
                continue
            kept.append(b)

        ordered = reading_order(kept)
        text = "\n\n".join(clean_text(b["text"]) for b in ordered)
        pages.append({"page": page.number + 1, "text": text})

    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract page-numbered text from a research paper PDF.")
    parser.add_argument("pdf", help="path to the PDF")
    parser.add_argument("--debug", action="store_true", help="print filtered header/footer text")
    args = parser.parse_args()

    for p in extract_pages(args.pdf, debug=args.debug):
        print(f"\n===== PAGE {p['page']} =====\n")
        print(p["text"])
