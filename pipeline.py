"""One command for the whole process, with every stage saved per paper.

Usage:
    python pipeline.py papers/ca_rag.pdf            # run everything (reuses saved stages)
    python pipeline.py papers/ca_rag.pdf --force    # redo the summary even if one is saved
    python pipeline.py papers/ca_rag.pdf --no-llm   # only extract + sections (no API call)

Saves to data/<paper name>/:
    pages.json      extracted text per page
    sections.json   sections with page numbers
    summary_<model>_<prompt hash>_<time>.md
    verify_<same name>.md   number / page-citation check of that summary
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from extract import extract_pages
from sections import section_text, split_sections
from summarize import SYSTEM_PROMPT, USER_TEMPLATE, build_paper_text

DATA_DIR = Path("data")


# ---------- saved stages ----------

def paper_dir(pdf_path: str) -> Path:
    d = DATA_DIR / Path(pdf_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_or_build(path: Path, build):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = build()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def get_pages(pdf_path: str):
    return _load_or_build(paper_dir(pdf_path) / "pages.json", lambda: extract_pages(pdf_path))


def get_sections(pdf_path: str):
    pages = get_pages(pdf_path)
    return _load_or_build(paper_dir(pdf_path) / "sections.json", lambda: split_sections(pages))


def prompt_hash() -> str:
    """Changing the prompt changes the hash, so old summaries are never reused by mistake."""
    return hashlib.md5((SYSTEM_PROMPT + USER_TEMPLATE).encode("utf-8")).hexdigest()[:6]


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def get_summary(pdf_path: str, llm, force: bool = False):
    """Return (summary_path, summary_text, was_cached)."""
    d = paper_dir(pdf_path)
    pattern = f"summary_{safe(llm.model)}_{prompt_hash()}_*.md"
    existing = sorted(d.glob(pattern))
    if existing and not force:
        return existing[-1], existing[-1].read_text(encoding="utf-8"), True

    paper = build_paper_text(get_sections(pdf_path))
    print(f"Sending ~{len(paper.split())} words to {llm.model}...")
    text = llm.complete(SYSTEM_PROMPT, USER_TEMPLATE.format(paper=paper))
    path = d / f"summary_{safe(llm.model)}_{prompt_hash()}_{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    return path, text, False


# ---------- verification ----------

CITATION = re.compile(r"[\[(]\s*(pp?\.[^\])]*)[\])]")
NUMBER = re.compile(r"(?<![\w.\-])\d+(?:\.\d+)?%?")


def parse_pages(citation_text: str) -> set:
    pages = set()
    for a, b, single in re.findall(r"(\d+)\s*[-\u2013]\s*(\d+)|(\d+)", citation_text):
        if single:
            pages.add(int(single))
        else:
            pages.update(range(int(a), int(b) + 1))
    return pages


def page_has_number(text: str, number: str) -> bool:
    num = number.rstrip("%")
    text = text.replace(",", "")
    return re.search(rf"(?<![\d.]){re.escape(num)}(?!\d|\.\d)", text) is not None


def verify_summary(summary: str, pages) -> str:
    """Check every meaningful number in each bullet against the pages it cites."""
    page_text = {p["page"]: p["text"] for p in pages}
    all_text = " ".join(page_text.values())
    lines_out, counts = [], {"ok": 0, "elsewhere": 0, "missing": 0}

    for line in summary.splitlines():
        if not line.lstrip().startswith(("-", "*")):
            continue
        cited = set()
        for group in CITATION.findall(line):
            cited |= parse_pages(group)
        body = CITATION.sub("", line)

        findings = []
        for num in NUMBER.findall(body):
            # Skip small whole numbers ("3 bullets", "Scenario 2"); they are noise.
            if "." not in num and "%" not in num and len(num) < 3:
                continue
            on_cited = [p for p in sorted(cited) if p in page_text and page_has_number(page_text[p], num)]
            anywhere = [p for p, t in sorted(page_text.items()) if page_has_number(t, num)]
            if on_cited:
                counts["ok"] += 1
                findings.append(f"  - OK       {num}: found on cited p. {on_cited[0]}")
            elif anywhere:
                counts["elsewhere"] += 1
                findings.append(f"  - CHECK    {num}: not on cited pages {sorted(cited)}, but appears on p. {anywhere[:6]}")
            elif page_has_number(all_text, num):
                counts["elsewhere"] += 1
                findings.append(f"  - CHECK    {num}: appears in the paper")
            else:
                counts["missing"] += 1
                findings.append(f"  - MISSING  {num}: not found anywhere in the paper text")

        if findings:
            lines_out.append(body.strip()[:110] + ("..." if len(body.strip()) > 110 else ""))
            lines_out.extend(findings)

    header = (f"Number check: {counts['ok']} OK, {counts['elsewhere']} on a different page than cited, "
              f"{counts['missing']} missing from the paper.\n"
              "OK = found on a cited page. It does NOT prove the number is attached to the right claim.\n")
    return header + "\n" + "\n".join(lines_out)


# ---------- command line ----------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the full paper pipeline.")
    parser.add_argument("pdf")
    parser.add_argument("--force", action="store_true", help="regenerate the summary")
    parser.add_argument("--no-llm", action="store_true", help="stop after extract + sections")
    args = parser.parse_args()

    pages = get_pages(args.pdf)
    print(f"[1/4] Extracted {len(pages)} pages -> {paper_dir(args.pdf) / 'pages.json'}")

    sections = get_sections(args.pdf)
    print("[2/4] Sections found:")
    for s in sections:
        nums = [p["page"] for p in s["paragraphs"]]
        print(f"      {s['title'][:48]:<48} pp. {min(nums)}-{max(nums)}   {len(section_text(s).split())} words")

    if args.no_llm:
        return

    from llm import LLMClient  # imported here so --no-llm works without a key
    path, summary, cached = get_summary(args.pdf, LLMClient(), force=args.force)
    print(f"[3/4] Summary {'loaded from' if cached else 'saved to'} {path}")

    report = verify_summary(summary, pages)
    vpath = path.with_name(path.name.replace("summary_", "verify_", 1))
    vpath.write_text(report, encoding="utf-8")
    print(f"[4/4] Verification saved to {vpath}\n")
    print(summary)
    print("\n" + "=" * 60 + "\nVERIFICATION\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
