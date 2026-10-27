"""Step 4: split an extracted paper into sections and drop the references.

Usage:
    python sections.py papers/ca_rag.pdf
"""
import argparse
import re

from extract import extract_pages

# "I. INTRODUCTION Large Language Models..." / "V. CONCLUSION AND FUTURE WORK This work..."
# Roman numeral, then an ALL-CAPS title, then (usually) the first word of the body text.
NUMBERED = re.compile(
    r"^(?P<num>[IVX]{1,5})\.\s+"
    r"(?P<title>[A-Z][A-Z0-9 ,&:/()\-]{2,}?)"
    r"(?=\s+[A-Z][a-z]|\s*$)"      # stop where body text starts (Capital + lowercase)
    r"\s*(?P<rest>.*)$",
    re.DOTALL,
)

# Headings with no number. ALL CAPS in IEEE Access, "Abstract—" style in many conferences.
UNNUMBERED = re.compile(
    r"^(?P<title>ABSTRACT|INDEX TERMS|KEYWORDS|ACKNOWLEDGMENTS?|REFERENCES|APPENDIX(?: [A-Z])?)"
    r"\b[\s:.\u2014\u2013\-]*(?P<rest>.*)$",
    re.DOTALL,
)
UNNUMBERED_DASH = re.compile(
    r"^(?P<title>Abstract|Index Terms|Keywords)\s*[\u2014\u2013\-:]\s*(?P<rest>.*)$",
    re.DOTALL | re.IGNORECASE,
)

SKIP_TITLES = {"INDEX TERMS", "KEYWORDS", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS"}
STOP_TITLE = "REFERENCES"


def match_heading(paragraph: str):
    """Return (title, rest_of_paragraph) if the paragraph starts with a heading, else None."""
    for pattern in (NUMBERED, UNNUMBERED, UNNUMBERED_DASH):
        m = pattern.match(paragraph)
        if m:
            title = m.group("title").strip().upper()
            if pattern is NUMBERED:
                title = f"{m.group('num')}. {title}"
            return title, m.group("rest").strip()
    return None


def split_sections(pages):
    """pages: output of extract_pages(). Returns a list of sections:
    {"title": str, "paragraphs": [{"page": int, "text": str}, ...]}
    Everything from REFERENCES onward is dropped.
    """
    sections = [{"title": "FRONT MATTER", "paragraphs": []}]

    for page in pages:
        for para in page["text"].split("\n\n"):
            para = para.strip()
            if not para:
                continue

            heading = match_heading(para)
            if heading:
                title, rest = heading
                if title == STOP_TITLE:
                    return _tidy(sections)
                sections.append({"title": title, "paragraphs": []})
                para = rest
                if not para:
                    continue

            sections[-1]["paragraphs"].append({"page": page["page"], "text": para})

    return _tidy(sections)


def _tidy(sections):
    """Drop empty sections and ones we don't want to summarise (keywords, acknowledgments)."""
    return [
        s for s in sections
        if s["paragraphs"] and not any(s["title"].endswith(t) or s["title"] == t for t in SKIP_TITLES)
    ]


def section_text(section) -> str:
    return "\n\n".join(p["text"] for p in section["paragraphs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a research paper into sections.")
    parser.add_argument("pdf", help="path to the PDF")
    args = parser.parse_args()

    for s in split_sections(extract_pages(args.pdf)):
        pages = [p["page"] for p in s["paragraphs"]]
        words = len(section_text(s).split())
        print(f"{s['title']:<45} pages {min(pages)}-{max(pages)}   {words} words")
