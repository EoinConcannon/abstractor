"""Map-reduce summarisation.

MAP:    summarise each section on its own, so the experiments section can't be skimmed.
REDUCE: combine the section notes into the final Problem / Approach / Key results /
        Limitations / Takeaway summary.

Each map result is saved as soon as it arrives, so if the free API fails halfway
(503, rate limit) the next run resumes instead of starting again.
"""
import json
import time
from pathlib import Path

from summarize import SYSTEM_PROMPT  # same rules as the single-call version

SKIP_TITLES = {"FRONT MATTER", "ABSTRACT"}   # abstract is passed to the reduce step separately

MAP_SYSTEM = """You are a careful research assistant. You will read ONE section of an academic paper and write notes for a later summary.

Rules:
- Use ONLY the section text. Do not add outside knowledge.
- Write 3-8 short bullets covering the claims, methods and results in this section. For a results section, include every headline number.
- Every bullet ends with the single page it comes from, like (p. 5).
- Quote numbers exactly. When a number comes from a table or experiment, say which one.
- If two numbers for the same metric differ, report both with their context.
- Include any limitation, caveat or assumption the authors state in this section (for example: only some models were tested, a quality trade-off, a simplifying assumption). Mark each with "Limitation:".
- Label illustrative or toy examples as such, so they are not mistaken for experimental results.
- Plain text only: no LaTeX, no backticks.
- Tables and equations may be garbled. If something depends on one, say so instead of guessing."""

MAP_USER = """Section: {title}{part}

Page numbers appear as [p.N] at the start of each paragraph.

--- SECTION START ---
{text}
--- SECTION END ---"""

REDUCE_USER = """Below are notes taken section by section from one research paper, plus the authors' abstract.
Write the final summary in exactly this Markdown structure:

## Problem
(2-3 bullets: what are the authors trying to solve, and why does it matter?)

## Approach
(3-5 bullets: what did they do, and what is new compared with prior work?)

## Key results
(3-6 bullets: the main findings with specific numbers. At least three must come from the experiments/results notes, not the abstract. Prefer measured results over toy or illustrative examples; if you use a toy example, say so.)

## Limitations
(2-4 bullets: weaknesses, trade-offs or untested cases, drawn from notes marked "Limitation:" and from the evaluation set-up. Mention future work only where it reveals a limitation, and label it "future work".)

## Takeaway
(1-2 sentences, ending with a page citation)

Keep the page citations from the notes. Do not add facts that are not in the notes.

--- ABSTRACT (for framing only) ---
{abstract}

--- SECTION NOTES ---
{notes}
--- END ---"""


def split_long(section, max_words: int = 3000):
    """Split a very long section into chunks at paragraph boundaries."""
    chunks, current, count = [], [], 0
    for para in section["paragraphs"]:
        words = len(para["text"].split())
        if current and count + words > max_words:
            chunks.append(current)
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        chunks.append(current)
    return chunks


def summarize_mapreduce(sections, llm, cache_path: Path, pause: float = 3.0,
                        max_words: int = 3000, progress=print) -> str:
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    work = []
    for s in sections:
        if s["title"] in SKIP_TITLES:
            continue
        chunks = split_long(s, max_words)
        for i, chunk in enumerate(chunks, 1):
            part = f" (part {i}/{len(chunks)})" if len(chunks) > 1 else ""
            work.append((s["title"], part, chunk))

    notes = []
    for n, (title, part, chunk) in enumerate(work, 1):
        key = f"{title}{part}"
        if key in cache:
            progress(f"  map {n}/{len(work)}: {key}  (saved)")
        else:
            progress(f"  map {n}/{len(work)}: {key}")
            text = "\n\n".join(f"[p.{p['page']}] {p['text']}" for p in chunk)
            cache[key] = llm.complete(MAP_SYSTEM, MAP_USER.format(title=title, part=part, text=text))
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(pause)  # be gentle with free-tier rate limits
        notes.append(f"### {key}\n{cache[key].strip()}")

    abstract = next(("\n\n".join(p["text"] for p in s["paragraphs"])
                     for s in sections if s["title"] == "ABSTRACT"), "(none found)")
    progress("  reduce: combining notes into the final summary")
    return llm.complete(SYSTEM_PROMPT, REDUCE_USER.format(abstract=abstract, notes="\n\n".join(notes)))
