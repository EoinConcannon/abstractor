import json
from pathlib import Path

import pytest

import mapreduce
import pipeline
from conftest import FakeLLM


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(mapreduce.time, "sleep", lambda s: None)


@pytest.fixture
def paper(tmp_path, monkeypatch, sample_pages):
    """Run inside a temp folder with the sample paper's pages already 'extracted' and saved."""
    monkeypatch.chdir(tmp_path)
    folder = Path("data") / "paper"
    folder.mkdir(parents=True)
    (folder / "pages.json").write_text(json.dumps(sample_pages), encoding="utf-8")
    return "papers/paper.pdf"   # never opened: pages.json already exists


# ---------- verification ----------

@pytest.mark.parametrize("text, expected", [
    ("pp. 6-7, 10", {6, 7, 10}),
    ("p. 1, p. 2", {1, 2}),
    ("pp. 9\u201310", {9, 10}),
    ("p. 5", {5}),
])
def test_parse_pages(text, expected):
    assert pipeline.parse_pages(text) == expected


@pytest.mark.parametrize("text, number, expected", [
    ("an F1 of 0.9089 here", "0.9089", True),
    ("an F1 of 10.9089 here", "0.9089", False),     # part of a longer number
    ("an F1 of 0.90891 here", "0.9089", False),
    ("used 1,024 tokens", "1024", True),            # thousands separator
    ("a 52% gain", "52%", True),
    ("a 152% gain", "52%", False),
])
def test_page_has_number(text, number, expected):
    assert pipeline.page_has_number(text, number) is expected


PAGES = [
    {"page": 1, "text": "Introduction only."},
    {"page": 2, "text": "We used ChatGPT-3.5 with three experts."},
    {"page": 3, "text": "The F1 was 0.9089, a gain of 52%."},
]


def test_verify_summary_reports_ok_check_and_missing():
    summary = (
        "- F1 of 0.9089 on the benchmark [p. 3].\n"
        "- A gain of 52% [p. 1].\n"
        "- Reached 0.5555 overall [p. 3].\n"
    )
    report = pipeline.verify_summary(summary, PAGES)
    assert "1 OK, 1 on a different page than cited, 1 missing" in report
    assert "OK       0.9089" in report
    assert "CHECK    52%" in report
    assert "MISSING  0.5555" in report


def test_verify_summary_ignores_small_numbers_and_model_names():
    report = pipeline.verify_summary("- Used ChatGPT-3.5 and 3 experts [p. 2].\n", PAGES)
    assert "0 OK, 0 on a different page than cited, 0 missing" in report
    assert "3.5" not in report.split("\n\n", 1)[-1]


def test_verify_summary_only_looks_at_bullets():
    report = pipeline.verify_summary("## Key results\nF1 of 0.5555 [p. 3].\n", PAGES)
    assert "0 missing" in report


# ---------- saved stages ----------

def test_prompt_hash_differs_between_methods():
    single, mr = pipeline.prompt_hash("single"), pipeline.prompt_hash("mapreduce")
    assert single != mr
    assert len(single) == len(mr) == 6


def test_get_sections_reads_the_saved_pages(paper):
    titles = [s["title"] for s in pipeline.get_sections(paper)]
    assert "II. METHOD" in titles
    assert Path("data/paper/sections.json").exists()


def test_get_pages_reuses_saved_file_until_refreshed(tmp_path, monkeypatch, sample_pdf):
    monkeypatch.chdir(tmp_path)
    pdf = str(sample_pdf)

    first = pipeline.get_pages(pdf)
    saved = pipeline.paper_dir(pdf) / "pages.json"
    saved.write_text(json.dumps([{"page": 1, "text": "SENTINEL"}]), encoding="utf-8")

    assert pipeline.get_pages(pdf)[0]["text"] == "SENTINEL"               # reused, not re-extracted
    assert pipeline.get_pages(pdf, refresh=True) == first                 # --refresh re-extracts


def test_single_summary_is_saved_and_then_reused(paper):
    llm = FakeLLM()
    assert pipeline.cached_summary(paper, llm.model, "single") is None

    path, text, cached = pipeline.get_summary(paper, llm, method="single")
    assert cached is False and path.exists() and "Problem" in text
    assert len(llm.calls) == 1

    _, _, cached_again = pipeline.get_summary(paper, llm, method="single")
    assert cached_again is True
    assert len(llm.calls) == 1                                            # no second API call


def test_a_different_model_does_not_reuse_the_summary(paper):
    pipeline.get_summary(paper, FakeLLM(model="model-a"), method="single")
    other = FakeLLM(model="model-b")
    _, _, cached = pipeline.get_summary(paper, other, method="single")
    assert cached is False and len(other.calls) == 1


def test_force_regenerates(paper):
    llm = FakeLLM()
    pipeline.get_summary(paper, llm, method="single")
    pipeline.get_summary(paper, llm, method="single", force=True)
    assert len(llm.calls) == 2


def test_mapreduce_makes_one_call_per_body_section_plus_one(paper):
    llm = FakeLLM()
    _, _, cached = pipeline.get_summary(paper, llm, method="mapreduce")
    assert cached is False
    assert len(llm.calls) == 5 + 1       # sections I-V, then the combining call


def test_failed_mapreduce_resumes_from_saved_notes(paper):
    failing = FakeLLM(fail_on_call=3)
    with pytest.raises(RuntimeError):
        pipeline.get_summary(paper, failing, method="mapreduce")

    notes_file = next(Path("data/paper").glob("map_*.json"))
    assert len(json.loads(notes_file.read_text(encoding="utf-8"))) == 2   # two sections were saved

    working = FakeLLM()
    pipeline.get_summary(paper, working, method="mapreduce")
    assert len(working.calls) == 3 + 1   # only the 3 missing sections, plus the combining call


def test_progress_messages_are_reported(paper):
    messages = []
    pipeline.get_summary(paper, FakeLLM(), method="mapreduce", progress=messages.append)
    assert any("map 1/5" in m for m in messages)
    assert any("reduce" in m for m in messages)
