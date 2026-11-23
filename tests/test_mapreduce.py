import json

import pytest

import mapreduce
from conftest import FakeLLM


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(mapreduce.time, "sleep", lambda s: None)


def para(words, page=1):
    return {"page": page, "text": " ".join(["word"] * words)}


def test_split_long_keeps_short_sections_whole():
    section = {"title": "II. METHOD", "paragraphs": [para(40), para(40)]}
    assert len(mapreduce.split_long(section, max_words=100)) == 1


def test_split_long_breaks_at_paragraph_boundaries():
    section = {"title": "VI. RESULTS", "paragraphs": [para(60), para(60), para(60)]}
    chunks = mapreduce.split_long(section, max_words=100)
    assert [len(c) for c in chunks] == [1, 1, 1]


def test_one_oversized_paragraph_is_kept_whole():
    section = {"title": "VI. RESULTS", "paragraphs": [para(500)]}
    assert len(mapreduce.split_long(section, max_words=100)) == 1


def test_front_matter_and_abstract_are_not_mapped(sample_sections, tmp_path):
    llm = FakeLLM()
    mapreduce.summarize_mapreduce(sample_sections, llm, tmp_path / "notes.json", progress=lambda m: None)
    map_prompts = [c for c in llm.calls if "SECTION START" in c]
    assert len(map_prompts) == 5
    assert not any("Section: ABSTRACT" in p or "Section: FRONT MATTER" in p for p in map_prompts)


def test_each_section_is_sent_with_page_tags(sample_sections, tmp_path):
    llm = FakeLLM()
    mapreduce.summarize_mapreduce(sample_sections, llm, tmp_path / "notes.json", progress=lambda m: None)
    method_prompt = next(c for c in llm.calls if "Section: II. METHOD" in c)
    assert "[p.2]" in method_prompt


def test_final_prompt_contains_the_abstract_and_every_sections_notes(sample_sections, tmp_path):
    llm = FakeLLM(reply=lambda user: "- note (p. 1)." if "SECTION START" in user else "FINAL SUMMARY")
    result = mapreduce.summarize_mapreduce(sample_sections, llm, tmp_path / "notes.json", progress=lambda m: None)

    assert result == "FINAL SUMMARY"
    final_prompt = llm.calls[-1]
    assert "F1 of 0.9089 on TriviaQA" in final_prompt            # the abstract, for framing
    for title in ["I. INTRODUCTION", "II. METHOD", "III. RESULTS", "V. CONCLUSION AND FUTURE WORK"]:
        assert f"### {title}" in final_prompt


def test_notes_are_saved_after_each_section(sample_sections, tmp_path):
    notes = tmp_path / "notes.json"
    with pytest.raises(RuntimeError):
        mapreduce.summarize_mapreduce(sample_sections, FakeLLM(fail_on_call=4), notes, progress=lambda m: None)
    assert len(json.loads(notes.read_text(encoding="utf-8"))) == 3


def test_saved_notes_are_not_requested_again(sample_sections, tmp_path):
    notes = tmp_path / "notes.json"
    mapreduce.summarize_mapreduce(sample_sections, FakeLLM(), notes, progress=lambda m: None)
    second = FakeLLM()
    mapreduce.summarize_mapreduce(sample_sections, second, notes, progress=lambda m: None)
    assert len(second.calls) == 1        # only the combining step
