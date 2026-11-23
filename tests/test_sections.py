import pytest

from sections import match_heading, section_text, split_sections


@pytest.mark.parametrize("paragraph, expected", [
    ("I. INTRODUCTION Large language models are popular.",
     ("I. INTRODUCTION", "Large language models are popular.")),
    ("V. CONCLUSION AND FUTURE WORK This work introduces a method.",
     ("V. CONCLUSION AND FUTURE WORK", "This work introduces a method.")),
    ("ABSTRACT We study retrieval.", ("ABSTRACT", "We study retrieval.")),
    ("Abstract\u2014We study retrieval.", ("ABSTRACT", "We study retrieval.")),
    ("REFERENCES", ("REFERENCES", "")),
])
def test_match_heading_recognises_headings(paragraph, expected):
    assert match_heading(paragraph) == expected


@pytest.mark.parametrize("paragraph", [
    "This is an ordinary sentence about methods.",
    "A. Chunking Strategy",                       # a subsection stays inside its parent section
    "The results in Section II are shown below.",
    "",
])
def test_match_heading_ignores_normal_text(paragraph):
    assert match_heading(paragraph) is None


def test_sections_are_found_in_order(sample_sections):
    assert [s["title"] for s in sample_sections] == [
        "FRONT MATTER",
        "ABSTRACT",
        "I. INTRODUCTION",
        "II. METHOD",
        "III. RESULTS",
        "IV. ONLINE SELECTION ALGORITHM: MODELING AND ANALYSIS",
        "V. CONCLUSION AND FUTURE WORK",
    ]


def test_drop_cap_is_rejoined(sample_sections):
    intro = next(s for s in sample_sections if s["title"] == "I. INTRODUCTION")
    assert intro["paragraphs"][0]["text"].startswith("LARGE language models")


def test_heading_that_wraps_onto_two_lines_is_one_title(sample_sections):
    section = next(s for s in sample_sections if s["title"].startswith("IV."))
    assert section["paragraphs"][0]["text"] == "This section details the modeling."


def test_references_and_everything_after_are_dropped(sample_sections):
    everything = " ".join(section_text(s) for s in sample_sections)
    assert "A. Author" not in everything
    assert "Another paper" not in everything   # page 5 is after REFERENCES
    assert sample_sections[-1]["title"] == "V. CONCLUSION AND FUTURE WORK"


def test_keywords_and_acknowledgments_are_dropped(sample_sections):
    titles = [s["title"] for s in sample_sections]
    assert "INDEX TERMS" not in titles
    assert "ACKNOWLEDGMENT" not in titles
    assert "thank the reviewers" not in " ".join(section_text(s) for s in sample_sections)


def test_subsection_stays_inside_its_section(sample_sections):
    method = next(s for s in sample_sections if s["title"] == "II. METHOD")
    assert "A. Chunking" in section_text(method)


def test_every_paragraph_keeps_its_page_number(sample_sections):
    results = next(s for s in sample_sections if s["title"] == "III. RESULTS")
    conclusion = next(s for s in sample_sections if s["title"].startswith("V."))
    assert {p["page"] for p in results["paragraphs"]} == {3}
    assert {p["page"] for p in conclusion["paragraphs"]} == {4}


def test_heading_in_the_middle_of_a_paragraph_is_not_a_heading():
    pages = [{"page": 1, "text": "I. INTRODUCTION Text. We discuss II. METHODS in the next part."}]
    sections = split_sections(pages)
    assert [s["title"] for s in sections] == ["I. INTRODUCTION"]


def test_empty_input_gives_no_sections():
    assert split_sections([]) == []
