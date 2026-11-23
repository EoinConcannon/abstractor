import pytest

import ask
import rag
from conftest import FakeLLM


@pytest.fixture
def index(tmp_path, sample_sections):
    return rag.build_index(tmp_path, sample_sections, embedder=None, max_words=40)


CHUNKS = [{"section": "III. RESULTS", "pages": [3], "text": "The F1 was 0.9089 with 1024 tokens."}]


# ---------- unsupported numbers ----------

def test_numbers_missing_from_the_excerpts_are_flagged():
    answer = "The F1 was 0.9089 and 0.77 (p. 3), using 1024 tokens."
    assert ask.unsupported_numbers(answer, CHUNKS) == ["0.77"]


def test_page_citations_are_not_mistaken_for_numbers():
    assert ask.unsupported_numbers("The F1 was 0.9089 (p. 12).", CHUNKS) == []


def test_single_digit_numbers_are_ignored():
    assert ask.unsupported_numbers("There were 3 experts and 9089 is not here? no.", CHUNKS) == ["9089"]


# ---------- exact word search ----------

def test_find_hits_is_case_insensitive_and_whole_word():
    pages = [{"page": 1, "text": "AI and ai are here, but MAIN and said are not."}]
    total, shown = ask.find_hits("AI", pages)
    assert total == 2
    assert all(page == 1 for page, _ in shown)


def test_find_hits_reports_overflow_on_a_busy_page():
    pages = [{"page": 4, "text": " ".join(["token"] * 6)}]
    total, shown = ask.find_hits("token", pages, max_per_page=3)
    assert total == 6
    assert len(shown) == 4
    assert "+3 more" in shown[-1][1]


def test_find_hits_with_no_match():
    assert ask.find_hits("blockchain", [{"page": 1, "text": "nothing relevant"}]) == (0, [])


def test_find_hits_treats_special_characters_literally():
    total, _ = ask.find_hits("F1 (macro)", [{"page": 1, "text": "we report F1 (macro) here"}])
    assert total == 1


# ---------- answering ----------

def test_excerpts_are_labelled_with_section_and_page():
    text = ask.format_excerpts(CHUNKS)
    assert "[Excerpt 1 | III. RESULTS | p. 3]" in text


def test_get_answer_sends_excerpts_and_question_to_the_llm(index):
    llm = FakeLLM(reply="The F1 was 0.9089 (p. 3).")
    result = ask.get_answer("What F1 was reached?", index, llm)

    assert result["answered"] is True
    assert result["answer"] == "The F1 was 0.9089 (p. 3)."
    assert result["chunks"]
    prompt = llm.calls[0]
    assert "[Excerpt 1 |" in prompt and "What F1 was reached?" in prompt


def test_get_answer_flags_numbers_the_excerpts_do_not_contain(index):
    llm = FakeLLM(reply="The F1 was 0.9089 (p. 3), but 0.4242 on another set (p. 3).")
    result = ask.get_answer("What F1 was reached?", index, llm)
    assert result["missing"] == ["0.4242"]


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
def test_a_declined_answer_is_recognised(index, apostrophe):
    llm = FakeLLM(reply=f"The retrieved excerpts don{apostrophe}t answer this. Look in the setup section.")
    result = ask.get_answer("What F1 was reached?", index, llm)
    assert result["answered"] is False
    assert result["missing"] == []


def test_no_matching_passages_means_no_llm_call(index):
    llm = FakeLLM()
    result = ask.get_answer("zzzz qqqq", index, llm)
    assert result["answered"] is False
    assert result["chunks"] == []
    assert llm.calls == []
