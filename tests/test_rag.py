import pytest

import rag


def words(chunk):
    return len(chunk["text"].split())


# ---------- chunking ----------

def test_chunks_never_cross_sections_and_skip_front_matter(sample_sections):
    chunks = rag.make_chunks(sample_sections, max_words=30)
    assert chunks
    assert all(c["section"] != "FRONT MATTER" for c in chunks)

    method_text = " ".join(c["text"] for c in chunks if c["section"] == "II. METHOD")
    assert "cosine similarity" in method_text
    assert "F1" not in method_text                      # nothing from the results section leaked in


def test_chunks_respect_the_size_limit(sample_sections):
    chunks = rag.make_chunks(sample_sections, max_words=30)
    assert all(words(c) <= 30 for c in chunks)


def test_chunks_keep_page_numbers(sample_sections):
    chunks = rag.make_chunks(sample_sections, max_words=30)
    results = [c for c in chunks if c["section"] == "III. RESULTS"]
    assert results and all(c["pages"] == [3] for c in results)


def test_chunk_ids_are_sequential(sample_sections):
    chunks = rag.make_chunks(sample_sections, max_words=30)
    assert [c["id"] for c in chunks] == list(range(len(chunks)))


def test_a_long_paragraph_is_split_at_sentence_boundaries():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    pieces = rag._split_long_paragraph(text, max_words=20)
    assert len(pieces) > 1
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


@pytest.mark.parametrize("pages, expected", [
    ([], "p. ?"), ([5], "p. 5"), ([5, 6], "pp. 5-6"), ([3, 7], "pp. 3-7"),
])
def test_format_pages(pages, expected):
    assert rag.format_pages(pages) == expected


# ---------- tokenising for keyword search ----------

def test_decimal_numbers_stay_in_one_token():
    assert "0.9089" in rag.tokenize("an F1 of 0.9089")


def test_plurals_are_normalised_and_stopwords_dropped():
    assert rag.tokenize("baselines") == rag.tokenize("baseline")
    assert rag.tokenize("What is the baseline?") == ["baseline"]


def test_words_that_only_look_like_plurals_are_left_alone():
    assert rag.tokenize("analysis") == ["analysis"]
    assert rag.tokenize("class") == ["class"]


# ---------- the index ----------

def test_keyword_only_index_finds_an_exact_number(tmp_path, sample_sections):
    index = rag.build_index(tmp_path, sample_sections, embedder=None, max_words=40)
    assert index.mode == "BM25 only"
    top = index.search("0.9089", k=3)
    assert top and "0.9089" in top[0]["text"]


def test_a_query_with_no_matching_words_returns_nothing(tmp_path, sample_sections):
    index = rag.build_index(tmp_path, sample_sections, embedder=None, max_words=40)
    assert index.search("zzzz qqqq", k=3) == []


def test_hybrid_index_reports_which_searches_found_a_chunk(tmp_path, sample_sections, fake_embedder):
    index = rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=40)
    assert index.mode == "hybrid (BM25 + vectors)"

    results = index.search("cosine similarity chunk size", k=3)
    relevant = [c for c in results if "cosine similarity" in c["text"]]
    assert relevant                                   # the right chunk is in the top 3
    assert set(relevant[0]["found_by"]) == {"bm25", "vector"}


def test_search_modes_can_be_isolated(tmp_path, sample_sections, fake_embedder):
    index = rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=40)
    only_bm25 = index.search("cosine similarity", k=3, mode="bm25")
    only_vector = index.search("cosine similarity", k=3, mode="vector")
    assert all(c["found_by"] == ["bm25"] for c in only_bm25)
    assert all(c["found_by"] == ["vector"] for c in only_vector)


def test_vectors_are_saved_and_reused(tmp_path, sample_sections, fake_embedder):
    rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=40)
    assert fake_embedder.embed_calls == 1
    assert (tmp_path / "index" / "embeddings.npy").exists()

    rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=40)
    assert fake_embedder.embed_calls == 1            # loaded from disk, not re-embedded


def test_changing_the_chunk_size_re_embeds(tmp_path, sample_sections, fake_embedder):
    rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=40)
    rag.build_index(tmp_path, sample_sections, fake_embedder, max_words=80)
    assert fake_embedder.embed_calls == 2
