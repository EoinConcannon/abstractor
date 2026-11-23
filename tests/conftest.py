"""Shared fixtures. Nothing here touches the network, a real PDF, or a real model."""
import zlib

import numpy as np
import pytest

import rag
from sections import split_sections


# ---------- fake pieces that stand in for the LLM and the embedding model ----------

class FakeLLM:
    """Records every prompt it receives and returns canned text."""

    def __init__(self, model="fake-model", fail_on_call=None, reply=None):
        self.model = model
        self.calls = []                  # the user prompt of each call
        self.fail_on_call = fail_on_call # 1-based call number that should raise
        self.reply = reply               # str, or callable(user_prompt) -> str

    def complete(self, system, user, temperature=0.2):
        self.calls.append(user)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated 503")
        if self.reply is not None:
            return self.reply(user) if callable(self.reply) else self.reply
        if "SECTION START" in user:
            return "- A note about this section (p. 1)."
        return "## Problem\n- A bullet (p. 1)."


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so vector search behaves sensibly without a model download."""

    def __init__(self):
        self.embed_calls = 0

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for word in rag.tokenize(text):
            v[zlib.crc32(word.encode()) % 64] += 1
        return v

    def embed(self, texts, batch_size=16):
        self.embed_calls += 1
        return (self._vec(t) for t in texts)

    def query_embed(self, queries):
        return (self._vec(q) for q in queries)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


# ---------- a small made-up paper, in the shape extract_pages() returns ----------

FILLER_INTRO = "Intro sentence about retrieval and cost. " * 8
FILLER_METHOD = "Method detail sentence. " * 8
FILLER_RESULT = "Result detail sentence. " * 8


@pytest.fixture
def sample_pages():
    return [
        {"page": 1, "text": (
            "Demo Paper on Budget Retrieval\n\nAuthor Name\n\n"
            "ABSTRACT We propose a method that reaches an F1 of 0.9089 on TriviaQA.\n\n"
            "INDEX TERMS retrieval, budget\n\n"
            "I. INTRODUCTION L\n\n"
            "ARGE language models are widely used for question answering. " + FILLER_INTRO)},
        {"page": 2, "text": (
            "II. METHOD We use cosine similarity with a chunk size of 1024 tokens. " + FILLER_METHOD +
            "\n\nA. Chunking\n\nChunks are split at sentence boundaries.")},
        {"page": 3, "text": (
            "III. RESULTS On TriviaQA the method reached an F1 of 0.9089, a gain of 52% over the baseline. "
            "We repeated each experiment 100 times. " + FILLER_RESULT +
            "\n\nIV. ONLINE SELECTION ALGORITHM: MODELING AND\n\nANALYSIS\n\n"
            "This section details the modeling.")},
        {"page": 4, "text": (
            "V. CONCLUSION AND FUTURE WORK We presented a method. Future work covers dynamic pricing.\n\n"
            "ACKNOWLEDGMENT The authors thank the reviewers.\n\n"
            "REFERENCES\n\n[1] A. Author, A paper, 2024.")},
        {"page": 5, "text": "[2] B. Author, Another paper, 2023."},
    ]


@pytest.fixture
def sample_sections(sample_pages):
    return split_sections(sample_pages)


# ---------- a real (tiny, generated) two-column PDF ----------

@pytest.fixture
def sample_pdf(tmp_path):
    import pymupdf

    path = tmp_path / "two_column.pdf"
    doc = pymupdf.open()
    for n in range(1, 5):
        page = doc.new_page()  # A4: 595 x 842
        page.insert_text((50, 30), "Journal of Testing, vol. 12", fontsize=8)   # running header
        page.insert_text((290, 825), str(100 + n), fontsize=8)                    # page number in footer
        y = 80
        if n == 1:
            page.insert_text(
                (50, y), "A full width title that spans both columns of the page for the test and keeps going",
                fontsize=10)
            y += 40
        for i in range(6):
            page.insert_text((50, y + i * 14), f"LEFT p{n} line {i} of the first column", fontsize=10)
            page.insert_text((320, y + i * 14), f"RIGHT p{n} line {i} of the second column", fontsize=10)
    doc.save(path)
    doc.close()
    return path
