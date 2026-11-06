"""Retrieval for follow-up questions: chunking, embeddings, BM25 and hybrid search.

Design choices (all deliberate, all worth explaining in the README):
- Chunks never cross section boundaries and keep their page numbers.
- Embeddings run locally (fastembed, ONNX) so retrieval costs nothing and uses no API quota.
- Vectors are stored as a plain numpy file. One paper is a few hundred chunks, so brute-force
  cosine similarity is instant and a vector database would be needless weight on an 8 GB laptop.
- BM25 keyword search runs alongside vector search and the two rankings are merged with
  reciprocal rank fusion. BM25 is what finds exact tokens like "0.9089" or "UCOSA".
- If fastembed isn't installed the system falls back to BM25 only, and says so.
"""
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # silence a harmless Windows warning

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard constant for reciprocal rank fusion


# ---------- chunking ----------

def _split_long_paragraph(text: str, max_words: int):
    """Split a very long paragraph at sentence boundaries."""
    if len(text.split()) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", text)
    pieces, current, count = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if current and count + w > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += w
    if current:
        pieces.append(" ".join(current))
    return pieces


def make_chunks(sections, max_words: int = 200):
    """Pack paragraphs into ~max_words chunks. Chunks stay inside one section."""
    chunks = []
    for section in sections:
        if section["title"] == "FRONT MATTER":
            continue
        current, pages, count = [], [], 0

        def flush():
            nonlocal current, pages, count
            if current:
                chunks.append({
                    "id": len(chunks),
                    "section": section["title"],
                    "pages": sorted(set(pages)),
                    "text": " ".join(current),
                })
            current, pages, count = [], [], 0

        for para in section["paragraphs"]:
            for piece in _split_long_paragraph(para["text"], max_words):
                w = len(piece.split())
                if current and count + w > max_words:
                    flush()
                current.append(piece)
                pages.append(para["page"])
                count += w
        flush()
    return chunks


def chunk_text_for_search(chunk) -> str:
    """Section title is included so a query like 'experimental setup' can match the right section."""
    return f"{chunk['section']}. {chunk['text']}"


def tokenize(text: str):
    """Lowercase words; decimals such as 0.9089 stay in one piece so exact numbers are searchable."""
    return re.findall(r"\d+\.\d+|\w+", text.lower())


def format_pages(pages) -> str:
    if not pages:
        return "p. ?"
    if len(pages) == 1:
        return f"p. {pages[0]}"
    return f"pp. {pages[0]}-{pages[-1]}"


# ---------- embeddings ----------

def load_embedder():
    """Return a fastembed model, or None if it isn't installed (BM25-only fallback)."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    # Keep the model inside the project (not in Windows Temp, which can be cleaned out).
    return TextEmbedding(EMBED_MODEL, cache_dir=str(Path(__file__).parent / "models"))


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ---------- the index ----------

class PaperIndex:
    def __init__(self, chunks, embeddings, embedder):
        self.chunks = chunks
        self.embeddings = embeddings          # None => BM25-only
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(chunk_text_for_search(c)) for c in chunks])

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + vectors)" if self.embeddings is not None else "BM25 only"

    def search(self, question: str, k: int = 6, pool: int = 20, mode: str = "hybrid"):
        """Return the top-k chunks, each with a 'score' and which searches found it.

        mode is "hybrid" (default), "bm25" or "vector"; eval.py uses the other two to compare them.
        """
        rankings = {}

        if mode in ("hybrid", "bm25"):
            bm = self.bm25.get_scores(tokenize(question))
            order = np.argsort(-bm)[:pool]
            rankings["bm25"] = [int(i) for i in order if bm[i] > 0]

        if mode in ("hybrid", "vector") and self.embeddings is not None and self.embedder is not None:
            q = np.array(list(self.embedder.query_embed([question]))[0], dtype=np.float32)
            q = q / max(np.linalg.norm(q), 1e-12)
            sims = self.embeddings @ q
            rankings["vector"] = [int(i) for i in np.argsort(-sims)[:pool]]

        scores, found_by = defaultdict(float), defaultdict(list)
        for name, ranked in rankings.items():
            for rank, i in enumerate(ranked):
                scores[i] += 1.0 / (RRF_K + rank + 1)
                found_by[i].append(name)

        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [{**self.chunks[i], "score": round(scores[i], 4), "found_by": found_by[i]} for i in top]


def build_index(paper_dir: Path, sections, embedder, max_words: int = 200, progress=print) -> PaperIndex:
    """Chunk the paper, embed it (once), and cache the vectors next to the paper's other data."""
    chunks = make_chunks(sections, max_words)
    index_dir = Path(paper_dir) / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path, vec_path = index_dir / "meta.json", index_dir / "embeddings.npy"

    texts = [chunk_text_for_search(c) for c in chunks]
    fingerprint = hashlib.md5(json.dumps(texts).encode("utf-8")).hexdigest()
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    if embedder is None:
        return PaperIndex(chunks, None, None)

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if vec_path.exists() and meta.get("fingerprint") == fingerprint and meta.get("model") == EMBED_MODEL:
        return PaperIndex(chunks, np.load(vec_path), embedder)

    progress(f"Embedding {len(chunks)} chunks locally (one-off per paper; slow on an old CPU)...")
    vectors = _normalise(np.array(list(embedder.embed(texts, batch_size=16)), dtype=np.float32))
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "model": EMBED_MODEL,
                                     "n_chunks": len(chunks)}), encoding="utf-8")
    return PaperIndex(chunks, vectors, embedder)
