from __future__ import annotations

"""
Author: Gourav Singla
Date: 2025-12-18
Description: Breakpoint-based semantic chunking using sentence embeddings; selects top chunks by similarity to a doc embedding.
Paper Inspiration: Qu et al. (2025) breakpoint semantic chunker.
"""

import os
import re
from functools import lru_cache

from .registry import register_method


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math

    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - _cosine_similarity(a, b)


def _sentence_split(text: str) -> list[str]:
    # Lightweight sentence splitter that works reasonably for PDFs (avoid model downloads).
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _chunks_from_breakpoints(sentences: list[str], break_indices: set[int]) -> list[str]:
    if not sentences:
        return []
    chunks: list[str] = []
    start = 0
    for i in range(1, len(sentences)):
        if i in break_indices:
            chunks.append(" ".join(sentences[start:i]).strip())
            start = i
    chunks.append(" ".join(sentences[start:]).strip())
    return [c for c in chunks if c]


@lru_cache(maxsize=2)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_embedder()
    # sentence-transformers returns numpy arrays; convert to lists to keep this module dependency-light.
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


@register_method(
    "semantic_breakpoint",
    description="Semantic chunking: split into sentences; insert breakpoints where embedding distance spikes; select most representative chunks under budget.",
)
def semantic_breakpoint(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        return " ".join(words[:max_words])

    threshold = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "0.55"))

    # Embed each sentence and break when consecutive distance exceeds threshold.
    sent_vecs = _embed_texts(sentences)
    breakpoints: set[int] = set()
    for i in range(1, len(sent_vecs)):
        if _cosine_distance(sent_vecs[i - 1], sent_vecs[i]) > threshold:
            breakpoints.add(i)

    semantic_chunks = _chunks_from_breakpoints(sentences, breakpoints)
    if not semantic_chunks:
        return " ".join(words[:max_words])

    # Select chunks under budget by similarity to a "document embedding" (computed from a prefix to bound cost).
    doc_prefix_words = int(os.getenv("SEMANTIC_DOC_PREFIX_WORDS", "4000"))
    doc_text = " ".join(words[: min(len(words), doc_prefix_words)])
    doc_vec = _embed_texts([doc_text])[0]

    chunk_vecs = _embed_texts(semantic_chunks)
    scored = []
    for idx, (chunk_text, chunk_vec) in enumerate(zip(semantic_chunks, chunk_vecs)):
        score = _cosine_similarity(doc_vec, chunk_vec)
        scored.append((score, idx, chunk_text))

    # Pick highest scoring chunks, then restore original order for readability.
    scored.sort(key=lambda x: x[0], reverse=True)
    selected_indices: list[int] = []
    selected_word_count = 0
    for _, idx, chunk_text in scored:
        chunk_words = len(chunk_text.split())
        if chunk_words == 0:
            continue
        if selected_word_count + chunk_words > max_words:
            continue
        selected_indices.append(idx)
        selected_word_count += chunk_words
        if selected_word_count >= max_words:
            break

    if not selected_indices:
        # Fallback: take first chunks sequentially.
        out: list[str] = []
        count = 0
        for c in semantic_chunks:
            cw = len(c.split())
            if count + cw > max_words:
                break
            out.append(c)
            count += cw
        return " ".join(out) if out else " ".join(words[:max_words])

    selected_indices.sort()
    output = " ".join(semantic_chunks[i] for i in selected_indices)
    out_words = output.split()
    return " ".join(out_words[:max_words])
