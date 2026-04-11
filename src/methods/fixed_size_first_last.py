from __future__ import annotations

"""
Author: Gourav Singla
Date: 2025-12-18
Description: Fixed-size sequential chunker that keeps a balanced mix of beginning and end chunks under a word budget.
Paper Inspiration: Fixed/positional baselines common in RAG chunking studies (e.g., Qu et al. 2025).
"""

from .registry import register_method


@register_method(
    "fixed_size_first_last",
    description="Fixed-size chunks; keep a balanced mix from the beginning and end under a word budget.",
)
def fixed_size_first_last(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""

    words = text.split()
    total_words = len(words)
    if total_words <= max_words:
        return text

    # Heuristic defaults: small enough to preserve some structure, big enough to be readable.
    chunk_size = min(300, max_words)
    if chunk_size <= 0:
        return ""

    chunks: list[list[str]] = []
    for i in range(0, total_words, chunk_size):
        chunks.append(words[i : i + chunk_size])

    # Budget split between start and end.
    first_budget = max_words // 2
    last_budget = max_words - first_budget

    first_chunks_count = max(1, first_budget // chunk_size) if first_budget > 0 else 0
    last_chunks_count = max(1, last_budget // chunk_size) if last_budget > 0 else 0

    # Select chunk indices (avoid duplicates for short documents).
    first_indices = list(range(0, min(first_chunks_count, len(chunks))))
    last_start = max(0, len(chunks) - last_chunks_count)
    last_indices = list(range(last_start, len(chunks)))

    selected_indices = sorted(set(first_indices + last_indices))
    selected_words: list[str] = []

    for idx in selected_indices:
        selected_words.extend(chunks[idx])
        if len(selected_words) >= max_words:
            selected_words = selected_words[:max_words]
            break

    return " ".join(selected_words)
