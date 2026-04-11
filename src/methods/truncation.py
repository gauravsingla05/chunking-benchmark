from __future__ import annotations

"""
Author: Gourav Singla
Date: 2025-12-18
Description: Baseline chunker that keeps only the first N words.
Paper Inspiration: Common fixed-order baseline in RAG/chunking comparisons (e.g., Qu et al. 2025).
"""

from .registry import register_method


@register_method(
    "truncation",
    description="Baseline: keep the first N words from the document.",
)
def truncation(text: str, max_words: int) -> str:
    # Simple positional baseline; trims at word budget.
    words = text.split()
    if max_words <= 0:
        return ""
    return " ".join(words[:max_words])
