from __future__ import annotations

"""
Author: Gourav Singla
Date: 2025-12-18
Description: Position-Aware Chunking (PAC) with quality signals and noise penalties; selects top overlapping windows under a word budget.
Paper Inspiration: Novel for this work; grounded in position importance (lost-in-the-middle, lead bias) and practical heuristics.
"""

import re
from typing import List, Dict, Iterable

from .registry import register_method


def _score_chunk(text: str, position: float) -> float:
    """
    Score a chunk based on position, quality signals (numbers/comparisons/method cues), and noise penalties.
    position: relative midpoint of the chunk in [0,1].

    Uses W-shaped position curve:
    - High score for intro (0-12%) and conclusion (88-100%)
    - Medium-high score for results/findings section (38-62%)
    - Floor score for background/discussion (no zeros)
    """
    score = 0.0

    # Position weighting: W-shaped curve that captures intro, results, and conclusion.
    # This addresses the "lost-in-the-middle" problem while still valuing the results section.
    if position < 0.12:
        # Abstract/Introduction - high importance
        score += 2.0
    elif position > 0.88:
        # Conclusion/Summary - high importance
        score += 2.0
    elif 0.38 <= position <= 0.62:
        # Results/Findings section - medium-high importance (peak at center)
        # Smooth bell curve: 1.2 at edges, 1.8 at center
        score += 1.2 + 0.6 * (1 - abs(position - 0.5) / 0.12)
    else:
        # Background (12-38%) and Discussion (62-88%) - floor score
        # Ensures these sections can still contribute if they have quality signals
        score += 0.6

    word_count = len(text.split())

    # Quality signals.
    if 100 <= word_count <= 800:
        score += 1.0  # reasonable chunk length
    if re.search(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", text):
        score += 1.5  # contains statistics/numbers
    if re.search(r"\bcompared?\b|\bversus\b|\bvs\.?\b|\bthan\b", text, re.IGNORECASE):
        score += 0.5  # comparative language
    if re.search(r"\b(introduction|conclusion|results?|findings?|summary|takeaways?)\b", text, re.IGNORECASE):
        score += 0.8  # heading/section cues
    if re.search(r"^[-•\u2022]", text.strip(), re.MULTILINE):
        score += 0.3  # bullet-like cues
    if len(re.findall(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", text)) >= 3:
        score += 0.8  # dense with statistics
    if re.search(r"\b(sample|response rate|survey|methodology|scopus|unpaywall|dataset)\b", text, re.IGNORECASE):
        score += 0.6  # methodology/reliability cues

    # Noise penalties (citations/urls/doi clutter).
    noise_patterns = [
        r"\[\d+\]",  # numeric citations
        r"\bet\s+al\.",  # et al.
        r"doi:\s*\d",
        r"https?://",
        r"\breferences\b",
        r"\backnowledg(e)?ments?\b",
        r"\bfaq\b",
        r"\bprocess flow\b",
        r"\bbenefits of\b",
    ]
    noise_count = 0
    for pattern in noise_patterns:
        noise_count += len(re.findall(pattern, text, re.IGNORECASE))

    if noise_count > 3:
        score -= 2.0
    elif noise_count > 1:
        score -= 1.0

    return score


def _jaccard_redundant(candidate: str, selected: Iterable[str], threshold: float = 0.6) -> bool:
    """
    Cheap redundancy gate: drop chunks whose token Jaccard overlap with already selected chunks
    is above the threshold, so we do not waste budget on near-duplicates.
    """
    cand_tokens = set(candidate.lower().split())
    if not cand_tokens:
        return False
    for text in selected:
        tokens = set(text.lower().split())
        if not tokens:
            continue
        overlap = len(cand_tokens & tokens) / max(1, len(cand_tokens | tokens))
        if overlap >= threshold:
            return True
    return False


@register_method(
    "pac_position_aware",
    description="Position-Aware Chunking: overlapping windows scored by position + quality signals − noise; keep best under budget.",
)
def pac_position_aware(
    text: str,
    max_words: int,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> str:
    if max_words <= 0:
        return ""

    words = text.split()
    total_words = len(words)
    if total_words <= max_words:
        return text

    # Adaptive chunk size for shorter/longer docs.
    # Keeps small docs from being over-fragmented, and large docs from using too few windows.
    if total_words < 3000:
        chunk_size = max(400, min(chunk_size, max_words))
    elif total_words > 10000:
        chunk_size = min(chunk_size + 200, 1400)

    overlap = max(100, min(overlap, chunk_size // 2))

    chunks: List[Dict] = []
    start = 0

    # Create overlapping windows and score them (position + quality − noise).
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        position = (start + (end - start) / 2) / total_words  # midpoint as fraction of doc length

        score = _score_chunk(chunk_text, position)
        chunks.append(
            {
                "text": chunk_text,
                "position": position,
                "score": score,
                "word_count": len(chunk_words),
            }
        )

        start = end - overlap
        if start >= total_words - overlap:
            break

    # Select highest scoring chunks until budget is filled (with redundancy gate).
    chunks.sort(key=lambda x: x["score"], reverse=True)
    selected: List[Dict] = []
    current_words = 0
    selected_texts: List[str] = []
    for chunk in chunks:
        if current_words + chunk["word_count"] > max_words:
            continue
        # Skip highly redundant chunks to preserve budget.
        if _jaccard_redundant(chunk["text"], selected_texts, threshold=0.6):
            continue
        selected.append(chunk)
        selected_texts.append(chunk["text"])
        current_words += chunk["word_count"]
        if current_words >= max_words:
            break

    if not selected:
        return " ".join(words[:max_words])

    # Reorder by position for coherent flow.
    selected.sort(key=lambda x: x["position"])

    # Ensure inclusion of at least one high-stat chunk and one methods chunk if available
    # to preserve key facts and reproducibility details.
    has_stats = any(len(re.findall(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", c["text"])) >= 3 for c in selected)
    has_methods = any(re.search(r"\b(sample|response rate|survey|methodology|scopus|unpaywall|dataset)\b", c["text"], re.IGNORECASE) for c in selected)
    if not has_stats or not has_methods:
        remaining = [c for c in chunks if c not in selected]
        remaining.sort(key=lambda c: c["score"], reverse=True)
        for c in remaining:
            if len(" ".join(output_words := (c["text"].split()))) > max_words:
                continue
            if not has_stats and len(re.findall(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", c["text"])) >= 3:
                selected.append(c)
                has_stats = True
            elif not has_methods and re.search(r"\b(sample|response rate|survey|methodology|scopus|unpaywall|dataset)\b", c["text"], re.IGNORECASE):
                selected.append(c)
                has_methods = True
            if has_stats and has_methods:
                break
        selected.sort(key=lambda x: x["position"])

    output_words: List[str] = []
    for chunk in selected:
        output_words.extend(chunk["text"].split())
        if len(output_words) >= max_words:
            break

    return " ".join(output_words[:max_words])
