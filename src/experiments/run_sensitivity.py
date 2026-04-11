#!/usr/bin/env python3
"""
W-curve sensitivity analysis: vary position scoring parameters in PAC.

Tests how changes to intro/conclusion/results weights affect chunk selection
and downstream quality. Answers reviewer question: "Why these specific parameter values?"

Usage:
    python run_sensitivity.py                  # All configs
    python run_sensitivity.py --configs baseline flat u_curve  # Specific ones
    python run_sensitivity.py --limit-docs 30  # Quick pilot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Iterable

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import SENSITIVITY_CONFIGS, DOCUMENTS_DIR, DATA_DIR, RESULTS_DIR, DEFAULT_BUDGET
from utils.text_io import iter_documents


def _score_chunk_parameterized(
    text: str,
    position: float,
    intro_weight: float,
    conclusion_weight: float,
    results_peak: float,
    floor_weight: float,
) -> float:
    """PAC scoring with configurable W-curve parameters."""
    score = 0.0

    # Parameterized W-curve
    if position < 0.12:
        score += intro_weight
    elif position > 0.88:
        score += conclusion_weight
    elif 0.38 <= position <= 0.62:
        score += floor_weight + (results_peak - floor_weight) * (1 - abs(position - 0.5) / 0.12)
    else:
        score += floor_weight

    word_count = len(text.split())

    # Quality signals (same as original PAC)
    if 100 <= word_count <= 800:
        score += 1.0
    if re.search(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", text):
        score += 1.5
    if re.search(r"\bcompared?\b|\bversus\b|\bvs\.?\b|\bthan\b", text, re.IGNORECASE):
        score += 0.5
    if re.search(r"\b(introduction|conclusion|results?|findings?|summary|takeaways?)\b", text, re.IGNORECASE):
        score += 0.8
    if re.search(r"^[-•\u2022]", text.strip(), re.MULTILINE):
        score += 0.3
    if len(re.findall(r"\d+%|\d+\.\d+|\d{1,3}(?:,\d{3})+", text)) >= 3:
        score += 0.8
    if re.search(r"\b(sample|response rate|survey|methodology|scopus|unpaywall|dataset)\b", text, re.IGNORECASE):
        score += 0.6

    # Noise penalties
    noise_patterns = [
        r"\[\d+\]", r"\bet\s+al\.", r"doi:\s*\d", r"https?://",
        r"\breferences\b", r"\backnowledg(e)?ments?\b",
    ]
    noise_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in noise_patterns)
    if noise_count > 3:
        score -= 2.0
    elif noise_count > 1:
        score -= 1.0

    return score


def _jaccard_redundant(candidate: str, selected: list[str], threshold: float = 0.6) -> bool:
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


def pac_with_params(text: str, max_words: int, params: dict) -> str:
    """Run PAC chunking with custom W-curve parameters."""
    words = text.split()
    total_words = len(words)
    if total_words <= max_words:
        return text

    chunk_size = 1000
    overlap = 200
    if total_words < 3000:
        chunk_size = max(400, min(chunk_size, max_words))
    elif total_words > 10000:
        chunk_size = min(chunk_size + 200, 1400)
    overlap = max(100, min(overlap, chunk_size // 2))

    chunks = []
    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_text = " ".join(words[start:end])
        position = (start + (end - start) / 2) / total_words

        score = _score_chunk_parameterized(
            chunk_text, position,
            intro_weight=params["intro"],
            conclusion_weight=params["conclusion"],
            results_peak=params["results_peak"],
            floor_weight=params["floor"],
        )
        chunks.append({
            "text": chunk_text, "position": position,
            "score": score, "word_count": end - start,
        })
        start = end - overlap
        if start >= total_words - overlap:
            break

    # Select highest scoring
    chunks.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    current_words = 0
    selected_texts = []
    for chunk in chunks:
        if current_words + chunk["word_count"] > max_words:
            continue
        if _jaccard_redundant(chunk["text"], selected_texts):
            continue
        selected.append(chunk)
        selected_texts.append(chunk["text"])
        current_words += chunk["word_count"]
        if current_words >= max_words:
            break

    if not selected:
        return " ".join(words[:max_words])

    selected.sort(key=lambda x: x["position"])
    output_words = []
    for chunk in selected:
        output_words.extend(chunk["text"].split())
        if len(output_words) >= max_words:
            break
    return " ".join(output_words[:max_words])


def run_sensitivity(configs: list[dict], budget: int, limit_docs: int | None = None):
    """Run PAC with different W-curve parameters on all documents."""
    documents = iter_documents(DATA_DIR, limit=limit_docs)
    if not documents:
        raise SystemExit(f"No documents in {DATA_DIR}")

    print(f"Documents: {len(documents)}")
    print(f"Configs: {[c['name'] for c in configs]}")
    print(f"Budget: {budget}")
    print("-" * 60)

    results = []

    for config in configs:
        print(f"\n=== Config: {config['name']} (intro={config['intro']}, concl={config['conclusion']}, res={config['results_peak']}, floor={config['floor']}) ===")

        for doc in documents:
            doc_id = doc.doc_id
            out_dir = DOCUMENTS_DIR / doc_id / "chunks_sensitivity"
            out_file = out_dir / f"pac_{config['name']}.txt"

            if out_file.exists():
                # Read existing for stats
                chunk_text = out_file.read_text()
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                start = time.time()
                chunk_text = pac_with_params(doc.text, budget, config)
                elapsed = time.time() - start
                out_file.write_text(chunk_text, encoding="utf-8")

            out_words = len(chunk_text.split())
            original_words = len(doc.text.split())

            results.append({
                "config": config["name"],
                "doc_id": doc_id,
                "original_words": original_words,
                "output_words": out_words,
                "reduction_pct": round((1 - out_words / original_words) * 100, 1) if original_words else 0,
            })

    # Save results summary
    summary_file = RESULTS_DIR / "sensitivity_analysis.json"
    summary_file.write_text(json.dumps({
        "configs": configs,
        "budget": budget,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"\nSensitivity results saved to: {summary_file}")

    # Print comparison table
    from collections import defaultdict
    config_words = defaultdict(list)
    for r in results:
        config_words[r["config"]].append(r["output_words"])

    print(f"\n{'Config':<15} {'Avg Words':>10} {'Min':>8} {'Max':>8}")
    print("-" * 45)
    for name, words in config_words.items():
        avg = sum(words) / len(words) if words else 0
        print(f"{name:<15} {avg:>10.0f} {min(words):>8} {max(words):>8}")


def main():
    parser = argparse.ArgumentParser(description="W-curve sensitivity analysis.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config names to run (default: all)")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--limit-docs", type=int, default=None)

    args = parser.parse_args()

    if args.configs:
        configs = [c for c in SENSITIVITY_CONFIGS if c["name"] in args.configs]
    else:
        configs = SENSITIVITY_CONFIGS

    limit = None if args.limit_docs == 0 else args.limit_docs
    run_sensitivity(configs, args.budget, limit)


if __name__ == "__main__":
    main()
