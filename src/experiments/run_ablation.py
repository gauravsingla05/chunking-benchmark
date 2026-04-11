#!/usr/bin/env python3
"""
Ablation study: run chunking with multiple word budgets (1K, 2K, 3K, 5K).

Tests how chunk budget affects downstream generation quality.
Only re-chunks + re-generates for new budgets; reuses existing 2K results.

Usage:
    python run_ablation.py                    # Chunk all budgets
    python run_ablation.py --budgets 1000 5000  # Specific budgets
    python run_ablation.py --generate --batch  # Also generate slides
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ABLATION_BUDGETS, CHUNKING_METHODS, DOCUMENTS_DIR, DATA_DIR, RESULTS_DIR
from utils.text_io import iter_documents

# Import chunking methods
import methods.truncation  # noqa: F401
import methods.fixed_size_first_last  # noqa: F401
import methods.semantic_breakpoint  # noqa: F401
import methods.pac_position_aware  # noqa: F401
from methods.registry import get_method


def run_ablation_chunking(budgets: list[int], limit_docs: int | None = None):
    """Run chunking for each budget, saving to budget-specific subdirectories."""
    documents = iter_documents(DATA_DIR, limit=limit_docs)
    if not documents:
        raise SystemExit(f"No documents found in {DATA_DIR}")

    print(f"Documents: {len(documents)}")
    print(f"Methods: {CHUNKING_METHODS}")
    print(f"Budgets: {budgets}")
    print(f"Total tasks: {len(documents) * len(CHUNKING_METHODS) * len(budgets)}")
    print("-" * 60)

    for budget in budgets:
        print(f"\n=== Budget: {budget} words ===")
        processed = 0

        for doc in documents:
            doc_id = doc.doc_id
            chunks_dir = DOCUMENTS_DIR / doc_id / f"chunks_budget_{budget}"

            for method_name in CHUNKING_METHODS:
                out_file = chunks_dir / f"{method_name}.txt"
                if out_file.exists():
                    continue  # resume

                chunks_dir.mkdir(parents=True, exist_ok=True)
                spec = get_method(method_name)
                start = time.time()
                chunk_text = spec.chunker(doc.text, budget)
                elapsed = time.time() - start

                out_file.write_text(chunk_text, encoding="utf-8")
                processed += 1

                out_words = len(chunk_text.split())
                reduction = (1 - out_words / len(doc.text.split())) * 100 if doc.text.split() else 0
                print(f"  {method_name:25s} | {doc_id[:30]:30s} | {out_words:5d}w ({reduction:.0f}% reduced) | {elapsed:.2f}s")

        print(f"  Processed {processed} new chunks for budget={budget}")

    # Save ablation metadata
    meta = {
        "budgets": budgets,
        "methods": CHUNKING_METHODS,
        "document_count": len(documents),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_file = RESULTS_DIR / "ablation_config.json"
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"\nAblation metadata saved to: {meta_file}")


def main():
    parser = argparse.ArgumentParser(description="Ablation study: chunking with multiple budgets.")
    parser.add_argument("--budgets", nargs="+", type=int, default=ABLATION_BUDGETS,
                        help=f"Word budgets to test (default: {ABLATION_BUDGETS})")
    parser.add_argument("--limit-docs", type=int, default=None, help="Limit documents (0=all)")

    args = parser.parse_args()
    limit = None if args.limit_docs == 0 else args.limit_docs
    run_ablation_chunking(args.budgets, limit)


if __name__ == "__main__":
    main()
