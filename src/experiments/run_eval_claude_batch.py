#!/usr/bin/env python3
"""
Claude slide evaluation in controlled batches.
Submits one batch at a time so you can monitor cost.

Usage:
    python run_eval_claude_batch.py --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.shared import (
    parse_json_response, read_pdf, find_source_pdf,
    get_anthropic_client, create_anthropic_batch, poll_anthropic_batch,
)
from experiments.run_evaluation_multi import (
    collect_eval_tasks, output_to_text, get_eval_prompt,
    EvaluationResult, DOCUMENTS_DIR,
)
from config import JUDGE_MODELS, DATA_DIR


def run_one_batch(batch_size: int = 500):
    judges = [j for j in JUDGE_MODELS if j["provider"] == "anthropic"]
    all_tasks = collect_eval_tasks("slides", judges)

    if not all_tasks:
        print("All Claude slide evaluations complete!")
        return 0

    print(f"Total remaining: {len(all_tasks)}")
    print(f"This batch: {min(batch_size, len(all_tasks))}")

    tasks = all_tasks[:batch_size]

    client = get_anthropic_client()
    if not client:
        print("Anthropic client not available")
        return -1

    # Build batch requests
    print("Building requests (reading PDFs)...")
    batch_requests = []
    task_map = {}
    skipped = 0

    for idx, t in enumerate(tasks):
        doc_id = t["doc_id"]
        gen_data = t["gen_data"]
        pdf_path = find_source_pdf(doc_id, DATA_DIR)
        if not pdf_path:
            skipped += 1
            continue
        doc_text = read_pdf(pdf_path)
        output_text = output_to_text("slides", gen_data.get("output", gen_data.get("slides", [])))
        prompt = get_eval_prompt("slides", doc_text, output_text)

        custom_id = f"ev_{idx}"
        batch_requests.append({
            "custom_id": custom_id,
            "model": t["judge"]["id"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        })
        task_map[custom_id] = t

    if skipped:
        print(f"Skipped {skipped} (missing PDFs)")

    print(f"Submitting {len(batch_requests)} requests...")
    batch_id = create_anthropic_batch(client, batch_requests, "slides eval")
    results = poll_anthropic_batch(client, batch_id)

    # Process results
    success = 0
    for r in results:
        t = task_map.get(r["custom_id"])
        if not t:
            continue
        gen_data = t["gen_data"]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            if r["result"]["type"] != "succeeded":
                continue
            text = r["result"]["message"]["content"][0]["text"]
            eval_data = parse_json_response(text)
            scores = {k: eval_data.get(k, {}) for k in eval_data if isinstance(eval_data.get(k), dict)}

            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=gen_data.get("method", ""), doc_id=t["doc_id"],
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_data.get("model", ""),
                judge_model=t["judge"]["id"], judge_provider="anthropic",
                task="slides", scores=scores,
                overall_score=eval_data.get("overall_score", 0),
                key_strengths=eval_data.get("key_strengths", []),
                key_weaknesses=eval_data.get("key_weaknesses", []),
                document_word_count=0, evaluation_time_seconds=0,
                timestamp=timestamp, success=True,
            )
            eval_dir = DOCUMENTS_DIR / t["doc_id"] / "evaluations" / "slides"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / t["eval_filename"]).write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False)
            )
            success += 1
        except Exception:
            pass

    remaining = len(all_tasks) - success
    print(f"\nBatch done: {success}/{len(batch_requests)} successful")
    print(f"Remaining: {remaining}")
    print(f"\nCheck your Anthropic billing, then run again for next batch.")
    return remaining


def main():
    parser = argparse.ArgumentParser(description="Claude eval in controlled batches.")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    run_one_batch(args.batch_size)


if __name__ == "__main__":
    main()
