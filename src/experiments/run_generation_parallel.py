#!/usr/bin/env python3
"""
Parallel Gemini generation wrapper. Speeds up Google API calls 10-15x.

Usage:
    # Parallel Gemini summaries (15 workers)
    python run_generation_parallel.py --task summary --workers 15

    # Parallel Gemini slides
    python run_generation_parallel.py --task slides --workers 15
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

from experiments.shared import generate, parse_json_response, run_parallel_google
from experiments.run_generation import (
    collect_tasks, get_prompt, get_output_dir,
    GenerationResult, GENERATION_MODELS,
)
from config import DOCUMENTS_DIR


def run_parallel(task_name: str, workers: int = 15):
    """Run Gemini generation in parallel."""
    models = [m for m in GENERATION_MODELS if m["provider"] == "google"]
    tasks = collect_tasks(task_name, models)

    if not tasks:
        print("Nothing to do — all outputs exist.")
        return

    print(f"Task: {task_name}")
    print(f"Tasks to process: {len(tasks)} (Gemini, {workers} parallel workers)")

    def _worker(t):
        content = t["chunk_file"].read_text(encoding="utf-8", errors="ignore")
        prompt = get_prompt(task_name, content)
        model_cfg = t["model"]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            response_text, elapsed = generate(
                provider=model_cfg["provider"],
                prompt=prompt,
                model=model_cfg["id"],
            )
            data = parse_json_response(response_text)

            if task_name == "slides":
                output = data.get("slides", [])
                item_count = len(output)
            else:
                output = data
                item_count = len(data.get("sections", []))

            result = GenerationResult(
                input_file=str(t["chunk_file"]),
                method=t["method"], doc_id=t["doc_id"], run_number=t["run"],
                model=model_cfg["id"], provider=model_cfg["provider"], task=task_name,
                output=output, input_words=len(content.split()),
                output_item_count=item_count,
                generation_time_seconds=elapsed, timestamp=timestamp, success=True,
            )
        except Exception as e:
            result = GenerationResult(
                input_file=str(t["chunk_file"]),
                method=t["method"], doc_id=t["doc_id"], run_number=t["run"],
                model=model_cfg["id"], provider=model_cfg["provider"], task=task_name,
                output=[], input_words=len(content.split()),
                output_item_count=0,
                generation_time_seconds=0, timestamp=timestamp,
                success=False, error=str(e),
            )

        # Save
        out_dir = get_output_dir(task_name, t["doc_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / t["filename"]
        out_file.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return {"success": result.success, "error": result.error}

    start = time.time()
    results = run_parallel_google(tasks, _worker, max_workers=workers)
    elapsed = time.time() - start

    success = sum(1 for r in results if r and r.get("success"))
    failed = len(results) - success
    print(f"\nDone: {success}/{len(results)} successful, {failed} failed in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Parallel Gemini generation.")
    parser.add_argument("--task", choices=["slides", "summary"], default="summary")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    run_parallel(args.task, args.workers)


if __name__ == "__main__":
    main()
