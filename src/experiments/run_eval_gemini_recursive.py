#!/usr/bin/env python3
"""
Focused Gemini slide judging for the recursive_character baseline only.

The general-purpose run_evaluation_multi.py with --judge gemini-2.0-flash
queues 3,000+ tasks because it also picks up missing legacy judgments from
prior runs. For the PeerJ revision we only need to judge the new
recursive_character method's slides; everything else already has Gemini
judgments stored under documents/<doc>/evaluations_gemini/ (legacy) or
documents/<doc>/evaluations/slides/ (current).

This script collects only the recursive_character slide files and judges
them with Gemini 2.0 Flash in parallel.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, DOCUMENTS_DIR  # noqa: E402
from experiments.shared import (  # noqa: E402
    find_source_pdf, generate, parse_json_response, read_pdf,
)
from experiments.run_evaluation_multi import (  # noqa: E402
    EvaluationResult, SLIDE_EVAL_PROMPT, get_eval_filename, output_to_text,
)

JUDGE = {
    "id": "gemini-2.0-flash",
    "provider": "google",
    "name": "Gemini Flash",
}
METHOD_PREFIX = "recursive_character__"


def collect_recursive_tasks() -> list[dict]:
    tasks = []
    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        slides_dir = doc_dir / "slides"
        if not slides_dir.exists():
            continue
        eval_dir = doc_dir / "evaluations" / "slides"
        existing = set()
        if eval_dir.exists():
            existing = {f.stem for f in eval_dir.glob("*.json")}

        for gen_file in sorted(slides_dir.glob(f"{METHOD_PREFIX}*.json")):
            try:
                gen_data = json.loads(gen_file.read_text())
                if not gen_data.get("success", False):
                    continue
            except Exception:
                continue

            eval_filename = get_eval_filename(gen_file.name, JUDGE["name"])
            if Path(eval_filename).stem in existing:
                continue
            tasks.append({
                "doc_id": doc_dir.name,
                "gen_file": gen_file,
                "gen_data": gen_data,
                "eval_filename": eval_filename,
                "eval_dir": eval_dir,
            })
    return tasks


def judge_one(task: dict) -> EvaluationResult | None:
    doc_id = task["doc_id"]
    gen_data = task["gen_data"]
    method = gen_data.get("method", "unknown")
    gen_model = gen_data.get("model", "unknown")
    try:
        pdf_path = find_source_pdf(doc_id, DATA_DIR)
        if not pdf_path:
            return None
        doc_text = read_pdf(pdf_path)
        word_count = len(doc_text.split())
        output_text = output_to_text("slides", gen_data.get("output", gen_data.get("slides", [])))
        prompt = SLIDE_EVAL_PROMPT.format(document_text=doc_text, output_text=output_text)
        response_text, elapsed = generate(JUDGE["provider"], prompt, JUDGE["id"])
        eval_data = parse_json_response(response_text)

        metric_keys = ["completeness", "accuracy", "statistics_retention",
                       "coherence", "relevance", "coverage_balance"]
        scores = {k: eval_data.get(k, {}) for k in metric_keys}

        result = EvaluationResult(
            generation_file=str(task["gen_file"]),
            method=method,
            doc_id=doc_id,
            run_number=gen_data.get("run_number", 1),
            generator_model=gen_model,
            judge_model=JUDGE["id"],
            judge_provider=JUDGE["provider"],
            task="slides",
            scores=scores,
            overall_score=eval_data.get("overall_score", 0),
            key_strengths=eval_data.get("key_strengths", []),
            key_weaknesses=eval_data.get("key_weaknesses", []),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            evaluation_time_seconds=elapsed,
            success=True,
            error=None,
            document_word_count=word_count,
        )
        eval_dir = task["eval_dir"]
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / task["eval_filename"]).write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False)
        )
        return result
    except Exception as exc:
        print(f"  fail {doc_id}/{Path(task['gen_file']).name}: {exc}", flush=True)
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    tasks = collect_recursive_tasks()
    print(f"Pending Gemini judgments for recursive_character: {len(tasks)}")
    if not tasks:
        return 0

    start = time.time()
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r and r.success:
                ok += 1
            else:
                fail += 1
            if i % 25 == 0 or i == len(tasks):
                rate = i / (time.time() - start)
                eta = (len(tasks) - i) / rate if rate else 0
                print(f"  [{i}/{len(tasks)}] ok={ok} fail={fail} rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

    print(f"Done. ok={ok}, fail={fail}, elapsed={(time.time()-start):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
