#!/usr/bin/env python3
"""
Parallel slide/summary evaluation: Gemini parallel + Claude batch.

Usage:
    python run_eval_parallel.py --task slides --workers 15
    python run_eval_parallel.py --task summary --workers 15
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
    generate, parse_json_response, read_pdf, find_source_pdf, slides_to_text,
    run_parallel_google, get_anthropic_client,
    create_anthropic_batch, poll_anthropic_batch,
)
from experiments.run_evaluation_multi import (
    collect_eval_tasks, output_to_text, get_eval_prompt,
    EvaluationResult, DOCUMENTS_DIR,
)
from config import JUDGE_MODELS, DATA_DIR


def run_parallel_eval(task_name: str, workers: int = 15):
    # Only use Gemini + Claude (skip OpenAI)
    judges = [j for j in JUDGE_MODELS if j["provider"] in ("google", "anthropic")]
    all_tasks = collect_eval_tasks(task_name, judges)

    if not all_tasks:
        print("Nothing to evaluate.")
        return

    by_provider = {"google": [], "anthropic": []}
    for t in all_tasks:
        by_provider[t["judge"]["provider"]].append(t)

    print(f"Task: {task_name}")
    print(f"Gemini evals: {len(by_provider['google'])}")
    print(f"Claude evals: {len(by_provider['anthropic'])}")

    # ── Gemini: parallel ──
    if by_provider["google"]:
        print(f"\n{'='*60}")
        print(f"Gemini judge ({len(by_provider['google'])} evals, {workers} parallel)")
        print(f"{'='*60}")

        def _gemini_worker(t):
            doc_id = t["doc_id"]
            gen_data = t["gen_data"]
            judge = t["judge"]
            method = gen_data.get("method", "unknown")

            pdf_path = find_source_pdf(doc_id, DATA_DIR)
            if not pdf_path:
                return {"error": f"PDF not found for {doc_id}"}

            doc_text = read_pdf(pdf_path)
            output_text = output_to_text(task_name, gen_data.get("output", gen_data.get("slides", [])))
            prompt = get_eval_prompt(task_name, doc_text, output_text)

            response_text, elapsed = generate(
                provider=judge["provider"], prompt=prompt, model=judge["id"],
            )
            eval_data = parse_json_response(response_text)

            scores = {k: eval_data.get(k, {}) for k in eval_data if isinstance(eval_data.get(k), dict)}
            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=method, doc_id=doc_id,
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_data.get("model", ""),
                judge_model=judge["id"], judge_provider=judge["provider"],
                task=task_name, scores=scores,
                overall_score=eval_data.get("overall_score", 0),
                key_strengths=eval_data.get("key_strengths", []),
                key_weaknesses=eval_data.get("key_weaknesses", []),
                document_word_count=len(doc_text.split()),
                evaluation_time_seconds=elapsed,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), success=True,
            )

            eval_dir = DOCUMENTS_DIR / doc_id / "evaluations" / task_name
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / t["eval_filename"]).write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False)
            )
            return {"success": True, "score": result.overall_score}

        start = time.time()
        results = run_parallel_google(by_provider["google"], _gemini_worker, max_workers=workers)
        elapsed = time.time() - start
        success = sum(1 for r in results if r and r.get("success"))
        print(f"Gemini done: {success}/{len(by_provider['google'])} in {elapsed:.0f}s")

    # ── Claude: batch ──
    if by_provider["anthropic"]:
        print(f"\n{'='*60}")
        print(f"Claude judge ({len(by_provider['anthropic'])} evals, batch)")
        print(f"{'='*60}")

        client = get_anthropic_client()
        if not client:
            print("Anthropic client not available — skipping")
            return

        batch_requests = []
        task_map = {}
        skipped = 0

        for idx, t in enumerate(by_provider["anthropic"]):
            doc_id = t["doc_id"]
            gen_data = t["gen_data"]
            pdf_path = find_source_pdf(doc_id, DATA_DIR)
            if not pdf_path:
                skipped += 1
                continue
            doc_text = read_pdf(pdf_path)
            output_text = output_to_text(task_name, gen_data.get("output", gen_data.get("slides", [])))
            prompt = get_eval_prompt(task_name, doc_text, output_text)

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

        # Split into chunks of 500 to stay under 256MB limit
        CHUNK_SIZE = 500
        all_batch_results = []
        for chunk_start in range(0, len(batch_requests), CHUNK_SIZE):
            chunk = batch_requests[chunk_start:chunk_start + CHUNK_SIZE]
            print(f"  Submitting batch {chunk_start // CHUNK_SIZE + 1} ({len(chunk)} requests)...")
            batch_id = create_anthropic_batch(client, chunk, f"{task_name} eval batch {chunk_start // CHUNK_SIZE + 1}")
            chunk_results = poll_anthropic_batch(client, batch_id)
            all_batch_results.extend(chunk_results)

        success = 0
        for r in all_batch_results:
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
                    task=task_name, scores=scores,
                    overall_score=eval_data.get("overall_score", 0),
                    key_strengths=eval_data.get("key_strengths", []),
                    key_weaknesses=eval_data.get("key_weaknesses", []),
                    document_word_count=0, evaluation_time_seconds=0,
                    timestamp=timestamp, success=True,
                )
                eval_dir = DOCUMENTS_DIR / t["doc_id"] / "evaluations" / task_name
                eval_dir.mkdir(parents=True, exist_ok=True)
                (eval_dir / t["eval_filename"]).write_text(
                    json.dumps(asdict(result), indent=2, ensure_ascii=False)
                )
                success += 1
            except Exception:
                pass

        print(f"Claude done: {success}/{len(batch_requests)}")


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation (Gemini + Claude).")
    parser.add_argument("--task", choices=["slides", "summary"], default="slides")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    run_parallel_eval(args.task, args.workers)


if __name__ == "__main__":
    main()
