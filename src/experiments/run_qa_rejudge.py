#!/usr/bin/env python3
"""
Re-judge existing Q&A answer files with a different judge (GPT-4o-mini or
Claude Sonnet 4) to address the reviewer objection that the original Gemini
question-generator/Gemini-judge pipeline is circular.

Reads existing answers from `results/qa/answers/<doc>__<method>__<gen>.json`
and writes new judgments to `results/qa/evaluations_<judge_short>/` so the
original Gemini-judged evaluations under `results/qa/evaluations/` remain
untouched for direct comparison.

Usage:
    python run_qa_rejudge.py --judge gpt-4o-mini --dry-run
    python run_qa_rejudge.py --judge gpt-4o-mini --workers 10
    python run_qa_rejudge.py --judge claude-sonnet-4 --workers 5

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment depending
on the chosen judge.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RESULTS_DIR  # noqa: E402
from experiments.shared import generate, parse_json_response  # noqa: E402

# The same prompt the Gemini judge uses, copied to keep the comparison clean.
QA_JUDGE_PROMPT = """You are evaluating whether an answer is correct by comparing it to the ground truth.

For each question, compare the candidate answer to the correct answer and judge:
- "correct": The candidate answer conveys the same key facts as the ground truth
- "partial": The candidate answer is partially correct but missing key details
- "incorrect": The candidate answer is wrong or contradicts the ground truth
- "unanswerable": The candidate correctly identified the question as unanswerable from the text

Respond ONLY with valid JSON in this exact format:
{{
  "judgments": [
    {{
      "id": 1,
      "verdict": "correct|partial|incorrect|unanswerable",
      "explanation": "Brief reason"
    }}
  ]
}}

QUESTIONS AND ANSWERS TO EVALUATE:
{qa_pairs}
"""

JUDGE_REGISTRY = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "dir_suffix": "gpt4o_mini",
        "estimated_cost_per_call_usd": 0.0008,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "dir_suffix": "claude_sonnet",
        "estimated_cost_per_call_usd": 0.012,
    },
}


def collect_tasks(judge_dir_suffix: str) -> tuple[list[dict], dict]:
    qa_dir = RESULTS_DIR / "qa"
    questions_dir = qa_dir / "questions"
    answers_dir = qa_dir / "answers"
    eval_dir = qa_dir / f"evaluations_{judge_dir_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    question_files = {f.stem: json.loads(f.read_text()) for f in questions_dir.glob("*.json")}
    answer_files = sorted(answers_dir.glob("*.json"))

    tasks: list[dict] = []
    for ans_file in answer_files:
        eval_file = eval_dir / ans_file.name
        if eval_file.exists():
            continue
        try:
            ans_data = json.loads(ans_file.read_text())
        except Exception:
            continue
        doc_id = ans_data.get("doc_id", "")
        if doc_id not in question_files:
            continue
        questions = question_files[doc_id].get("questions", [])
        answers = ans_data.get("answers", [])
        if not questions or not answers:
            continue

        answer_map = {a["id"]: a for a in answers}
        qa_lines = []
        for q in questions:
            ans = answer_map.get(q["id"], {})
            qa_lines.append(
                f"Q{q['id']}: {q['question']}\n"
                f"  Correct answer: {q['answer']}\n"
                f"  Candidate answer: {ans.get('answer', 'NO ANSWER')}\n"
                f"  Confidence: {ans.get('confidence', 'unknown')}"
            )

        tasks.append({
            "ans_file": ans_file,
            "eval_file": eval_file,
            "ans_data": ans_data,
            "qa_pairs_text": "\n\n".join(qa_lines),
            "questions": questions,
        })

    return tasks, question_files


def judge_one(task: dict, provider: str, model_id: str) -> dict:
    prompt = QA_JUDGE_PROMPT.format(qa_pairs=task["qa_pairs_text"])
    try:
        response_text, elapsed = generate(provider, prompt, model_id, max_tokens=2048)
        data = parse_json_response(response_text)
        result = {
            "doc_id": task["ans_data"]["doc_id"],
            "method": task["ans_data"]["method"],
            "model": task["ans_data"]["model"],
            "judge": model_id,
            "judgments": data.get("judgments", []),
            "evaluation_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        task["eval_file"].write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as exc:
        return {"error": str(exc), "ans_file": str(task["ans_file"])}


def summarize(eval_dir: Path, question_files: dict, judge_label: str) -> None:
    from collections import defaultdict

    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "unanswerable": 0, "total": 0}))
    for eval_file in eval_dir.glob("*.json"):
        try:
            data = json.loads(eval_file.read_text())
        except Exception:
            continue
        method = data.get("method", "unknown")
        gen = data.get("model", "unknown")
        for j in data.get("judgments", []):
            verdict = j.get("verdict", "incorrect")
            if verdict not in {"correct", "partial", "incorrect", "unanswerable"}:
                continue
            stats[method][gen][verdict] += 1
            stats[method][gen]["total"] += 1

    print(f"\n=== Q&A accuracy under {judge_label} judge ===")
    print(f"{'Method':<25} {'Generator':<35} {'Acc%':>7}")
    for method in sorted(stats):
        for gen in sorted(stats[method]):
            s = stats[method][gen]
            total = s["total"] or 1
            acc = 100.0 * (s["correct"] + 0.5 * s["partial"]) / total
            print(f"{method:<25} {gen:<35} {acc:>6.2f}")

    print(f"\n=== Aggregate per method ({judge_label}) ===")
    per_method_totals = defaultdict(lambda: {"correct": 0, "partial": 0, "total": 0})
    for method, gens in stats.items():
        for gen, s in gens.items():
            per_method_totals[method]["correct"] += s["correct"]
            per_method_totals[method]["partial"] += s["partial"]
            per_method_totals[method]["total"] += s["total"]
    method_order = sorted(per_method_totals.keys(), key=lambda m: -(per_method_totals[m]["correct"] + 0.5 * per_method_totals[m]["partial"]) / max(1, per_method_totals[m]["total"]))
    for method in method_order:
        s = per_method_totals[method]
        total = s["total"] or 1
        acc = 100.0 * (s["correct"] + 0.5 * s["partial"]) / total
        print(f"  {method:<25} acc = {acc:5.2f}%   n = {total}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=list(JUDGE_REGISTRY.keys()), default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="Count tasks and estimate cost without making API calls")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N tasks (for testing)")
    args = ap.parse_args()

    judge_cfg = JUDGE_REGISTRY[args.judge]
    tasks, question_files = collect_tasks(judge_cfg["dir_suffix"])
    eval_dir = RESULTS_DIR / "qa" / f"evaluations_{judge_cfg['dir_suffix']}"

    print(f"Judge: {args.judge} (provider={judge_cfg['provider']}, model={judge_cfg['model_id']})")
    print(f"Existing eval files in {eval_dir.name}/: {len(list(eval_dir.glob('*.json')))}")
    print(f"Pending tasks: {len(tasks)}")
    cost_estimate = len(tasks) * judge_cfg["estimated_cost_per_call_usd"]
    print(f"Estimated cost: ${cost_estimate:.2f}")

    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to execute.")
        return 0

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"Limiting to first {len(tasks)} tasks")

    if not tasks:
        print("Nothing to do.")
        summarize(eval_dir, question_files, args.judge)
        return 0

    start = time.time()
    success = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, t, judge_cfg["provider"], judge_cfg["model_id"]): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if "error" in result:
                failures += 1
            else:
                success += 1
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                eta = (len(tasks) - i) / rate if rate else 0
                print(f"  [{i}/{len(tasks)}] success={success} failed={failures} rate={rate:.1f}/s eta={eta:.0f}s")

    print(f"\nDone. success={success}, failed={failures}, elapsed={(time.time()-start):.1f}s")
    summarize(eval_dir, question_files, args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
