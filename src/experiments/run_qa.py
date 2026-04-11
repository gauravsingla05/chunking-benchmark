#!/usr/bin/env python3
"""
Q&A pipeline: Generate questions from full PDFs, then test if chunked text can answer them.

Stage 1: Generate 10 factual questions + ground-truth answers from full PDF (Gemini, parallel)
Stage 2: Answer questions using only chunked text (multiple models)
Stage 3: Evaluate correctness (automated comparison)

Usage:
    # Stage 1: Generate questions (parallel Gemini, fast)
    python run_qa.py --stage questions --workers 15

    # Stage 2: Answer from chunks (parallel Gemini + Claude batch)
    python run_qa.py --stage answers --workers 15

    # Stage 3: Evaluate correctness
    python run_qa.py --stage evaluate

    # All stages
    python run_qa.py --stage all --workers 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.shared import (
    generate, parse_json_response, read_pdf, find_source_pdf,
    get_google_client, get_anthropic_client, get_openai_client,
    run_parallel_google, create_anthropic_batch, poll_anthropic_batch,
    retry_with_backoff,
)
from config import DOCUMENTS_DIR, DATA_DIR, RESULTS_DIR, GENERATION_MODELS

# ─── Prompts ──────────────────────────────────────────────────

QUESTION_GEN_PROMPT = """You are an expert at creating factual comprehension questions from documents.

Read the FULL document below and generate exactly 10 questions that:
1. Have specific, factual answers found in the document
2. Cover different sections (intro, methodology, results, conclusion)
3. Include at least 3 questions about specific numbers/statistics
4. Include at least 2 questions about methodology
5. Include at least 2 questions about findings/conclusions
6. Have short, definitive answers (1-2 sentences max)

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "What was the sample size of the study?",
      "answer": "The study included 2,450 participants.",
      "category": "methodology",
      "source_section": "methods"
    }},
    ...
  ]
}}

Categories: "statistics", "methodology", "findings", "background", "conclusion"

DOCUMENT:
{document_text}"""

QA_ANSWER_PROMPT = """Answer each question below using ONLY the provided text. If the text does not contain enough information to answer, say "UNANSWERABLE".

Be specific and concise. Use exact numbers and facts from the text.

Return ONLY valid JSON:
{{
  "answers": [
    {{
      "id": 1,
      "answer": "Your answer based on the text",
      "confidence": "high|medium|low|unanswerable"
    }},
    ...
  ]
}}

TEXT:
{chunk_text}

QUESTIONS:
{questions_text}"""

QA_JUDGE_PROMPT = """You are evaluating whether an answer is correct by comparing it to the ground truth.

For each question, compare the candidate answer to the correct answer and judge:
- "correct": The candidate answer conveys the same key facts as the ground truth
- "partial": The candidate answer is partially correct but missing key details
- "incorrect": The candidate answer is wrong or contradicts the ground truth
- "unanswerable": The candidate correctly identified the question as unanswerable from the text

Return ONLY valid JSON:
{{
  "judgments": [
    {{
      "id": 1,
      "verdict": "correct|partial|incorrect|unanswerable",
      "explanation": "brief reason"
    }},
    ...
  ]
}}

QUESTIONS AND ANSWERS:
{qa_pairs}"""


# ─── Stage 1: Question Generation ─────────────────────────────

def generate_questions_parallel(workers: int = 15):
    """Generate questions from full PDFs using parallel Gemini calls."""
    qa_dir = RESULTS_DIR / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    questions_dir = qa_dir / "questions"
    questions_dir.mkdir(exist_ok=True)

    # Collect docs that need questions
    tasks = []
    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir() or not (doc_dir / "chunks").exists():
            continue
        doc_id = doc_dir.name
        out_file = questions_dir / f"{doc_id}.json"
        if out_file.exists():
            continue
        pdf_path = find_source_pdf(doc_id, DATA_DIR)
        if not pdf_path:
            continue
        tasks.append({"doc_id": doc_id, "pdf_path": pdf_path, "out_file": out_file})

    if not tasks:
        print("All questions already generated.")
        return

    print(f"Generating questions for {len(tasks)} documents ({workers} parallel workers)")

    def _worker(task):
        doc_text = read_pdf(task["pdf_path"])
        # Truncate to ~50K words to fit context
        words = doc_text.split()
        if len(words) > 50000:
            doc_text = " ".join(words[:50000])

        prompt = QUESTION_GEN_PROMPT.format(document_text=doc_text)
        response_text, elapsed = generate("google", prompt, "gemini-2.0-flash", max_tokens=2048)
        data = parse_json_response(response_text)

        result = {
            "doc_id": task["doc_id"],
            "questions": data.get("questions", []),
            "question_count": len(data.get("questions", [])),
            "document_words": len(words),
            "generation_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        task["out_file"].write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    results = run_parallel_google(tasks, _worker, max_workers=workers)
    success = sum(1 for r in results if r and "error" not in r)
    print(f"\nQuestions generated: {success}/{len(tasks)}")


# ─── Stage 2: Answer Questions from Chunks ─────────────────────

def answer_questions_parallel(workers: int = 15):
    """Answer questions using chunked text. Gemini parallel + Claude batch."""
    qa_dir = RESULTS_DIR / "qa"
    questions_dir = qa_dir / "questions"
    answers_dir = qa_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    # Load all question files
    question_files = {f.stem: json.loads(f.read_text()) for f in questions_dir.glob("*.json")}
    if not question_files:
        print("No questions found. Run --stage questions first.")
        return

    # Collect answer tasks: for each (doc, method, model) that doesn't have answers yet
    models_to_run = [m for m in GENERATION_MODELS if m["provider"] in ("google", "anthropic")]
    tasks_by_provider = {"google": [], "anthropic": []}

    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if doc_id not in question_files:
            continue

        chunks_dir = doc_dir / "chunks"
        if not chunks_dir.exists():
            continue

        questions = question_files[doc_id].get("questions", [])
        if not questions:
            continue

        questions_text = "\n".join(f"Q{q['id']}: {q['question']}" for q in questions)

        for chunk_file in sorted(chunks_dir.glob("*.txt")):
            method = chunk_file.stem
            for model in models_to_run:
                safe_model = model["name"].replace(" ", "_").replace("-", "").lower()
                out_file = answers_dir / f"{doc_id}__{method}__{safe_model}.json"
                if out_file.exists():
                    continue

                tasks_by_provider[model["provider"]].append({
                    "doc_id": doc_id,
                    "method": method,
                    "chunk_file": chunk_file,
                    "model": model,
                    "questions": questions,
                    "questions_text": questions_text,
                    "out_file": out_file,
                })

    total = sum(len(t) for t in tasks_by_provider.values())
    if total == 0:
        print("All answers already generated.")
        return

    # ── Gemini: parallel ──
    if tasks_by_provider["google"]:
        gemini_tasks = tasks_by_provider["google"]
        print(f"\nGemini answers: {len(gemini_tasks)} tasks ({workers} parallel)")

        def _gemini_worker(task):
            chunk_text = task["chunk_file"].read_text(encoding="utf-8", errors="ignore")
            prompt = QA_ANSWER_PROMPT.format(
                chunk_text=chunk_text,
                questions_text=task["questions_text"],
            )
            response_text, elapsed = generate("google", prompt, "gemini-2.0-flash", max_tokens=2048)
            data = parse_json_response(response_text)

            result = {
                "doc_id": task["doc_id"],
                "method": task["method"],
                "model": task["model"]["id"],
                "answers": data.get("answers", []),
                "generation_time": elapsed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            task["out_file"].write_text(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        results = run_parallel_google(gemini_tasks, _gemini_worker, max_workers=workers)
        success = sum(1 for r in results if r and "error" not in r)
        print(f"Gemini answers: {success}/{len(gemini_tasks)}")

    # ── Anthropic: batch ──
    if tasks_by_provider["anthropic"]:
        anthropic_tasks = tasks_by_provider["anthropic"]
        print(f"\nClaude answers: {len(anthropic_tasks)} tasks (batch)")

        client = get_anthropic_client()
        if not client:
            print("Anthropic client not available — skipping Claude answers")
        else:
            batch_requests = []
            task_map = {}
            for idx, task in enumerate(anthropic_tasks):
                chunk_text = task["chunk_file"].read_text(encoding="utf-8", errors="ignore")
                prompt = QA_ANSWER_PROMPT.format(
                    chunk_text=chunk_text,
                    questions_text=task["questions_text"],
                )
                custom_id = f"qa_{idx}"
                batch_requests.append({
                    "custom_id": custom_id,
                    "model": task["model"]["id"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                })
                task_map[custom_id] = task

            batch_id = create_anthropic_batch(client, batch_requests, "QA answers")
            results = poll_anthropic_batch(client, batch_id)

            success = 0
            for r in results:
                task = task_map.get(r["custom_id"])
                if not task:
                    continue
                try:
                    if r["result"]["type"] != "succeeded":
                        continue
                    text = r["result"]["message"]["content"][0]["text"]
                    data = parse_json_response(text)
                    result = {
                        "doc_id": task["doc_id"],
                        "method": task["method"],
                        "model": task["model"]["id"],
                        "answers": data.get("answers", []),
                        "generation_time": 0,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    task["out_file"].write_text(json.dumps(result, indent=2, ensure_ascii=False))
                    success += 1
                except Exception:
                    pass
            print(f"Claude answers: {success}/{len(anthropic_tasks)}")


# ─── Stage 3: Evaluate Correctness ────────────────────────────

def evaluate_answers(workers: int = 15):
    """Compare answers to ground truth using Gemini as judge (parallel)."""
    qa_dir = RESULTS_DIR / "qa"
    questions_dir = qa_dir / "questions"
    answers_dir = qa_dir / "answers"
    eval_dir = qa_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    question_files = {f.stem: json.loads(f.read_text()) for f in questions_dir.glob("*.json")}
    answer_files = list(answers_dir.glob("*.json"))

    if not answer_files:
        print("No answers found. Run --stage answers first.")
        return

    # Collect eval tasks
    tasks = []
    for ans_file in sorted(answer_files):
        eval_file = eval_dir / ans_file.name
        if eval_file.exists():
            continue

        ans_data = json.loads(ans_file.read_text())
        doc_id = ans_data.get("doc_id", "")
        if doc_id not in question_files:
            continue

        questions = question_files[doc_id].get("questions", [])
        answers = ans_data.get("answers", [])
        if not questions or not answers:
            continue

        # Build comparison pairs
        answer_map = {a["id"]: a for a in answers}
        qa_pairs = []
        for q in questions:
            ans = answer_map.get(q["id"], {})
            qa_pairs.append(
                f"Q{q['id']}: {q['question']}\n"
                f"  Correct answer: {q['answer']}\n"
                f"  Candidate answer: {ans.get('answer', 'NO ANSWER')}\n"
                f"  Confidence: {ans.get('confidence', 'unknown')}"
            )

        tasks.append({
            "ans_file": ans_file,
            "eval_file": eval_file,
            "ans_data": ans_data,
            "qa_pairs_text": "\n\n".join(qa_pairs),
            "questions": questions,
        })

    if not tasks:
        print("All evaluations already complete.")
        _print_results(eval_dir, question_files)
        return

    print(f"Evaluating {len(tasks)} answer sets ({workers} parallel)")

    def _eval_worker(task):
        prompt = QA_JUDGE_PROMPT.format(qa_pairs=task["qa_pairs_text"])
        response_text, elapsed = generate("google", prompt, "gemini-2.0-flash", max_tokens=2048)
        data = parse_json_response(response_text)

        result = {
            "doc_id": task["ans_data"]["doc_id"],
            "method": task["ans_data"]["method"],
            "model": task["ans_data"]["model"],
            "judgments": data.get("judgments", []),
            "evaluation_time": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        task["eval_file"].write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    results = run_parallel_google(tasks, _eval_worker, max_workers=workers)
    success = sum(1 for r in results if r and "error" not in r)
    print(f"Evaluated: {success}/{len(tasks)}")

    _print_results(eval_dir, question_files)


def _print_results(eval_dir: Path, question_files: dict):
    """Print Q&A accuracy results grouped by method and model."""
    from collections import defaultdict

    # method -> model -> {correct, partial, incorrect, unanswerable, total}
    stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "unanswerable": 0, "total": 0}))

    # Also by category
    category_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for eval_file in eval_dir.glob("*.json"):
        data = json.loads(eval_file.read_text())
        method = data.get("method", "unknown")
        model = data.get("model", "unknown")
        doc_id = data.get("doc_id", "")
        questions = {q["id"]: q for q in question_files.get(doc_id, {}).get("questions", [])}

        for j in data.get("judgments", []):
            verdict = j.get("verdict", "incorrect")
            stats[method][model][verdict] = stats[method][model].get(verdict, 0) + 1
            stats[method][model]["total"] += 1

            # Category breakdown
            q = questions.get(j["id"], {})
            cat = q.get("category", "unknown")
            category_stats[method][cat]["total"] += 1
            if verdict == "correct":
                category_stats[method][cat]["correct"] += 1

    if not stats:
        print("No evaluation results yet.")
        return

    print(f"\n{'='*70}")
    print("Q&A ACCURACY RESULTS")
    print(f"{'='*70}")

    # By method × model
    print(f"\n{'Method':<25} {'Model':<25} {'Correct':>8} {'Partial':>8} {'Wrong':>8} {'N/A':>6} {'Acc%':>7}")
    print("-" * 90)

    for method in sorted(stats):
        for model in sorted(stats[method]):
            s = stats[method][model]
            total = s["total"]
            correct = s.get("correct", 0)
            partial = s.get("partial", 0)
            incorrect = s.get("incorrect", 0)
            na = s.get("unanswerable", 0)
            acc = (correct + 0.5 * partial) / total * 100 if total else 0
            print(f"{method:<25} {model:<25} {correct:>8} {partial:>8} {incorrect:>8} {na:>6} {acc:>6.1f}%")

    # Summary by method (averaged across models)
    print(f"\n{'Method':<25} {'Avg Accuracy':>12}")
    print("-" * 40)
    for method in sorted(stats):
        accs = []
        for model, s in stats[method].items():
            total = s["total"]
            if total:
                accs.append((s.get("correct", 0) + 0.5 * s.get("partial", 0)) / total * 100)
        if accs:
            print(f"{method:<25} {sum(accs)/len(accs):>11.1f}%")

    # Save summary
    summary_file = RESULTS_DIR / "qa" / "qa_results_summary.json"
    summary_file.write_text(json.dumps({
        "by_method_model": {m: {mod: dict(s) for mod, s in models.items()} for m, models in stats.items()},
        "by_method_category": {m: {cat: dict(s) for cat, s in cats.items()} for m, cats in category_stats.items()},
    }, indent=2))
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Q&A evaluation pipeline.")
    parser.add_argument("--stage", choices=["questions", "answers", "evaluate", "all"], default="all")
    parser.add_argument("--workers", type=int, default=15, help="Parallel workers for Gemini")

    args = parser.parse_args()

    if args.stage in ("questions", "all"):
        print(f"\n{'='*60}")
        print("STAGE 1: Generate Questions from Full PDFs")
        print(f"{'='*60}")
        generate_questions_parallel(args.workers)

    if args.stage in ("answers", "all"):
        print(f"\n{'='*60}")
        print("STAGE 2: Answer Questions from Chunks")
        print(f"{'='*60}")
        answer_questions_parallel(args.workers)

    if args.stage in ("evaluate", "all"):
        print(f"\n{'='*60}")
        print("STAGE 3: Evaluate Answer Correctness")
        print(f"{'='*60}")
        evaluate_answers(args.workers)


if __name__ == "__main__":
    main()
