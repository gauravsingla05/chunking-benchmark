#!/usr/bin/env python3
"""
Multi-judge evaluation: GPT-4o, Claude Sonnet, Gemini Flash evaluate all generated outputs.

Evaluates both slide decks and summaries against source PDFs.
Supports batch APIs for OpenAI/Anthropic (50% savings).

Usage:
    # Evaluate all slides with all judges, batch mode
    python run_evaluation_multi.py --task slides --batch

    # Single judge, realtime
    python run_evaluation_multi.py --task slides --judge gemini-2.0-flash --judge-provider google

    # Evaluate summaries
    python run_evaluation_multi.py --task summary --batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.shared import (
    generate, parse_json_response, read_pdf, find_source_pdf, slides_to_text,
    get_openai_client, get_anthropic_client,
    create_openai_batch, create_anthropic_batch,
    poll_openai_batch, poll_anthropic_batch,
)
from config import JUDGE_MODELS, DOCUMENTS_DIR, DATA_DIR, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT


# ─── Evaluation Prompts ──────────────────────────────────────

SLIDE_EVAL_PROMPT = """You are an expert evaluator assessing the quality of a slide presentation generated from a source document.

## FULL SOURCE DOCUMENT:
{document_text}

## GENERATED SLIDE DECK:
{output_text}

## Evaluation Task

Compare the slide deck against the FULL source document above. Evaluate on these criteria:

### 1. Completeness (1-5)
Does the deck cover the most important information from the source document?
- 5: Covers all critical facts and main points
- 3: Covers main points but misses some important details
- 1: Missing critical information

### 2. Accuracy (1-5)
Is the information factually consistent with the source? No hallucinations?
- 5: All facts accurate, faithful to source
- 3: Minor inaccuracies
- 1: Major factual errors or hallucinations

### 3. Statistics Retention (1-5)
Are key numbers, percentages, and quantitative findings preserved?
- 5: All important statistics included accurately
- 3: Some statistics included, some missing
- 1: Most statistics missing or wrong

### 4. Coherence (1-5)
Does the deck flow logically? Clear structure from intro to conclusion?
- 5: Excellent flow and logical structure
- 3: Adequate structure, some awkward transitions
- 1: Disorganized or confusing

### 5. Relevance (1-5)
Is the content relevant and valuable? Captures the document's main purpose?
- 5: All content directly relevant
- 3: Mostly relevant, some filler
- 1: Much irrelevant content

### 6. Coverage Balance (1-5)
Does the deck represent content from ALL parts of the document?
- 5: Balanced coverage across entire document
- 3: Some sections over/under-represented
- 1: Only covers beginning or specific sections

Return ONLY valid JSON:
{{
  "completeness": {{"score": <1-5>, "justification": "<brief>"}},
  "accuracy": {{"score": <1-5>, "justification": "<brief>"}},
  "statistics_retention": {{"score": <1-5>, "justification": "<brief>"}},
  "coherence": {{"score": <1-5>, "justification": "<brief>"}},
  "relevance": {{"score": <1-5>, "justification": "<brief>"}},
  "coverage_balance": {{"score": <1-5>, "justification": "<brief>"}},
  "overall_score": <average of 6 scores, 1 decimal>,
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_weaknesses": ["<weakness 1>", "<weakness 2>"]
}}"""

SUMMARY_EVAL_PROMPT = """You are an expert evaluator assessing the quality of a document summary.

## FULL SOURCE DOCUMENT:
{document_text}

## GENERATED SUMMARY:
{output_text}

## Evaluation Task

Compare the summary against the FULL source document. Evaluate on these criteria:

### 1. Completeness (1-5)
Does the summary cover the most important information?
- 5: Covers all critical facts and main points
- 3: Covers main points but misses important details
- 1: Missing critical information

### 2. Accuracy (1-5)
Is all information factually consistent with the source?
- 5: All facts accurate
- 3: Minor inaccuracies
- 1: Major factual errors

### 3. Statistics Retention (1-5)
Are key numbers and quantitative findings preserved?
- 5: All important statistics included
- 3: Some included, some missing
- 1: Most missing or wrong

### 4. Coherence (1-5)
Is the summary well-organized and logical?
- 5: Excellent structure and flow
- 3: Adequate structure
- 1: Disorganized

### 5. Conciseness (1-5)
Is the summary appropriately concise without losing key information?
- 5: Perfectly balanced — comprehensive yet concise
- 3: Somewhat wordy or too brief
- 1: Extremely verbose or missing critical details

### 6. Coverage Balance (1-5)
Does the summary represent ALL parts of the document?
- 5: Balanced coverage
- 3: Some sections over/under-represented
- 1: Only covers parts of the document

Return ONLY valid JSON:
{{
  "completeness": {{"score": <1-5>, "justification": "<brief>"}},
  "accuracy": {{"score": <1-5>, "justification": "<brief>"}},
  "statistics_retention": {{"score": <1-5>, "justification": "<brief>"}},
  "coherence": {{"score": <1-5>, "justification": "<brief>"}},
  "conciseness": {{"score": <1-5>, "justification": "<brief>"}},
  "coverage_balance": {{"score": <1-5>, "justification": "<brief>"}},
  "overall_score": <average of 6 scores, 1 decimal>,
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_weaknesses": ["<weakness 1>", "<weakness 2>"]
}}"""


@dataclass
class EvaluationResult:
    generation_file: str
    method: str
    doc_id: str
    run_number: int
    generator_model: str
    judge_model: str
    judge_provider: str
    task: str
    scores: dict
    overall_score: float
    key_strengths: list
    key_weaknesses: list
    document_word_count: int
    evaluation_time_seconds: float
    timestamp: str
    success: bool
    error: Optional[str] = None


def output_to_text(task: str, output_data: list | dict) -> str:
    """Convert generation output to text for evaluation prompt."""
    if task == "slides":
        if isinstance(output_data, list):
            return slides_to_text(output_data)
        return slides_to_text(output_data.get("slides", []))
    else:  # summary
        if isinstance(output_data, dict):
            parts = []
            if output_data.get("title"):
                parts.append(f"Title: {output_data['title']}")
            if output_data.get("abstract"):
                parts.append(f"Abstract: {output_data['abstract']}")
            for sec in output_data.get("sections", []):
                parts.append(f"\n## {sec.get('heading', 'Section')}")
                parts.append(sec.get("content", ""))
                if sec.get("key_statistics"):
                    parts.append(f"Statistics: {', '.join(sec['key_statistics'])}")
            if output_data.get("key_findings"):
                parts.append(f"\nKey Findings: {', '.join(output_data['key_findings'])}")
            if output_data.get("conclusion"):
                parts.append(f"\nConclusion: {output_data['conclusion']}")
            return "\n".join(parts)
        return str(output_data)


def get_eval_prompt(task: str, doc_text: str, output_text: str) -> str:
    if task == "slides":
        return SLIDE_EVAL_PROMPT.format(document_text=doc_text, output_text=output_text)
    return SUMMARY_EVAL_PROMPT.format(document_text=doc_text, output_text=output_text)


def get_eval_filename(gen_filename: str, judge_name: str) -> str:
    """e.g. truncation__run_1__gpt4o__judge_geminiflash.json"""
    safe_judge = judge_name.replace("-", "").replace(".", "").replace(" ", "_").lower()
    stem = Path(gen_filename).stem
    return f"{stem}__judge_{safe_judge}.json"


def collect_eval_tasks(task: str, judges: list[dict]) -> list[dict]:
    """Collect all (generation_output, judge) pairs that need evaluation."""
    tasks = []

    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        task_dir = doc_dir / task
        if not task_dir.exists():
            continue

        doc_id = doc_dir.name
        eval_dir = doc_dir / "evaluations" / task
        existing = set()
        if eval_dir.exists():
            existing = {f.stem for f in eval_dir.glob("*.json")}

        for gen_file in sorted(task_dir.glob("*.json")):
            # Load generation to check it succeeded
            try:
                gen_data = json.loads(gen_file.read_text())
                if not gen_data.get("success", False):
                    continue
            except Exception:
                continue

            for judge in judges:
                eval_filename = get_eval_filename(gen_file.name, judge["name"])
                if Path(eval_filename).stem in existing:
                    continue

                tasks.append({
                    "doc_id": doc_id,
                    "gen_file": gen_file,
                    "gen_data": gen_data,
                    "judge": judge,
                    "eval_filename": eval_filename,
                })

    return tasks


def run_realtime_eval(task_name: str, tasks: list[dict], delay: float) -> list[EvaluationResult]:
    results = []
    total = len(tasks)

    for i, t in enumerate(tasks, 1):
        doc_id = t["doc_id"]
        judge = t["judge"]
        gen_data = t["gen_data"]
        method = gen_data.get("method", "unknown")
        gen_model = gen_data.get("model", "unknown")

        print(f"[{i}/{total}] Judge:{judge['name']} | Gen:{gen_model} | {method} | {doc_id}...", end=" ", flush=True)

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            # Read source PDF
            pdf_path = find_source_pdf(doc_id, DATA_DIR)
            if not pdf_path:
                raise ValueError(f"Source PDF not found for {doc_id}")
            doc_text = read_pdf(pdf_path)
            word_count = len(doc_text.split())

            # Convert output to text
            output_text = output_to_text(task_name, gen_data.get("output", gen_data.get("slides", [])))
            prompt = get_eval_prompt(task_name, doc_text, output_text)

            response_text, elapsed = generate(
                provider=judge["provider"], prompt=prompt, model=judge["id"],
            )
            eval_data = parse_json_response(response_text)

            # Extract scores
            metric_keys = ["completeness", "accuracy", "statistics_retention", "coherence",
                           "relevance" if task_name == "slides" else "conciseness", "coverage_balance"]
            scores = {}
            for key in metric_keys:
                scores[key] = eval_data.get(key, {})

            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=method, doc_id=doc_id,
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_model,
                judge_model=judge["id"], judge_provider=judge["provider"],
                task=task_name, scores=scores,
                overall_score=eval_data.get("overall_score", 0),
                key_strengths=eval_data.get("key_strengths", []),
                key_weaknesses=eval_data.get("key_weaknesses", []),
                document_word_count=word_count,
                evaluation_time_seconds=elapsed,
                timestamp=timestamp, success=True,
            )
            print(f"OK Score:{result.overall_score:.1f} ({elapsed:.1f}s)")

        except Exception as e:
            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=method, doc_id=doc_id,
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_model,
                judge_model=judge["id"], judge_provider=judge["provider"],
                task=task_name, scores={}, overall_score=0,
                key_strengths=[], key_weaknesses=[],
                document_word_count=0, evaluation_time_seconds=0,
                timestamp=timestamp, success=False, error=str(e),
            )
            print(f"FAIL: {e}")

        # Save
        eval_dir = DOCUMENTS_DIR / doc_id / "evaluations" / task_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / t["eval_filename"]).write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False)
        )
        results.append(result)

        if delay > 0 and i < total:
            time.sleep(delay)

    return results


def run_batch_eval(task_name: str, tasks: list[dict]) -> list[EvaluationResult]:
    """Batch evaluation using OpenAI/Anthropic batch APIs + realtime for Google."""
    by_provider: dict[str, list] = {"openai": [], "anthropic": [], "google": []}
    for t in tasks:
        by_provider[t["judge"]["provider"]].append(t)

    all_results = []

    # Google: realtime
    if by_provider["google"]:
        print(f"\nGoogle judge ({len(by_provider['google'])} evals) — realtime")
        all_results.extend(run_realtime_eval(task_name, by_provider["google"], delay=3.0))

    # OpenAI: batch
    if by_provider["openai"]:
        print(f"\nOpenAI judge ({len(by_provider['openai'])} evals) — batch API")
        client = get_openai_client()
        if not client:
            raise SystemExit("OpenAI client not available")

        batch_requests = []
        task_map = {}
        for idx, t in enumerate(by_provider["openai"]):
            doc_id = t["doc_id"]
            gen_data = t["gen_data"]
            pdf_path = find_source_pdf(doc_id, DATA_DIR)
            if not pdf_path:
                continue
            doc_text = read_pdf(pdf_path)
            output_text = output_to_text(task_name, gen_data.get("output", gen_data.get("slides", [])))
            prompt = get_eval_prompt(task_name, doc_text, output_text)

            custom_id = f"e_{idx}"
            batch_requests.append({
                "custom_id": custom_id,
                "model": t["judge"]["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            })
            task_map[custom_id] = t

        if batch_requests:
            batch_id = create_openai_batch(client, batch_requests, f"{task_name} evaluation")
            results = poll_openai_batch(client, batch_id, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT)
            all_results.extend(_process_openai_batch_results(results, task_map, task_name))

    # Anthropic: batch
    if by_provider["anthropic"]:
        print(f"\nAnthropic judge ({len(by_provider['anthropic'])} evals) — batch API")
        client = get_anthropic_client()
        if not client:
            raise SystemExit("Anthropic client not available")

        batch_requests = []
        task_map = {}
        for idx, t in enumerate(by_provider["anthropic"]):
            doc_id = t["doc_id"]
            gen_data = t["gen_data"]
            pdf_path = find_source_pdf(doc_id, DATA_DIR)
            if not pdf_path:
                continue
            doc_text = read_pdf(pdf_path)
            output_text = output_to_text(task_name, gen_data.get("output", gen_data.get("slides", [])))
            prompt = get_eval_prompt(task_name, doc_text, output_text)

            custom_id = f"e_{idx}"
            batch_requests.append({
                "custom_id": custom_id,
                "model": t["judge"]["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
            })
            task_map[custom_id] = t

        if batch_requests:
            batch_id = create_anthropic_batch(client, batch_requests, f"{task_name} evaluation")
            results = poll_anthropic_batch(client, batch_id, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT)
            all_results.extend(_process_anthropic_batch_results(results, task_map, task_name))

    return all_results


def _process_openai_batch_results(results, task_map, task_name):
    processed = []
    for r in results:
        t = task_map.get(r["custom_id"])
        if not t:
            continue
        gen_data = t["gen_data"]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            body = r["response"]["body"]
            text = body["choices"][0]["message"]["content"]
            eval_data = parse_json_response(text)
            scores = {k: eval_data.get(k, {}) for k in eval_data if isinstance(eval_data.get(k), dict)}
            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=gen_data.get("method", ""), doc_id=t["doc_id"],
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_data.get("model", ""),
                judge_model=t["judge"]["id"], judge_provider="openai",
                task=task_name, scores=scores,
                overall_score=eval_data.get("overall_score", 0),
                key_strengths=eval_data.get("key_strengths", []),
                key_weaknesses=eval_data.get("key_weaknesses", []),
                document_word_count=0, evaluation_time_seconds=0,
                timestamp=timestamp, success=True,
            )
        except Exception as e:
            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=gen_data.get("method", ""), doc_id=t["doc_id"],
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_data.get("model", ""),
                judge_model=t["judge"]["id"], judge_provider="openai",
                task=task_name, scores={}, overall_score=0,
                key_strengths=[], key_weaknesses=[],
                document_word_count=0, evaluation_time_seconds=0,
                timestamp=timestamp, success=False, error=str(e),
            )
        eval_dir = DOCUMENTS_DIR / t["doc_id"] / "evaluations" / task_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / t["eval_filename"]).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        processed.append(result)
    return processed


def _process_anthropic_batch_results(results, task_map, task_name):
    processed = []
    for r in results:
        t = task_map.get(r["custom_id"])
        if not t:
            continue
        gen_data = t["gen_data"]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            if r["result"]["type"] != "succeeded":
                raise RuntimeError(r["result"].get("error", "failed"))
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
        except Exception as e:
            result = EvaluationResult(
                generation_file=str(t["gen_file"]),
                method=gen_data.get("method", ""), doc_id=t["doc_id"],
                run_number=gen_data.get("run_number", 1),
                generator_model=gen_data.get("model", ""),
                judge_model=t["judge"]["id"], judge_provider="anthropic",
                task=task_name, scores={}, overall_score=0,
                key_strengths=[], key_weaknesses=[],
                document_word_count=0, evaluation_time_seconds=0,
                timestamp=timestamp, success=False, error=str(e),
            )
        eval_dir = DOCUMENTS_DIR / t["doc_id"] / "evaluations" / task_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / t["eval_filename"]).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        processed.append(result)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Multi-judge evaluation with batch API support.")
    parser.add_argument("--task", choices=["slides", "summary"], default="slides")
    parser.add_argument("--judge", type=str, default=None, help="Specific judge model ID")
    parser.add_argument("--judge-provider", type=str, default=None)
    parser.add_argument("--batch", action="store_true", help="Use batch APIs")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.judge and args.judge_provider:
        judges = [{"id": args.judge, "provider": args.judge_provider, "name": args.judge}]
    elif args.judge:
        judges = [j for j in JUDGE_MODELS if j["id"] == args.judge]
    else:
        judges = JUDGE_MODELS

    print(f"Task: {args.task}")
    print(f"Judges: {[j['name'] for j in judges]}")

    tasks = collect_eval_tasks(args.task, judges)
    if args.limit:
        tasks = tasks[:args.limit]

    if not tasks:
        print("Nothing to evaluate — all evaluations already exist.")
        return

    print(f"Evaluations to process: {len(tasks)}")

    if args.batch:
        results = run_batch_eval(args.task, tasks)
    else:
        results = run_realtime_eval(args.task, tasks, args.delay)

    success = sum(1 for r in results if r.success)
    print(f"\nDone: {success}/{len(results)} successful")


if __name__ == "__main__":
    main()
