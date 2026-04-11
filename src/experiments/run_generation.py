#!/usr/bin/env python3
"""
Multi-model slide generation with batch API support.

Generates slides from chunked documents using GPT-4o, Claude Sonnet, and Gemini Flash.
Supports OpenAI and Anthropic batch APIs for 50% cost savings.

Usage:
    # All models, batch mode (cheapest)
    python run_generation.py --task slides --batch

    # Single model, realtime
    python run_generation.py --task slides --model gpt-4o --provider openai

    # Summarization task
    python run_generation.py --task summary --batch

    # Resume interrupted run
    python run_generation.py --task slides --batch --resume
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
    generate, parse_json_response, get_openai_client, get_anthropic_client,
    create_openai_batch, create_anthropic_batch,
    poll_openai_batch, poll_anthropic_batch,
)
from config import GENERATION_MODELS, DOCUMENTS_DIR, DATA_DIR, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT


# ─── Prompts ──────────────────────────────────────────────────

SLIDE_PROMPT = """You are a senior domain consultant and presentation author. Produce a complete, professional deck based on the provided document content.

TASK: Create a presentation that captures the key insights from the document below.

SLIDE COUNT: Generate exactly 7-10 slides for a balanced presentation.
- 1 title slide
- 5-7 content slides (mix of bullet_points, charts, diagrams as appropriate)
- 1 conclusion slide

CONTENT DEPTH:
- Bullet titles: 3-5 words
- Bullet body: 15-25 words
- Body text: 30-50 words
- 4-5 bullets per slide

SLIDE TYPES AVAILABLE:
- "title" - Opening slide with title and body
- "bullet_points" - Main content slides with bullet points
- "chart" - Data visualization (bar, line, pie, doughnut)
- "conclusion" - Closing slide summarizing key takeaways

RULES:
1. Extract the most important information from the document
2. Create substantive, evidence-based content
3. Include specific numbers, statistics, and examples from the document
4. Return ONLY valid JSON with a "slides" array

OUTPUT FORMAT:
{{
  "slides": [
    {{
      "slide_number": 1,
      "type": "title",
      "title": "Presentation Title",
      "body": "Subtitle or context",
      "image_keywords": ["keyword1", "keyword2", "keyword3"]
    }},
    {{
      "slide_number": 2,
      "type": "bullet_points",
      "title": "Section Title",
      "bullet_points": [
        {{ "title": "Point 1", "body": "Explanation with specifics" }},
        {{ "title": "Point 2", "body": "Explanation with specifics" }}
      ],
      "image_keywords": ["keyword1", "keyword2"]
    }},
    {{
      "slide_number": N,
      "type": "conclusion",
      "title": "Key Takeaways",
      "body": "Summary of main points",
      "image_keywords": ["keyword1", "keyword2"]
    }}
  ]
}}

DOCUMENT CONTENT:
{document_content}"""

SUMMARY_PROMPT = """You are an expert summarizer. Produce a comprehensive, structured summary of the provided document.

TASK: Summarize the document below, preserving key findings, methodology, statistics, and conclusions.

REQUIREMENTS:
1. Cover ALL major sections of the document proportionally
2. Preserve specific numbers, percentages, and quantitative findings
3. Include methodology details
4. Maintain factual accuracy — do not add information not in the source
5. Target length: 500-800 words
6. Return ONLY valid JSON

OUTPUT FORMAT:
{{
  "title": "Document title or topic",
  "abstract": "1-2 sentence overview",
  "sections": [
    {{
      "heading": "Section name",
      "content": "Summary of this section (50-150 words)",
      "key_statistics": ["stat1", "stat2"]
    }}
  ],
  "key_findings": ["finding1", "finding2", "finding3"],
  "methodology": "Brief methodology description",
  "limitations": ["limitation1", "limitation2"],
  "conclusion": "Main conclusion in 2-3 sentences",
  "word_count": <number of words in the full summary>
}}

DOCUMENT CONTENT:
{document_content}"""


@dataclass(frozen=True)
class GenerationResult:
    input_file: str
    method: str
    doc_id: str
    run_number: int
    model: str
    provider: str
    task: str
    output: list | dict  # slides list or summary dict
    input_words: int
    output_item_count: int  # slide count or section count
    generation_time_seconds: float
    timestamp: str
    success: bool
    error: Optional[str] = None


def get_prompt(task: str, content: str) -> str:
    if task == "slides":
        return SLIDE_PROMPT.format(document_content=content)
    elif task == "summary":
        return SUMMARY_PROMPT.format(document_content=content)
    raise ValueError(f"Unknown task: {task}")


def get_output_dir(task: str, doc_id: str) -> Path:
    return DOCUMENTS_DIR / doc_id / task


def get_output_filename(method: str, run: int, model_name: str) -> str:
    """e.g. truncation__run_1__gpt4o.json"""
    safe_name = model_name.replace("-", "").replace(".", "").replace(" ", "_").lower()
    return f"{method}__run_{run}__{safe_name}.json"


def get_existing_outputs(task: str) -> set[str]:
    """Get set of existing output keys: '{method}__{doc_id}__run_{N}__{model}'."""
    existing = set()
    if not DOCUMENTS_DIR.exists():
        return existing
    for doc_dir in DOCUMENTS_DIR.iterdir():
        if not doc_dir.is_dir():
            continue
        task_dir = doc_dir / task
        if not task_dir.exists():
            continue
        for f in task_dir.glob("*.json"):
            # Parse: method__run_N__modelname.json
            key = f"{doc_dir.name}/{f.stem}"
            existing.add(key)
    return existing


def collect_tasks(task: str, models: list[dict], runs: int = 1) -> list[dict]:
    """Collect all (doc, method, model, run) combinations that need processing."""
    existing = get_existing_outputs(task)
    tasks = []

    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        chunks_dir = doc_dir / "chunks"
        if not chunks_dir.exists():
            continue

        doc_id = doc_dir.name
        for chunk_file in sorted(chunks_dir.glob("*.txt")):
            method = chunk_file.stem
            for model_cfg in models:
                for run_num in range(1, runs + 1):
                    filename = get_output_filename(method, run_num, model_cfg["name"])
                    key = f"{doc_id}/{Path(filename).stem}"
                    if key in existing:
                        continue
                    tasks.append({
                        "doc_id": doc_id,
                        "method": method,
                        "chunk_file": chunk_file,
                        "model": model_cfg,
                        "run": run_num,
                        "filename": filename,
                    })

    return tasks


def run_realtime(task_name: str, tasks: list[dict], delay: float) -> list[GenerationResult]:
    """Process tasks one-by-one with real-time API calls."""
    results = []
    total = len(tasks)

    for i, t in enumerate(tasks, 1):
        content = t["chunk_file"].read_text(encoding="utf-8", errors="ignore")
        prompt = get_prompt(task_name, content)
        model_cfg = t["model"]

        print(f"[{i}/{total}] {model_cfg['name']} | {t['method']} | {t['doc_id']}...", end=" ", flush=True)

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
            print(f"OK ({item_count} items, {elapsed:.1f}s)")

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
            print(f"FAIL: {e}")

        # Save result
        out_dir = get_output_dir(task_name, t["doc_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / t["filename"]
        out_file.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(result)

        if delay > 0 and i < total:
            time.sleep(delay)

    return results


def run_batch(task_name: str, tasks: list[dict]) -> list[GenerationResult]:
    """Process tasks via batch APIs (OpenAI + Anthropic) and realtime for Google."""
    # Split tasks by provider
    by_provider: dict[str, list] = {"openai": [], "anthropic": [], "google": []}
    for t in tasks:
        by_provider[t["model"]["provider"]].append(t)

    all_results = []

    # ── Google: realtime (no batch API) ──
    if by_provider["google"]:
        print(f"\n{'='*60}")
        print(f"Google ({len(by_provider['google'])} tasks) — realtime (no batch API)")
        print(f"{'='*60}")
        all_results.extend(run_realtime(task_name, by_provider["google"], delay=4.0))

    # ── OpenAI: batch API (50% off) ──
    if by_provider["openai"]:
        print(f"\n{'='*60}")
        print(f"OpenAI ({len(by_provider['openai'])} tasks) — batch API")
        print(f"{'='*60}")

        client = get_openai_client()
        if not client:
            raise SystemExit("OpenAI client not available")

        # Build batch requests
        batch_requests = []
        task_map = {}
        for idx, t in enumerate(by_provider["openai"]):
            content = t["chunk_file"].read_text(encoding="utf-8", errors="ignore")
            prompt = get_prompt(task_name, content)
            custom_id = f"oai_{idx}"
            batch_requests.append({
                "custom_id": custom_id,
                "model": t["model"]["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
            })
            task_map[custom_id] = t

        batch_id = create_openai_batch(client, batch_requests, f"{task_name} generation")
        results = poll_openai_batch(client, batch_id, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT)

        for r in results:
            t = task_map.get(r["custom_id"])
            if not t:
                continue
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                body = r["response"]["body"]
                response_text = body["choices"][0]["message"]["content"]
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
                    model=t["model"]["id"], provider="openai", task=task_name,
                    output=output, input_words=0, output_item_count=item_count,
                    generation_time_seconds=0, timestamp=timestamp, success=True,
                )
            except Exception as e:
                result = GenerationResult(
                    input_file=str(t["chunk_file"]),
                    method=t["method"], doc_id=t["doc_id"], run_number=t["run"],
                    model=t["model"]["id"], provider="openai", task=task_name,
                    output=[], input_words=0, output_item_count=0,
                    generation_time_seconds=0, timestamp=timestamp,
                    success=False, error=str(e),
                )

            out_dir = get_output_dir(task_name, t["doc_id"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / t["filename"]
            out_file.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            all_results.append(result)

    # ── Anthropic: batch API (50% off) ──
    if by_provider["anthropic"]:
        print(f"\n{'='*60}")
        print(f"Anthropic ({len(by_provider['anthropic'])} tasks) — batch API")
        print(f"{'='*60}")

        client = get_anthropic_client()
        if not client:
            raise SystemExit("Anthropic client not available")

        batch_requests = []
        task_map = {}
        for idx, t in enumerate(by_provider["anthropic"]):
            content = t["chunk_file"].read_text(encoding="utf-8", errors="ignore")
            prompt = get_prompt(task_name, content)
            custom_id = f"ant_{idx}"
            batch_requests.append({
                "custom_id": custom_id,
                "model": t["model"]["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
            })
            task_map[custom_id] = t

        batch_id = create_anthropic_batch(client, batch_requests, f"{task_name} generation")
        results = poll_anthropic_batch(client, batch_id, BATCH_POLL_INTERVAL, BATCH_MAX_WAIT)

        for r in results:
            t = task_map.get(r["custom_id"])
            if not t:
                continue
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                if r["result"]["type"] != "succeeded":
                    raise RuntimeError(r["result"].get("error", "Batch request failed"))
                response_text = r["result"]["message"]["content"][0]["text"]
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
                    model=t["model"]["id"], provider="anthropic", task=task_name,
                    output=output, input_words=0, output_item_count=item_count,
                    generation_time_seconds=0, timestamp=timestamp, success=True,
                )
            except Exception as e:
                result = GenerationResult(
                    input_file=str(t["chunk_file"]),
                    method=t["method"], doc_id=t["doc_id"], run_number=t["run"],
                    model=t["model"]["id"], provider="anthropic", task=task_name,
                    output=[], input_words=0, output_item_count=0,
                    generation_time_seconds=0, timestamp=timestamp,
                    success=False, error=str(e),
                )

            out_dir = get_output_dir(task_name, t["doc_id"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / t["filename"]
            out_file.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-model generation with batch API support.")
    parser.add_argument("--task", choices=["slides", "summary"], default="slides")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID (default: all)")
    parser.add_argument("--provider", type=str, default=None, help="Specific provider")
    parser.add_argument("--batch", action="store_true", help="Use batch APIs (50% off for OpenAI/Anthropic)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between realtime calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--resume", action="store_true", default=True)

    args = parser.parse_args()

    # Select models — always match from config to keep consistent filenames
    if args.model:
        models = [m for m in GENERATION_MODELS if m["id"] == args.model]
        if not models and args.provider:
            models = [{"id": args.model, "provider": args.provider, "name": args.model}]
    elif args.provider:
        models = [m for m in GENERATION_MODELS if m["provider"] == args.provider]
    else:
        models = GENERATION_MODELS

    if not models:
        raise SystemExit(f"No model found matching: {args.model}")

    print(f"Task: {args.task}")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Batch mode: {args.batch}")

    tasks = collect_tasks(args.task, models, args.runs)
    if args.limit:
        tasks = tasks[:args.limit]

    if not tasks:
        print("Nothing to do — all outputs already exist.")
        return

    print(f"Tasks to process: {len(tasks)}")

    if args.batch:
        results = run_batch(args.task, tasks)
    else:
        results = run_realtime(args.task, tasks, args.delay)

    # Summary
    success = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(results)} successful")


if __name__ == "__main__":
    main()
