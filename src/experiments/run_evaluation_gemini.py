#!/usr/bin/env python3
"""
Author: Gourav Singla
Date: 2025-12-21
Description: Evaluate generated slide decks using Gemini with direct PDF comparison.

This script compares slides directly against the FULL source PDF, eliminating
any potential bias from intermediate summary extraction.

Usage:
    # Evaluate all slides in documents folder
    python run_evaluation_gemini.py --docs-dir results/documents/ --source-dir data/raw/

    # With custom delay
    python run_evaluation_gemini.py --docs-dir results/documents/ --source-dir data/raw/ --delay 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Load environment variables from backend .env file
def load_env():
    """Load .env file from backend if it exists."""
    env_paths = [
        Path(__file__).resolve().parents[4] / "SlideMaker-Backend" / ".env",
        Path.home() / "Projects" / "Slidemaker app" / "SlideMaker-Backend" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            print(f"Loaded env from: {env_path}")
            return
    print("Warning: No .env file found. Set GOOGLE_API_KEY manually.")

load_env()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of a slide presentation generated from a source document.

## FULL SOURCE DOCUMENT:
{document_text}

## GENERATED SLIDE DECK:
{slides_text}

## Evaluation Task

Compare the slide deck against the FULL source document above. Evaluate on these criteria:

### 1. Completeness (1-5)
Does the deck cover the most important information from the source document?
- 5: Covers all critical facts and main points from the entire document
- 3: Covers main points but misses some important details
- 1: Missing critical information from the document

### 2. Accuracy (1-5)
Is the information factually consistent with the source document? No hallucinations or misrepresentations?
- 5: All facts accurate, faithful to the source
- 3: Minor inaccuracies or slight misrepresentations
- 1: Major factual errors or hallucinations

### 3. Statistics Retention (1-5)
Are key numbers, percentages, and quantitative findings from the source preserved?
- 5: All important statistics from the document included accurately
- 3: Some statistics included, some missing
- 1: Most statistics from the document missing or wrong

### 4. Coherence (1-5)
Does the deck flow logically? Clear structure from intro to conclusion?
- 5: Excellent flow and logical structure
- 3: Adequate structure, some awkward transitions
- 1: Disorganized or confusing flow

### 5. Relevance (1-5)
Is the content relevant and valuable? Does it capture the document's main purpose?
- 5: All content directly relevant and valuable
- 3: Mostly relevant, some filler
- 1: Much irrelevant or low-value content

### 6. Coverage Balance (1-5)
Does the deck represent content from ALL parts of the document (beginning, middle, end)?
- 5: Balanced coverage across the entire document
- 3: Some sections over/under-represented
- 1: Only covers beginning or specific sections

## Response Format

Return ONLY a valid JSON object with this exact structure:
{{
  "completeness": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "accuracy": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "statistics_retention": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "coherence": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "relevance": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "coverage_balance": {{
    "score": <1-5>,
    "justification": "<brief explanation>"
  }},
  "overall_score": <average of all 6 scores, 1 decimal>,
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_weaknesses": ["<weakness 1>", "<weakness 2>"],
  "missing_from_beginning": ["<important info from doc start not in slides>"],
  "missing_from_middle": ["<important info from doc middle not in slides>"],
  "missing_from_end": ["<important info from doc end not in slides>"]
}}
"""


@dataclass
class GeminiEvaluationResult:
    """Result of evaluating a single slide deck with Gemini."""
    generation_file: str
    method: str
    doc_id: str
    run_number: int
    completeness: dict
    accuracy: dict
    statistics_retention: dict
    coherence: dict
    relevance: dict
    coverage_balance: dict
    overall_score: float
    key_strengths: list
    key_weaknesses: list
    missing_from_beginning: list
    missing_from_middle: list
    missing_from_end: list
    document_word_count: int
    evaluation_time_seconds: float
    timestamp: str
    model: str
    evaluator: str  # "gemini_direct_pdf"
    success: bool
    error: Optional[str] = None


def get_gemini_client():
    """Configure and return Gemini client."""
    if genai is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def read_pdf(path: Path) -> str:
    """Read PDF using pdftotext."""
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


def find_source_pdf(doc_id: str, source_dir: Path) -> Optional[Path]:
    """Find the source PDF for a document ID."""
    # Try exact match first
    pdf_path = source_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        return pdf_path

    # Try case-insensitive search
    for f in source_dir.iterdir():
        if f.suffix.lower() == ".pdf" and f.stem.lower() == doc_id.lower():
            return f

    return None


def slides_to_text(slides: list) -> str:
    """Convert slides JSON to readable text for evaluation."""
    lines = []
    for slide in slides:
        lines.append(f"--- Slide {slide.get('slide_number', '?')}: {slide.get('type', 'unknown')} ---")
        if slide.get('title'):
            lines.append(f"Title: {slide['title']}")
        if slide.get('body'):
            lines.append(f"Body: {slide['body']}")
        if slide.get('bullet_points'):
            for bp in slide['bullet_points']:
                lines.append(f"  - {bp.get('title', '')}: {bp.get('body', '')}")
        lines.append("")
    return "\n".join(lines)


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 60.0,
    max_delay: float = 300.0,
):
    """
    Retry a function with exponential backoff for rate limit errors.

    Args:
        func: Function to call (should return result)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (default 60s for rate limits)
        max_delay: Maximum delay between retries (default 5 minutes)
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a rate limit error
            is_rate_limit = any(phrase in error_str for phrase in [
                "rate limit", "rate_limit", "too many requests", "429",
                "quota exceeded", "resource exhausted", "overloaded"
            ])

            if not is_rate_limit or attempt == max_retries:
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 10), max_delay)
            last_exception = e
            print(f"\n  ⚠️  Rate limit hit. Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)

    raise last_exception


def evaluate_with_gemini(
    client,
    document_text: str,
    slides: list,
    model: str = "gemini-2.0-flash",
    max_retries: int = 3,
) -> tuple[dict, float]:
    """Evaluate slides using Gemini against full document text with retry logic."""

    slides_text = slides_to_text(slides)

    prompt = EVALUATION_PROMPT.format(
        document_text=document_text,
        slides_text=slides_text,
    )

    def _call():
        start = time.time()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2048,
            ),
        )
        elapsed = time.time() - start

        response_text = response.text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        evaluation = json.loads(response_text)
        return evaluation, elapsed

    return retry_with_backoff(_call, max_retries=max_retries)


def parse_slides_path(slides_path: Path) -> tuple[str, str, int]:
    """Parse method, doc_id, run_number from structure: documents/{doc_id}/slides/{method}__run_N.json."""
    stem = slides_path.stem
    doc_id = slides_path.parent.parent.name

    if "__run_" in stem:
        parts = stem.rsplit("__run_", 1)
        method = parts[0]
        run_num = int(parts[1])
        return method, doc_id, run_num
    return stem, doc_id, 1


def evaluate_generation(
    generation_path: Path,
    source_dir: Path,
    client,
    model: str,
) -> GeminiEvaluationResult:
    """Evaluate a single generation file using Gemini with full PDF."""

    method, doc_id, run_number = parse_slides_path(generation_path)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        # Load generation
        with open(generation_path) as f:
            generation_data = json.load(f)

        slides = generation_data.get("slides", [])
        if not slides:
            raise ValueError("No slides in generation file")

        # Find and read source PDF
        pdf_path = find_source_pdf(doc_id, source_dir)
        if not pdf_path:
            raise ValueError(f"Could not find source PDF for {doc_id} in {source_dir}")

        document_text = read_pdf(pdf_path)
        word_count = len(document_text.split())

        # Evaluate with Gemini
        evaluation, elapsed = evaluate_with_gemini(client, document_text, slides, model)

        return GeminiEvaluationResult(
            generation_file=str(generation_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            completeness=evaluation.get("completeness", {}),
            accuracy=evaluation.get("accuracy", {}),
            statistics_retention=evaluation.get("statistics_retention", {}),
            coherence=evaluation.get("coherence", {}),
            relevance=evaluation.get("relevance", {}),
            coverage_balance=evaluation.get("coverage_balance", {}),
            overall_score=evaluation.get("overall_score", 0.0),
            key_strengths=evaluation.get("key_strengths", []),
            key_weaknesses=evaluation.get("key_weaknesses", []),
            missing_from_beginning=evaluation.get("missing_from_beginning", []),
            missing_from_middle=evaluation.get("missing_from_middle", []),
            missing_from_end=evaluation.get("missing_from_end", []),
            document_word_count=word_count,
            evaluation_time_seconds=elapsed,
            timestamp=timestamp,
            model=model,
            evaluator="gemini_direct_pdf",
            success=True,
        )

    except Exception as e:
        return GeminiEvaluationResult(
            generation_file=str(generation_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            completeness={},
            accuracy={},
            statistics_retention={},
            coherence={},
            relevance={},
            coverage_balance={},
            overall_score=0.0,
            key_strengths=[],
            key_weaknesses=[],
            missing_from_beginning=[],
            missing_from_middle=[],
            missing_from_end=[],
            document_word_count=0,
            evaluation_time_seconds=0.0,
            timestamp=timestamp,
            model=model,
            evaluator="gemini_direct_pdf",
            success=False,
            error=str(e),
        )


def get_existing_evaluations(docs_dir: Path) -> set[str]:
    """Get set of already completed (method, doc_id, run) combinations."""
    existing = set()
    for doc_dir in docs_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        eval_dir = doc_dir / "evaluations_gemini"
        if not eval_dir.exists():
            continue
        for f in eval_dir.glob("*.json"):
            stem = f.stem
            if "__run_" in stem:
                parts = stem.rsplit("__run_", 1)
                if len(parts) == 2:
                    method = parts[0]
                    run_num = parts[1]
                    existing.add(f"{method}__{doc_dir.name}__run_{run_num}")
    return existing


def run(
    docs_dir: Path,
    source_dir: Path,
    output_dir: Path,
    model: str,
    delay: float,
    resume: bool = True,
    start: int = 0,
    limit: int | None = None,
) -> list[GeminiEvaluationResult]:
    """Run Gemini evaluation on all generation files."""

    client = get_gemini_client()
    if not client:
        raise SystemExit("Gemini client not available. Set GOOGLE_API_KEY.")

    # Find all slide generation files in documents/{doc_id}/slides/
    generation_files = []

    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        slides_dir = doc_dir / "slides"
        if slides_dir.exists():
            for f in sorted(slides_dir.glob("*.json")):
                generation_files.append(f)

    if not generation_files:
        raise SystemExit(f"No slide .json files found in {docs_dir}/*/slides/")

    # Apply start and limit for batch processing
    if start > 0:
        generation_files = generation_files[start:]
    if limit is not None and limit > 0:
        generation_files = generation_files[:limit]

    if not generation_files:
        raise SystemExit(f"No files to process after applying start={start}, limit={limit}")

    # Get existing evaluations for resume
    existing = get_existing_evaluations(docs_dir) if resume else set()

    results: list[GeminiEvaluationResult] = []
    total = len(generation_files)

    print(f"Evaluating {total} generation files with Gemini (direct PDF)")
    if start > 0 or limit is not None:
        print(f"Batch: start={start}, limit={limit}")
    print(f"Model: {model}")
    print(f"Documents dir: {docs_dir}")
    print(f"Source PDFs: {source_dir}")
    print(f"Output: {output_dir}")
    if resume:
        print(f"Resume mode: skipping {len(existing)} existing evaluations")
    print("-" * 60)

    completed = 0
    for i, gen_file in enumerate(generation_files, 1):
        method, doc_id, run_num = parse_slides_path(gen_file)
        task_key = f"{method}__{doc_id}__run_{run_num}"

        # Skip if already exists (resume mode)
        if task_key in existing:
            completed += 1
            print(f"[{i}/{total}] SKIP (exists): {method} / {doc_id} (run {run_num})")
            continue

        completed += 1
        print(f"[{i}/{total}] Evaluating: {method} / {doc_id} (run {run_num})...", end=" ", flush=True)

        result = evaluate_generation(gen_file, source_dir, client, model)

        if result.success:
            print(f"OK Score: {result.overall_score:.1f}/5 ({result.document_word_count} words) in {result.evaluation_time_seconds:.1f}s")
        else:
            print(f"X Error: {result.error}")

        results.append(result)

        # Save individual evaluation to doc folder
        eval_dir = docs_dir / doc_id / "evaluations_gemini"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"{method}__run_{run_num}.json"
        eval_file.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Delay between API calls
        if delay > 0 and i < total:
            time.sleep(delay)

    # Save aggregated results
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save all results
    results_file = output_dir / f"gemini_evaluations_{eval_timestamp}.json"
    results_file.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate summary by method
    summary = generate_summary(results)
    summary_file = output_dir / f"gemini_summary_{eval_timestamp}.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary table
    print("-" * 60)
    print("\n## Gemini Evaluation Summary by Method\n")
    print(f"{'Method':<25} {'Complete':>8} {'Accuracy':>8} {'Stats':>8} {'Coherence':>8} {'Relevance':>8} {'Balance':>8} {'Overall':>8}")
    print("-" * 105)
    for method_name, scores in summary["by_method"].items():
        print(f"{method_name:<25} {scores['completeness']:>8.2f} {scores['accuracy']:>8.2f} {scores['statistics_retention']:>8.2f} {scores['coherence']:>8.2f} {scores['relevance']:>8.2f} {scores['coverage_balance']:>8.2f} {scores['overall']:>8.2f}")

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    return results


def generate_summary(results: list[GeminiEvaluationResult]) -> dict:
    """Generate summary statistics by method."""
    from collections import defaultdict

    method_scores = defaultdict(lambda: {
        "completeness": [],
        "accuracy": [],
        "statistics_retention": [],
        "coherence": [],
        "relevance": [],
        "coverage_balance": [],
        "overall": [],
    })

    for r in results:
        if not r.success:
            continue
        method_scores[r.method]["completeness"].append(r.completeness.get("score", 0))
        method_scores[r.method]["accuracy"].append(r.accuracy.get("score", 0))
        method_scores[r.method]["statistics_retention"].append(r.statistics_retention.get("score", 0))
        method_scores[r.method]["coherence"].append(r.coherence.get("score", 0))
        method_scores[r.method]["relevance"].append(r.relevance.get("score", 0))
        method_scores[r.method]["coverage_balance"].append(r.coverage_balance.get("score", 0))
        method_scores[r.method]["overall"].append(r.overall_score)

    summary = {"by_method": {}}
    for method, scores in method_scores.items():
        avg_scores = {
            metric: sum(vals) / len(vals) if vals else 0
            for metric, vals in scores.items()
        }
        summary["by_method"][method] = avg_scores

    # Overall stats
    all_successful = [r for r in results if r.success]
    summary["total_evaluated"] = len(all_successful)
    summary["total_failed"] = len(results) - len(all_successful)
    summary["evaluator"] = "gemini_direct_pdf"

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated slide decks using Gemini with direct PDF comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation_gemini.py --docs-dir results/documents/ --source-dir data/raw/
  python run_evaluation_gemini.py --docs-dir results/documents/ --source-dir data/raw/ --delay 5
        """,
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Documents directory containing {doc_id}/slides/",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing source PDF files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for aggregated evaluation results. Default: results/evaluations/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between API calls (default: 2)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip existing evaluations (regenerate all)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this file index (0-indexed, for batch processing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to this many files (for batch processing)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent.parent.parent
        args.output_dir = script_dir / "results" / "evaluations"

    run(
        docs_dir=args.docs_dir,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model=args.model,
        delay=args.delay,
        resume=not args.no_resume,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
