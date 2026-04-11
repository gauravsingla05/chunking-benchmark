#!/usr/bin/env python3
"""
Author: Gourav Singla
Date: 2025-12-20
Description: Evaluate generated slide decks using Claude as judge.

Usage:
    # Evaluate all generations in a folder
    python run_evaluation.py --generations results/generations/20251220-104554/ --source-dir data/raw/

    # Evaluate with custom delay
    python run_evaluation.py --generations results/generations/20251220-104554/ --source-dir data/raw/ --delay 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    print("Warning: No .env file found. Set ANTHROPIC_API_KEY manually.")

load_env()

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of a slide presentation generated from a source document.

## Ground Truth Key Facts from Source Document:
{key_facts}

## Statistics from Source Document:
{statistics}

## Main Findings from Source Document:
{main_findings}

## Generated Slide Deck:
{slides_json}

## Evaluation Task

Evaluate the slide deck on these criteria based on how well it covers the key facts above. For each, provide a score and brief justification.

### 1. Completeness (1-5)
Does the deck cover the critical and high-importance facts from the source?
- 5: Covers all critical facts and most high-importance facts
- 3: Covers most critical facts, misses some high-importance ones
- 1: Missing critical information

### 2. Accuracy (1-5)
Is the information factually consistent with the key facts? No hallucinations or misrepresentations?
- 5: All facts accurate, no errors
- 3: Minor inaccuracies or slight misrepresentations
- 1: Major factual errors or hallucinations

### 3. Statistics Retention (1-5)
Are key numbers, percentages, and quantitative findings from the source preserved?
- 5: All important statistics included accurately
- 3: Some statistics included, some missing
- 1: Most statistics missing or wrong

### 4. Coherence (1-5)
Does the deck flow logically? Clear structure from intro to conclusion?
- 5: Excellent flow and logical structure
- 3: Adequate structure, some awkward transitions
- 1: Disorganized or confusing flow

### 5. Relevance (1-5)
Is the content relevant and valuable? No filler or off-topic material?
- 5: All content directly relevant and valuable
- 3: Mostly relevant, some filler
- 1: Much irrelevant or low-value content

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
  "overall_score": <average of all scores, 1 decimal>,
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_weaknesses": ["<weakness 1>", "<weakness 2>"],
  "missing_critical_facts": ["<critical fact not covered 1>", "<critical fact not covered 2>"],
  "facts_covered_count": <number of key facts covered>,
  "facts_total_count": <total number of key facts>
}}
"""


@dataclass
class EvaluationResult:
    """Result of evaluating a single slide deck."""
    generation_file: str
    method: str
    doc_id: str
    run_number: int
    completeness: dict
    accuracy: dict
    statistics_retention: dict
    coherence: dict
    relevance: dict
    overall_score: float
    key_strengths: list
    key_weaknesses: list
    missing_critical_facts: list
    facts_covered_count: int
    facts_total_count: int
    evaluation_time_seconds: float
    timestamp: str
    success: bool
    error: Optional[str] = None


def get_anthropic_client() -> Optional[Anthropic]:
    """Get Anthropic client if available."""
    if Anthropic is None:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def load_summary(docs_dir: Path, doc_id: str) -> Optional[dict]:
    """Load summary.json for a document."""
    summary_path = docs_dir / doc_id / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def format_key_facts(key_facts: list) -> str:
    """Format key facts for the prompt."""
    lines = []
    for i, fact in enumerate(key_facts, 1):
        importance = fact.get("importance", "medium")
        category = fact.get("category", "")
        fact_text = fact.get("fact", "")
        lines.append(f"{i}. [{importance.upper()}] [{category}] {fact_text}")
    return "\n".join(lines)


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
                lines.append(f"  • {bp.get('title', '')}: {bp.get('body', '')}")
        lines.append("")
    return "\n".join(lines)


def evaluate_with_claude(
    client: Anthropic,
    summary: dict,
    slides: list,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[dict, float]:
    """Evaluate slides using Claude against extracted summary."""

    key_facts = summary.get("key_facts", [])
    statistics = summary.get("statistics", [])
    main_findings = summary.get("main_findings", [])

    slides_text = slides_to_text(slides)

    prompt = EVALUATION_PROMPT.format(
        key_facts=format_key_facts(key_facts),
        statistics="\n".join(f"- {s}" for s in statistics),
        main_findings="\n".join(f"- {f}" for f in main_findings),
        slides_json=slides_text,
    )

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    response_text = response.content[0].text.strip()

    # Handle potential markdown code blocks
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
    # Add total facts count
    evaluation["facts_total_count"] = len(key_facts)
    return evaluation, elapsed


def parse_generation_filename(filename: str) -> tuple[str, str, int]:
    """Parse method, doc_id, run_number from filename like 'method__doc_id__run_1.json'."""
    stem = Path(filename).stem
    # Format: method__doc_id__run_N
    if "__run_" in stem:
        parts = stem.rsplit("__run_", 1)
        method_doc = parts[0]
        run_num = int(parts[1])
        if "__" in method_doc:
            method, doc_id = method_doc.split("__", 1)
            return method, doc_id, run_num
    return "unknown", stem, 1


def parse_slides_path(slides_path: Path) -> tuple[str, str, int]:
    """Parse method, doc_id, run_number from new structure: documents/{doc_id}/slides/{method}__run_N.json."""
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
    docs_dir: Path,
    client: Anthropic,
    model: str,
    use_new_structure: bool = False,
) -> EvaluationResult:
    """Evaluate a single generation file."""

    if use_new_structure:
        method, doc_id, run_number = parse_slides_path(generation_path)
    else:
        method, doc_id, run_number = parse_generation_filename(generation_path.name)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        # Load generation
        with open(generation_path) as f:
            generation_data = json.load(f)

        slides = generation_data.get("slides", [])
        if not slides:
            raise ValueError("No slides in generation file")

        # Load summary.json (ground truth)
        summary = load_summary(docs_dir, doc_id)
        if not summary:
            raise ValueError(f"Could not load summary.json for {doc_id}")

        # Evaluate with Claude
        evaluation, elapsed = evaluate_with_claude(client, summary, slides, model)

        return EvaluationResult(
            generation_file=str(generation_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            completeness=evaluation.get("completeness", {}),
            accuracy=evaluation.get("accuracy", {}),
            statistics_retention=evaluation.get("statistics_retention", {}),
            coherence=evaluation.get("coherence", {}),
            relevance=evaluation.get("relevance", {}),
            overall_score=evaluation.get("overall_score", 0.0),
            key_strengths=evaluation.get("key_strengths", []),
            key_weaknesses=evaluation.get("key_weaknesses", []),
            missing_critical_facts=evaluation.get("missing_critical_facts", []),
            facts_covered_count=evaluation.get("facts_covered_count", 0),
            facts_total_count=evaluation.get("facts_total_count", 0),
            evaluation_time_seconds=elapsed,
            timestamp=timestamp,
            success=True,
        )

    except Exception as e:
        return EvaluationResult(
            generation_file=str(generation_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            completeness={},
            accuracy={},
            statistics_retention={},
            coherence={},
            relevance={},
            overall_score=0.0,
            key_strengths=[],
            key_weaknesses=[],
            missing_critical_facts=[],
            facts_covered_count=0,
            facts_total_count=0,
            evaluation_time_seconds=0.0,
            timestamp=timestamp,
            success=False,
            error=str(e),
        )


def run(
    docs_dir: Path,
    output_dir: Path,
    model: str,
    delay: float,
) -> list[EvaluationResult]:
    """Run evaluation on all generation files using new folder structure."""

    client = get_anthropic_client()
    if not client:
        raise SystemExit("Anthropic client not available. Set ANTHROPIC_API_KEY.")

    # Find all slide generation files in documents/{doc_id}/slides/
    generation_files = []
    use_new_structure = False

    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        slides_dir = doc_dir / "slides"
        if slides_dir.exists():
            use_new_structure = True
            for f in sorted(slides_dir.glob("*.json")):
                generation_files.append(f)

    if not generation_files:
        raise SystemExit(f"No slide .json files found in {docs_dir}/*/slides/")

    results: list[EvaluationResult] = []
    total = len(generation_files)

    print(f"Evaluating {total} generation files")
    print(f"Model: {model}")
    print(f"Documents dir: {docs_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    for i, gen_file in enumerate(generation_files, 1):
        method, doc_id, run_num = parse_slides_path(gen_file)
        print(f"[{i}/{total}] Evaluating: {method} / {doc_id} (run {run_num})...", end=" ", flush=True)

        result = evaluate_generation(gen_file, docs_dir, client, model, use_new_structure=True)

        if result.success:
            print(f"✓ Score: {result.overall_score:.1f}/5 ({result.facts_covered_count}/{result.facts_total_count} facts) in {result.evaluation_time_seconds:.1f}s")
        else:
            print(f"✗ Error: {result.error}")

        results.append(result)

        # Save individual evaluation to doc folder
        eval_dir = docs_dir / doc_id / "evaluations"
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
    results_file = output_dir / f"evaluations_{eval_timestamp}.json"
    results_file.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate summary by method
    summary = generate_summary(results)
    summary_file = output_dir / f"summary_{eval_timestamp}.json"
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Print summary table
    print("-" * 60)
    print("\n## Summary by Method\n")
    print(f"{'Method':<25} {'Complete':>8} {'Accuracy':>8} {'Stats':>8} {'Coherence':>8} {'Relevance':>8} {'Overall':>8} {'Facts%':>8}")
    print("-" * 95)
    for method_name, scores in summary["by_method"].items():
        facts_pct = scores.get("facts_coverage_pct", 0)
        print(f"{method_name:<25} {scores['completeness']:>8.2f} {scores['accuracy']:>8.2f} {scores['statistics_retention']:>8.2f} {scores['coherence']:>8.2f} {scores['relevance']:>8.2f} {scores['overall']:>8.2f} {facts_pct:>7.1f}%")

    print(f"\nResults saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    return results


def generate_summary(results: list[EvaluationResult]) -> dict:
    """Generate summary statistics by method."""
    from collections import defaultdict

    method_scores = defaultdict(lambda: {
        "completeness": [],
        "accuracy": [],
        "statistics_retention": [],
        "coherence": [],
        "relevance": [],
        "overall": [],
        "facts_covered": [],
        "facts_total": [],
    })

    for r in results:
        if not r.success:
            continue
        method_scores[r.method]["completeness"].append(r.completeness.get("score", 0))
        method_scores[r.method]["accuracy"].append(r.accuracy.get("score", 0))
        method_scores[r.method]["statistics_retention"].append(r.statistics_retention.get("score", 0))
        method_scores[r.method]["coherence"].append(r.coherence.get("score", 0))
        method_scores[r.method]["relevance"].append(r.relevance.get("score", 0))
        method_scores[r.method]["overall"].append(r.overall_score)
        method_scores[r.method]["facts_covered"].append(r.facts_covered_count)
        method_scores[r.method]["facts_total"].append(r.facts_total_count)

    summary = {"by_method": {}}
    for method, scores in method_scores.items():
        avg_scores = {
            metric: sum(vals) / len(vals) if vals else 0
            for metric, vals in scores.items()
            if metric not in ["facts_covered", "facts_total"]
        }
        # Calculate facts coverage percentage
        total_covered = sum(scores["facts_covered"])
        total_facts = sum(scores["facts_total"])
        avg_scores["facts_coverage_pct"] = (total_covered / total_facts * 100) if total_facts > 0 else 0
        summary["by_method"][method] = avg_scores

    # Overall stats
    all_successful = [r for r in results if r.success]
    summary["total_evaluated"] = len(all_successful)
    summary["total_failed"] = len(results) - len(all_successful)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated slide decks using Claude as judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using new folder structure (documents/{doc_id}/slides/)
  python run_evaluation.py --docs-dir results/documents/

  # With custom output directory
  python run_evaluation.py --docs-dir results/documents/ --output-dir results/evaluations/
        """,
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Documents directory containing {doc_id}/slides/ and {doc_id}/summary.json",
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
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between API calls (default: 2)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent.parent.parent
        args.output_dir = script_dir / "results" / "evaluations"

    run(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        model=args.model,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
