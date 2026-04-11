#!/usr/bin/env python3
"""
Author: Gourav Singla
Date: 2025-12-20
Description: Run LLM generation on pre-chunked documents for research evaluation.

Usage:
    # Process all chunked outputs from a specific run
    python run_llm.py --input results/runs/20251220-101033/outputs/ --delay 10

    # Process a single file
    python run_llm.py --input results/runs/20251220-101033/outputs/pac__doc1.txt

    # Multiple runs for variance analysis
    python run_llm.py --input results/runs/20251220-101033/outputs/ --runs 3 --delay 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    print("Warning: No .env file found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY manually.")

load_env()

# LLM imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# Verbosity instructions for slide count
VERBOSITY_INSTRUCTIONS = """
SLIDE COUNT: Generate exactly 7-10 slides for a balanced presentation.
- 1 title slide
- 5-7 content slides (mix of bullet_points, charts, diagrams as appropriate)
- 1 conclusion slide

CONTENT DEPTH:
- Bullet titles: 3-5 words
- Bullet body: 15-25 words
- Body text: 30-50 words
- 4-5 bullets per slide
"""

# Simplified monolithic prompt for research evaluation
SLIDE_GENERATION_PROMPT = """
You are a senior domain consultant and presentation author. Produce a complete, professional deck based on the provided document content.

TASK: Create a presentation that captures the key insights from the document below.

{verbosity_instructions}

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
{document_content}
"""


@dataclass(frozen=True)
class GenerationResult:
    """Result of a single LLM generation."""
    input_file: str
    method: str
    doc_id: str
    run_number: int
    model: str
    slides: list
    input_words: int
    output_slide_count: int
    generation_time_seconds: float
    timestamp: str
    success: bool
    error: Optional[str] = None


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client if available."""
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> Optional[Anthropic]:
    """Get Anthropic client if available."""
    if Anthropic is None:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


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


def generate_with_openai(client: OpenAI, prompt: str, model: str = "gpt-4o", max_retries: int = 3) -> tuple[str, float]:
    """Generate slides using OpenAI with retry logic."""
    def _call():
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=4096,
        )
        elapsed = time.time() - start
        return response.choices[0].message.content, elapsed

    return retry_with_backoff(_call, max_retries=max_retries)


def generate_with_anthropic(client: Anthropic, prompt: str, model: str = "claude-sonnet-4-20250514", max_retries: int = 3) -> tuple[str, float]:
    """Generate slides using Anthropic Claude with retry logic."""
    def _call():
        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start
        return response.content[0].text, elapsed

    return retry_with_backoff(_call, max_retries=max_retries)


def parse_filename(filename: str) -> tuple[str, str]:
    """Parse method and doc_id from filename like 'pac_position_aware__doc_name.txt' (old format)
    or just 'method.txt' (new format)."""
    stem = Path(filename).stem
    if "__" in stem:
        # Old format: method__doc_id.txt
        parts = stem.split("__", 1)
        return parts[0], parts[1]
    # New format: method.txt (doc_id comes from parent folder)
    return stem, ""


def parse_chunk_path(chunk_path: Path) -> tuple[str, str]:
    """Parse method and doc_id from new folder structure: documents/{doc_id}/chunks/{method}.txt."""
    method = chunk_path.stem
    # Go up to get doc_id: chunks -> doc_id
    doc_id = chunk_path.parent.parent.name
    return method, doc_id


def generate_slides(
    input_path: Path,
    output_dir: Path,
    model: str,
    run_number: int,
    provider: str = "openai",
    use_new_structure: bool = False,
) -> GenerationResult:
    """Generate slides from a chunked text file."""

    # Read input
    content = input_path.read_text(encoding="utf-8", errors="ignore")
    input_words = len(content.split())

    # Parse method and doc_id based on folder structure
    if use_new_structure:
        method, doc_id = parse_chunk_path(input_path)
    else:
        method, doc_id = parse_filename(input_path.name)

    # Build prompt
    prompt = SLIDE_GENERATION_PROMPT.format(
        verbosity_instructions=VERBOSITY_INSTRUCTIONS,
        document_content=content,
    )

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        # Get appropriate client
        if provider == "openai":
            client = get_openai_client()
            if not client:
                raise ValueError("OpenAI client not available. Set OPENAI_API_KEY.")
            response_text, elapsed = generate_with_openai(client, prompt, model)
        elif provider == "anthropic":
            client = get_anthropic_client()
            if not client:
                raise ValueError("Anthropic client not available. Set ANTHROPIC_API_KEY.")
            response_text, elapsed = generate_with_anthropic(client, prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Parse JSON response
        # Handle potential markdown code blocks
        if response_text.strip().startswith("```"):
            lines = response_text.strip().split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        slides_data = json.loads(response_text)
        slides = slides_data.get("slides", [])

        result = GenerationResult(
            input_file=str(input_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            model=model,
            slides=slides,
            input_words=input_words,
            output_slide_count=len(slides),
            generation_time_seconds=elapsed,
            timestamp=timestamp,
            success=True,
        )

    except Exception as e:
        result = GenerationResult(
            input_file=str(input_path),
            method=method,
            doc_id=doc_id,
            run_number=run_number,
            model=model,
            slides=[],
            input_words=input_words,
            output_slide_count=0,
            generation_time_seconds=0.0,
            timestamp=timestamp,
            success=False,
            error=str(e),
        )

    return result


def save_result(result: GenerationResult, output_dir: Path, use_new_structure: bool = False) -> Path:
    """Save generation result to JSON file."""

    if use_new_structure:
        # New structure: documents/{doc_id}/slides/{method}__run_N.json
        slides_dir = output_dir / result.doc_id / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.method}__run_{result.run_number}.json"
        output_path = slides_dir / filename
    else:
        # Old structure: generations/{timestamp}/method__doc_id__run_N.json
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.method}__{result.doc_id}__run_{result.run_number}.json"
        output_path = output_dir / filename

    output_path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def get_existing_outputs(output_dir: Path, use_new_structure: bool = False) -> set[str]:
    """Get set of already completed (method, doc_id, run) combinations."""
    existing = set()
    if not output_dir.exists():
        return existing

    if use_new_structure:
        # New structure: documents/{doc_id}/slides/{method}__run_N.json
        for doc_dir in output_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            slides_dir = doc_dir / "slides"
            if not slides_dir.exists():
                continue
            for f in slides_dir.glob("*.json"):
                stem = f.stem
                if "__run_" in stem:
                    parts = stem.rsplit("__run_", 1)
                    if len(parts) == 2:
                        method = parts[0]
                        run_num = parts[1]
                        existing.add(f"{method}__{doc_dir.name}__run_{run_num}")
    else:
        # Old structure: generations/{timestamp}/method__doc_id__run_N.json
        for f in output_dir.glob("*.json"):
            stem = f.stem
            if "__run_" in stem:
                parts = stem.rsplit("__run_", 1)
                if len(parts) == 2:
                    method_doc = parts[0]
                    run_num = parts[1]
                    existing.add(f"{method_doc}__run_{run_num}")

    return existing


def run(
    input_path: Path,
    output_dir: Path,
    model: str,
    provider: str,
    runs: int,
    delay: float,
    resume: bool,
    start: int = 0,
    limit: int | None = None,
) -> list[GenerationResult]:
    """Run LLM generation on input files."""

    # Detect if using new folder structure
    # New structure: documents/{doc_id}/chunks/{method}.txt
    # Old structure: runs/{id}/outputs/{method}__{doc_id}.txt
    use_new_structure = False
    input_files = []

    if input_path.is_file():
        input_files = [input_path]
        # Check if it's in new structure
        if input_path.parent.name == "chunks":
            use_new_structure = True
    elif input_path.is_dir():
        # Check if this is documents/ directory (new structure)
        # by looking for {doc_id}/chunks/ subdirectories
        for doc_dir in input_path.iterdir():
            chunks_dir = doc_dir / "chunks"
            if chunks_dir.is_dir():
                use_new_structure = True
                input_files.extend(sorted(chunks_dir.glob("*.txt")))

        # If not new structure, try old structure
        if not input_files:
            input_files = sorted(input_path.glob("*.txt"))

    if not input_files:
        raise SystemExit(f"No .txt files found in {input_path}")

    # Apply start and limit for batch processing
    if start > 0:
        input_files = input_files[start:]
    if limit is not None and limit > 0:
        input_files = input_files[:limit]

    if not input_files:
        raise SystemExit(f"No files to process after applying start={start}, limit={limit}")

    # Get existing outputs for resume
    existing = get_existing_outputs(output_dir, use_new_structure) if resume else set()

    results: list[GenerationResult] = []
    total_tasks = len(input_files) * runs
    completed = 0

    print(f"Processing {len(input_files)} files × {runs} runs = {total_tasks} total generations")
    if start > 0 or limit is not None:
        print(f"Batch: start={start}, limit={limit}")
    print(f"Model: {provider}/{model}")
    print(f"Output: {output_dir}")
    print(f"Structure: {'new (per-document)' if use_new_structure else 'old (flat)'}")
    if resume:
        print(f"Resume mode: skipping {len(existing)} existing outputs")
    print("-" * 60)

    for run_num in range(1, runs + 1):
        for input_file in input_files:
            if use_new_structure:
                method, doc_id = parse_chunk_path(input_file)
            else:
                method, doc_id = parse_filename(input_file.name)

            task_key = f"{method}__{doc_id}__run_{run_num}"

            # Skip if already exists (resume mode)
            if task_key in existing:
                completed += 1
                print(f"[{completed}/{total_tasks}] SKIP (exists): {task_key}")
                continue

            completed += 1
            print(f"[{completed}/{total_tasks}] Generating: {method} / {doc_id} (run {run_num})...", end=" ", flush=True)

            result = generate_slides(
                input_path=input_file,
                output_dir=output_dir,
                model=model,
                run_number=run_num,
                provider=provider,
                use_new_structure=use_new_structure,
            )

            if result.success:
                print(f"✓ {result.output_slide_count} slides in {result.generation_time_seconds:.1f}s")
            else:
                print(f"✗ Error: {result.error}")

            # Save result
            saved_path = save_result(result, output_dir, use_new_structure)
            results.append(result)

            # Delay between API calls
            if delay > 0 and completed < total_tasks:
                time.sleep(delay)

    # Summary
    print("-" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} successful")
    print(f"Results saved to: {output_dir}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM generation on pre-chunked documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all chunked outputs from a run
  python run_llm.py --input results/runs/20251220/outputs/

  # Single file test
  python run_llm.py --input results/runs/20251220/outputs/pac__doc1.txt

  # Multiple runs with delay
  python run_llm.py --input results/runs/20251220/outputs/ --runs 3 --delay 10

  # Use Claude instead of GPT-4
  python run_llm.py --input results/runs/20251220/outputs/ --provider anthropic --model claude-sonnet-4-20250514
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file or directory containing chunked .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generation results. Default: results/generations/<timestamp>/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per file for variance analysis (default: 1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between API calls (default: 5)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip existing outputs (regenerate all)",
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
        run_id = time.strftime("%Y%m%d-%H%M%S")
        script_dir = Path(__file__).resolve().parent.parent.parent
        args.output_dir = script_dir / "results" / "generations" / run_id

    run(
        input_path=args.input,
        output_dir=args.output_dir,
        model=args.model,
        provider=args.provider,
        runs=args.runs,
        delay=args.delay,
        resume=not args.no_resume,
        start=args.start,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
