#!/usr/bin/env python3
"""
Author: Gourav Singla
Date: 2025-12-20
Description: Extract detailed summaries from PDFs using Gemini (cost-effective for long docs).

Usage:
    # Process all PDFs in data/raw/
    python run_summary.py --input data/raw/ --output results/documents/

    # Process a single PDF
    python run_summary.py --input data/raw/doc_A.pdf --output results/documents/
"""

from __future__ import annotations

import argparse
import json
import os
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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SUMMARY_PROMPT = """You are an expert document analyst. Analyze this document thoroughly and extract a comprehensive summary.

IMPORTANT: This summary will be used as ground truth to evaluate how well different text chunking methods preserve important information. Be thorough and include ALL key facts, statistics, and findings.

## Document Content:
{document_text}

## Task

Extract and return a JSON object with this exact structure:

{{
  "title": "Document title or main topic",
  "document_type": "research_paper | business_report | news_article | technical_doc | medical_study | other",
  "domain": "technology | business | medical | science | policy | education | other",
  "total_words": <approximate word count>,

  "key_facts": [
    {{
      "fact": "Specific factual statement with numbers/names",
      "importance": "critical | high | medium",
      "category": "finding | methodology | background | statistic | conclusion"
    }}
  ],

  "statistics": [
    "71% of users preferred option A",
    "$2.3M in cost savings",
    "p < 0.05, n = 500"
  ],

  "main_findings": [
    "Primary finding or conclusion 1",
    "Primary finding or conclusion 2"
  ],

  "methodology": "Brief description of methods, data sources, or approach used",

  "conclusions": [
    "Key takeaway or recommendation 1",
    "Key takeaway or recommendation 2"
  ],

  "entities": {{
    "organizations": ["Company A", "University B"],
    "people": ["Author Name", "Expert quoted"],
    "locations": ["California", "Global"],
    "time_periods": ["2016-2019", "Q4 2024"]
  }},

  "abstract_or_summary": "If the document has an abstract or executive summary, include it here verbatim or paraphrased"
}}

## Guidelines

1. **Be exhaustive with key_facts** - Include 15-30 facts for a typical document
2. **Preserve ALL statistics** - Every number, percentage, dollar amount matters
3. **Include context** - Facts should be understandable standalone
4. **Mark importance accurately**:
   - critical: Core findings, main conclusions, headline statistics
   - high: Supporting evidence, secondary findings
   - medium: Background info, context, minor details
5. **Categorize correctly**:
   - finding: Research results, discoveries
   - methodology: How the study/analysis was done
   - background: Context, prior work, setting
   - statistic: Numerical data, measurements
   - conclusion: Recommendations, implications

Return ONLY the JSON object, no additional text or markdown.
"""


@dataclass
class SummaryResult:
    """Result of summarizing a document."""
    doc_id: str
    source_path: str
    title: str
    document_type: str
    domain: str
    total_words: int
    key_facts: list
    statistics: list
    main_findings: list
    methodology: str
    conclusions: list
    entities: dict
    abstract_or_summary: str
    extraction_time_seconds: float
    extracted_at: str
    model: str
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


def get_openai_client():
    """Configure and return OpenAI client."""
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


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


def read_document(path: Path) -> str:
    """Read document content."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_summary_with_gemini(
    client,
    document_text: str,
    model: str = "gemini-2.0-flash",
) -> tuple[dict, float]:
    """Extract summary using Gemini."""

    prompt = SUMMARY_PROMPT.format(document_text=document_text)

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=4096,
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

    summary = json.loads(response_text)
    return summary, elapsed


def extract_summary_with_openai(
    client,
    document_text: str,
    model: str = "gpt-4o",
) -> tuple[dict, float]:
    """Extract summary using OpenAI."""

    prompt = SUMMARY_PROMPT.format(document_text=document_text)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
    )
    elapsed = time.time() - start

    response_text = response.choices[0].message.content.strip()

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

    summary = json.loads(response_text)
    return summary, elapsed


def process_document(
    source_path: Path,
    output_dir: Path,
    model: str,
    client,
    use_openai: bool = False,
) -> SummaryResult:
    """Process a single document and save summary."""

    doc_id = source_path.stem
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Create document folder
    doc_folder = output_dir / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Read document
        document_text = read_document(source_path)
        word_count = len(document_text.split())

        # Extract summary
        if use_openai:
            summary, elapsed = extract_summary_with_openai(client, document_text, model)
        else:
            summary, elapsed = extract_summary_with_gemini(client, document_text, model)

        result = SummaryResult(
            doc_id=doc_id,
            source_path=str(source_path),
            title=summary.get("title", doc_id),
            document_type=summary.get("document_type", "unknown"),
            domain=summary.get("domain", "unknown"),
            total_words=summary.get("total_words", word_count),
            key_facts=summary.get("key_facts", []),
            statistics=summary.get("statistics", []),
            main_findings=summary.get("main_findings", []),
            methodology=summary.get("methodology", ""),
            conclusions=summary.get("conclusions", []),
            entities=summary.get("entities", {}),
            abstract_or_summary=summary.get("abstract_or_summary", ""),
            extraction_time_seconds=elapsed,
            extracted_at=timestamp,
            model=model,
            success=True,
        )

    except Exception as e:
        result = SummaryResult(
            doc_id=doc_id,
            source_path=str(source_path),
            title="",
            document_type="",
            domain="",
            total_words=0,
            key_facts=[],
            statistics=[],
            main_findings=[],
            methodology="",
            conclusions=[],
            entities={},
            abstract_or_summary="",
            extraction_time_seconds=0,
            extracted_at=timestamp,
            model=model,
            success=False,
            error=str(e),
        )

    # Save summary
    summary_path = doc_folder / "summary.json"
    summary_path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def run(
    input_path: Path,
    output_dir: Path,
    model: str,
    delay: float,
    skip_existing: bool,
) -> list[SummaryResult]:
    """Run summary extraction on documents."""

    # Try Gemini first, fallback to OpenAI
    use_openai = False
    client = get_gemini_client()
    if not client:
        print("Gemini not available, trying OpenAI...")
        client = get_openai_client()
        use_openai = True
        if model.startswith("gemini"):
            model = "gpt-4o"  # Switch to OpenAI model
    if not client:
        raise SystemExit("No LLM client available. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

    # Determine input files
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
        )
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not input_files:
        raise SystemExit(f"No documents found in {input_path}")

    results: list[SummaryResult] = []
    total = len(input_files)

    print(f"Processing {total} documents")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    for i, doc_path in enumerate(input_files, 1):
        doc_id = doc_path.stem

        # Skip if exists
        if skip_existing:
            existing_summary = output_dir / doc_id / "summary.json"
            if existing_summary.exists():
                print(f"[{i}/{total}] SKIP (exists): {doc_id}")
                continue

        print(f"[{i}/{total}] Extracting summary: {doc_id}...", end=" ", flush=True)

        result = process_document(doc_path, output_dir, model, client, use_openai)

        if result.success:
            fact_count = len(result.key_facts)
            print(f"✓ {fact_count} facts, {result.total_words} words in {result.extraction_time_seconds:.1f}s")
        else:
            print(f"✗ Error: {result.error}")

        results.append(result)

        # Delay between API calls
        if delay > 0 and i < total:
            time.sleep(delay)

    # Summary
    print("-" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"Completed: {success_count}/{len(results)} successful")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract detailed summaries from documents using Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_summary.py --input data/raw/ --output results/documents/
  python run_summary.py --input data/raw/doc_A.pdf --output results/documents/
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file or directory containing PDFs/TXT/MD",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for document folders. Default: results/documents/",
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
        "--no-skip",
        action="store_true",
        help="Don't skip existing summaries (regenerate all)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output is None:
        script_dir = Path(__file__).resolve().parent.parent.parent
        args.output = script_dir / "results" / "documents"

    run(
        input_path=args.input,
        output_dir=args.output,
        model=args.model,
        delay=args.delay,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
