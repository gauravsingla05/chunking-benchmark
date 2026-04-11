"""
Shared utilities for experiment scripts: env loading, client creation, retry logic, batch APIs.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Optional

# ─── Env Loading ──────────────────────────────────────────────

def load_env():
    """Load .env file from backend if it exists."""
    from config import ENV_PATHS
    for env_path in ENV_PATHS:
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
    print("Warning: No .env file found.")

load_env()

# ─── LLM Client Factories ────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


def get_openai_client() -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None


def get_anthropic_client() -> Optional[Anthropic]:
    if Anthropic is None:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key) if api_key else None


def get_google_client():
    if genai is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else None


# ─── Retry Logic ──────────────────────────────────────────────

def retry_with_backoff(func, max_retries=3, base_delay=60.0, max_delay=300.0):
    """Retry with exponential backoff on rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(p in error_str for p in [
                "rate limit", "rate_limit", "too many requests", "429",
                "quota exceeded", "resource exhausted", "overloaded"
            ])
            if not is_rate_limit or attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 10), max_delay)
            print(f"\n  ⚠️  Rate limit. Waiting {delay:.0f}s (retry {attempt + 1}/{max_retries})...")
            time.sleep(delay)


# ─── Generation Helpers ───────────────────────────────────────

def generate_openai(client, prompt: str, model: str = "gpt-4o", max_tokens: int = 4096) -> tuple[str, float]:
    def _call():
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content, time.time() - start
    return retry_with_backoff(_call)


def generate_anthropic(client, prompt: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096) -> tuple[str, float]:
    def _call():
        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text, time.time() - start
    return retry_with_backoff(_call)


def generate_google(client, prompt: str, model: str = "gemini-2.0-flash", max_tokens: int = 4096) -> tuple[str, float]:
    def _call():
        start = time.time()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text, time.time() - start
    return retry_with_backoff(_call)


def generate(provider: str, prompt: str, model: str, max_tokens: int = 4096) -> tuple[str, float]:
    """Unified generation interface. Returns (response_text, elapsed_seconds)."""
    if provider == "openai":
        client = get_openai_client()
        if not client:
            raise ValueError("OpenAI client not available. Set OPENAI_API_KEY.")
        return generate_openai(client, prompt, model, max_tokens)
    elif provider == "anthropic":
        client = get_anthropic_client()
        if not client:
            raise ValueError("Anthropic client not available. Set ANTHROPIC_API_KEY.")
        return generate_anthropic(client, prompt, model, max_tokens)
    elif provider == "google":
        client = get_google_client()
        if not client:
            raise ValueError("Google client not available. Set GOOGLE_API_KEY.")
        return generate_google(client, prompt, model, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── JSON Parsing Helper ─────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)
    return json.loads(text)


# ─── PDF Reading ──────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    """Read PDF text using pdftotext."""
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout


def find_source_pdf(doc_id: str, source_dir: Path) -> Optional[Path]:
    """Find the source PDF for a document ID."""
    pdf_path = source_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    for f in source_dir.iterdir():
        if f.suffix.lower() == ".pdf" and f.stem.lower() == doc_id.lower():
            return f
    return None


# ─── Batch API Support ────────────────────────────────────────

def create_openai_batch(client, requests: list[dict], description: str = "") -> str:
    """
    Submit an OpenAI batch job. Returns batch_id.

    Each request: {"custom_id": str, "model": str, "messages": [...], "max_tokens": int}
    """
    import tempfile

    # Write JSONL
    lines = []
    for req in requests:
        lines.append(json.dumps({
            "custom_id": req["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": req["model"],
                "messages": req["messages"],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": req.get("max_tokens", 4096),
            }
        }))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        jsonl_path = f.name

    # Upload file
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )

    os.unlink(jsonl_path)
    print(f"  OpenAI batch created: {batch.id} ({len(requests)} requests)")
    return batch.id


def create_anthropic_batch(client, requests: list[dict], description: str = "") -> str:
    """
    Submit an Anthropic message batch. Returns batch_id.

    Each request: {"custom_id": str, "model": str, "messages": [...], "max_tokens": int}
    """
    batch_requests = []
    for req in requests:
        batch_requests.append({
            "custom_id": req["custom_id"],
            "params": {
                "model": req["model"],
                "max_tokens": req.get("max_tokens", 4096),
                "messages": req["messages"],
            }
        })

    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  Anthropic batch created: {batch.id} ({len(requests)} requests)")
    return batch.id


def poll_openai_batch(client, batch_id: str, poll_interval: int = 60, max_wait: int = 86400) -> list[dict]:
    """Poll OpenAI batch until complete. Returns list of results."""
    elapsed = 0
    while elapsed < max_wait:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status == "completed":
            # Download results
            content = client.files.content(batch.output_file_id)
            results = []
            for line in content.text.strip().split("\n"):
                results.append(json.loads(line))
            print(f"  OpenAI batch {batch_id} completed: {len(results)} results")
            return results
        elif status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {batch_id} {status}")
        else:
            print(f"  Batch {batch_id}: {status} ({batch.request_counts}) — waiting {poll_interval}s...")
            time.sleep(poll_interval)
            elapsed += poll_interval

    raise TimeoutError(f"OpenAI batch {batch_id} did not complete within {max_wait}s")


def poll_anthropic_batch(client, batch_id: str, poll_interval: int = 60, max_wait: int = 86400) -> list[dict]:
    """Poll Anthropic batch until complete. Returns list of results."""
    elapsed = 0
    while elapsed < max_wait:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        if status == "ended":
            results = []
            for result in client.messages.batches.results(batch_id):
                results.append({
                    "custom_id": result.custom_id,
                    "result": {
                        "type": result.result.type,
                        "message": {
                            "content": [{"text": result.result.message.content[0].text}]
                        } if result.result.type == "succeeded" else None,
                        "error": str(result.result.error) if result.result.type != "succeeded" else None,
                    }
                })
            print(f"  Anthropic batch {batch_id} ended: {len(results)} results")
            return results
        else:
            counts = batch.request_counts
            print(f"  Batch {batch_id}: {status} (done={counts.succeeded + counts.errored}/{counts.processing + counts.succeeded + counts.errored}) — waiting {poll_interval}s...")
            time.sleep(poll_interval)
            elapsed += poll_interval

    raise TimeoutError(f"Anthropic batch {batch_id} did not complete within {max_wait}s")


# ─── Parallel Execution ───────────────────────────────────

def run_parallel_google(tasks: list[dict], worker_fn, max_workers: int = 10, delay_per_worker: float = 1.0):
    """
    Run Google API tasks in parallel using ThreadPoolExecutor.

    tasks: list of dicts with task data
    worker_fn: function(task_dict) -> result
    max_workers: concurrent threads (10-20 recommended for Gemini)

    Returns list of results in order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = [None] * len(tasks)
    lock = threading.Lock()
    completed = [0]

    def _worker(idx_task):
        idx, task = idx_task
        # Stagger start to avoid burst
        time.sleep(idx % max_workers * delay_per_worker)
        try:
            result = worker_fn(task)
            with lock:
                completed[0] += 1
                if completed[0] % 20 == 0:
                    print(f"  [{completed[0]}/{len(tasks)}] completed...", flush=True)
            return idx, result
        except Exception as e:
            return idx, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, (i, t)) for i, t in enumerate(tasks)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


def slides_to_text(slides: list) -> str:
    """Convert slides JSON to readable text for evaluation."""
    lines = []
    for slide in slides:
        lines.append(f"--- Slide {slide.get('slide_number', '?')}: {slide.get('type', 'unknown')} ---")
        if slide.get("title"):
            lines.append(f"Title: {slide['title']}")
        if slide.get("body"):
            lines.append(f"Body: {slide['body']}")
        if slide.get("bullet_points"):
            for bp in slide["bullet_points"]:
                lines.append(f"  - {bp.get('title', '')}: {bp.get('body', '')}")
        lines.append("")
    return "\n".join(lines)
