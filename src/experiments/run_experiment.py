from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methods import list_methods  # noqa: E402
from methods.registry import get_method  # noqa: E402
import methods.truncation  # noqa: F401, E402
import methods.fixed_size_first_last  # noqa: F401, E402
import methods.semantic_breakpoint  # noqa: F401, E402
import methods.pac_position_aware  # noqa: F401, E402
from utils.text_io import iter_documents  # noqa: E402


@dataclass(frozen=True)
class ExperimentResult:
    doc_id: str
    method: str
    original_words: int
    output_words: int
    token_reduction_pct: float
    processing_time_seconds: float


def token_reduction_pct(original_words: int, output_words: int) -> float:
    if original_words <= 0:
        return 0.0
    return (1 - (output_words / original_words)) * 100


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run(
    *,
    input_dir: Path,
    output_dir: Path,
    methods: list[str],
    max_words: int,
    limit_docs: int | None,
    save_outputs: bool,
) -> list[ExperimentResult]:
    documents = iter_documents(input_dir, limit=limit_docs)
    if not documents:
        raise SystemExit(f"No documents found in {input_dir} (expected .pdf/.txt/.md).")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / "runs" / run_id
    _ensure_dir(run_dir)

    # New folder structure: results/documents/{doc_id}/chunks/
    docs_dir = output_dir / "documents"

    results: list[ExperimentResult] = []

    for doc in documents:
        original_words = len(doc.text.split())

        # Create document folder with chunks subdirectory
        if save_outputs:
            doc_chunks_dir = docs_dir / doc.doc_id / "chunks"
            _ensure_dir(doc_chunks_dir)

        for method_name in methods:
            spec = get_method(method_name)
            start = time.time()
            processed = spec.chunker(doc.text, max_words)
            elapsed = time.time() - start

            out_words = len(processed.split())
            results.append(
                ExperimentResult(
                    doc_id=doc.doc_id,
                    method=method_name,
                    original_words=original_words,
                    output_words=out_words,
                    token_reduction_pct=token_reduction_pct(original_words, out_words),
                    processing_time_seconds=elapsed,
                )
            )

            if save_outputs:
                # Save to documents/{doc_id}/chunks/{method}.txt
                out_path = doc_chunks_dir / f"{method_name}.txt"
                out_path.write_text(processed, encoding="utf-8", errors="ignore")

    (run_dir / "experiment_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "methods": methods,
                "max_words": max_words,
                "limit_docs": limit_docs,
                "save_outputs": save_outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {run_dir / 'experiment_results.json'}")
    if save_outputs:
        print(f"Chunks saved to: {docs_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunking experiments on local documents.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research-paper/data/raw"),
        help="Directory containing .pdf/.txt/.md documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research-paper/results"),
        help="Directory to write results.",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method name to run. Repeat for multiple. Use 'all' to run all registered methods.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2000,
        help="Max words budget for the chunked output.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=10,
        help="Max number of docs to process (for pilot runs). Use 0 for none.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Also save the extracted output text for each doc+method to results/runs/<id>/outputs/.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print available methods and exit.",
    )
    args = parser.parse_args()

    if args.list_methods:
        for spec in list_methods():
            print(f"{spec.name}: {spec.description}")
        return

    methods = args.methods or ["truncation"]
    if "all" in methods:
        methods = [m.name for m in list_methods()]

    limit_docs = None if args.limit_docs == 0 else args.limit_docs
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        methods=methods,
        max_words=args.max_words,
        limit_docs=limit_docs,
        save_outputs=args.save_outputs,
    )


if __name__ == "__main__":
    main()
