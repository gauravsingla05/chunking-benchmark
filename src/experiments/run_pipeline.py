#!/usr/bin/env python3
"""
Master pipeline orchestrator for the full research experiment.

Runs all experiment stages in order:
  1. Chunk documents (4 methods × default budget)
  2. Ablation chunking (4 methods × 4 budgets)
  3. W-curve sensitivity analysis (11 configs)
  4. Generate slides (3 models × 4 methods)
  5. Generate summaries (3 models × 4 methods)
  6. Evaluate slides (3 judges × all generation outputs)
  7. Evaluate summaries (3 judges × all generation outputs)
  8. Compute statistics
  9. Export human evaluation pairs

Each stage is idempotent — re-running skips completed work.

Usage:
    # Full pipeline with batch APIs
    python run_pipeline.py --batch

    # Specific stages
    python run_pipeline.py --stages chunk generate evaluate

    # Dry run — show what would be done
    python run_pipeline.py --dry-run

    # Quick pilot (5 docs)
    python run_pipeline.py --limit-docs 5 --stages chunk generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    GENERATION_MODELS, JUDGE_MODELS, CHUNKING_METHODS,
    DEFAULT_BUDGET, ABLATION_BUDGETS, SENSITIVITY_CONFIGS,
    DOCUMENTS_DIR, DATA_DIR, RESULTS_DIR,
)


ALL_STAGES = [
    "chunk",       # 1. Chunk with default budget
    "ablation",    # 2. Chunk with multiple budgets
    "sensitivity", # 3. W-curve parameter variations
    "generate",    # 4. Generate slides + summaries
    "evaluate",    # 5. Multi-judge evaluation
    "stats",       # 6. Statistical analysis
    "human_eval",  # 7. Export human eval pairs
]


def print_banner(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def stage_chunk(limit_docs):
    """Stage 1: Chunk all documents with default budget."""
    print_banner("STAGE 1: Chunking (default budget)")
    from experiments.run_experiment import run as run_chunking
    run_chunking(
        input_dir=DATA_DIR,
        output_dir=RESULTS_DIR,
        methods=CHUNKING_METHODS,
        max_words=DEFAULT_BUDGET,
        limit_docs=limit_docs,
        save_outputs=True,
    )


def stage_ablation(limit_docs):
    """Stage 2: Ablation chunking with multiple budgets."""
    print_banner("STAGE 2: Ablation Chunking")
    from experiments.run_ablation import run_ablation_chunking
    run_ablation_chunking(ABLATION_BUDGETS, limit_docs)


def stage_sensitivity(limit_docs):
    """Stage 3: W-curve sensitivity analysis."""
    print_banner("STAGE 3: W-curve Sensitivity Analysis")
    from experiments.run_sensitivity import run_sensitivity
    run_sensitivity(SENSITIVITY_CONFIGS, DEFAULT_BUDGET, limit_docs)


def stage_generate(batch: bool, limit_docs_approx: int | None):
    """Stage 4: Generate slides + summaries with all models."""
    from experiments.run_generation import collect_tasks, run_batch, run_realtime

    for task in ["slides", "summary"]:
        print_banner(f"STAGE 4: Generate {task}")
        tasks = collect_tasks(task, GENERATION_MODELS, runs=1)
        if not tasks:
            print(f"  All {task} generation already complete.")
            continue

        print(f"  {len(tasks)} tasks to process")
        if batch:
            run_batch(task, tasks)
        else:
            run_realtime(task, tasks, delay=5.0)


def stage_evaluate(batch: bool):
    """Stage 5: Evaluate with all judges."""
    from experiments.run_evaluation_multi import collect_eval_tasks, run_batch_eval, run_realtime_eval

    for task in ["slides", "summary"]:
        print_banner(f"STAGE 5: Evaluate {task}")
        tasks = collect_eval_tasks(task, JUDGE_MODELS)
        if not tasks:
            print(f"  All {task} evaluations already complete.")
            continue

        print(f"  {len(tasks)} evaluations to process")
        if batch:
            run_batch_eval(task, tasks)
        else:
            run_realtime_eval(task, tasks, delay=3.0)


def stage_stats():
    """Stage 6: Compute statistics."""
    print_banner("STAGE 6: Statistical Analysis")
    from analysis.compute_statistics_v2 import main as compute_stats
    compute_stats()


def stage_human_eval():
    """Stage 7: Export human evaluation pairs."""
    print_banner("STAGE 7: Human Evaluation Export")
    from experiments.run_human_eval import select_pairs, export_csv, export_html, export_answer_key
    output_dir = RESULTS_DIR / "human_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = select_pairs(30)
    if pairs:
        export_csv(pairs, output_dir)
        export_html(pairs, output_dir)
        export_answer_key(pairs, output_dir)
    else:
        print("  No eligible pairs yet — run generation first.")


def estimate_costs(dry_run: bool = True):
    """Estimate API costs for remaining work."""
    from experiments.run_generation import collect_tasks
    from experiments.run_evaluation_multi import collect_eval_tasks

    print_banner("COST ESTIMATE")

    # Count tasks per provider
    gen_counts = {"openai": 0, "anthropic": 0, "google": 0}
    eval_counts = {"openai": 0, "anthropic": 0, "google": 0}

    for task in ["slides", "summary"]:
        for t in collect_tasks(task, GENERATION_MODELS):
            gen_counts[t["model"]["provider"]] += 1
        for t in collect_eval_tasks(task, JUDGE_MODELS):
            eval_counts[t["judge"]["provider"]] += 1

    # Cost per call estimates (input + output tokens)
    # Generation: ~4K input + 3K output tokens per call
    # Evaluation: ~8K input + 800 output tokens per call
    gen_costs = {
        "openai":    {"input_per_1m": 2.50, "output_per_1m": 10.00},  # gpt-4o
        "anthropic": {"input_per_1m": 3.00, "output_per_1m": 15.00},  # claude sonnet
        "google":    {"input_per_1m": 0.10, "output_per_1m": 0.40},   # gemini flash
    }

    total_cost = 0
    print(f"{'Provider':<12} {'Gen Tasks':>10} {'Eval Tasks':>11} {'Est. Cost':>10}")
    print("-" * 47)

    for provider in ["openai", "anthropic", "google"]:
        gc = gen_counts[provider]
        ec = eval_counts[provider]
        rates = gen_costs[provider]

        # Generation: ~4K input, ~3K output tokens per call
        gen_cost = gc * (4000 * rates["input_per_1m"] / 1e6 + 3000 * rates["output_per_1m"] / 1e6)
        # Evaluation: ~8K input, ~800 output tokens per call
        eval_cost = ec * (8000 * rates["input_per_1m"] / 1e6 + 800 * rates["output_per_1m"] / 1e6)

        # 50% batch discount for openai/anthropic
        if provider in ("openai", "anthropic"):
            gen_cost *= 0.5
            eval_cost *= 0.5

        cost = gen_cost + eval_cost
        total_cost += cost
        print(f"{provider:<12} {gc:>10} {ec:>11} ${cost:>8.2f}")

    print("-" * 47)
    print(f"{'TOTAL':<12} {sum(gen_counts.values()):>10} {sum(eval_counts.values()):>11} ${total_cost:>8.2f}")
    print(f"\n(Estimates include 50% batch discount for OpenAI/Anthropic)")


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for research experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stages: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES,
                        choices=ALL_STAGES, help="Stages to run")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch APIs for OpenAI/Anthropic (50% savings)")
    parser.add_argument("--limit-docs", type=int, default=None,
                        help="Limit documents for pilot runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show cost estimate without running")

    args = parser.parse_args()
    limit = None if args.limit_docs == 0 else args.limit_docs

    print(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stages: {args.stages}")
    print(f"Batch mode: {args.batch}")
    if limit:
        print(f"Document limit: {limit}")

    estimate_costs()

    if args.dry_run:
        print("\n[DRY RUN] No work performed.")
        return

    start_time = time.time()

    for stage in args.stages:
        if stage == "chunk":
            stage_chunk(limit)
        elif stage == "ablation":
            stage_ablation(limit)
        elif stage == "sensitivity":
            stage_sensitivity(limit)
        elif stage == "generate":
            stage_generate(args.batch, limit)
        elif stage == "evaluate":
            stage_evaluate(args.batch)
        elif stage == "stats":
            stage_stats()
        elif stage == "human_eval":
            stage_human_eval()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
