#!/usr/bin/env python3
"""
Statistical analysis for multi-model, multi-judge experiments.

Handles:
- Multiple generation models (GPT-4o, Claude, Gemini)
- Multiple judge models (GPT-4o, Claude, Gemini)
- Multiple tasks (slides, summaries)
- Ablation budgets
- Sensitivity analysis
- Paired t-tests, effect sizes, cross-judge agreement

Usage:
    python compute_statistics_v2.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DOCUMENTS_DIR, RESULTS_DIR, CHUNKING_METHODS, SLIDE_METRICS, SUMMARY_METRICS

METRICS_BY_TASK = {
    "slides": SLIDE_METRICS + ["overall_score"],
    "summary": SUMMARY_METRICS + ["overall_score"],
}


def load_evaluations(task: str = "slides") -> list[dict]:
    """Load all evaluation results for a task from documents/*/evaluations/{task}/."""
    results = []
    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        eval_dir = doc_dir / "evaluations" / task
        if not eval_dir.exists():
            # Also check legacy evaluations_gemini for slides
            if task == "slides":
                eval_dir = doc_dir / "evaluations_gemini"
                if not eval_dir.exists():
                    continue
            else:
                continue

        for eval_file in sorted(eval_dir.glob("*.json")):
            try:
                data = json.loads(eval_file.read_text())
                if not data.get("success", False):
                    continue
                # Normalize fields
                data.setdefault("task", task)
                data.setdefault("judge_model", data.get("model", "gemini-2.0-flash"))
                data.setdefault("generator_model", "gpt-4o")  # legacy
                results.append(data)
            except Exception as e:
                print(f"Error loading {eval_file}: {e}")
    return results


def compute_descriptive(evaluations: list[dict], group_by: list[str], metrics: list[str]) -> dict:
    """Compute mean, std, 95% CI grouped by specified keys."""
    groups = defaultdict(lambda: defaultdict(list))

    for ev in evaluations:
        key_parts = []
        for field in group_by:
            val = ev.get(field, "unknown")
            key_parts.append(str(val))
        group_key = " | ".join(key_parts)

        for metric in metrics:
            if metric == "overall_score":
                score = ev.get("overall_score")
            else:
                score_obj = ev.get("scores", {}).get(metric, ev.get(metric, {}))
                if isinstance(score_obj, dict):
                    score = score_obj.get("score")
                else:
                    score = score_obj
            if score is not None:
                groups[group_key][metric].append(float(score))

    results = {}
    for group_key, metric_scores in groups.items():
        results[group_key] = {}
        for metric, scores in metric_scores.items():
            n = len(scores)
            if n < 2:
                continue
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            se = std / np.sqrt(n)
            ci_95 = 1.96 * se
            results[group_key][metric] = {
                "n": n,
                "mean": round(float(mean), 3),
                "std": round(float(std), 3),
                "se": round(float(se), 4),
                "ci_95_lower": round(float(mean - ci_95), 3),
                "ci_95_upper": round(float(mean + ci_95), 3),
            }
    return results


def compute_paired_tests(evaluations: list[dict], baseline_method: str = "truncation") -> dict:
    """Paired t-tests: baseline vs each other method, controlling for doc_id and judge."""
    # Group scores by (doc_id, judge_model, generator_model, method)
    scores_by_key = defaultdict(lambda: defaultdict(list))
    for ev in evaluations:
        doc_id = ev.get("doc_id", "")
        judge = ev.get("judge_model", "unknown")
        gen = ev.get("generator_model", "unknown")
        method = ev.get("method", "")
        overall = ev.get("overall_score", 0)
        if overall:
            scores_by_key[(doc_id, judge, gen)][method].append(float(overall))

    results = {}
    for method in CHUNKING_METHODS:
        if method == baseline_method:
            continue

        baseline_scores = []
        method_scores = []

        for key, method_dict in scores_by_key.items():
            if baseline_method in method_dict and method in method_dict:
                # Average across runs if multiple
                baseline_scores.append(np.mean(method_dict[baseline_method]))
                method_scores.append(np.mean(method_dict[method]))

        n = len(baseline_scores)
        if n < 5:
            continue

        baseline_arr = np.array(baseline_scores)
        method_arr = np.array(method_scores)

        t_stat, p_value = stats.ttest_rel(baseline_arr, method_arr)
        diff = baseline_arr - method_arr
        cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0

        results[f"{baseline_method}_vs_{method}"] = {
            "n": n,
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 5),
            "cohens_d": round(cohens_d, 4),
            "significant_05": bool(p_value < 0.05),
            "significant_01": bool(p_value < 0.01),
            "baseline_mean": round(float(baseline_arr.mean()), 3),
            "method_mean": round(float(method_arr.mean()), 3),
        }

    return results


def compute_judge_agreement(evaluations: list[dict]) -> dict:
    """Compute inter-judge agreement (Pearson correlation between judge pairs)."""
    # Group by (doc_id, method, generator) → {judge: overall_score}
    groups = defaultdict(dict)
    for ev in evaluations:
        key = (ev.get("doc_id"), ev.get("method"), ev.get("generator_model"))
        judge = ev.get("judge_model", "unknown")
        score = ev.get("overall_score", 0)
        if score:
            groups[key][judge] = float(score)

    # Get all judges
    all_judges = set()
    for judge_dict in groups.values():
        all_judges.update(judge_dict.keys())
    all_judges = sorted(all_judges)

    if len(all_judges) < 2:
        return {}

    correlations = {}
    for i, judge_a in enumerate(all_judges):
        for judge_b in all_judges[i + 1:]:
            scores_a = []
            scores_b = []
            for judge_dict in groups.values():
                if judge_a in judge_dict and judge_b in judge_dict:
                    scores_a.append(judge_dict[judge_a])
                    scores_b.append(judge_dict[judge_b])

            if len(scores_a) >= 10:
                r, p = stats.pearsonr(scores_a, scores_b)
                correlations[f"{judge_a}_vs_{judge_b}"] = {
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "n": len(scores_a),
                }

    return correlations


def compute_cross_model_consistency(evaluations: list[dict]) -> dict:
    """Check if chunking method rankings are consistent across generation models."""
    # Group by (generator_model, method) → [overall_scores]
    scores = defaultdict(lambda: defaultdict(list))
    for ev in evaluations:
        gen = ev.get("generator_model", "unknown")
        method = ev.get("method", "")
        overall = ev.get("overall_score", 0)
        if overall and method:
            scores[gen][method].append(float(overall))

    results = {}
    for gen, method_scores in scores.items():
        method_means = {}
        for method, s in method_scores.items():
            method_means[method] = round(float(np.mean(s)), 3)
        # Rank methods
        ranked = sorted(method_means.items(), key=lambda x: x[1], reverse=True)
        results[gen] = {
            "method_means": method_means,
            "ranking": [m for m, _ in ranked],
        }

    return results


def print_summary(task: str, desc_stats: dict, t_tests: dict, agreement: dict, consistency: dict):
    """Print human-readable summary."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {task.upper()}")
    print(f"{'='*80}")

    print("\n--- Descriptive Statistics (by Method) ---")
    for group, metrics in desc_stats.items():
        overall = metrics.get("overall_score", {})
        if overall:
            print(f"  {group:<50s}: {overall['mean']:.3f} ± {overall['std']:.3f} (n={overall['n']})")

    print("\n--- Paired T-Tests ---")
    for comparison, result in t_tests.items():
        sig = "**" if result["significant_01"] else ("*" if result["significant_05"] else "")
        print(f"  {comparison}: t={result['t_statistic']:.3f}, p={result['p_value']:.4f}{sig}, d={result['cohens_d']:.3f}")

    if agreement:
        print("\n--- Judge Agreement ---")
        for pair, result in agreement.items():
            print(f"  {pair}: r={result['pearson_r']:.3f}, p={result['p_value']:.4f} (n={result['n']})")

    if consistency:
        print("\n--- Cross-Model Rankings ---")
        for gen, data in consistency.items():
            print(f"  {gen}: {' > '.join(data['ranking'])}")


def main():
    all_results = {}

    for task in ["slides", "summary"]:
        print(f"\nLoading {task} evaluations...")
        evaluations = load_evaluations(task)
        if not evaluations:
            print(f"  No evaluations found for {task}")
            continue

        print(f"  Loaded {len(evaluations)} evaluations")
        metrics = METRICS_BY_TASK[task]

        # Descriptive stats by method
        desc_by_method = compute_descriptive(evaluations, ["method"], metrics)

        # Descriptive stats by method × generator
        desc_by_method_gen = compute_descriptive(evaluations, ["method", "generator_model"], metrics)

        # Descriptive stats by method × judge
        desc_by_method_judge = compute_descriptive(evaluations, ["method", "judge_model"], metrics)

        # Paired t-tests
        t_tests = compute_paired_tests(evaluations)

        # Judge agreement
        agreement = compute_judge_agreement(evaluations)

        # Cross-model consistency
        consistency = compute_cross_model_consistency(evaluations)

        print_summary(task, desc_by_method, t_tests, agreement, consistency)

        all_results[task] = {
            "n_evaluations": len(evaluations),
            "descriptive_by_method": desc_by_method,
            "descriptive_by_method_generator": desc_by_method_gen,
            "descriptive_by_method_judge": desc_by_method_judge,
            "paired_t_tests": t_tests,
            "judge_agreement": agreement,
            "cross_model_consistency": consistency,
        }

    # Save
    output_file = RESULTS_DIR / "statistical_analysis_v2.json"
    output_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
