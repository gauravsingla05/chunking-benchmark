#!/usr/bin/env python3
"""
Statistical Analysis for Document Chunking Research Paper
Computes means, standard deviations, confidence intervals, and p-values
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics
from scipy import stats
import numpy as np

# Paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "documents"

METRICS = [
    "completeness",
    "accuracy",
    "statistics_retention",
    "coherence",
    "relevance",
    "coverage_balance",
    "overall_score"
]

METHODS = [
    "truncation",
    "fixed_size_first_last",
    "semantic_breakpoint",
    "pac_position_aware"
]


def load_all_evaluations():
    """Load all evaluation JSON files into a structured dict."""
    data = defaultdict(lambda: defaultdict(list))
    doc_count = 0

    for doc_dir in sorted(RESULTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue

        eval_dir = doc_dir / "evaluations_gemini"
        if not eval_dir.exists():
            continue

        doc_methods = set()
        for eval_file in sorted(eval_dir.glob("*.json")):
            try:
                with open(eval_file) as f:
                    result = json.load(f)

                if not result.get("success", False):
                    continue

                method = result["method"]
                doc_id = result["doc_id"]
                doc_methods.add(method)

                for metric in METRICS:
                    if metric == "overall_score":
                        score = result.get("overall_score")
                    else:
                        score = result.get(metric, {}).get("score")

                    if score is not None:
                        data[method][metric].append(score)

                # Store doc_id for paired tests
                data[method]["_doc_ids"].append(doc_id)

            except Exception as e:
                print(f"Error loading {eval_file}: {e}")

        if doc_methods:
            doc_count += 1

    print(f"Loaded evaluations from {doc_count} documents")
    return data


def compute_descriptive_stats(data):
    """Compute mean, std, and 95% CI for each method and metric."""
    results = {}

    for method in METHODS:
        results[method] = {}
        for metric in METRICS:
            scores = data[method][metric]
            if len(scores) < 2:
                continue

            n = len(scores)
            mean = statistics.mean(scores)
            std = statistics.stdev(scores)
            se = std / (n ** 0.5)
            ci_95 = 1.96 * se

            results[method][metric] = {
                "n": n,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "se": round(se, 3),
                "ci_95_lower": round(mean - ci_95, 2),
                "ci_95_upper": round(mean + ci_95, 2)
            }

    return results


def compute_paired_t_tests(data):
    """Compute paired t-tests between truncation and other methods."""
    results = {}
    baseline = "truncation"

    for method in METHODS:
        if method == baseline:
            continue

        results[f"{baseline}_vs_{method}"] = {}

        for metric in METRICS:
            baseline_scores = data[baseline][metric]
            method_scores = data[method][metric]

            # Need same number of samples for paired test
            n = min(len(baseline_scores), len(method_scores))
            if n < 5:
                continue

            baseline_arr = np.array(baseline_scores[:n])
            method_arr = np.array(method_scores[:n])

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_arr, method_arr)

            # Effect size (Cohen's d for paired samples)
            diff = baseline_arr - method_arr
            cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

            results[f"{baseline}_vs_{method}"][metric] = {
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "cohens_d": round(cohens_d, 3),
                "significant_05": p_value < 0.05,
                "significant_01": p_value < 0.01
            }

    return results


def print_latex_table(desc_stats):
    """Print LaTeX-formatted table for the paper."""
    print("\n" + "="*80)
    print("LATEX TABLE (with standard deviations)")
    print("="*80)

    print(r"""
\begin{table}[h]
\centering
\caption{Evaluation Results by Chunking Method (Mean $\pm$ SD)}
\label{tab:results}
\begin{tabular}{lccccccc}
\toprule
\textbf{Method} & \textbf{Comp.} & \textbf{Acc.} & \textbf{Stats} & \textbf{Coh.} & \textbf{Rel.} & \textbf{Bal.} & \textbf{Overall} \\
\midrule""")

    method_names = {
        "truncation": "Truncation",
        "fixed_size_first_last": "Fixed",
        "semantic_breakpoint": "Semantic",
        "pac_position_aware": "PAC"
    }

    for method in METHODS:
        row = [method_names[method]]
        for metric in METRICS:
            if metric in desc_stats.get(method, {}):
                s = desc_stats[method][metric]
                row.append(f"{s['mean']:.2f}$\\pm${s['std']:.2f}")
            else:
                row.append("-")
        print(" & ".join(row) + r" \\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def print_summary(desc_stats, t_tests):
    """Print human-readable summary."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)

    for method in METHODS:
        print(f"\n{method.upper()}")
        print("-" * 40)
        for metric in METRICS:
            if metric in desc_stats.get(method, {}):
                s = desc_stats[method][metric]
                print(f"  {metric:20s}: {s['mean']:.2f} +/- {s['std']:.2f} (95% CI: [{s['ci_95_lower']:.2f}, {s['ci_95_upper']:.2f}]) n={s['n']}")

    print("\n" + "="*80)
    print("PAIRED T-TESTS (Truncation vs Others)")
    print("="*80)

    for comparison, metrics in t_tests.items():
        print(f"\n{comparison}")
        print("-" * 40)
        for metric, result in metrics.items():
            sig = "**" if result["significant_01"] else ("*" if result["significant_05"] else "")
            print(f"  {metric:20s}: t={result['t_statistic']:6.3f}, p={result['p_value']:.4f}{sig}, d={result['cohens_d']:.3f}")


def main():
    print("Loading evaluation data...")
    data = load_all_evaluations()

    print("\nComputing descriptive statistics...")
    desc_stats = compute_descriptive_stats(data)

    print("Computing paired t-tests...")
    t_tests = compute_paired_t_tests(data)

    print_summary(desc_stats, t_tests)
    print_latex_table(desc_stats)

    # Save results
    output = {
        "descriptive_statistics": desc_stats,
        "paired_t_tests": t_tests
    }

    output_file = RESULTS_DIR.parent / "statistical_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
