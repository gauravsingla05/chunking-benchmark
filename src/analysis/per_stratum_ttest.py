#!/usr/bin/env python3
"""
Per-stratum paired t-tests addressing the reviewer objection that the PAC
"medium-document niche" (5,000-10,000 words) rests on a 0.06-point mean
difference without per-stratum significance testing.

For each length bucket, compares PAC against each other chunking method using a
paired t-test over (doc_id, judge, generator) triples, the same pairing used
in full_analysis.py.

Outputs results/per_stratum_ttest.json with per-bucket statistics.
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

from analysis.full_analysis import load_slide_evaluations, length_bucket  # noqa: E402
from config import RESULTS_DIR  # noqa: E402

METHODS = ["truncation", "fixed_size_first_last", "semantic_breakpoint", "pac_position_aware", "recursive_character"]


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def main() -> None:
    evals = load_slide_evaluations()

    # paired[bucket][(doc, judge, gen)][method] = score
    paired: dict = defaultdict(lambda: defaultdict(dict))

    for ev in evals:
        method = ev.get("method", "")
        if method not in METHODS:
            continue
        score = ev.get("overall_score")
        if not score:
            continue
        wc = ev.get("document_word_count", 0)
        if not wc:
            continue
        bucket = length_bucket(wc)
        judge = ev.get("judge_model", ev.get("model", "unknown"))
        gen = ev.get("generator_model", "gpt-4o")
        doc_id = ev.get("doc_id", "")
        if not doc_id:
            continue
        paired[bucket][(doc_id, judge, gen)][method] = score

    report: dict = {}
    pac = "pac_position_aware"

    for bucket in ["<5K", "5-10K", "10-20K", "20K+"]:
        bucket_records = paired.get(bucket, {})
        bucket_report = {
            "n_unique_keys": len(bucket_records),
            "method_means": {},
            "pac_vs_method": {},
        }

        # Per-method descriptive within bucket
        per_method_scores: dict = defaultdict(list)
        for _key, scores in bucket_records.items():
            for m, s in scores.items():
                per_method_scores[m].append(s)
        for m, vals in per_method_scores.items():
            if vals:
                bucket_report["method_means"][m] = {
                    "n": len(vals),
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                }

        # Paired t-tests: PAC vs each other method
        for method in METHODS:
            if method == pac:
                continue
            a_vals, b_vals = [], []
            for _key, scores in bucket_records.items():
                if pac in scores and method in scores:
                    a_vals.append(scores[pac])
                    b_vals.append(scores[method])
            if len(a_vals) >= 5:
                a = np.array(a_vals)
                b = np.array(b_vals)
                t_stat, p_val = stats.ttest_rel(a, b)
                bucket_report["pac_vs_method"][method] = {
                    "n_pairs": len(a_vals),
                    "pac_mean": round(float(a.mean()), 4),
                    "other_mean": round(float(b.mean()), 4),
                    "diff_mean": round(float((a - b).mean()), 4),
                    "t": round(float(t_stat), 3),
                    "p": round(float(p_val), 5),
                    "cohens_d": round(cohens_d_paired(a, b), 3),
                    "significant_at_0.05": bool(p_val < 0.05),
                }

        report[bucket] = bucket_report

    out = RESULTS_DIR / "per_stratum_ttest.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
