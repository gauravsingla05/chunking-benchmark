#!/usr/bin/env python3
"""
Comprehensive analysis: aggregates all data and computes every statistic needed for paper.

Outputs:
- results/full_analysis.json (all stats)
- results/paper_tables.tex (LaTeX tables)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DOCUMENTS_DIR, RESULTS_DIR

METHODS = ["truncation", "fixed_size_first_last", "semantic_breakpoint", "pac_position_aware"]
METHOD_NAMES = {
    "truncation": "Truncation",
    "fixed_size_first_last": "Fixed-Size",
    "semantic_breakpoint": "Semantic",
    "pac_position_aware": "PAC",
}


def load_slide_evaluations():
    """Load all slide evaluations from documents/*/evaluations/slides/."""
    results = []
    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        eval_dir = doc_dir / "evaluations" / "slides"
        if not eval_dir.exists():
            # Legacy
            eval_dir = doc_dir / "evaluations_gemini"
            if not eval_dir.exists():
                continue
        for f in eval_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                if d.get("success"):
                    results.append(d)
            except Exception:
                pass
    return results


def load_qa_evaluations():
    """Load Q&A evaluations."""
    qa_dir = RESULTS_DIR / "qa" / "evaluations"
    questions_dir = RESULTS_DIR / "qa" / "questions"

    # Get doc word counts
    doc_lengths = {}
    for f in questions_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            doc_lengths[d["doc_id"]] = d.get("document_words", 0)
        except Exception:
            pass

    results = []
    for f in qa_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            d["document_words"] = doc_lengths.get(d.get("doc_id", ""), 0)
            results.append(d)
        except Exception:
            pass
    return results


def length_bucket(wc: int) -> str:
    if wc < 5000:
        return "<5K"
    elif wc < 10000:
        return "5-10K"
    elif wc < 20000:
        return "10-20K"
    else:
        return "20K+"


# ─── Main Slide Analysis ──────────────────────────────────────

def analyze_slides(evals: list) -> dict:
    """Compute slide stats: by method, by judge, by generator, by length."""
    by_method = defaultdict(list)
    by_method_judge = defaultdict(lambda: defaultdict(list))
    by_method_gen = defaultdict(lambda: defaultdict(list))
    by_method_length = defaultdict(lambda: defaultdict(list))

    # For paired tests (need same doc/judge/gen pairs)
    paired = defaultdict(lambda: defaultdict(float))  # (doc, judge, gen) -> {method: score}

    for ev in evals:
        method = ev.get("method", "")
        if method not in METHODS:
            continue
        score = ev.get("overall_score", 0)
        if not score:
            continue
        judge = ev.get("judge_model", ev.get("model", "unknown"))
        gen = ev.get("generator_model", "gpt-4o")  # legacy
        wc = ev.get("document_word_count", 0)
        doc_id = ev.get("doc_id", "")

        by_method[method].append(score)
        by_method_judge[method][judge].append(score)
        by_method_gen[method][gen].append(score)
        if wc:
            by_method_length[method][length_bucket(wc)].append(score)

        if doc_id and judge and gen:
            paired[(doc_id, judge, gen)][method] = score

    # Descriptive stats
    desc = {}
    for method, scores in by_method.items():
        if len(scores) >= 2:
            desc[method] = {
                "n": len(scores),
                "mean": round(float(np.mean(scores)), 3),
                "std": round(float(np.std(scores, ddof=1)), 3),
                "ci_95": round(float(1.96 * np.std(scores, ddof=1) / np.sqrt(len(scores))), 3),
            }

    # Paired t-tests vs truncation
    paired_tests = {}
    baseline = "truncation"
    for method in METHODS:
        if method == baseline:
            continue
        b_scores, m_scores = [], []
        for key, methods_dict in paired.items():
            if baseline in methods_dict and method in methods_dict:
                b_scores.append(methods_dict[baseline])
                m_scores.append(methods_dict[method])
        if len(b_scores) >= 5:
            t, p = stats.ttest_rel(b_scores, m_scores)
            diff = np.array(b_scores) - np.array(m_scores)
            d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0
            paired_tests[f"{baseline}_vs_{method}"] = {
                "n": len(b_scores),
                "t": round(float(t), 3),
                "p": round(float(p), 5),
                "cohens_d": round(d, 3),
                "significant": bool(p < 0.05),
            }

    # By judge
    judge_stats = {}
    for method, judges in by_method_judge.items():
        judge_stats[method] = {}
        for judge, scores in judges.items():
            if len(scores) >= 2:
                judge_stats[method][judge] = {
                    "n": len(scores),
                    "mean": round(float(np.mean(scores)), 3),
                    "std": round(float(np.std(scores, ddof=1)), 3),
                }

    # By generator
    gen_stats = {}
    for method, gens in by_method_gen.items():
        gen_stats[method] = {}
        for gen, scores in gens.items():
            if len(scores) >= 2:
                gen_stats[method][gen] = {
                    "n": len(scores),
                    "mean": round(float(np.mean(scores)), 3),
                }

    # By length
    length_stats = {}
    for method, lengths in by_method_length.items():
        length_stats[method] = {}
        for bucket, scores in lengths.items():
            if len(scores) >= 2:
                length_stats[method][bucket] = {
                    "n": len(scores),
                    "mean": round(float(np.mean(scores)), 3),
                }

    # Judge agreement (Pearson correlation)
    judges_list = sorted({ev.get("judge_model", ev.get("model", "")) for ev in evals if ev.get("judge_model") or ev.get("model")})
    judge_agreement = {}
    if len(judges_list) >= 2:
        # Group by (doc, method, gen) -> {judge: score}
        triples = defaultdict(dict)
        for ev in evals:
            judge = ev.get("judge_model", ev.get("model", ""))
            score = ev.get("overall_score", 0)
            if not judge or not score:
                continue
            key = (ev.get("doc_id"), ev.get("method"), ev.get("generator_model", "gpt-4o"))
            triples[key][judge] = score

        for i, ja in enumerate(judges_list):
            for jb in judges_list[i+1:]:
                a_scores, b_scores = [], []
                for triple in triples.values():
                    if ja in triple and jb in triple:
                        a_scores.append(triple[ja])
                        b_scores.append(triple[jb])
                if len(a_scores) >= 10:
                    r, p = stats.pearsonr(a_scores, b_scores)
                    judge_agreement[f"{ja}_vs_{jb}"] = {
                        "n": len(a_scores),
                        "pearson_r": round(float(r), 3),
                        "p_value": round(float(p), 5),
                    }

    return {
        "descriptive": desc,
        "paired_tests": paired_tests,
        "by_judge": judge_stats,
        "by_generator": gen_stats,
        "by_length": length_stats,
        "judge_agreement": judge_agreement,
    }


# ─── Q&A Analysis ──────────────────────────────────────────

def analyze_qa(evals: list) -> dict:
    """Compute Q&A accuracy stats."""
    by_method = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0, "unanswerable": 0, "total": 0})
    by_method_model = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "partial": 0, "total": 0}))
    by_method_length = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "partial": 0, "total": 0}))

    for ev in evals:
        method = ev.get("method", "")
        if method not in METHODS:
            continue
        model = ev.get("model", "unknown")
        wc = ev.get("document_words", 0)
        bucket = length_bucket(wc) if wc else None

        for j in ev.get("judgments", []):
            verdict = j.get("verdict", "incorrect")
            by_method[method][verdict] = by_method[method].get(verdict, 0) + 1
            by_method[method]["total"] += 1
            if verdict in ("correct", "partial"):
                by_method_model[method][model][verdict] += 1
            by_method_model[method][model]["total"] += 1
            if bucket:
                if verdict in ("correct", "partial"):
                    by_method_length[method][bucket][verdict] += 1
                by_method_length[method][bucket]["total"] += 1

    # Compute accuracy: (correct + 0.5 * partial) / total
    method_accuracy = {}
    for method, stats_d in by_method.items():
        total = stats_d["total"]
        if total:
            acc = (stats_d.get("correct", 0) + 0.5 * stats_d.get("partial", 0)) / total * 100
            method_accuracy[method] = {
                "n": total,
                "accuracy": round(acc, 1),
                "correct": stats_d.get("correct", 0),
                "partial": stats_d.get("partial", 0),
                "incorrect": stats_d.get("incorrect", 0),
                "unanswerable": stats_d.get("unanswerable", 0),
            }

    model_accuracy = {}
    for method, models in by_method_model.items():
        model_accuracy[method] = {}
        for model, stats_d in models.items():
            total = stats_d["total"]
            if total:
                acc = (stats_d.get("correct", 0) + 0.5 * stats_d.get("partial", 0)) / total * 100
                model_accuracy[method][model] = {"n": total, "accuracy": round(acc, 1)}

    length_accuracy = {}
    for method, lengths in by_method_length.items():
        length_accuracy[method] = {}
        for bucket, stats_d in lengths.items():
            total = stats_d["total"]
            if total:
                acc = (stats_d.get("correct", 0) + 0.5 * stats_d.get("partial", 0)) / total * 100
                length_accuracy[method][bucket] = {"n": total, "accuracy": round(acc, 1)}

    return {
        "by_method": method_accuracy,
        "by_method_model": model_accuracy,
        "by_method_length": length_accuracy,
    }


# ─── Sensitivity Analysis ──────────────────────────────

def analyze_sensitivity() -> dict:
    """Load and analyze W-curve sensitivity results."""
    f = RESULTS_DIR / "sensitivity_analysis.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    by_config = defaultdict(list)
    for r in data.get("results", []):
        by_config[r["config"]].append(r["output_words"])

    summary = {}
    for config, words in by_config.items():
        summary[config] = {
            "mean_words": round(float(np.mean(words)), 0),
            "std_words": round(float(np.std(words)), 0),
            "n": len(words),
        }
    return summary


# ─── Human Evaluation ─────────────────────────────────

def load_human_eval() -> dict:
    f = RESULTS_DIR / "human_evaluation_results.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


# ─── Main ──────────────────────────────────────────────

def main():
    print("Loading data...")
    slide_evals = load_slide_evaluations()
    qa_evals = load_qa_evaluations()
    print(f"  {len(slide_evals)} slide evaluations")
    print(f"  {len(qa_evals)} Q&A evaluations")

    print("Analyzing slides...")
    slide_stats = analyze_slides(slide_evals)

    print("Analyzing Q&A...")
    qa_stats = analyze_qa(qa_evals)

    print("Loading sensitivity...")
    sens_stats = analyze_sensitivity()

    print("Loading human eval...")
    human_eval = load_human_eval()

    # Save full analysis
    full = {
        "slide_evaluations": {
            "total": len(slide_evals),
            "stats": slide_stats,
        },
        "qa_evaluations": {
            "total": len(qa_evals),
            "stats": qa_stats,
        },
        "sensitivity": sens_stats,
        "human_evaluation": human_eval,
    }
    out_file = RESULTS_DIR / "full_analysis.json"
    out_file.write_text(json.dumps(full, indent=2))
    print(f"\nSaved: {out_file}")

    # Print key summaries
    print("\n" + "="*70)
    print("SLIDES — DESCRIPTIVE STATS BY METHOD")
    print("="*70)
    for method in METHODS:
        d = slide_stats["descriptive"].get(method, {})
        if d:
            print(f"  {METHOD_NAMES[method]:<15}: {d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})")

    print("\n" + "="*70)
    print("SLIDES — BY DOCUMENT LENGTH (PAC's niche?)")
    print("="*70)
    print(f"  {'Length':<10}", end="")
    for m in METHODS:
        print(f"{METHOD_NAMES[m]:>12}", end="")
    print()
    for bucket in ["<5K", "5-10K", "10-20K", "20K+"]:
        print(f"  {bucket:<10}", end="")
        for m in METHODS:
            v = slide_stats["by_length"].get(m, {}).get(bucket, {})
            print(f"{v.get('mean', 0):>12.3f}", end="")
        print()

    print("\n" + "="*70)
    print("SLIDES — JUDGE AGREEMENT (Pearson r)")
    print("="*70)
    for pair, d in slide_stats["judge_agreement"].items():
        print(f"  {pair}: r={d['pearson_r']:.3f} (n={d['n']})")

    print("\n" + "="*70)
    print("Q&A — ACCURACY BY METHOD")
    print("="*70)
    for method in METHODS:
        d = qa_stats["by_method"].get(method, {})
        if d:
            print(f"  {METHOD_NAMES[method]:<15}: {d['accuracy']:.1f}% (n={d['n']})")

    print("\n" + "="*70)
    print("Q&A — BY DOCUMENT LENGTH")
    print("="*70)
    print(f"  {'Length':<10}", end="")
    for m in METHODS:
        print(f"{METHOD_NAMES[m]:>12}", end="")
    print()
    for bucket in ["<5K", "5-10K", "10-20K", "20K+"]:
        print(f"  {bucket:<10}", end="")
        for m in METHODS:
            v = qa_stats["by_method_length"].get(m, {}).get(bucket, {})
            print(f"{v.get('accuracy', 0):>11.1f}%", end="")
        print()

    print("\n" + "="*70)
    print("HUMAN EVALUATION (50 pairs)")
    print("="*70)
    if human_eval.get("win_rates"):
        for metric, rates in human_eval["win_rates"].items():
            pac_rate = rates.get("pac_position_aware", 0) * 100
            print(f"  {metric:<15}: PAC wins {pac_rate:.0f}%")

    print("\n" + "="*70)
    print("PAIRED T-TESTS (vs Truncation)")
    print("="*70)
    for comp, d in slide_stats["paired_tests"].items():
        sig = "***" if d["p"] < 0.001 else ("**" if d["p"] < 0.01 else ("*" if d["p"] < 0.05 else ""))
        print(f"  {comp}: t={d['t']:.3f}, p={d['p']:.4f}{sig}, d={d['cohens_d']:.3f}")


if __name__ == "__main__":
    main()
