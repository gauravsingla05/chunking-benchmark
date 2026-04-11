#!/usr/bin/env python3
"""
Human evaluation export: generates side-by-side comparison pairs for human raters.

Selects 30 random documents, creates pairs of (position-aware vs semantic/truncation)
slide decks for blind evaluation. Exports to CSV + HTML for easy rating.

Usage:
    python run_human_eval.py                     # Export 30 pairs
    python run_human_eval.py --pairs 20          # Export 20 pairs
    python run_human_eval.py --import ratings.csv # Import completed ratings
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DOCUMENTS_DIR, RESULTS_DIR, HUMAN_EVAL_PAIRS, HUMAN_EVAL_SEED
from experiments.shared import slides_to_text


def select_pairs(num_pairs: int, generator: str = "gpt4o", seed: int = HUMAN_EVAL_SEED) -> list[dict]:
    """Select random document pairs for human evaluation.

    For each document, create a blind comparison:
    - Deck A: pac_position_aware chunking
    - Deck B: truncation or semantic_breakpoint (alternating)

    Returns list of {doc_id, deck_a_file, deck_b_file, deck_a_method, deck_b_method, label_hidden}
    """
    rng = random.Random(seed)

    # Find documents that have slides for both PAC and baseline methods
    eligible = []
    for doc_dir in sorted(DOCUMENTS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        slides_dir = doc_dir / "slides"
        if not slides_dir.exists():
            continue

        # Find matching generation files
        pac_files = list(slides_dir.glob(f"pac_position_aware__run_1__{generator}.json"))
        trunc_files = list(slides_dir.glob(f"truncation__run_1__{generator}.json"))
        sem_files = list(slides_dir.glob(f"semantic_breakpoint__run_1__{generator}.json"))

        if pac_files and (trunc_files or sem_files):
            eligible.append({
                "doc_id": doc_dir.name,
                "pac": pac_files[0] if pac_files else None,
                "truncation": trunc_files[0] if trunc_files else None,
                "semantic": sem_files[0] if sem_files else None,
            })

    if len(eligible) < num_pairs:
        print(f"Warning: Only {len(eligible)} eligible documents (requested {num_pairs})")
        num_pairs = len(eligible)

    selected = rng.sample(eligible, num_pairs)

    pairs = []
    for i, doc in enumerate(selected):
        # Alternate baseline: truncation vs semantic
        if i % 2 == 0 and doc["truncation"]:
            baseline_file = doc["truncation"]
            baseline_method = "truncation"
        elif doc["semantic"]:
            baseline_file = doc["semantic"]
            baseline_method = "semantic_breakpoint"
        elif doc["truncation"]:
            baseline_file = doc["truncation"]
            baseline_method = "truncation"
        else:
            continue

        # Randomly assign PAC to A or B (blind)
        if rng.random() < 0.5:
            pairs.append({
                "pair_id": i + 1,
                "doc_id": doc["doc_id"],
                "deck_a_file": str(doc["pac"]),
                "deck_b_file": str(baseline_file),
                "deck_a_method": "pac_position_aware",
                "deck_b_method": baseline_method,
            })
        else:
            pairs.append({
                "pair_id": i + 1,
                "doc_id": doc["doc_id"],
                "deck_a_file": str(baseline_file),
                "deck_b_file": str(doc["pac"]),
                "deck_a_method": baseline_method,
                "deck_b_method": "pac_position_aware",
            })

    return pairs


def export_csv(pairs: list[dict], output_dir: Path):
    """Export rating sheet as CSV."""
    csv_file = output_dir / "human_eval_rating_sheet.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "doc_id",
            "completeness_preference", "accuracy_preference",
            "statistics_preference", "coherence_preference",
            "relevance_preference", "overall_preference",
            "confidence", "notes"
        ])
        for pair in pairs:
            writer.writerow([
                pair["pair_id"], pair["doc_id"],
                "", "", "", "", "", "",  # A or B
                "",  # 1-5
                "",
            ])
    print(f"Rating sheet: {csv_file}")
    return csv_file


def export_html(pairs: list[dict], output_dir: Path):
    """Export side-by-side HTML for easier human evaluation."""
    html_file = output_dir / "human_eval_pairs.html"

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Human Evaluation - Slide Comparison</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }
.pair { border: 2px solid #e5e7eb; border-radius: 12px; margin: 30px 0; padding: 20px; }
.pair-header { font-size: 18px; font-weight: 700; margin-bottom: 15px; color: #374151; }
.decks { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.deck { background: #f9fafb; border-radius: 8px; padding: 15px; }
.deck h3 { color: #4f46e5; margin-top: 0; }
.slide { background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; margin: 10px 0; }
.slide-title { font-weight: 600; color: #1f2937; }
.slide-type { font-size: 12px; color: #9ca3af; }
.bullet { margin: 4px 0 4px 16px; }
.bullet-title { font-weight: 500; }
</style></head><body>
<h1>Human Evaluation: Slide Deck Comparison</h1>
<p>For each pair, compare Deck A vs Deck B on: Completeness, Accuracy, Statistics Retention, Coherence, Relevance, Overall Quality.</p>
<p>Record your preference (A or B) and confidence (1-5) in the rating sheet CSV.</p>
"""]

    for pair in pairs:
        html_parts.append(f'<div class="pair">')
        html_parts.append(f'<div class="pair-header">Pair {pair["pair_id"]} — Document: {pair["doc_id"][:50]}</div>')
        html_parts.append('<div class="decks">')

        for label, file_key in [("A", "deck_a_file"), ("B", "deck_b_file")]:
            try:
                data = json.loads(Path(pair[file_key]).read_text())
                slides = data.get("output", data.get("slides", []))
                if isinstance(slides, dict):
                    slides = slides.get("slides", [])
            except Exception:
                slides = []

            html_parts.append(f'<div class="deck"><h3>Deck {label}</h3>')
            for slide in slides:
                stype = slide.get("type", "unknown")
                html_parts.append(f'<div class="slide">')
                html_parts.append(f'<span class="slide-type">[{stype}]</span> ')
                html_parts.append(f'<span class="slide-title">{slide.get("title", "")}</span>')
                if slide.get("body"):
                    html_parts.append(f'<p>{slide["body"]}</p>')
                if slide.get("bullet_points"):
                    for bp in slide["bullet_points"]:
                        html_parts.append(f'<div class="bullet"><span class="bullet-title">{bp.get("title", "")}</span>: {bp.get("body", "")}</div>')
                html_parts.append('</div>')
            html_parts.append('</div>')

        html_parts.append('</div></div>')

    html_parts.append('</body></html>')
    html_file.write_text("\n".join(html_parts))
    print(f"HTML viewer: {html_file}")
    return html_file


def export_answer_key(pairs: list[dict], output_dir: Path):
    """Export the hidden answer key (which deck is PAC vs baseline)."""
    key_file = output_dir / "human_eval_answer_key.json"
    key_data = []
    for pair in pairs:
        key_data.append({
            "pair_id": pair["pair_id"],
            "doc_id": pair["doc_id"],
            "deck_a_method": pair["deck_a_method"],
            "deck_b_method": pair["deck_b_method"],
        })
    key_file.write_text(json.dumps(key_data, indent=2))
    print(f"Answer key (DO NOT share with raters): {key_file}")


def import_ratings(csv_path: Path, answer_key_path: Path):
    """Import completed human ratings and compute results."""
    # Load answer key
    with open(answer_key_path) as f:
        key = {item["pair_id"]: item for item in json.load(f)}

    # Load ratings
    ratings = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = int(row["pair_id"])
            if pair_id not in key:
                continue
            answer = key[pair_id]

            # Convert A/B preferences to method preferences
            prefs = {}
            for metric in ["completeness", "accuracy", "statistics", "coherence", "relevance", "overall"]:
                pref = row.get(f"{metric}_preference", "").strip().upper()
                if pref not in ("A", "B"):
                    continue
                chosen_method = answer["deck_a_method"] if pref == "A" else answer["deck_b_method"]
                prefs[metric] = chosen_method

            ratings.append({
                "pair_id": pair_id,
                "doc_id": answer["doc_id"],
                "preferences": prefs,
                "confidence": int(row.get("confidence", 3)),
                "notes": row.get("notes", ""),
            })

    # Compute win rates
    from collections import defaultdict
    wins = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)

    for rating in ratings:
        for metric, winner in rating["preferences"].items():
            wins[metric][winner] += 1
            total[metric] += 1

    print("\n" + "=" * 60)
    print("HUMAN EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal rated pairs: {len(ratings)}")

    print(f"\n{'Metric':<20} {'PAC Wins':>10} {'Baseline Wins':>14} {'PAC Win%':>10}")
    print("-" * 56)
    for metric in ["completeness", "accuracy", "statistics", "coherence", "relevance", "overall"]:
        pac_wins = wins[metric].get("pac_position_aware", 0)
        other_wins = total[metric] - pac_wins
        pct = (pac_wins / total[metric] * 100) if total[metric] else 0
        print(f"{metric:<20} {pac_wins:>10} {other_wins:>14} {pct:>9.1f}%")

    # Save results
    results_file = RESULTS_DIR / "human_evaluation_results.json"
    results_file.write_text(json.dumps({
        "ratings": ratings,
        "win_rates": {
            metric: {
                method: count / total[metric] if total[metric] else 0
                for method, count in method_wins.items()
            }
            for metric, method_wins in wins.items()
        },
        "total_pairs": len(ratings),
    }, indent=2))
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Human evaluation export/import.")
    parser.add_argument("--pairs", type=int, default=HUMAN_EVAL_PAIRS)
    parser.add_argument("--generator", type=str, default="gpt4o",
                        help="Generator model name suffix in filenames")
    parser.add_argument("--import-ratings", type=Path, default=None,
                        help="Import completed CSV ratings")
    parser.add_argument("--answer-key", type=Path, default=None,
                        help="Answer key JSON (for import)")

    args = parser.parse_args()
    output_dir = RESULTS_DIR / "human_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.import_ratings:
        key_path = args.answer_key or (output_dir / "human_eval_answer_key.json")
        import_ratings(args.import_ratings, key_path)
    else:
        pairs = select_pairs(args.pairs, args.generator)
        if not pairs:
            print("No eligible pairs found. Run generation first.")
            return
        print(f"Selected {len(pairs)} pairs for human evaluation")
        export_csv(pairs, output_dir)
        export_html(pairs, output_dir)
        export_answer_key(pairs, output_dir)
        print(f"\nFiles saved to: {output_dir}")
        print("\nInstructions:")
        print("1. Open human_eval_pairs.html in a browser")
        print("2. Fill in human_eval_rating_sheet.csv (A or B for each metric)")
        print("3. Run: python run_human_eval.py --import-ratings human_eval_rating_sheet.csv")


if __name__ == "__main__":
    main()
