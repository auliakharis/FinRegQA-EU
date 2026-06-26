
"""
Step 3 + 4: Extract citations for every item in every judge-result file
and compute Precision / Recall / F1 per example, then aggregate by file.

Outputs
-------
For each input file  output/<dir>/<name>.json  →  output/<dir>/citation_<name>.json
Plus a summary CSV   output/citation_scores_summary.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from citation_parser import (
    Citation,
    citation_scores,
    extract_citations,
    parse_legal_act,
)

# ---------------------------------------------------------------------------
# File registry
# ---------------------------------------------------------------------------

INPUT_FILES: list[tuple[Path, str, str]] = [
    # (path, subdir_label, corpus_label)
    (Path("output/small_judge_result/judge_results_esma.json"), "small", "esma"),
    (Path("output/small_judge_result/judge_results_eba.json"),  "small", "eba"),
    (Path("output/big_judge_result/judge_results_esma_qa_api.json"), "big", "esma"),
    (Path("output/big_judge_result/judge_results_eba_qa_api.json"),  "big", "eba"),
]


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def _cit_strings(cits: list[Citation]) -> list[str]:
    return [str(c) for c in cits]


def process_item(item: dict) -> dict:
    legal_act       = item.get("meta", {}).get("legal_act", "")
    ground_truth    = item.get("ground_truth", "")
    candidate       = item.get("candidate_answer", "")

    gt_meta   = parse_legal_act(legal_act)
    gt_text   = extract_citations(ground_truth)
    cand      = extract_citations(candidate)

    scores_inst = citation_scores(gt_meta, cand, level="instrument")
    scores_full = citation_scores(gt_text, cand, level="full")

    return {
        "gt_meta_citations":      _cit_strings(gt_meta),
        "gt_text_citations":      _cit_strings(gt_text),
        "candidate_citations":    _cit_strings(cand),
        "scores_instrument": scores_inst,   # uses legal_act as GT
        "scores_full":       scores_full,   # uses ground_truth text as GT
    }


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Mean P/R/F1 across all items for both scoring levels."""
    totals: dict[str, dict[str, float]] = {
        "scores_instrument": {"precision": 0, "recall": 0, "f1": 0},
        "scores_full":       {"precision": 0, "recall": 0, "f1": 0},
    }
    n = len(results)
    if n == 0:
        return totals
    for r in results:
        for level in totals:
            for metric in ("precision", "recall", "f1"):
                totals[level][metric] += r[level][metric]
    return {
        level: {k: round(v / n, 4) for k, v in metrics.items()}
        for level, metrics in totals.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    summary_rows: list[dict] = []

    for path, size_label, corpus_label in INPUT_FILES:
        if not path.exists():
            print(f"[skip] {path}")
            continue

        with path.open() as f:
            data: list[dict] = json.load(f)

        print(f"Processing {path.name} ({len(data)} items)...")

        augmented: list[dict] = []
        citation_results: list[dict] = []

        for item in data:
            cit = process_item(item)
            citation_results.append(cit)
            augmented.append({**item, "citation": cit})

        agg = aggregate(citation_results)

        # Write per-item output
        out_path = path.parent / f"citation_{path.name}"
        with out_path.open("w") as f:
            json.dump(augmented, f, indent=2)
        print(f"  → {out_path}")

        # Collect summary row
        model = (data[0].get("meta", {}).get("answerer_model", "")
                 or data[0].get("meta", {}).get("judge_model", "")
                 or size_label)
        summary_rows.append({
            "file":    path.name,
            "size":    size_label,
            "corpus":  corpus_label,
            "model":   model,
            "n":       len(data),
            **{f"inst_{k}": v
               for k, v in agg["scores_instrument"].items()},
            **{f"full_{k}": v
               for k, v in agg["scores_full"].items()},
        })

    # Write summary CSV
    summary_path = Path("output/citation_scores_summary.csv")
    if summary_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary → {summary_path}")

    # Print to console too
    print("\n=== Aggregate Citation Scores ===")
    for row in summary_rows:
        print(f"\n{row['file']}  (n={row['n']})")
        print(f"  Instrument-level  P={row['inst_precision']:.3f}  "
              f"R={row['inst_recall']:.3f}  F1={row['inst_f1']:.3f}")
        print(f"  Full-level        P={row['full_precision']:.3f}  "
              f"R={row['full_recall']:.3f}  F1={row['full_f1']:.3f}")


if __name__ == "__main__":
    main()
