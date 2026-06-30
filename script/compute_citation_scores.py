"""
Extract citations for every (question, answerer_model) pair in
output/judge_results_train_api.jsonl and compute Precision / Recall / F1,
then aggregate by regulator x answerer_model.

Questions whose ground_truth is just a "question deleted" stub are skipped
(no real reference citations to compare against).

Outputs
-------
Per-pair output      output/judge_results_train_api_citation.json
Summary CSV          output/citation_scores_summary.csv
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

INPUT_FILE = Path("output/judge_results_train_api.jsonl")
DELETED_QUESTION_IDS = {
    "ESMA_ESMA_QA_1580",
    "ESMA_ESMA_QA_1589",
    "ESMA_ESMA_QA_1668",
    "ESMA_ESMA_QA_1569",
}


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def _cit_strings(cits: list[Citation]) -> list[str]:
    return [str(c) for c in cits]


def process_pair(legal_act: str, ground_truth: str, candidate: str) -> dict:
    gt_meta = parse_legal_act(legal_act)
    gt_text = extract_citations(ground_truth)
    cand = extract_citations(candidate)

    scores_inst = citation_scores(gt_meta, cand, level="instrument")
    scores_full = citation_scores(gt_text, cand, level="full")

    return {
        "gt_meta_citations": _cit_strings(gt_meta),
        "gt_text_citations": _cit_strings(gt_text),
        "candidate_citations": _cit_strings(cand),
        "scores_instrument": scores_inst,   # uses legal_act as GT
        "scores_full": scores_full,          # uses ground_truth text as GT
    }


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Mean P/R/F1 across all pairs for both scoring levels."""
    totals: dict[str, dict[str, float]] = {
        "scores_instrument": {"precision": 0, "recall": 0, "f1": 0},
        "scores_full": {"precision": 0, "recall": 0, "f1": 0},
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
    if not INPUT_FILE.exists():
        print(f"[skip] {INPUT_FILE}")
        return

    records = []
    with INPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_skipped_deleted = sum(1 for r in records if r["question_id"] in DELETED_QUESTION_IDS)
    records = [r for r in records if r["question_id"] not in DELETED_QUESTION_IDS]

    print(f"Processing {INPUT_FILE.name} "
          f"({len(records)} questions, skipped {n_skipped_deleted} deleted-question stubs)...")

    augmented: list[dict] = []
    # citation_results grouped by (regulator, answerer_model) for aggregation
    grouped: dict[tuple[str, str], list[dict]] = {}

    for rec in records:
        qid = rec["question_id"]
        regulator = rec["regulator"]
        legal_act = rec.get("meta", {}).get("legal_act", "")
        ground_truth = rec.get("ground_truth", "")

        for answerer, ans in rec.get("answers", {}).items():
            candidate = ans.get("text", "")
            cit = process_pair(legal_act, ground_truth, candidate)

            augmented.append({
                "question_id": qid,
                "regulator": regulator,
                "answerer_model": answerer,
                "legal_act": legal_act,
                "citation": cit,
            })

            key = (regulator, answerer)
            grouped.setdefault(key, []).append(cit)

    # Write per-pair output
    out_path = INPUT_FILE.parent / f"{INPUT_FILE.stem}_citation.json"
    with out_path.open("w") as f:
        json.dump(augmented, f, indent=2)
    print(f"  -> {out_path}")

    # Collect summary rows: one per (regulator, answerer_model)
    summary_rows: list[dict] = []
    for (regulator, answerer), cits in sorted(grouped.items()):
        agg = aggregate(cits)
        summary_rows.append({
            "regulator": regulator,
            "answerer_model": answerer,
            "n": len(cits),
            **{f"inst_{k}": v for k, v in agg["scores_instrument"].items()},
            **{f"full_{k}": v for k, v in agg["scores_full"].items()},
        })

    # Write summary CSV
    summary_path = Path("output/citation_scores_summary.csv")
    if summary_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary -> {summary_path}")

    # Print to console too
    print("\n=== Aggregate Citation Scores ===")
    for row in summary_rows:
        print(f"\n{row['regulator']} / {row['answerer_model']}  (n={row['n']})")
        print(f"  Instrument-level  P={row['inst_precision']:.3f}  "
              f"R={row['inst_recall']:.3f}  F1={row['inst_f1']:.3f}")
        print(f"  Full-level        P={row['full_precision']:.3f}  "
              f"R={row['full_recall']:.3f}  F1={row['full_f1']:.3f}")


if __name__ == "__main__":
    main()
