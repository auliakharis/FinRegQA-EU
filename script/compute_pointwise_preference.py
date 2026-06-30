"""
Turn pointwise judge scores into pairwise (head-to-head) preferences.

Each judge scores both answerer models on the same question independently
(pointwise scoring, see judge_api.py). For every (question, judge) pair we
compare the two answerers' "overall" score (mean of the 4 score dimensions:
accuracy, completeness, topic_coherence, citation_quality) and decide:

  - which answerer the judge preferred (the one with the higher overall score)
  - whether that preference is "strong" (score gap >= STRONG_PREFERENCE_THRESHOLD)
    or "weak" (0 < gap < threshold), or a "tie" (gap == 0)

Each dimension is on a 1-5 scale, so the "overall" gap ranges from 0 to 4
(one model scoring all 5s, the other all 1s). STRONG_PREFERENCE_THRESHOLD
controls how big that gap has to be before we call it a clear/decisive
preference rather than a marginal one.

Questions with an incomplete score or a 'question deleted' ground-truth
stub are excluded (same filters as compute_similarity.py /
compute_citation_scores.py).

Outputs
-------
output/pointwise_preference_comparisons.json  - one row per (question, judge)
output/pointwise_preference_summary.csv        - aggregated win/tie/strength stats per judge
output/pointwise_preference_consensus.json     - per-question majority vote across judges
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_API_FILE = Path("output/judge_results_train_api.jsonl")
DIMS = ["accuracy", "completeness", "topic_coherence", "citation_quality"]
DELETED_QUESTION_IDS = {
    "ESMA_ESMA_QA_1580",
    "ESMA_ESMA_QA_1589",
    "ESMA_ESMA_QA_1668",
    "ESMA_ESMA_QA_1569",
}

STRONG_PREFERENCE_THRESHOLD = 1.5  # overall-score gap (0-4 scale) for a "strong"/decisive preference
TIE_EPSILON = 1e-9                 # gap <= this counts as an exact tie


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Build per-(question, judge) comparisons
# ---------------------------------------------------------------------------

def build_comparisons(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    counters = {
        "skipped_deleted_questions": 0,
        "skipped_missing_answerer_pair": 0,
        "skipped_incomplete_score": 0,
    }

    comparisons: list[dict] = []
    for rec in records:
        qid = rec["question_id"]
        if qid in DELETED_QUESTION_IDS:
            counters["skipped_deleted_questions"] += 1
            continue

        regulator = rec["regulator"]
        answers = rec.get("answers", {})
        answerers = sorted(answers.keys())
        if len(answerers) != 2:
            counters["skipped_missing_answerer_pair"] += 1
            continue
        a1, a2 = answerers

        judge_scores_1 = answers[a1].get("judge_scores", {})
        judge_scores_2 = answers[a2].get("judge_scores", {})
        judges = sorted(set(judge_scores_1) & set(judge_scores_2))

        for judge in judges:
            sc1, sc2 = judge_scores_1[judge], judge_scores_2[judge]
            if any(sc1.get(d) is None for d in DIMS) or any(sc2.get(d) is None for d in DIMS):
                counters["skipped_incomplete_score"] += 1
                continue

            overall_1 = sum(float(sc1[d]) for d in DIMS) / len(DIMS)
            overall_2 = sum(float(sc2[d]) for d in DIMS) / len(DIMS)
            diff = overall_1 - overall_2
            abs_diff = abs(diff)

            if abs_diff <= TIE_EPSILON:
                preferred, strength = "tie", "tie"
            else:
                preferred = a1 if diff > 0 else a2
                strength = "strong" if abs_diff >= STRONG_PREFERENCE_THRESHOLD else "weak"

            comparisons.append({
                "question_id": qid,
                "regulator": regulator,
                "judge_model": judge,
                "answerer_1": a1,
                "answerer_2": a2,
                "overall_1": round(overall_1, 4),
                "overall_2": round(overall_2, 4),
                "diff": round(diff, 4),
                "abs_diff": round(abs_diff, 4),
                "preferred": preferred,
                "preference_strength": strength,
            })

    return comparisons, counters


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(comparisons: list[dict], group_keys: list[str]) -> list[dict]:
    """Win/tie/strength counts, grouped by the given keys (e.g. ["judge_model"])."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in comparisons:
        key = tuple(c[k] for k in group_keys)
        groups[key].append(c)

    rows = []
    for key, items in sorted(groups.items()):
        n = len(items)
        a1, a2 = items[0]["answerer_1"], items[0]["answerer_2"]
        wins_1 = sum(1 for c in items if c["preferred"] == a1)
        wins_2 = sum(1 for c in items if c["preferred"] == a2)
        ties = sum(1 for c in items if c["preferred"] == "tie")
        strong = sum(1 for c in items if c["preference_strength"] == "strong")
        weak = sum(1 for c in items if c["preference_strength"] == "weak")

        row = dict(zip(group_keys, key))
        row.update({
            "n": n,
            f"wins_{a1}": wins_1,
            f"wins_{a2}": wins_2,
            "ties": ties,
            "strong_preference": strong,
            "weak_preference": weak,
            "strong_preference_rate": round(strong / n, 4) if n else 0.0,
            "mean_abs_diff": round(sum(c["abs_diff"] for c in items) / n, 4) if n else 0.0,
        })
        rows.append(row)
    return rows


def question_consensus(comparisons: list[dict]) -> list[dict]:
    """For each question, do the 3 judges agree on who they prefer?"""
    by_question: dict[str, list[dict]] = defaultdict(list)
    for c in comparisons:
        by_question[c["question_id"]].append(c)

    rows = []
    for qid, items in sorted(by_question.items()):
        regulator = items[0]["regulator"]
        votes = Counter(c["preferred"] for c in items)
        n_judges = len(items)
        top_choice, top_count = votes.most_common(1)[0]
        unanimous = top_count == n_judges
        unanimous_strong = unanimous and top_choice != "tie" and all(
            c["preference_strength"] == "strong" for c in items
        )
        rows.append({
            "question_id": qid,
            "regulator": regulator,
            "n_judges": n_judges,
            "majority_preferred": top_choice,
            "majority_count": top_count,
            "unanimous": unanimous,
            "unanimous_strong": unanimous_strong,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    records = load_records(TRAIN_API_FILE)
    comparisons, counters = build_comparisons(records)

    print(f"Skipped {counters['skipped_deleted_questions']} deleted-question stubs")
    print(f"Skipped {counters['skipped_missing_answerer_pair']} questions without exactly 2 answerers")
    print(f"Skipped {counters['skipped_incomplete_score']} judge comparisons with an incomplete score")
    print(f"Total (question, judge) comparisons: {len(comparisons)}")
    print(f"Strong-preference threshold: gap >= {STRONG_PREFERENCE_THRESHOLD} "
          f"(on a 0-4 overall-score-gap scale)\n")

    # Per-judge summary
    by_judge = aggregate(comparisons, ["judge_model"])
    print("=== Preference summary by judge ===")
    for row in by_judge:
        a1_key = [k for k in row if k.startswith("wins_")][0]
        a2_key = [k for k in row if k.startswith("wins_")][1]
        print(f"\n{row['judge_model']}  (n={row['n']})")
        print(f"  {a1_key[5:]:45s}: {row[a1_key]}")
        print(f"  {a2_key[5:]:45s}: {row[a2_key]}")
        print(f"  ties                                         : {row['ties']}")
        print(f"  strong preference                            : {row['strong_preference']} "
              f"({row['strong_preference_rate']*100:.1f}%)")
        print(f"  weak preference                              : {row['weak_preference']}")
        print(f"  mean |overall_1 - overall_2|                 : {row['mean_abs_diff']}")

    # Per-judge x regulator summary
    by_judge_regulator = aggregate(comparisons, ["judge_model", "regulator"])

    # Overall summary (across all judges)
    overall = aggregate(comparisons, [])
    print("\n=== Overall (all judges pooled) ===")
    print(overall[0] if overall else "no data")

    # Per-question consensus across the 3 judges
    consensus = question_consensus(comparisons)
    n_unanimous = sum(1 for r in consensus if r["unanimous"])
    n_unanimous_strong = sum(1 for r in consensus if r["unanimous_strong"])
    print(f"\n=== Cross-judge consensus ({len(consensus)} questions) ===")
    print(f"  Unanimous (all judges agree on same preferred/tie): {n_unanimous} "
          f"({n_unanimous / len(consensus) * 100:.1f}%)")
    print(f"  Unanimous AND all strong preferences               : {n_unanimous_strong} "
          f"({n_unanimous_strong / len(consensus) * 100:.1f}%)")

    # ---- Save outputs -------------------------------------------------------
    out_dir = TRAIN_API_FILE.parent

    with (out_dir / "pointwise_preference_comparisons.json").open("w") as f:
        json.dump(comparisons, f, indent=2)

    with (out_dir / "pointwise_preference_consensus.json").open("w") as f:
        json.dump(consensus, f, indent=2)

    summary_rows = by_judge + by_judge_regulator + overall
    summary_path = out_dir / "pointwise_preference_summary.csv"
    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row})
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nSaved -> {out_dir / 'pointwise_preference_comparisons.json'}")
    print(f"Saved -> {out_dir / 'pointwise_preference_consensus.json'}")
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
