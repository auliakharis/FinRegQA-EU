"""
Build one unified report across all adversarial-corruption eval folders.

Judge model names drifted between runs (e.g. "google/gemma-4-31B-it-bdoan"
vs "google/gemma-4-31B-it" refer to the same served model, redeployed by
different lab members under different endpoint names), so scores are
aggregated by canonical judge id (see eval_dpo.JUDGE_ALIASES) instead of
relying on exact key match like eval_dpo.py's report_phase does.

Usage
-----
    python script/unified_report.py
    python script/unified_report.py --output_dirs output/eval_hallucinate_citation output/eval_law_swap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_dpo import canonical_judge

DIMS = ["accuracy", "completeness", "topic_coherence", "citation_quality"]

DEFAULT_DIRS = [
    Path("output/llama/eval_sft"),
    Path("output/llama/eval_hallucinate_citation"),
    Path("output/llama/eval_hallucinate_text"),
    Path("output/llama/eval_article_swap"),
    Path("output/llama/eval_law_swap"),
    Path("output/llama/eval_dpo_all"),
    Path("output/llama/eval_gr_dpo"),
    Path("output/llama/eval_dr_dpo"),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def citation_f1(pred: str, ref: str) -> float | None:
    try:
        from citation_parser import find_citation_spans
    except ImportError:
        return None

    def spans(text: str) -> set[str]:
        return {text[s:e].lower().strip() for s, e, *_ in find_citation_spans(text)}

    pred_c, ref_c = spans(pred), spans(ref)
    if not ref_c and not pred_c:
        return 1.0
    if not ref_c or not pred_c:
        return 0.0
    tp = len(pred_c & ref_c)
    p, r = tp / len(pred_c), tp / len(ref_c)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_folder_report(folder: Path) -> dict:
    answers = load_jsonl(folder / "eval_answers.jsonl")
    scored = load_jsonl(folder / "eval_judge_scores.jsonl")
    answers_by_qid = {r["question_id"]: r for r in answers}

    dim_sums = {"baseline": {d: 0.0 for d in DIMS}, "dpo": {d: 0.0 for d in DIMS}}
    dim_counts = {"baseline": {d: 0 for d in DIMS}, "dpo": {d: 0 for d in DIMS}}
    judges_seen: set[str] = set()
    wins = {"baseline": 0, "dpo": 0, "tie": 0}
    n_judged = 0

    for row in scored:
        norm_scores = {"baseline": {}, "dpo": {}}
        for model_key in ("baseline", "dpo"):
            for raw_judge, sc in row["scores"].get(model_key, {}).items():
                jm = canonical_judge(raw_judge)
                judges_seen.add(jm)
                norm_scores[model_key][jm] = sc
                for d in DIMS:
                    v = sc.get(d)
                    if v is not None:
                        dim_sums[model_key][d] += float(v)
                        dim_counts[model_key][d] += 1

        def overall(key: str) -> float | None:
            vals = []
            for sc in norm_scores[key].values():
                vs = [sc.get(d) for d in DIMS if sc.get(d) is not None]
                if vs:
                    vals.append(sum(vs) / len(vs))
            return sum(vals) / len(vals) if vals else None

        b, d = overall("baseline"), overall("dpo")
        if b is not None and d is not None:
            n_judged += 1
            gap = d - b
            if abs(gap) < 1e-9:
                wins["tie"] += 1
            elif gap > 0:
                wins["dpo"] += 1
            else:
                wins["baseline"] += 1

    cit_f1 = {"baseline": [], "dpo": []}
    for row in answers:
        ref = row.get("ground_truth", "")
        for key in ("baseline", "dpo"):
            pred = row.get(key, {}).get("text", "")
            if pred:
                f1 = citation_f1(pred, ref)
                if f1 is not None:
                    cit_f1[key].append(f1)

    def mean_dim(key: str, d: str) -> float | None:
        c = dim_counts[key][d]
        return dim_sums[key][d] / c if c else None

    judge_scores = {
        key: {d: mean_dim(key, d) for d in DIMS} for key in ("baseline", "dpo")
    }
    for key in ("baseline", "dpo"):
        vals = [v for v in judge_scores[key].values() if v is not None]
        judge_scores[key]["overall"] = sum(vals) / len(vals) if vals else None

    return {
        "n_questions": len(answers),
        "n_judged": n_judged,
        "judges": sorted(judges_seen),
        "judge_scores": judge_scores,
        "win_rate": {
            "baseline": wins["baseline"] / n_judged if n_judged else None,
            "dpo": wins["dpo"] / n_judged if n_judged else None,
            "tie": wins["tie"] / n_judged if n_judged else None,
        },
        "citation_f1": {
            key: (sum(vs) / len(vs) if vs else None) for key, vs in cit_f1.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dirs", nargs="+", default=DEFAULT_DIRS, type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/unified_report.json"))
    args = parser.parse_args()

    report = {str(d): build_folder_report(d) for d in args.output_dirs}

    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved unified report → {args.out}\n")

    # --- Print comparison table -------------------------------------------
    header = f"{'Corruption type':<26}{'Judges':>8}{'Base ovr':>10}{'DPO ovr':>10}{'DPO win%':>10}{'Base F1':>10}{'DPO F1':>10}"
    print(header)
    print("-" * len(header))
    for d, r in report.items():
        name = Path(d).name.replace("eval_", "")
        n_judges = len(r["judges"])
        b_ovr = r["judge_scores"]["baseline"]["overall"]
        d_ovr = r["judge_scores"]["dpo"]["overall"]
        win = r["win_rate"]["dpo"]
        b_f1 = r["citation_f1"]["baseline"]
        d_f1 = r["citation_f1"]["dpo"]
        print(
            f"{name:<26}{n_judges:>8}"
            f"{b_ovr:>10.3f}{d_ovr:>10.3f}"
            f"{win:>9.1%}"
            f"{b_f1:>10.3f}{d_f1:>10.3f}"
        )


if __name__ == "__main__":
    main()
