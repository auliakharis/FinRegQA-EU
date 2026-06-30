"""
Generate DPO (chosen/rejected) pairs from the pointwise judge preferences,
with the "rejected" (less-preferred) answer deliberately corrupted to make
the contrast against "chosen" sharper and less ambiguous.

Source of preference: output/pointwise_preference_consensus.json (the
majority vote of the 3 judges per question — see compute_pointwise_preference.py).
Only questions with a non-tie majority and majority_count >= --min-majority
are used, so the chosen/rejected assignment itself is reasonably confident
before any corruption is applied.

Corruption types (operate on the rejected answer's text)
----------------------------------------------------------
law_swap              swap every mention of the most-cited regulation/directive
                       for a different, real-but-wrong one (e.g. CRR -> BRRD)
article_swap          swap cited article numbers for different ones (a real
                       citation-confusion error, not a fabrication)
hallucinate_citation  append a sentence with a fabricated, made-up citation
hallucinate_text      append a fabricated factual claim (no citation)
combined              apply all four corruptions together

Each corruption is independently optional in the sense that if it doesn't
find anything to corrupt (e.g. law_swap on a text with no citations at all),
that variant is skipped for that example rather than emitting an unchanged
"rejected" text.

Output: one JSONL row per (question, corruption variant) in
output/dpo_corruption_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from citation_parser import ACRONYM_TO_FULL, _norm_acronym, canonicalize_instrument, find_citation_spans

CONSENSUS_FILE = Path("output/pointwise_preference_consensus.json")
RECORDS_FILE = Path("output/judge_results_train_api.jsonl")
OUTPUT_FILE = Path("output/dpo_corruption_pairs.jsonl")

LAW_KINDS = {"regulation_eu", "regulation_short", "directive", "acronym", "guideline"}
CORRUPTION_ORDER = ["law_swap", "article_swap", "hallucinate_citation", "hallucinate_text"]

HALLUCINATED_CLAIMS = [
    "Notably, this requirement was permanently waived for institutions headquartered "
    "outside the EU following the latest amendment.",
    "It should also be noted that the relevant competent authority may, at its sole "
    "discretion, exempt branches with fewer than 50 employees from this obligation.",
    "This obligation does not apply during the first three years following the "
    "entity's initial authorisation.",
    "In practice, supervisory guidance has confirmed that this provision is no longer "
    "enforced following the 2021 simplification package.",
    "This requirement was subsequently superseded by a transitional carve-out "
    "applicable to all subsidiaries established before 2018.",
]


# ---------------------------------------------------------------------------
# Individual corruptions — each returns (new_text, applied: bool)
# ---------------------------------------------------------------------------

def _canonical_key_for_match(kind: str, m: re.Match) -> str:
    if kind in ("regulation_eu", "regulation_short", "directive"):
        return canonicalize_instrument(m.group(1))
    if kind == "acronym":
        return _norm_acronym(m.group(1))
    if kind == "guideline":
        return m.group(1).upper()
    return m.group(0).upper()


def corrupt_law_swap(text: str, rng: random.Random) -> tuple[str, bool]:
    """Swap every mention of the most-cited instrument for a different real one."""
    spans = find_citation_spans(text)
    law_spans = [s for s in spans if s[2] in LAW_KINDS]
    if not law_spans:
        return text, False

    keys = [_canonical_key_for_match(kind, m) for _, _, kind, m in law_spans]
    target = Counter(keys).most_common(1)[0][0]

    candidates = [a for a in ACRONYM_TO_FULL if a != target]
    if not candidates:
        return text, False
    replacement_acronym = rng.choice(candidates)
    replacement_full = ACRONYM_TO_FULL[replacement_acronym]

    new_text = text
    changed = False
    for (start, end, kind, _m), key in sorted(
        zip(law_spans, keys), key=lambda t: t[0][0], reverse=True
    ):
        if key != target:
            continue
        # Bare acronym mentions (e.g. "CRR", or the "(CRR)" in a parenthetical)
        # get swapped for the new acronym; numbered/full citations get the new
        # full name — keeps the corrupted text grammatically coherent.
        replacement_text = replacement_acronym if kind == "acronym" else replacement_full
        new_text = new_text[:start] + replacement_text + new_text[end:]
        changed = True
    return new_text, changed


def corrupt_article_swap(text: str, rng: random.Random) -> tuple[str, bool]:
    """Swap cited article numbers for different ones (citation confusion)."""
    spans = find_citation_spans(text)
    article_spans = [s for s in spans if s[2] == "article"]
    if not article_spans:
        return text, False

    numbers_in_order = []
    for _, _, _, m in article_spans:
        if m.group(1) not in numbers_in_order:
            numbers_in_order.append(m.group(1))

    if len(numbers_in_order) >= 2:
        shuffled = numbers_in_order[1:] + numbers_in_order[:1]  # cyclic shift, no fixed points
        mapping = dict(zip(numbers_in_order, shuffled))
    else:
        old = numbers_in_order[0]
        old_int = int(re.sub(r"[a-zA-Z]", "", old) or 0)
        new_int = old_int
        while new_int == old_int:
            new_int = rng.randint(1, 600)
        mapping = {old: str(new_int)}

    new_text = text
    changed = False
    for start, end, kind, m in sorted(article_spans, key=lambda s: s[0], reverse=True):
        old_num = m.group(1)
        new_num = mapping.get(old_num, old_num)
        if new_num == old_num:
            continue
        num_start, num_end = m.start(1), m.end(1)
        new_text = new_text[:num_start] + new_num + new_text[num_end:]
        changed = True
    return new_text, changed


def corrupt_hallucinate_citation(text: str, rng: random.Random) -> tuple[str, bool]:
    """Append a sentence citing a fabricated regulation/article that doesn't exist."""
    fake_num = f"{rng.randint(9000, 9999)}/{rng.randint(2030, 2099)}"
    fake_article = rng.randint(500, 999)
    sentence = (
        f" This conclusion is further reinforced by Regulation (EU) No {fake_num}, "
        f"Article {fake_article}, which explicitly mandates additional disclosure "
        f"requirements not otherwise referenced here."
    )
    return text.rstrip() + sentence, True


def corrupt_hallucinate_text(text: str, rng: random.Random) -> tuple[str, bool]:
    """Append a fabricated factual claim (no citation, just made-up content)."""
    claim = rng.choice(HALLUCINATED_CLAIMS)
    return text.rstrip() + " " + claim, True


CORRUPTIONS = {
    "law_swap": corrupt_law_swap,
    "article_swap": corrupt_article_swap,
    "hallucinate_citation": corrupt_hallucinate_citation,
    "hallucinate_text": corrupt_hallucinate_text,
}


def apply_corruptions(text: str, types: set[str], rng: random.Random) -> tuple[str, list[str]]:
    """Apply the requested corruption types (in CORRUPTION_ORDER) and report which stuck."""
    applied: list[str] = []
    out = text
    for name in CORRUPTION_ORDER:
        if name not in types:
            continue
        out, ok = CORRUPTIONS[name](out, rng)
        if ok:
            applied.append(name)
    return out, applied


def build_variant(rejected_text: str, variant: str, rng: random.Random) -> tuple[str, list[str]]:
    types = set(CORRUPTION_ORDER) if variant == "combined" else {variant}
    return apply_corruptions(rejected_text, types, rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["question_id"]] = rec
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corruptions", nargs="+",
        default=CORRUPTION_ORDER + ["combined"],
        choices=CORRUPTION_ORDER + ["combined"],
        help="Which corruption variants to generate (default: all + combined).",
    )
    parser.add_argument(
        "--min-majority", type=int, default=2,
        help="Require at least this many of the 3 judges to agree on the preferred "
             "answer before using the question (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    consensus = json.load(CONSENSUS_FILE.open())
    records = load_jsonl(RECORDS_FILE)

    n_skipped_tie = 0
    n_skipped_low_confidence = 0
    n_questions_used = 0
    variant_counts: Counter = Counter()
    rows: list[dict] = []

    for c in consensus:
        if c["majority_preferred"] == "tie":
            n_skipped_tie += 1
            continue
        if c["majority_count"] < args.min_majority:
            n_skipped_low_confidence += 1
            continue

        qid = c["question_id"]
        rec = records[qid]
        answerers = list(rec["answers"].keys())
        chosen_model = c["majority_preferred"]
        if chosen_model not in answerers:
            continue
        rejected_model = next(a for a in answerers if a != chosen_model)

        chosen_text = rec["answers"][chosen_model]["text"]
        rejected_text = rec["answers"][rejected_model]["text"]
        n_questions_used += 1

        for variant in args.corruptions:
            corrupted_text, applied = build_variant(rejected_text, variant, rng)
            if not applied:
                continue  # corruption wasn't applicable to this text (e.g. no citations)

            variant_counts[variant] += 1
            rows.append({
                "question_id": qid,
                "regulator": c["regulator"],
                "question": rec["question"],
                "ground_truth": rec["ground_truth"],
                "chosen": {"model": chosen_model, "text": chosen_text},
                "rejected": {
                    "model": rejected_model,
                    "text": corrupted_text,
                    "original_text": rejected_text,
                    "corruption_variant": variant,
                    "corruptions_applied": applied,
                },
                "majority_count": c["majority_count"],
                "unanimous": c["unanimous"],
            })

    print(f"Questions skipped (tie / no majority preference): {n_skipped_tie}")
    print(f"Questions skipped (majority_count < {args.min_majority}): {n_skipped_low_confidence}")
    print(f"Questions used as a base for corruption: {n_questions_used}")
    print(f"DPO rows generated: {len(rows)}")
    print("\nBy corruption variant:")
    for variant in args.corruptions:
        print(f"  {variant:22s}: {variant_counts[variant]}")

    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
