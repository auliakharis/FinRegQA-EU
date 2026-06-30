"""
Citation extractor for EU financial regulatory texts.

Parses structured citations from free-text fields and legal_act metadata,
returning normalised Citation objects for downstream P/R/F1 computation.

Extraction pipeline
-------------------
1. Run regex patterns (most-specific first) to collect raw matches.
2. Associate bare Article references with the preceding instrument.
3. Normalise acronyms to a canonical uppercase form.
4. Deduplicate on full_key().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Acronym ↔ full-name lookup  (step 2 in the larger pipeline)
# ---------------------------------------------------------------------------

ACRONYM_TO_FULL: dict[str, str] = {
    "CRR":      "Regulation (EU) No 575/2013",
    "CRR2":     "Regulation (EU) 2019/876",
    "CRD":      "Directive 2013/36/EU",
    "CRDIV":    "Directive 2013/36/EU",
    "BRRD":     "Directive 2014/59/EU",
    "BRRD2":    "Directive 2019/879",
    "MIFID":    "Directive 2014/65/EU",
    "MIFIDII":  "Directive 2014/65/EU",
    "MIFIR":    "Regulation (EU) No 600/2014",
    "EMIR":     "Regulation (EU) No 648/2012",
    "UCITS":    "Directive 2009/65/EC",
    "UCITSV":   "Directive 2014/91/EU",
    "AIFMD":    "Directive 2011/61/EU",
    "PSD2":     "Directive 2015/2366/EU",
    "MAR":      "Regulation (EU) No 596/2014",
    "BMR":      "Regulation (EU) 2016/1011",
    "SSR":      "Regulation (EU) No 236/2012",
    "SFDR":     "Regulation (EU) 2019/2088",
    "ECSPR":    "Regulation (EU) 2020/1503",
    "DORA":     "Regulation (EU) 2022/2554",
    "PSD":      "Directive 2007/64/EC",          # original PSD
    "SRMR":     "Regulation (EU) No 806/2014",   # Single Resolution Mechanism
    "DGSD":     "Directive 2014/49/EU",          # Deposit Guarantee Schemes
    "MCD":      "Directive 2014/17/EU",          # Mortgage Credit Directive
    "PRIIPS":   "Regulation (EU) No 1286/2014",  # PRIIPs KID
    "CSDR":     "Regulation (EU) No 909/2014",   # Central Securities Depositories
    "SFTR":     "Regulation (EU) 2015/2365",     # Securities Financing Transactions
    "MICAR":    "Regulation (EU) 2023/1114",     # Markets in Crypto-Assets
    "MICA":     "Regulation (EU) 2023/1114",     # alias
    "EBAR":     "Regulation (EU) No 1093/2010",  # EBA founding regulation
    "ESMAR":    "Regulation (EU) No 1095/2010",  # ESMA founding regulation
    "AMLD":     "Directive (EU) 2015/849",       # 4th AMLD
    "AMLD5":    "Directive (EU) 2018/843",
    "AMLD6":    "Directive (EU) 2018/1673",
}

FULL_TO_ACRONYM: dict[str, str] = {v: k for k, v in ACRONYM_TO_FULL.items()}

# ---------------------------------------------------------------------------
# Step 2: number → canonical acronym lookup
# ---------------------------------------------------------------------------

# Extract regulation/directive numbers from the full-name strings so that
# '575/2013' and 'CRR' both resolve to the same canonical key 'CRR'.
_NUM_RE = re.compile(r"(\d{3,4}/\d{2,4}(?:/E[UC])?)")


def _build_number_to_acronym() -> dict[str, str]:
    result: dict[str, str] = {}
    for acronym, full_name in ACRONYM_TO_FULL.items():
        for m in _NUM_RE.finditer(full_name):
            num = m.group(1)
            if num not in result:
                result[num] = acronym
    return result


_NUMBER_TO_ACRONYM: dict[str, str] = _build_number_to_acronym()


def canonicalize_instrument(instrument: str) -> str:
    """
    Return the canonical acronym for an instrument identifier.

    Handles both forms so P/R/F1 is computed on a single key space:
      '575/2013'  → 'CRR'
      'CRR'       → 'CRR'
      '2013/36/EU'→ 'CRD'
      'unknown'   → 'unknown'  (returned as-is)
    """
    upper = instrument.upper().replace(" ", "")
    if upper in ACRONYM_TO_FULL:
        return upper
    return _NUMBER_TO_ACRONYM.get(instrument, instrument)


def _norm_acronym(raw: str) -> str:
    """'MiFID II' → 'MIFIDII', 'CRD IV' → 'CRDIV'."""
    return re.sub(r"[\s\-]+", "", raw).upper()


# ---------------------------------------------------------------------------
# Regex patterns  (ordered: most specific first)
# ---------------------------------------------------------------------------

# Each entry: (kind, compiled_pattern)
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Regulation (EU) No 575/2013  /  Regulation (EU) 2020/1503
    ("regulation_eu", re.compile(
        r"Regulation\s+\(EU\)\s+(?:No\.?\s+)?(\d{3,4}/\d{4})",
        re.IGNORECASE,
    )),
    # Bare year-first form: Regulation 2020/1503
    ("regulation_short", re.compile(
        r"\bRegulation\s+(\d{4}/\d{1,4})\b",
        re.IGNORECASE,
    )),
    # Directive 2014/59/EU or Directive 2009/65/EC
    ("directive", re.compile(
        r"Directive\s+(\d{4}/\d+/E[UC])",
        re.IGNORECASE,
    )),
    # Article / Art. with optional (paragraph)(sub-paragraph)
    # e.g. Article 425(2)(c), Art. 3 (1) (21), Article 39a
    ("article", re.compile(
        r"Art(?:icle)?\.?\s+(\d+[a-z]?)"
        r"(?:\s*\((\d+)\))?"
        r"(?:\s*\(([a-z\d]+)\))?",
        re.IGNORECASE,
    )),
    # EBA/GL/2017/12, ESMA/ITS/2016/03, etc.
    ("guideline", re.compile(
        r"\b((?:EBA|ESMA|EIOPA)/(?:GL|REC|ITS|RTS|OP|CP|BS)/\d{4}/\d+)\b",
        re.IGNORECASE,
    )),
    # Common acronyms (catch-all, listed longest first to avoid partial match)
    ("acronym", re.compile(
        r"\b(CRR2|BRRD2|MiFID\s*II|MiFIR|EMIR|UCITS\s*V|AIFMD|CRDIV|"
        r"CRR|BRRD|MiFID|UCITS|CRD\s*IV|CRD|PSD2|PSD|MAR|BMR|SSR|"
        r"SFDR|ECSPR|DORA)\b",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    kind: str                       # "article", "regulation_eu", "directive", …
    instrument: str                 # normalised instrument id (or "" for bare articles)
    article: Optional[str] = None
    paragraph: Optional[str] = None
    subparagraph: Optional[str] = None

    def instrument_key(self) -> str:
        """Coarse key used for partial (instrument-level) matching."""
        return self.instrument.upper().replace(" ", "")

    def full_key(self) -> str:
        """Fine key: instrument + full article hierarchy."""
        key = self.instrument.upper().replace(" ", "")
        if self.article:
            key += f":art{self.article.lower()}"
        if self.paragraph:
            key += f":({self.paragraph})"
        if self.subparagraph:
            key += f":({self.subparagraph.lower()})"
        return key

    def __str__(self) -> str:
        parts = [self.instrument] if self.instrument else []
        if self.article:
            parts.append(f"Art {self.article}")
        if self.paragraph:
            parts[-1] += f"({self.paragraph})"
        if self.subparagraph:
            parts[-1] += f"({self.subparagraph})"
        return " ".join(parts) or f"[{self.kind}]"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _build_citation(kind: str, m: re.Match) -> Citation:
    if kind == "article":
        return Citation(
            kind=kind,
            instrument="",
            article=m.group(1),
            paragraph=m.group(2),
            subparagraph=m.group(3),
        )
    if kind in ("regulation_eu", "regulation_short", "directive"):
        raw = m.group(1)
        canon = canonicalize_instrument(raw)
        # Promote kind to "acronym" when the number resolved to a known acronym
        new_kind = "acronym" if canon != raw else kind
        return Citation(kind=new_kind, instrument=canon)
    if kind == "guideline":
        return Citation(kind=kind, instrument=m.group(1).upper())
    if kind == "acronym":
        return Citation(kind=kind, instrument=_norm_acronym(m.group(1)))
    return Citation(kind=kind, instrument=m.group(0))


def find_citation_spans(text: str) -> list[tuple[int, int, str, re.Match]]:
    """
    Return (start, end, kind, match) for every raw citation pattern match in
    *text*, sorted by position. Unlike extract_citations(), this keeps the
    regex Match object (and its span) so callers can edit the raw string —
    e.g. to swap a cited article/instrument for a deliberately wrong one.
    """
    if not text:
        return []
    spans: list[tuple[int, int, str, re.Match]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), kind, m))
    spans.sort(key=lambda t: t[0])
    return spans


def extract_citations(text: str) -> list[Citation]:
    """
    Extract all citations from *text*.

    Articles are attached to the most recently mentioned instrument so that
    'Regulation (EU) No 575/2013 Article 425(2)(c)' becomes one traceable unit.
    """
    if not text:
        return []

    # Collect all matches with their start positions
    raw: list[tuple[int, str, re.Match]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), kind, m))
    raw.sort(key=lambda t: t[0])

    citations: list[Citation] = []
    last_instrument: str = ""
    last_instrument_kind: str = ""

    for _pos, kind, m in raw:
        c = _build_citation(kind, m)

        if kind in ("regulation_eu", "regulation_short", "directive",
                    "guideline", "acronym"):
            last_instrument = c.instrument
            last_instrument_kind = kind
            citations.append(c)

        elif kind == "article":
            if last_instrument:
                c = Citation(
                    kind="article",
                    instrument=last_instrument,
                    article=c.article,
                    paragraph=c.paragraph,
                    subparagraph=c.subparagraph,
                )
            citations.append(c)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[Citation] = []
    for c in citations:
        key = c.full_key()
        if key not in seen:
            seen.add(key)
            result.append(c)

    return result


def parse_legal_act(legal_act: str) -> list[Citation]:
    """
    Parse citations from a structured legal_act metadata string, e.g.
      'Directive 2013/36/EU (CRD)'
      'Regulation (EU) No 575/2013 (CRR)'
      'Markets in Financial Instruments Regulation (MiFIR) Regulation (EU) No 600/2014'
    """
    return extract_citations(legal_act)


# ---------------------------------------------------------------------------
# P / R / F1 helpers
# ---------------------------------------------------------------------------

def _jaccard_sets(gt: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    tp = len(gt & pred)
    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(gt)   if gt   else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def citation_scores(
    gt_citations: list[Citation],
    pred_citations: list[Citation],
    level: str = "full",          # "full" | "instrument"
) -> dict[str, float]:
    """
    Compute Precision / Recall / F1 between two citation lists.

    level="full"       → exact match including article/paragraph
    level="instrument" → coarse match on instrument only (partial credit)
    """
    key_fn = Citation.full_key if level == "full" else Citation.instrument_key
    gt_keys   = {key_fn(c) for c in gt_citations}
    pred_keys = {key_fn(c) for c in pred_citations}
    p, r, f1 = _jaccard_sets(gt_keys, pred_keys)
    return {"precision": p, "recall": r, "f1": f1,
            "gt_count": len(gt_keys), "pred_count": len(pred_keys)}


# ---------------------------------------------------------------------------
# Quick manual test on the first 50 questions of judge_results_train_api.jsonl
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    DATA_FILE = Path("output/judge_results_train_api.jsonl")

    if not DATA_FILE.exists():
        print(f"[skip] {DATA_FILE} not found")
    else:
        records = []
        with DATA_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        print(f"\n{'='*70}")
        print(f"FILE: {DATA_FILE.name}  ({len(records)} questions, testing first 50)")
        print("=" * 70)

        for rec in records[:50]:
            legal_act    = rec.get("meta", {}).get("legal_act", "")
            ground_truth = rec.get("ground_truth", "")

            gt_meta = parse_legal_act(legal_act)
            gt_text = extract_citations(ground_truth)

            print(f"\n--- {rec['question_id']} ---")
            print(f"  legal_act   : {legal_act[:80]}")
            print(f"  GT (meta)   : {[str(c) for c in gt_meta]}")
            print(f"  GT (text)   : {[str(c) for c in gt_text[:5]]}")

            for answerer, ans in rec.get("answers", {}).items():
                cand = extract_citations(ans.get("text", ""))
                scores_inst = citation_scores(gt_meta, cand, level="instrument")
                scores_full = citation_scores(gt_text, cand, level="full")

                print(f"  [{answerer}]")
                print(f"    Candidate   : {[str(c) for c in cand[:5]]}")
                print(f"    Scores(inst): P={scores_inst['precision']:.2f}  "
                      f"R={scores_inst['recall']:.2f}  F1={scores_inst['f1']:.2f}")
                print(f"    Scores(full): P={scores_full['precision']:.2f}  "
                      f"R={scores_full['recall']:.2f}  F1={scores_full['f1']:.2f}")
