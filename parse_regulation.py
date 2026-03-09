"""
EU Financial Regulation Parser (v2)
====================================
Parses EU regulation PDFs and extracts articles into structured JSON format.
Designed for: DORA, GDPR, MiFID II, MiCA, NIS2

Improvements over v1:
  - Skips preambles, recitals, and definitions sections (non-substantive content)
  - Stricter article matching: only captures "Article XX" headers with titles
  - Narrower, domain-specific keywords tied to actual cross-regulation overlap zones
  - Filters out amendment/transitional articles that aren't useful for question generation

Output structure:
{
  "regulation": "DORA",
  "total_articles_found": 42,
  "articles_after_filtering": 35,
  "articles": [
    {
      "article_number": "19",
      "title": "Reporting of major ICT-related incidents...",
      "text": "...",
      "chapter": "III",
      "overlap_zones": ["incident_reporting"],
      "char_count": 2450
    }
  ]
}

Usage:
    python parse_regulations.py --input_dir ./data --output_dir ./output --regulations DORA GDPR
    python parse_regulations.py --input_dir ./data --output_dir ./output --regulations DORA GDPR MiFID2 MiCA NIS2
"""

import re
import json
import argparse
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber is required. Install with: pip install pdfplumber")
    exit(1)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REGULATION_FILES = {
    "DORA":    "DORA.pdf",
    "GDPR":    "GDPR.pdf",
    "MiFID2":  "MiFID2.pdf",
    "MiCA":    "MiCA.pdf",
    "NIS2":    "NIS2.pdf",
}

# ─────────────────────────────────────────────
# SECTIONS TO SKIP
# These patterns identify non-substantive content
# that should be excluded before article parsing.
# EUR-Lex PDFs typically have:
#   - Preamble (Having regard to..., Whereas...)
#   - Recitals (numbered (1), (2), ... before Article 1)
#   - Definitions article (usually Article 2 or 3)
#   - Amendment articles (amending other regulations)
#   - Final/transitional provisions
# ─────────────────────────────────────────────
SKIP_ARTICLE_PATTERNS = {
    # Titles that indicate non-substantive articles
    "title_patterns": [
        r"^Subject\s*matter$",                  # Art 1 - just states what the regulation is about
        r"^Scope$",                             # Art 2 - scope listing (useful as reference but not for questions)
        r"^Definitions?$",                      # Art 3 - definitions list
        r"^Amendment",                          # Articles amending other regulations
        r"^Repeal",                             # Repeal of prior legislation
        r"^Entry\s*into\s*force",               # Entry into force dates
        r"^Transitional\s*(provisions?|measures?|arrangements?)$",  # Transitional rules
        r"^Review$",                            # Review clauses
        r"^Transposition",                      # Directive transposition deadlines
        r"^Addressees$",                        # "This regulation is addressed to..."
        r"^Exercise\s*of\s*the\s*delegation$",  # Delegation power procedures
        r"^Committee\s*procedure$",             # Comitology procedures
    ],
}

# Article number ranges to skip per regulation (amendments, final provisions, etc.)
# These are regulation-specific because each regulation has different
# final/amendment article numbers
SKIP_ARTICLE_RANGES = {
    "DORA":   {"skip_above": 56},   # Art 57-64 are delegated acts & transitional provisions
    "GDPR":   {"skip_above": 91},   # Art 92-99 are delegated acts, final provisions
    "MiFID2": {"skip_above": 90},   # Final and amendment articles
    "MiCA":   {"skip_above": 140},  # Art 141+ are amendments and transitional
    "NIS2":   {"skip_above": 41},   # Art 42-46 are transposition, amendments, final
}


# ─────────────────────────────────────────────
# CROSS-REGULATION OVERLAP KEYWORDS (v2)
#
# These are NARROWER and more specific than v1.
# Each zone targets the actual legal provisions
# where two or more regulations interact.
# Keywords require domain-specific compound phrases
# rather than generic single words.
# ─────────────────────────────────────────────
OVERLAP_KEYWORDS = {
    "incident_reporting": {
        "description": "Where DORA, GDPR, and NIS2 incident reporting obligations overlap",
        "keywords": [
            "major ict-related incident",
            "ict-related incident",
            "personal data breach",
            "data breach notification",
            "incident reporting",
            "incident notification",
            "notify the competent authority",
            "notify the supervisory authority",
            "without undue delay",
            "72 hours",
            "initial notification",
            "intermediate report",
            "final report",
            "significant incident",
        ],
        "min_hits": 2,  # Minimum keyword matches to qualify
    },
    "third_party_ict_risk": {
        "description": "Where DORA third-party oversight meets NIS2 supply chain and GDPR processor rules",
        "keywords": [
            "ict third-party",
            "critical ict third-party",
            "third-party service provider",
            "critical third-party provider",
            "outsourcing",
            "subcontracting",
            "contractual arrangement",
            "exit strategy",
            "right of access, inspection",
            "right to audit",
            "service level",
            "concentration risk",
            "supply chain security",
            "lead overseer",
        ],
        "min_hits": 2,
    },
    "data_protection_vs_retention": {
        "description": "Where GDPR data subject rights conflict with MiFID II/DORA record-keeping",
        "keywords": [
            "right to erasure",
            "right to be forgotten",
            "data minimisation",
            "record-keeping",
            "retention period",
            "retain records",
            "recording of telephone",
            "recording of electronic communications",
            "lawful basis",
            "legal obligation",
            "legitimate interest",
            "data subject rights",
        ],
        "min_hits": 2,
    },
    "crypto_asset_classification": {
        "description": "Where MiCA and MiFID II scope boundaries create classification questions",
        "keywords": [
            "crypto-asset",
            "asset-referenced token",
            "e-money token",
            "utility token",
            "transferable security",
            "financial instrument",
            "crypto-asset service provider",
            "white paper",
            "distributed ledger technology",
            "does not apply to crypto-asset",
            "qualify as financial instrument",
            "significant token",
        ],
        "min_hits": 2,
    },
    "supervisory_authority_jurisdiction": {
        "description": "Where supervisory authority roles overlap or conflict across frameworks",
        "keywords": [
            "competent authority",
            "lead supervisory authority",
            "one-stop-shop",
            "european banking authority",
            "european securities and markets authority",
            "national competent authority",
            "supervisory powers",
            "cross-border",
            "lex specialis",
            "sector-specific",
            "oversight framework",
            "designation of critical",
        ],
        "min_hits": 2,
    },
    "operational_resilience_testing": {
        "description": "Where DORA testing requirements interact with NIS2 risk management",
        "keywords": [
            "digital operational resilience testing",
            "threat-led penetration testing",
            "vulnerability assessment",
            "resilience testing",
            "risk management measures",
            "cybersecurity risk-management",
            "proportionality",
            "simplified ict risk management",
            "testing programme",
            "risk-based approach",
        ],
        "min_hits": 2,
    },
}


# ─────────────────────────────────────────────
# STEP 1 — Extract raw text from PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF file using pdfplumber."""
    print(f"  Extracting text from: {pdf_path}")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- PAGE {i+1} ---\n{text}"
    print(f"  Extracted {len(full_text):,} characters from {len(pdf.pages)} pages")
    return full_text


# ─────────────────────────────────────────────
# STEP 2 — Strip preamble and recitals
# ─────────────────────────────────────────────
def strip_preamble(text: str) -> str:
    """
    Remove everything before the first 'Article 1' header.

    EUR-Lex regulation PDFs follow a consistent structure:
      1. Title and citation block
      2. "Having regard to..." preamble
      3. "Whereas:" followed by numbered recitals (1), (2), ...
      4. "HAS ADOPTED THIS REGULATION:" or similar
      5. Article 1 begins

    We skip everything before Article 1 because recitals
    are interpretive context, not binding provisions.
    They cannot serve as ground truth for MCQ answers.
    """
    # Find the first "Article 1" that starts a line (with optional title after it)
    match = re.search(
        r'\n\s*(Article\s+1)\s*\n',
        text,
        re.IGNORECASE
    )
    if match:
        preamble_length = match.start()
        print(f"  Stripped preamble: {preamble_length:,} characters removed")
        return text[match.start():]
    else:
        print("  WARNING: Could not locate 'Article 1' — returning full text")
        return text


# ─────────────────────────────────────────────
# STEP 3 — Parse articles from cleaned text
# ─────────────────────────────────────────────
def parse_articles(text: str, regulation_name: str) -> list[dict]:
    """
    Parse individual articles from regulation text.

    Improved matching strategy:
    - Matches "Article XX" followed by a title line
    - Captures everything until the next "Article XX" or end of text
    - Skips articles matching skip patterns (definitions, amendments, etc.)
    """
    articles = []
    skipped = {"preamble": 0, "title_pattern": 0, "range": 0, "too_short": 0}

    # Pattern explanation:
    #   Article\s+(\d+[a-z]?)  — "Article 19" or "Article 3a" (for inserted articles)
    #   \s*\n                   — newline after article number
    #   ([^\n]+)\n              — title on the next line (non-empty)
    #   (.*?)                   — body text (non-greedy)
    #   (?=Article\s+\d|$)      — until next article or end
    article_pattern = re.compile(
        r'Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n(.*?)(?=\nArticle\s+\d|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    matches = list(article_pattern.finditer(text))
    print(f"  Raw matches found: {len(matches)}")

    skip_above = SKIP_ARTICLE_RANGES.get(regulation_name, {}).get("skip_above", 999)

    for match in matches:
        article_number = match.group(1).strip()
        title = match.group(2).strip()
        body = match.group(3).strip()

        # Clean up body text
        body = re.sub(r'--- PAGE \d+ ---', '', body)
        body = re.sub(r'\n{3,}', '\n\n', body)
        body = body.strip()

        # --- FILTER 1: Skip by article number range ---
        try:
            art_num = int(re.match(r'(\d+)', article_number).group(1))
            if art_num > skip_above:
                skipped["range"] += 1
                continue
        except (ValueError, AttributeError):
            pass

        # --- FILTER 2: Skip by title pattern ---
        should_skip = False
        for pattern in SKIP_ARTICLE_PATTERNS["title_patterns"]:
            if re.search(pattern, title, re.IGNORECASE):
                should_skip = True
                skipped["title_pattern"] += 1
                break
        if should_skip:
            continue

        # --- FILTER 3: Skip very short articles (likely parsing artifacts) ---
        if len(body) < 100:
            skipped["too_short"] += 1
            continue

        article = {
            "regulation": regulation_name,
            "article_number": article_number,
            "title": title,
            "text": body,
            "chapter": extract_chapter(text, match.start()),
            "overlap_zones": detect_overlap_zones(title + " " + body),
            "char_count": len(body),
        }
        articles.append(article)

    print(f"  Skipped: {skipped['title_pattern']} by title, "
          f"{skipped['range']} by range, "
          f"{skipped['too_short']} too short")
    print(f"  Final articles: {len(articles)}")

    return articles


def extract_chapter(full_text: str, article_position: int) -> str:
    """Find the chapter/title that contains this article by looking backwards."""
    text_before = full_text[:article_position]

    # Try CHAPTER first (DORA, MiCA, NIS2 style)
    chapter_matches = list(re.finditer(
        r'(?:CHAPTER|TITLE|SECTION)\s+(I{1,4}V?|VI{0,3}|IX|X{0,3}|\d+)\s*\n([^\n]+)',
        text_before, re.IGNORECASE
    ))
    if chapter_matches:
        last = chapter_matches[-1]
        section_type = last.group(0).split()[0].strip().title()
        return f"{section_type} {last.group(1).strip()} — {last.group(2).strip()}"

    return "Unknown"


# ─────────────────────────────────────────────
# STEP 4 — Overlap zone detection (v2 - stricter)
# ─────────────────────────────────────────────
def detect_overlap_zones(text: str) -> list[str]:
    """
    Detect which cross-regulation overlap zones this article belongs to.

    v2 improvements:
    - Uses compound phrases instead of single words
    - Requires min_hits threshold per zone (default 2)
    - Returns zone names only when meaningfully matched
    """
    zones = []
    text_lower = text.lower()

    for zone_name, zone_config in OVERLAP_KEYWORDS.items():
        hits = sum(1 for kw in zone_config["keywords"] if kw.lower() in text_lower)
        if hits >= zone_config["min_hits"]:
            zones.append(zone_name)

    return zones


# ─────────────────────────────────────────────
# STEP 5 — Build cross-regulation overlap mapping
# ─────────────────────────────────────────────
def build_overlap_mapping(all_regulations: dict[str, list[dict]]) -> dict:
    """
    Find articles across different regulations that share overlap zones.
    Only includes zones where 2+ regulations have relevant articles.
    """
    print("\n" + "=" * 60)
    print("CROSS-REGULATION OVERLAP MAPPING")
    print("=" * 60)

    mapping = {}

    for zone_name, zone_config in OVERLAP_KEYWORDS.items():
        zone_articles = {}

        for reg_name, articles in all_regulations.items():
            relevant = [a for a in articles if zone_name in a["overlap_zones"]]
            if relevant:
                zone_articles[reg_name] = [
                    {
                        "article_number": a["article_number"],
                        "title": a["title"],
                        "text": a["text"][:800],  # Truncated preview for mapping file
                        "full_text_char_count": a["char_count"],
                    }
                    for a in relevant
                ]

        # Only include zones with overlap across 2+ regulations
        if len(zone_articles) >= 2:
            mapping[zone_name] = {
                "description": zone_config["description"],
                "regulations_involved": list(zone_articles.keys()),
                "article_counts": {r: len(arts) for r, arts in zone_articles.items()},
                "articles": zone_articles,
            }
            regs = list(zone_articles.keys())
            counts = [len(zone_articles[r]) for r in regs]
            print(f"\n  [{zone_name}]")
            print(f"    {zone_config['description']}")
            print(f"    Regulations: {regs}")
            print(f"    Article counts: {dict(zip(regs, counts))}")
            for reg in regs:
                for art in zone_articles[reg]:
                    print(f"      {reg} Art. {art['article_number']}: {art['title']}")

    return mapping


# ─────────────────────────────────────────────
# STEP 6 — Export article pairs for LLM prompting
# ─────────────────────────────────────────────
def export_article_pairs(
    mapping: dict,
    all_regulations: dict[str, list[dict]],
    output_path: str
):
    """
    Export article pairs ready for question generation prompts.
    Uses FULL article text (not truncated) from the parsed regulations.
    """
    pairs = []

    for zone_name, zone_data in mapping.items():
        reg_names = zone_data["regulations_involved"]
        articles_by_reg = zone_data["articles"]

        for i in range(len(reg_names)):
            for j in range(i + 1, len(reg_names)):
                reg_a = reg_names[i]
                reg_b = reg_names[j]

                # Get full article text from the original parsed data
                full_articles_a = {
                    a["article_number"]: a
                    for a in all_regulations[reg_a]
                    if zone_name in a["overlap_zones"]
                }
                full_articles_b = {
                    a["article_number"]: a
                    for a in all_regulations[reg_b]
                    if zone_name in a["overlap_zones"]
                }

                for art_num_a, art_a in full_articles_a.items():
                    for art_num_b, art_b in full_articles_b.items():
                        pair = {
                            "pair_id": f"{reg_a}_Art{art_num_a}_x_{reg_b}_Art{art_num_b}",
                            "overlap_zone": zone_name,
                            "regulation_a": {
                                "name": reg_a,
                                "article_number": art_num_a,
                                "title": art_a["title"],
                                "text": art_a["text"],
                            },
                            "regulation_b": {
                                "name": reg_b,
                                "article_number": art_num_b,
                                "title": art_b["title"],
                                "text": art_b["text"],
                            },
                            "generation_prompt": build_generation_prompt(
                                reg_a, art_a, reg_b, art_b, zone_name
                            ),
                        }
                        pairs.append(pair)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(pairs)} article pairs to: {output_path}")

    # Print summary
    zone_counts = {}
    for p in pairs:
        z = p["overlap_zone"]
        zone_counts[z] = zone_counts.get(z, 0) + 1
    print("  Pairs per overlap zone:")
    for z, c in sorted(zone_counts.items()):
        print(f"    {z}: {c} pairs")

    return pairs


def build_generation_prompt(reg_a, art_a, reg_b, art_b, zone) -> str:
    """Build the question generation prompt for a given article pair."""
    zone_display = zone.replace("_", " ").title()

    return f"""You are a legal expert in EU financial regulation. Your task is to generate
scenario-based multiple-choice questions that test cross-regulation reasoning.

OVERLAP ZONE: {zone_display}
This tests how {reg_a} and {reg_b} interact or conflict on this topic.

---
REGULATION A: {reg_a}
Article {art_a['article_number']}: {art_a['title']}

{art_a['text']}

---
REGULATION B: {reg_b}
Article {art_b['article_number']}: {art_b['title']}

{art_b['text']}

---
INSTRUCTIONS:
Generate 3 multiple-choice questions. Each question MUST:

1. Present a REALISTIC SCENARIO involving a specific entity type (bank, insurer,
   CASP, investment firm, etc.) in a specific EU Member State facing a concrete
   situation (cyberattack, product launch, client request, regulatory filing, etc.)

2. REQUIRE reasoning across BOTH {reg_a} and {reg_b} — a question answerable
   from only one regulation is NOT acceptable.

3. Have exactly ONE correct answer and THREE plausible but wrong distractors.

4. Each distractor must represent a DISTINCT, NAMEABLE reasoning error:
   - Distractor type 1: Applies {reg_a} correctly but ignores {reg_b}
   - Distractor type 2: Applies {reg_b} correctly but ignores {reg_a}
   - Distractor type 3: Plausible but factually wrong (wrong timeline,
     wrong authority, wrong threshold, or fabricated exemption)

5. Cite SPECIFIC ARTICLE NUMBERS for the correct answer and each distractor.

OUTPUT FORMAT (strict JSON):
{{
  "questions": [
    {{
      "scenario": "A [entity type] in [EU country] [specific situation]...",
      "question": "What is/are the [specific regulatory question]?",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "correct_answer": "B",
      "explanation": "Detailed explanation citing specific articles...",
      "source_articles": {{
        "{reg_a}": ["Art. X(Y)"],
        "{reg_b}": ["Art. X(Y)"]
      }},
      "distractor_analysis": {{
        "A": {{"error_type": "ignores_{reg_b.lower()}", "explanation": "..."}},
        "C": {{"error_type": "ignores_{reg_a.lower()}", "explanation": "..."}},
        "D": {{"error_type": "fabricated_rule", "explanation": "..."}}
      }},
      "reasoning_type": "cross_regulation_overlap|framework_precedence|classification_boundary|timeline_conflict|authority_identification",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}
"""


# ─────────────────────────────────────────────
# STEP 7 — Print parsing diagnostics
# ─────────────────────────────────────────────
def print_diagnostics(all_regulations: dict[str, list[dict]]):
    """Print a summary of what was parsed for quality checking."""
    print("\n" + "=" * 60)
    print("PARSING DIAGNOSTICS")
    print("=" * 60)

    for reg_name, articles in all_regulations.items():
        print(f"\n  {reg_name}: {len(articles)} articles")

        # Articles with overlap zones
        with_zones = [a for a in articles if a["overlap_zones"]]
        print(f"    With overlap zones: {len(with_zones)}")

        # Zone distribution
        zone_counts = {}
        for a in articles:
            for z in a["overlap_zones"]:
                zone_counts[z] = zone_counts.get(z, 0) + 1
        if zone_counts:
            for z, c in sorted(zone_counts.items()):
                print(f"      {z}: {c} articles")

        # List first 5 articles as sanity check
        print(f"    First 5 articles:")
        for a in articles[:5]:
            zones_str = ", ".join(a["overlap_zones"]) if a["overlap_zones"] else "none"
            print(f"      Art. {a['article_number']}: {a['title'][:60]}  [{zones_str}]")

        # Flag any potential issues
        art_nums = [a["article_number"] for a in articles]
        if "1" in art_nums:
            print(f"    ⚠  Article 1 (Subject matter) was NOT filtered — check skip patterns")
        if "3" in art_nums and reg_name in ("DORA", "GDPR", "MiCA"):
            print(f"    ⚠  Article 3 (Definitions) was NOT filtered — check skip patterns")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parse EU regulation PDFs into structured JSON for benchmark generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse DORA and GDPR
  python parse_regulations.py --input_dir ./data --regulations DORA GDPR

  # Parse all five regulations
  python parse_regulations.py --input_dir ./data --regulations DORA GDPR MiFID2 MiCA NIS2

  # Custom output directory
  python parse_regulations.py --input_dir ./data --output_dir ./parsed --regulations DORA GDPR
        """
    )
    parser.add_argument("--input_dir", default=".", help="Directory containing regulation PDFs (in a 'data' subfolder)")
    parser.add_argument("--output_dir", default="output", help="Output directory for JSON files")
    parser.add_argument("--regulations", nargs="+", default=["DORA", "GDPR", "MiCA", "MiFID2", "NIS2", "SFDR"],
                        help="Which regulations to parse (default: DORA GDPR)")
    parser.add_argument("--keep_definitions", action="store_true",
                        help="Keep definitions articles (Article 2/3) — excluded by default")
    parser.add_argument("--keep_scope", action="store_true",
                        help="Keep scope articles — excluded by default")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full article list during parsing")
    args = parser.parse_args()

    # Allow user to override skip patterns
    if args.keep_definitions:
        SKIP_ARTICLE_PATTERNS["title_patterns"] = [
            p for p in SKIP_ARTICLE_PATTERNS["title_patterns"]
            if "Definitions" not in p
        ]
        print("NOTE: Keeping definitions articles (--keep_definitions)")

    if args.keep_scope:
        SKIP_ARTICLE_PATTERNS["title_patterns"] = [
            p for p in SKIP_ARTICLE_PATTERNS["title_patterns"]
            if "Scope" not in p
        ]
        print("NOTE: Keeping scope articles (--keep_scope)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_regulations = {}

    # Parse each regulation
    for reg_name in args.regulations:
        if reg_name not in REGULATION_FILES:
            print(f"WARNING: {reg_name} not in config — available: {list(REGULATION_FILES.keys())}")
            continue

        pdf_path = Path(args.input_dir) / "data" / REGULATION_FILES[reg_name]
        if not pdf_path.exists():
            # Also try without the data/ subfolder
            pdf_path = Path(args.input_dir) / REGULATION_FILES[reg_name]
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping {reg_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"PARSING: {reg_name}")
        print(f"{'=' * 60}")

        # Extract, strip preamble, parse
        raw_text = extract_text_from_pdf(str(pdf_path))
        clean_text = strip_preamble(raw_text)
        articles = parse_articles(clean_text, reg_name)
        all_regulations[reg_name] = articles

        # Save individual regulation articles
        out_path = Path(args.output_dir) / f"{reg_name.lower()}_articles.json"
        output_data = {
            "regulation": reg_name,
            "total_articles_found": len(articles),
            "overlap_zone_summary": {},
            "articles": articles,
        }

        # Compute overlap zone summary
        for a in articles:
            for z in a["overlap_zones"]:
                if z not in output_data["overlap_zone_summary"]:
                    output_data["overlap_zone_summary"][z] = []
                output_data["overlap_zone_summary"][z].append(
                    f"Art. {a['article_number']}: {a['title']}"
                )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(articles)} articles to {out_path}")

        # Verbose: print all articles
        if args.verbose:
            for a in articles:
                zones = ", ".join(a["overlap_zones"]) if a["overlap_zones"] else "-"
                print(f"    Art. {a['article_number']:>3}: {a['title'][:55]:<55} [{zones}]")

    # Diagnostics
    print_diagnostics(all_regulations)

    # Build cross-regulation mapping
    if len(all_regulations) >= 2:
        mapping = build_overlap_mapping(all_regulations)

        mapping_path = Path(args.output_dir) / "overlap_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"\nSaved overlap mapping to {mapping_path}")

        # Export article pairs with generation prompts
        pairs_path = Path(args.output_dir) / "article_pairs_for_generation.json"
        export_article_pairs(mapping, all_regulations, str(pairs_path))

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review the parsed articles in {args.output_dir}/<regulation>_articles.json")
    print(f"  2. Check overlap_mapping.json for cross-regulation article pairs")
    print(f"  3. Feed article_pairs_for_generation.json into your LLM to generate MCQs")
    print(f"  4. Manually validate generated questions against the source regulation texts")


if __name__ == "__main__":
    main()