"""
EU Financial Regulation Parser
================================
Parses EU regulation PDFs and extracts articles into structured JSON format.
Designed for: DORA, GDPR, MiFID II, MiCA, NIS2, SFDR

Output structure:
{
  "regulation": "DORA",
  "articles": [
    {
      "article_number": "19",
      "title": "Reporting of major ICT-related incidents...",
      "text": "...",
      "chapter": "IV",
      "keywords": ["incident", "reporting", "ICT"]
    }
  ]
}

Usage:
    python parse_regulations.py --input DORA.pdf --regulation DORA --output dora_articles.json
"""

import re
import json
import argparse
from pathlib import Path
import pdfplumber


# ─────────────────────────────────────────────
# CONFIG — Add your regulation files here
# ─────────────────────────────────────────────
REGULATION_FILES = {
    "DORA":    "DORA.pdf",
    "GDPR":    "GDPR.pdf",
    "MiFID2":  "MiFID2.pdf",
    "MiCA":    "MiCA.pdf",
    "NIS2":    "NIS2.pdf",
    "SFDR":    "SFDR.pdf",
}

# Keywords relevant to cross-regulation overlap zones
# Used to flag articles that are likely relevant for question generation
OVERLAP_KEYWORDS = {
    "incident_reporting": [
        "incident", "reporting", "notification", "notify", "report", "breach",
        "cyber", "ICT", "72 hours", "4 hours", "24 hours"
    ],
    "third_party_risk": [
        "third party", "third-party", "processor", "subprocessor", "outsourcing",
        "vendor", "service provider", "contractual"
    ],
    "data_protection": [
        "personal data", "data subject", "controller", "processor",
        "consent", "lawful basis", "data breach"
    ],
    "crypto_assets": [
        "crypto", "crypto-asset", "digital asset", "token", "DLT",
        "distributed ledger", "stablecoin", "CASP"
    ],
    "compliance_obligations": [
        "obligation", "shall", "must", "comply", "compliance",
        "competent authority", "supervisory", "penalty", "fine"
    ],
    "esg_sustainability": [
        "sustainability", "ESG", "sustainable", "green", "taxonomy",
        "disclosure", "SFDR", "Article 8", "Article 9"
    ],
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
# STEP 2 — Parse articles from raw text
# ─────────────────────────────────────────────
def parse_articles(text: str, regulation_name: str) -> list[dict]:
    """
    Parse individual articles from regulation text.
    Handles common EUR-Lex formatting patterns.
    """
    articles = []

    # Pattern to match article headers like:
    # "Article 19" or "Article 19\nTitle of article"
    article_pattern = re.compile(
        r'Article\s+(\d+)\s*\n([^\n]*)\n(.*?)(?=Article\s+\d+\s*\n|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    matches = list(article_pattern.finditer(text))
    print(f"  Found {len(matches)} articles in {regulation_name}")

    for match in matches:
        article_number = match.group(1).strip()
        title = match.group(2).strip()
        body = match.group(3).strip()

        # Clean up text — remove page markers, excessive whitespace
        body = re.sub(r'--- PAGE \d+ ---', '', body)
        body = re.sub(r'\n{3,}', '\n\n', body)
        body = body.strip()

        # Skip very short articles (likely parsing artifacts)
        if len(body) < 50:
            continue

        article = {
            "regulation": regulation_name,
            "article_number": article_number,
            "title": title,
            "text": body,
            "chapter": extract_chapter(text, match.start()),
            "keywords_found": detect_keywords(title + " " + body),
            "overlap_zones": detect_overlap_zones(title + " " + body),
            "char_count": len(body)
        }
        articles.append(article)

    return articles


def extract_chapter(full_text: str, article_position: int) -> str:
    """Find the chapter that contains this article by looking backwards."""
    text_before = full_text[:article_position]
    chapter_matches = list(re.finditer(
        r'CHAPTER\s+(I{1,4}V?|VI{0,3}|IX|X{0,3})\s*\n([^\n]+)',
        text_before, re.IGNORECASE
    ))
    if chapter_matches:
        last = chapter_matches[-1]
        return f"Chapter {last.group(1)} - {last.group(2).strip()}"
    return "Unknown"


def detect_keywords(text: str) -> list[str]:
    """Find all overlap keywords present in this article."""
    found = []
    text_lower = text.lower()
    for zone, keywords in OVERLAP_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(kw)
    return list(set(found))


def detect_overlap_zones(text: str) -> list[str]:
    """Detect which overlap zones this article belongs to."""
    zones = []
    text_lower = text.lower()
    for zone, keywords in OVERLAP_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 2:  # At least 2 keyword hits to qualify
            zones.append(zone)
    return zones


# ─────────────────────────────────────────────
# STEP 3 — Build cross-regulation mapping
# ─────────────────────────────────────────────
def build_overlap_mapping(all_regulations: dict[str, list[dict]]) -> dict:
    """
    Find articles across different regulations that share overlap zones.
    This is the core input for cross-regulation question generation.
    """
    print("\nBuilding cross-regulation overlap mapping...")
    mapping = {}

    for zone in OVERLAP_KEYWORDS.keys():
        zone_articles = {}
        for reg_name, articles in all_regulations.items():
            relevant = [a for a in articles if zone in a["overlap_zones"]]
            if relevant:
                zone_articles[reg_name] = [
                    {
                        "article_number": a["article_number"],
                        "title": a["title"],
                        "text": a["text"][:500] + "..."  # Preview only
                    }
                    for a in relevant
                ]
        if len(zone_articles) >= 2:  # Only include zones with 2+ regulations
            mapping[zone] = zone_articles
            regs = list(zone_articles.keys())
            counts = [len(zone_articles[r]) for r in regs]
            print(f"  [{zone}] Found overlap across: {regs} ({counts} articles each)")

    return mapping


# ─────────────────────────────────────────────
# STEP 4 — Export article pairs for prompting
# ─────────────────────────────────────────────
def export_article_pairs(mapping: dict, output_path: str):
    """
    Export article pairs ready to be fed into question generation prompts.
    Each pair contains two articles from different regulations on the same topic.
    """
    pairs = []

    for zone, regulations in mapping.items():
        reg_names = list(regulations.keys())

        # Create pairs between all regulation combinations
        for i in range(len(reg_names)):
            for j in range(i + 1, len(reg_names)):
                reg_a = reg_names[i]
                reg_b = reg_names[j]

                for art_a in regulations[reg_a]:
                    for art_b in regulations[reg_b]:
                        pair = {
                            "overlap_zone": zone,
                            "regulation_a": reg_a,
                            "article_a": art_a,
                            "regulation_b": reg_b,
                            "article_b": art_b,
                            "generation_prompt": build_generation_prompt(
                                reg_a, art_a, reg_b, art_b, zone
                            )
                        }
                        pairs.append(pair)

    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nExported {len(pairs)} article pairs to: {output_path}")
    return pairs


def build_generation_prompt(reg_a, art_a, reg_b, art_b, zone) -> str:
    """Build the question generation prompt for a given article pair."""
    return f"""You are a legal expert in EU financial regulation.

Below are two regulatory articles that both relate to the topic of: {zone.replace('_', ' ').upper()}

---
REGULATION A: {reg_a} — Article {art_a['article_number']}: {art_a['title']}
{art_a['text']}

---
REGULATION B: {reg_b} — Article {art_b['article_number']}: {art_b['title']}
{art_b['text']}

---
Your task: Generate 3 multiple choice questions that:
1. CANNOT be answered by reading only one of the above articles
2. Require understanding how {reg_a} and {reg_b} interact or conflict
3. Have ONE clearly correct answer and THREE plausible but wrong distractors

For each distractor, use a specific error type:
- Distractor 1: Correct under {reg_a} but ignores {reg_b}
- Distractor 2: Correct under {reg_b} but ignores {reg_a}
- Distractor 3: Plausible but factually wrong (e.g., wrong timeframe, wrong authority)

Output format (JSON):
{{
  "questions": [
    {{
      "question": "...",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "correct_answer": "A",
      "explanation": "...",
      "citations": {{
        "{reg_a}": "Article X",
        "{reg_b}": "Article Y"
      }},
      "distractor_errors": {{
        "B": "Ignores {reg_b} requirement",
        "C": "Ignores {reg_a} requirement",
        "D": "Wrong timeframe/authority"
      }}
    }}
  ]
}}
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Parse EU regulation PDFs")
    parser.add_argument("--input_dir", default=".", help="Directory containing regulation PDFs")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--regulations", nargs="+", default=["DORA", "GDPR"],
                        help="Which regulations to parse (default: DORA GDPR)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)
    all_regulations = {}

    # Parse each regulation
    for reg_name in args.regulations:
        if reg_name not in REGULATION_FILES:
            print(f"Warning: {reg_name} not in config, skipping")
            continue

        pdf_path = Path(args.input_dir) / REGULATION_FILES[reg_name]
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found, skipping")
            continue

        print(f"\nParsing {reg_name}...")
        raw_text = extract_text_from_pdf(str(pdf_path))
        articles = parse_articles(raw_text, reg_name)
        all_regulations[reg_name] = articles

        # Save individual regulation articles
        out_path = Path(args.output_dir) / f"{reg_name.lower()}_articles.json"
        with open(out_path, "w") as f:
            json.dump({"regulation": reg_name, "articles": articles}, f, indent=2)
        print(f"  Saved {len(articles)} articles to {out_path}")

    # Build cross-regulation mapping
    if len(all_regulations) >= 2:
        mapping = build_overlap_mapping(all_regulations)

        mapping_path = Path(args.output_dir) / "overlap_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"\nSaved overlap mapping to {mapping_path}")

        # Export article pairs with generation prompts
        pairs_path = Path(args.output_dir) / "article_pairs_for_generation.json"
        export_article_pairs(mapping, str(pairs_path))

    print("\nDone! Next step: feed article_pairs_for_generation.json into your LLM to generate MCQ questions.")


if __name__ == "__main__":
    main()