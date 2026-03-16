"""
ESMA Q&A PDF to JSON Parser
============================
Parses ESMA Q&A PDF documents and extracts entries that have published answers.

Usage:
    python parse_esma_qa.py ESMA_QA.pdf                    # outputs esma_qa.json
    python parse_esma_qa.py ESMA_QA.pdf -o my_output.json  # custom output name

Requirements:
    pip install pdfplumber
"""

import argparse
import json
import re
import sys

import pdfplumber


def extract_full_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse excessive newlines, strip lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def parse_qa_entries(full_text: str) -> list[dict]:
    """
    Parse ESMA Q&A entries from extracted PDF text.

    Each entry starts with:
        Submission Date
        ESMA_QA_XXXX DD/MM/YYYY

    The answer block starts with:
        ESMA Answer
        DD-MM-YYYY
        Original language

    And ends when the next entry (Submission Date + ESMA_QA_XXXX) begins.

    Only entries with a published "ESMA Answer" are included.
    """
    # Split text into sections at each new QA entry boundary
    sections = re.split(r"(?=Submission Date\s*\n\s*ESMA_QA_\d+)", full_text)

    results = []
    for section in sections:
        # Check this is an actual QA entry (not TOC or preamble)
        id_match = re.match(r"Submission Date\s*\n\s*(ESMA_QA_\d+)", section)
        if not id_match:
            continue

        qa_id = id_match.group(1)

        # Skip entries without a published answer
        if "ESMA Answer" not in section:
            continue

        # Extract subject matter (between "Subject Matter" and "Question" or "Additional")
        sm_match = re.search(
            r"Subject Matter\s*\n(.+?)(?:\n(?:Question|Additional))",
            section,
            re.DOTALL,
        )
        subject_matter = clean_text(sm_match.group(1)) if sm_match else ""

        # Extract question (between "Question" and "ESMA Answer")
        q_match = re.search(
            r"\nQuestion\s*\n(.*?)(?=ESMA Answer)", section, re.DOTALL
        )
        question = clean_text(q_match.group(1)) if q_match else ""

        # Extract answer (after "ESMA Answer" + date + optional "Original language")
        a_match = re.search(
            r"ESMA Answer\s*\n\d{2}-\d{2}-\d{4}\s*\n(?:Original language\s*\n)?(.*)",
            section,
            re.DOTALL,
        )
        answer = clean_text(a_match.group(1)) if a_match else ""

        if answer:
            results.append(
                {
                    "id": qa_id,
                    "subject_matter": subject_matter,
                    "question": question,
                    "answer": answer,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse ESMA Q&A PDF into JSON format."
    )
    parser.add_argument("pdf_path", help="Path to the ESMA Q&A PDF file", default = "data/ESMA_QA.pdf")
    parser.add_argument(
        "-o",
        "--output",
        default="esma_qa.json",
        help="Output JSON file path (default: esma_qa.json)",
    )
    args = parser.parse_args()

    print(f"Reading PDF: {args.pdf_path}")
    full_text = extract_full_text(args.pdf_path)
    print(f"Extracted {len(full_text):,} characters")

    results = parse_qa_entries(full_text)
    print(f"Parsed {len(results)} Q&A entries with answers")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()