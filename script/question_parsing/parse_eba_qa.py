"""
EBA Single Rulebook Q&A PDF to JSON Parser
============================================
Parses EBA Q&A PDF documents and extracts entries that have a "Final answer".

Usage:
    python parse_eba_qa.py EBA_QA.pdf                    # outputs eba_qa.json
    python parse_eba_qa.py EBA_QA.pdf -o my_output.json  # custom output name

Requirements:
    pip install pdfplumber
"""

import argparse
import json
import re
import sys

import pdfplumber

pdf_path = "data/EBA_QA.pdf"


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
    Parse EBA Q&A entries from extracted PDF text.

    Each entry starts with:
        Question ID YYYY_NNNN
        Status Final Q&A

    The answer block is between "Final answer" and "Answer prepared by".

    Only entries with a "Final answer" are included.
    """
    # Remove page number lines like "1 of 42"
    full_text = re.sub(r"\n\d+ of \d+\n", "\n", full_text)

    # Split text into sections at each new QA entry boundary
    sections = re.split(r"(?=Question ID \d{4}_\d{4}\nStatus)", full_text)

    results = []
    for section in sections:
        # Check this is an actual QA entry (not TOC or preamble)
        id_match = re.match(r"Question ID (\d{4}_\d{4})\nStatus", section)
        if not id_match:
            continue

        qa_id = id_match.group(1)

        # Skip entries without a final answer
        if "Final answer" not in section:
            continue

        # Extract subject matter (between "Subject matter" and "Question\n")
        sm_match = re.search(
            r"Subject matter\s+(.+?)(?:\nQuestion\n)",
            section,
            re.DOTALL,
        )
        subject_matter = clean_text(sm_match.group(1)) if sm_match else ""

        # Extract question (between "Question\n" and "Background on the" or "Final answer")
        q_match = re.search(
            r"\nQuestion\n(.*?)(?=Background on the\n|Final answer)",
            section,
            re.DOTALL,
        )
        question = clean_text(q_match.group(1)) if q_match else ""

        # Extract answer (between "Final answer" and "Answer prepared by")
        a_match = re.search(
            r"Final answer\n(.*?)(?=Answer prepared by)",
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
        description="Parse EBA Single Rulebook Q&A PDF into JSON format."
    )
    parser.add_argument("pdf_path", help="Path to the EBA Q&A PDF file")
    parser.add_argument(
        "-o",
        "--output",
        default="output/eba_qa.json",
        help="Output JSON file path (default: output/eba_qa.json)",
    )
    args = parser.parse_args()

    print(f"Reading PDF: {args.pdf_path}")
    full_text = extract_full_text(args.pdf_path)
    print(f"Extracted {len(full_text):,} characters")

    results = parse_qa_entries(full_text)
    print(f"Parsed {len(results)} Q&A entries with final answers")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()