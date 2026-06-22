import hashlib
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINAL_DATA_DIR = DATA_DIR / "final_data"
SPLITS_DIR = DATA_DIR / "splits"

SOURCES = {
    "EBA": FINAL_DATA_DIR / "eba_qa_web.json",
    "ESMA": FINAL_DATA_DIR / "esma_qa_web.json",
}

# Rename source-specific field names to a shared schema across regulators.
FIELD_ALIASES = {
    "level1_regulation": "legal_act",
    "final_answer": "answer",
}

# Common output field order; any extra source-specific fields (e.g. esma's
# "url") are appended after these.
COMMON_FIELDS = [
    "question_id",
    "regulator",
    "id",
    "status",
    "legal_act",
    "topic",
    "subject_matter",
    "question",
    "answer",
]


def load_all_questions():
    questions = []
    for regulator, path in SOURCES.items():
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for record in records:
            renamed = {FIELD_ALIASES.get(k, k): v for k, v in record.items()}
            renamed["regulator"] = regulator
            renamed["question_id"] = f"{regulator}_{renamed['id']}"
            ordered = {k: renamed[k] for k in COMMON_FIELDS if k in renamed}
            ordered.update(
                {k: v for k, v in renamed.items() if k not in COMMON_FIELDS}
            )
            questions.append(ordered)
    return questions


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_manifest(manifest, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def split_data(seed=42):
    questions = load_all_questions()

    # Topic has many singleton classes, so stratifying on regulator+topic
    # would make train_test_split fail (each class needs >=2 members for
    # a 3-way split). Stratifying on regulator alone keeps EBA/ESMA
    # proportions fair across train/val/test, which is the main goal here.
    regulators = [q["regulator"] for q in questions]

    train, temp = train_test_split(
        questions, test_size=0.30, stratify=regulators, random_state=seed
    )
    temp_regulators = [q["regulator"] for q in temp]
    val, test = train_test_split(
        temp, test_size=0.50,  # 15% / 15% of the full set
        stratify=temp_regulators, random_state=seed,
    )

    save_jsonl(train, SPLITS_DIR / "train_questions.jsonl")
    save_jsonl(val, SPLITS_DIR / "val_questions.jsonl")
    save_jsonl(test, SPLITS_DIR / "test_questions.jsonl")

    all_ids = sorted(q["question_id"] for q in questions)
    split_hash = hashlib.sha256("|".join(all_ids).encode("utf-8")).hexdigest()

    save_manifest(
        {
            "seed": seed,
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "split_hash": split_hash,
            "date_created": datetime.now().isoformat(),
        },
        SPLITS_DIR / "split_manifest.json",
    )

    return train, val, test


if __name__ == "__main__":
    train, val, test = split_data()
    print(f"train={len(train)} val={len(val)} test={len(test)}")
