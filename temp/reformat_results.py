"""Convert NDJSON judge results to a pretty-printed JSON array."""

import json
import sys
from pathlib import Path


def reformat(path: str):
    p = Path(path)
    records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Reformatted {len(records)} records → {p}")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["output/judge_results_esma.json"]:
        reformat(f)
