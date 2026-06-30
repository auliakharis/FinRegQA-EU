"""
Compute sentence-embedding cosine similarity between ground_truth and
candidate_answer for output/judge_results_train_api.jsonl. Saves results
and reports Pearson/Spearman correlations with each judge score dimension.

Record shape: one line per question, with multiple answerer models each
scored by multiple judge models (see script/judge_exploration.ipynb). We:
  1. skip judgments with any missing score dimension
  2. skip questions whose ground_truth is just a "question deleted" stub
     (no real reference answer to compare against)
  3. use sentence-embedding cosine similarity instead of TF-IDF, since
     TF-IDF only captures lexical (word) overlap — embeddings capture
     semantic/content overlap even when the candidate paraphrases the
     ground truth in different words.
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    EMBED_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    EMBED_DEVICE = "mps"
else:
    EMBED_DEVICE = "cpu"
print(f"Embedding device: {EMBED_DEVICE}")

TRAIN_API_FILE = "output/judge_results_train_api.jsonl"
TRAIN_API_DIMS = ["accuracy", "completeness", "topic_coherence", "citation_quality"]
DELETED_QUESTION_IDS = {
    "ESMA_ESMA_QA_1580",
    "ESMA_ESMA_QA_1589",
    "ESMA_ESMA_QA_1668",
    "ESMA_ESMA_QA_1569",
}

print(f"\n{'='*60}")
print(f"File: {TRAIN_API_FILE}")
print('='*60)

records = []
with open(TRAIN_API_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

rows = []
n_skipped_deleted_questions = 0
n_skipped_deleted_judgments = 0
n_skipped_incomplete = 0
for rec in records:
    qid = rec["question_id"]
    if qid in DELETED_QUESTION_IDS:
        n_skipped_deleted_questions += 1
        n_skipped_deleted_judgments += sum(
            len(ans.get("judge_scores", {})) for ans in rec.get("answers", {}).values()
        )
        continue
    for answerer, ans in rec.get("answers", {}).items():
        candidate = ans.get("text", "")
        for judge, sc in ans.get("judge_scores", {}).items():
            if any(sc.get(d) is None for d in TRAIN_API_DIMS):
                n_skipped_incomplete += 1
                continue
            row = {
                "question_id": qid,
                "answerer_model": answerer,
                "judge_model": judge,
                "ground_truth": rec["ground_truth"],
                "candidate_answer": candidate,
            }
            for d in TRAIN_API_DIMS:
                row[d] = float(sc[d])
            rows.append(row)

print(f"Skipped {n_skipped_deleted_questions} deleted-question stubs "
      f"({n_skipped_deleted_judgments} judgments)")
print(f"Skipped {n_skipped_incomplete} judgments with an incomplete score")
print(f"Remaining judgments: {len(rows)}")

# Similarity only depends on (question, answerer), not on the judge — compute it
# once per unique pair to avoid wasted embedding work, then broadcast to every row.
pair_texts = {}
for r in rows:
    key = (r["question_id"], r["answerer_model"])
    pair_texts.setdefault(key, (r["ground_truth"], r["candidate_answer"]))

pair_keys = list(pair_texts.keys())
texts_gt = [pair_texts[k][0] for k in pair_keys]
texts_ca = [pair_texts[k][1] for k in pair_keys]

embedder = SentenceTransformer("all-mpnet-base-v2", device=EMBED_DEVICE)
emb_gt = embedder.encode(texts_gt, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
emb_ca = embedder.encode(texts_ca, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

n_pairs = len(pair_keys)
pair_sims_arr = np.sum(emb_gt * emb_ca, axis=1)  # normalized embeddings -> dot product = cosine sim
sim_by_pair = {key: float(pair_sims_arr[i]) for i, key in enumerate(pair_keys)}

for r in rows:
    r["similarity"] = sim_by_pair[(r["question_id"], r["answerer_model"])]

pair_sims = list(sim_by_pair.values())
summary = {
    "mean": round(sum(pair_sims) / len(pair_sims), 4),
    "min": round(min(pair_sims), 4),
    "max": round(max(pair_sims), 4),
    "count_pairs": n_pairs,
    "count_judgments": len(rows),
    "skipped_deleted_questions": n_skipped_deleted_questions,
    "skipped_deleted_judgments": n_skipped_deleted_judgments,
    "skipped_incomplete_score": n_skipped_incomplete,
}
print(f"\nMean: {summary['mean']}  Min: {summary['min']}  Max: {summary['max']}  "
      f"Pairs: {n_pairs}  Judgments: {len(rows)}")

correlations = {}
print("\nCorrelations with judge scores (Pearson / Spearman):")
for dim in TRAIN_API_DIMS:
    sim_vals = [r["similarity"] for r in rows]
    judge_vals = [r[dim] for r in rows]
    pearson_r, pearson_p = pearsonr(sim_vals, judge_vals)
    spearman_r, spearman_p = spearmanr(sim_vals, judge_vals)
    correlations[dim] = {
        "n": len(rows),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
    }
    print(f"  {dim:18s}  n={len(rows):5d}  pearson={pearson_r:+.4f} (p={pearson_p:.3e})  "
          f"spearman={spearman_r:+.4f} (p={spearman_p:.3e})")

summary["correlations"] = correlations

results = [
    {
        "question_id": r["question_id"],
        "answerer_model": r["answerer_model"],
        "judge_model": r["judge_model"],
        "similarity": round(r["similarity"], 4),
        **{d: r[d] for d in TRAIN_API_DIMS},
    }
    for r in rows
]

out_path = TRAIN_API_FILE.replace(".jsonl", "_semantic_similarity.json")
with open(out_path, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)
print(f"\nSaved -> {out_path}")
