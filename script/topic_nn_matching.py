"""
topic_nn_matching.py

Zero-shot nearest-neighbor topic matching against LLM judge `topic_coherence`.

Approach:
  1. Embed all unique GT topic labels (154 categories from train_questions.jsonl)
  2. Embed each candidate answer
  3. Compute cosine similarity from each answer to ALL topic labels at once
     (single matrix multiply — fast)
  4. Derive two metrics per answer:
       sim_to_gt_topic  — cosine sim between the answer and its own GT topic
       topic_margin     — sim_to_gt_topic minus the highest sim to any OTHER topic
                          (positive = correctly on-topic, negative = drifted)
  5. Correlate both metrics against all four LLM judge dims

Requirements:
    pip install --timeout=300 sentence-transformers scipy
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_FILE           = "output/judge_results_train_api.jsonl"
TRAIN_QUESTIONS_FILE = "data/splits/train_questions.jsonl"
OUTPUT_FILE          = "output/topic_nn_matching_results.json"
JUDGE_DIMS           = ["accuracy", "completeness", "topic_coherence", "citation_quality"]
DELETED_QUESTION_IDS = {
    "ESMA_ESMA_QA_1580", "ESMA_ESMA_QA_1589",
    "ESMA_ESMA_QA_1668", "ESMA_ESMA_QA_1569",
}

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load GT topic labels
# ---------------------------------------------------------------------------
qid_to_topic: dict[str, str] = {}
with open(TRAIN_QUESTIONS_FILE) as f:
    for line in f:
        rec = json.loads(line)
        topic = rec.get("topic", "").strip()
        if topic:
            qid_to_topic[rec["question_id"]] = topic

unique_topics = sorted(set(qid_to_topic.values()))
topic_to_idx  = {t: i for i, t in enumerate(unique_topics)}
print(f"Unique GT topic labels : {len(unique_topics)}")
print(f"Questions with a topic : {len(qid_to_topic)}")

# ---------------------------------------------------------------------------
# Load and flatten judge results
# ---------------------------------------------------------------------------
records = []
with open(JSONL_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

rows = []
n_skipped = 0
for rec in records:
    qid = rec["question_id"]
    if qid in DELETED_QUESTION_IDS or qid not in qid_to_topic:
        n_skipped += 1
        continue
    for answerer, ans in rec.get("answers", {}).items():
        candidate = ans.get("text", "")
        for judge, sc in ans.get("judge_scores", {}).items():
            if any(sc.get(d) is None for d in JUDGE_DIMS):
                continue
            rows.append({
                "question_id":      qid,
                "answerer_model":   answerer,
                "judge_model":      judge,
                "candidate_answer": candidate,
                "gt_topic":         qid_to_topic[qid],
                "gt_topic_idx":     topic_to_idx[qid_to_topic[qid]],
                **{d: float(sc[d]) for d in JUDGE_DIMS},
            })

print(f"Skipped records        : {n_skipped}")
print(f"Total judgments        : {len(rows)}")

# Unique (question_id, answerer) pairs — metrics don't depend on the judge
pair_map: dict = {}
for r in rows:
    key = (r["question_id"], r["answerer_model"])
    pair_map.setdefault(key, {
        "candidate_answer": r["candidate_answer"],
        "gt_topic":         r["gt_topic"],
        "gt_topic_idx":     r["gt_topic_idx"],
    })

pair_keys    = list(pair_map.keys())
candidates   = [pair_map[k]["candidate_answer"] for k in pair_keys]
gt_topics    = [pair_map[k]["gt_topic"]         for k in pair_keys]
gt_topic_idx = [pair_map[k]["gt_topic_idx"]     for k in pair_keys]
n_pairs      = len(pair_keys)
print(f"Unique (question, answerer) pairs: {n_pairs}")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

print(f"\nEmbedding {len(unique_topics)} topic labels ...")
topic_embs = embedder.encode(
    unique_topics, batch_size=32, show_progress_bar=False, normalize_embeddings=True
)  # shape: (n_topics, 768)

print(f"Embedding {n_pairs} candidate answers ...")
answer_embs = embedder.encode(
    candidates, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)  # shape: (n_pairs, 768)

# ---------------------------------------------------------------------------
# Similarity matrix  (n_pairs × n_topics)
# Single matrix multiply — normalized embeddings → dot product = cosine sim
# ---------------------------------------------------------------------------
sim_matrix = answer_embs @ topic_embs.T   # (n_pairs, n_topics)

# For each answer: similarity to its own GT topic and to every other topic
sim_to_gt      = sim_matrix[np.arange(n_pairs), gt_topic_idx]   # (n_pairs,)

# Margin: sim to GT minus the highest sim to any OTHER topic
sim_matrix_no_gt = sim_matrix.copy()
sim_matrix_no_gt[np.arange(n_pairs), gt_topic_idx] = -np.inf
sim_to_best_other = sim_matrix_no_gt.max(axis=1)                 # (n_pairs,)
topic_margin = sim_to_gt - sim_to_best_other                     # (n_pairs,)

# Predicted topic (nearest neighbor)
predicted_idx    = sim_matrix.argmax(axis=1)                     # (n_pairs,)
predicted_topics = [unique_topics[i] for i in predicted_idx]
topic_accuracy   = float(np.mean(predicted_idx == np.array(gt_topic_idx)))

print(f"\nTopic retrieval accuracy (NN predicted == GT): {topic_accuracy:.3%}")
print(f"sim_to_gt_topic  — mean={sim_to_gt.mean():.4f}  "
      f"min={sim_to_gt.min():.4f}  max={sim_to_gt.max():.4f}")
print(f"topic_margin     — mean={topic_margin.mean():.4f}  "
      f"min={topic_margin.min():.4f}  max={topic_margin.max():.4f}")

# Store per-pair results
sim_gt_by_pair       = dict(zip(pair_keys, sim_to_gt.tolist()))
margin_by_pair       = dict(zip(pair_keys, topic_margin.tolist()))
predicted_by_pair    = dict(zip(pair_keys, predicted_topics))

# ---------------------------------------------------------------------------
# Attach to rows
# ---------------------------------------------------------------------------
for r in rows:
    key = (r["question_id"], r["answerer_model"])
    r["sim_to_gt_topic"]  = sim_gt_by_pair[key]
    r["topic_margin"]     = margin_by_pair[key]
    r["predicted_topic"]  = predicted_by_pair[key]

# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Correlations with LLM judge scores")
print("=" * 60)


def report_corr(x_vals, y_vals, label: str) -> dict:
    x, y = np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"  {label:<50s}  pearson={pr:+.4f} (p={pp:.2e})  "
          f"spearman={sr:+.4f} (p={sp:.2e})")
    return {
        "n":          int(len(x)),
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
    }


sim_gt_vals    = [r["sim_to_gt_topic"] for r in rows]
margin_vals    = [r["topic_margin"]    for r in rows]
corr_results: dict = {}

print(f"\n  {'Metric vs LLM judge dim':<50s}  {'Pearson':>25}   {'Spearman':>25}")
print("  " + "-" * 110)

print("\n  --- sim_to_gt_topic vs LLM judge dims ---")
for dim in JUDGE_DIMS:
    llm_vals = [r[dim] for r in rows]
    corr_results[f"sim_to_gt_topic_vs_{dim}"] = report_corr(
        sim_gt_vals, llm_vals, f"sim_to_gt_topic   vs  {dim}"
    )

print("\n  --- topic_margin vs LLM judge dims ---")
for dim in JUDGE_DIMS:
    llm_vals = [r[dim] for r in rows]
    corr_results[f"topic_margin_vs_{dim}"] = report_corr(
        margin_vals, llm_vals, f"topic_margin      vs  {dim}"
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results_out = [
    {
        "question_id":     r["question_id"],
        "answerer_model":  r["answerer_model"],
        "judge_model":     r["judge_model"],
        "gt_topic":        r["gt_topic"],
        "predicted_topic": r["predicted_topic"],
        "topic_correct":   r["predicted_topic"] == r["gt_topic"],
        "sim_to_gt_topic": round(r["sim_to_gt_topic"], 4),
        "topic_margin":    round(r["topic_margin"],    4),
        **{d: r[d] for d in JUDGE_DIMS},
    }
    for r in rows
]

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_unique_topics":    len(unique_topics),
        "n_pairs":            n_pairs,
        "n_judgments":        len(rows),
        "topic_nn_accuracy":  round(topic_accuracy, 4),
        "correlations":       corr_results,
        "results":            results_out,
    }, f, indent=2)

print(f"\nSaved -> {OUTPUT_FILE}")
