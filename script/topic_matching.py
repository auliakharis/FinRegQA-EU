"""
topic_matching.py

Validate automated topic-matching metrics against LLM judge `topic_coherence`
scores, using the ground-truth topic labels already present in
data/splits/train_questions.jsonl  (field: `topic`).

Two automated metrics:

  1. BERTTopic match
     Fit BERTTopic on all candidate answers.  For each answer, retrieve the
     top-keyword string of its assigned topic cluster.  Embed those keywords
     and the ground-truth `topic` label with a sentence-transformer and
     compute their cosine similarity.
     → captures whether the answer is actually about the right topic

  2. BERT direct similarity
     Embed the candidate answer and the ground-truth `topic` label directly,
     compute cosine similarity.
     → simpler baseline: does the answer semantically align with the topic?

Both are correlated (Pearson + Spearman) against all four LLM judge dims, with
`topic_coherence` expected to show the highest correlation.

Requirements (not in requirements.txt — install once):
    pip install --timeout=300 bertopic sentence-transformers scipy umap-learn hdbscan
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_FILE            = "output/judge_results_train_api.jsonl"
TRAIN_QUESTIONS_FILE  = "data/splits/train_questions.jsonl"
OUTPUT_FILE           = "output/topic_matching_results.json"
JUDGE_DIMS            = ["accuracy", "completeness", "topic_coherence", "citation_quality"]
DELETED_QUESTION_IDS  = {
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
# Load ground-truth topic labels  (one label per question_id)
# ---------------------------------------------------------------------------
topic_labels: dict[str, str] = {}
with open(TRAIN_QUESTIONS_FILE) as f:
    for line in f:
        rec = json.loads(line)
        topic_labels[rec["question_id"]] = rec.get("topic", "").strip()

print(f"Loaded topic labels for {len(topic_labels)} questions")

# ---------------------------------------------------------------------------
# Load judge results and flatten to one row per (question, answerer, judge)
# ---------------------------------------------------------------------------
records = []
with open(JSONL_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

rows = []
n_skipped_deleted  = 0
n_skipped_no_topic = 0
n_skipped_score    = 0

for rec in records:
    qid = rec["question_id"]
    if qid in DELETED_QUESTION_IDS:
        n_skipped_deleted += 1
        continue
    if qid not in topic_labels or not topic_labels[qid]:
        n_skipped_no_topic += 1
        continue
    for answerer, ans in rec.get("answers", {}).items():
        candidate = ans.get("text", "")
        for judge, sc in ans.get("judge_scores", {}).items():
            if any(sc.get(d) is None for d in JUDGE_DIMS):
                n_skipped_score += 1
                continue
            rows.append({
                "question_id":      qid,
                "answerer_model":   answerer,
                "judge_model":      judge,
                "candidate_answer": candidate,
                "gt_topic":         topic_labels[qid],
                **{d: float(sc[d]) for d in JUDGE_DIMS},
            })

print(f"Skipped — deleted: {n_skipped_deleted}, "
      f"missing topic: {n_skipped_no_topic}, "
      f"incomplete scores: {n_skipped_score}")
print(f"Total judgments kept: {len(rows)}")

# Unique (question_id, answerer) pairs — automated metrics don't depend on judge
pair_map: dict = {}
for r in rows:
    key = (r["question_id"], r["answerer_model"])
    pair_map.setdefault(key, {
        "candidate_answer": r["candidate_answer"],
        "gt_topic":         r["gt_topic"],
    })

pair_keys  = list(pair_map.keys())
candidates = [pair_map[k]["candidate_answer"] for k in pair_keys]
gt_topics  = [pair_map[k]["gt_topic"]         for k in pair_keys]
n_pairs    = len(pair_keys)
print(f"Unique (question, answerer) pairs: {n_pairs}")

# ---------------------------------------------------------------------------
# Shared embedding model
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

# Pre-encode the GT topic labels once — reused in both parts
print("\nEncoding ground-truth topic labels ...")
emb_gt_topics = embedding_model.encode(
    gt_topics, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)

# ---------------------------------------------------------------------------
# Part 1: BERTTopic — fit on answers, compare extracted topic with GT label
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part 1: BERTTopic topic match")
print("=" * 60)

print(f"Fitting BERTTopic on {n_pairs} candidate answers ...")
# Remove stopwords and bare numbers from topic keywords — without this,
# c-TF-IDF picks up function words and article/regulation numbers as top terms.
vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=2,
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",   # letters only, min length 2
)
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=10,
    nr_topics="auto",
    verbose=False,
)
assigned_topics, _ = topic_model.fit_transform(candidates)

n_outliers_raw = sum(1 for t in assigned_topics if t == -1)
print(f"Before outlier reduction — topics: {len(topic_model.get_topics()) - 1}  "
      f"outliers: {n_outliers_raw}")

# Reassign every outlier (-1) to its nearest topic by embedding distance
assigned_topics = topic_model.reduce_outliers(candidates, assigned_topics)
topic_model.update_topics(candidates, topics=assigned_topics)

n_topics   = len(topic_model.get_topics())
n_outliers = sum(1 for t in assigned_topics if t == -1)
print(f"After  outlier reduction — topics: {n_topics}  outliers: {n_outliers}")


def topic_keywords(model: BERTopic, topic_id: int, top_n: int = 10) -> str:
    words = model.get_topic(topic_id)
    return " ".join(w for w, _ in words[:top_n]) if words else ""


answer_topic_texts = []
for i, tid in enumerate(assigned_topics):
    kw = topic_keywords(topic_model, tid)
    answer_topic_texts.append(kw if kw else candidates[i])

print("Embedding BERTTopic keyword strings ...")
emb_topic_kw = embedding_model.encode(
    answer_topic_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)

bertopic_sims = np.sum(emb_topic_kw * emb_gt_topics, axis=1).tolist()
bertopic_by_pair      = dict(zip(pair_keys, bertopic_sims))
bertopic_keywords_by_pair = dict(zip(pair_keys, answer_topic_texts))

print(f"BERTTopic match — mean={np.mean(bertopic_sims):.4f}  "
      f"min={np.min(bertopic_sims):.4f}  max={np.max(bertopic_sims):.4f}")

# ---------------------------------------------------------------------------
# Part 2: BERT direct similarity — candidate answer vs GT topic label
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Part 2: BERT direct similarity  (answer vs GT topic label)")
print("=" * 60)

print("Encoding candidate answers ...")
emb_candidates = embedding_model.encode(
    candidates, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)

bert_sims = np.sum(emb_candidates * emb_gt_topics, axis=1).tolist()
bert_by_pair = dict(zip(pair_keys, bert_sims))

print(f"BERT similarity   — mean={np.mean(bert_sims):.4f}  "
      f"min={np.min(bert_sims):.4f}  max={np.max(bert_sims):.4f}")

# ---------------------------------------------------------------------------
# Attach scores to rows and compute correlations
# ---------------------------------------------------------------------------
for r in rows:
    key = (r["question_id"], r["answerer_model"])
    r["bertopic_match"]    = bertopic_by_pair[key]
    r["bert_similarity"]   = bert_by_pair[key]
    r["bertopic_keywords"] = bertopic_keywords_by_pair[key]

print("\n" + "=" * 60)
print("Part 3: Correlations with LLM judge scores")
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


bertopic_vals = [r["bertopic_match"]  for r in rows]
bert_sim_vals = [r["bert_similarity"] for r in rows]
corr_results: dict = {}

print(f"\n  {'Metric vs LLM judge dim':<50s}  {'Pearson':>25}   {'Spearman':>25}")
print("  " + "-" * 110)

print("\n  --- BERTTopic match vs LLM judge dims ---")
for dim in JUDGE_DIMS:
    llm_vals = [r[dim] for r in rows]
    corr_results[f"bertopic_match_vs_{dim}"] = report_corr(
        bertopic_vals, llm_vals, f"BERTTopic match   vs  {dim}"
    )

print("\n  --- BERT direct similarity vs LLM judge dims ---")
for dim in JUDGE_DIMS:
    llm_vals = [r[dim] for r in rows]
    corr_results[f"bert_similarity_vs_{dim}"] = report_corr(
        bert_sim_vals, llm_vals, f"BERT similarity   vs  {dim}"
    )

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results_out = [
    {
        "question_id":       r["question_id"],
        "answerer_model":    r["answerer_model"],
        "judge_model":       r["judge_model"],
        "gt_topic":          r["gt_topic"],
        "bertopic_keywords": r["bertopic_keywords"],
        "bertopic_match":    round(r["bertopic_match"],  4),
        "bert_similarity":   round(r["bert_similarity"], 4),
        **{d: r[d] for d in JUDGE_DIMS},
    }
    for r in rows
]

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "n_pairs":       n_pairs,
        "n_judgments":   len(rows),
        "n_bert_topics":     n_topics,
        "n_outliers_before": n_outliers_raw,
        "n_outliers_after":  n_outliers,
        "correlations":  corr_results,
        "results":       results_out,
    }, f, indent=2)
print(f"\nSaved -> {OUTPUT_FILE}")
