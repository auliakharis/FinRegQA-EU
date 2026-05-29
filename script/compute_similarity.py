"""
Compute TF-IDF cosine similarity between ground_truth and candidate_answer
for all judge result files (big/small, EBA/ESMA). Saves per-file JSON results
and reports Pearson/Spearman correlations with each judge score dimension.
"""

import json
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILES = [
    "output/big_judge_result/judge_results_eba_qa_api.json",
    "output/big_judge_result/judge_results_esma_qa_api.json",
    "output/small_judge_result/judge_results_eba.json",
    "output/small_judge_result/judge_results_esma.json",
]

for input_file in FILES:
    print(f"\n{'='*60}")
    print(f"File: {input_file}")
    print('='*60)

    with open(input_file) as f:
        data = json.load(f)

    texts_gt = [item["ground_truth"] for item in data]
    texts_ca = [item["candidate_answer"] for item in data]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts_gt + texts_ca)

    n = len(data)
    scores = []
    results = []
    for i in range(n):
        sim = float(cosine_similarity(tfidf[i], tfidf[n + i])[0][0])
        scores.append(sim)
        results.append({"id": data[i].get("id", i), "similarity": round(sim, 4)})
        print(f"[{i:03d}] id={str(data[i].get('id', i)):30s}  similarity={sim:.4f}")

    summary = {
        "mean": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "count": n,
    }
    print(f"\nMean: {summary['mean']}  Min: {summary['min']}  Max: {summary['max']}  Count: {n}")

    # Correlations with judge score dimensions (skip None values)
    score_keys = [k for k in data[0]["scores"] if k != "reasoning"]
    correlations = {}
    print("\nCorrelations with judge scores (Pearson / Spearman):")
    for key in score_keys:
        pairs = [(s, float(item["scores"][key])) for s, item in zip(scores, data) if item["scores"].get(key) is not None]
        sim_vals, judge_vals = zip(*pairs)
        pearson_r, pearson_p = pearsonr(sim_vals, judge_vals)
        spearman_r, spearman_p = spearmanr(sim_vals, judge_vals)
        correlations[key] = {
            "n": len(pairs),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
        }
        print(f"  {key:15s}  n={len(pairs):5d}  pearson={pearson_r:+.4f} (p={pearson_p:.3e})  spearman={spearman_r:+.4f} (p={spearman_p:.3e})")

    summary["correlations"] = correlations

    out_path = input_file.replace(".json", "_similarity.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")
