"""
Compute TF-IDF cosine similarity between ground_truth and candidate_answer
for all judge result files (big/small, EBA/ESMA). Saves per-file JSON results.
"""

import json
import os
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

    out_path = input_file.replace(".json", "_similarity.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved -> {out_path}")
