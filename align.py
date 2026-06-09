import json
import numpy as np

PAPERS_FILE = "papers.json"

print("Loading embeddings...")
aims_embedding = np.load("aims_embedding.npy")
paper_embeddings = np.load("paper_embeddings.npy")

with open(PAPERS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]    

papers = [p for p in papers if p.get("year") and int(p["year"]) >= 2015]

print("Computing alignment scores...")
scores = np.dot(paper_embeddings, aims_embedding)

for i, paper in enumerate(papers):
    paper["alignment_score"] = float(scores[i])

scores_array = np.array(scores)
print(f"\nAlignment Score Statistics:")
print(f"Mean:   {scores_array.mean():.4f}")
print(f"Std:    {scores_array.std():.4f}")
print(f"Min:    {scores_array.min():.4f}")
print(f"Max:    {scores_array.max():.4f}")
print(f"Median: {np.median(scores_array):.4f}")

threshold = np.percentile(scores_array, 5)
print(f"\nOutlier threshold (5th percentile): {threshold:.4f}")
outliers = [p for p in papers if p["alignment_score"] < threshold]
print(f"Outliers: {len(outliers)}")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump({"papers": papers, "total": len(papers)}, f, ensure_ascii=False, indent=2)

print("\nDone. Results saved to results.json")