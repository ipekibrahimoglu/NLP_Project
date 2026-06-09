import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

AIMS_AND_SCOPE = """
Natural Language Engineering is an open access journal which meets the needs
of professionals and researchers working in all areas of natural language
processing (NLP). Its aim is to bridge the gap between traditional
computational linguistics research and the implementation of practical
applications with potential real-world use. The journal publishes original
research articles on a broad range of methods and resources applied in NLP,
language processing tasks and NLP applications, including machine translation,
translation technology, sentiment analysis, information retrieval, question
answering, text summarisation, text simplification, and speech processing.
"""

with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
papers = [p for p in papers if p.get("year", 0) >= 2015]
abstracts = [p["abstract"] for p in papers]

print(f"Loaded {len(papers)} papers")

models = {
    "MiniLM-L6"  : "all-MiniLM-L6-v2",
    "MPNet-Base" : "all-mpnet-base-v2",
    "SPECTER"    : "allenai-specter",
}

results = {}
for model_name, model_id in models.items():
    print(f"\nLoading model: {model_name} ({model_id})...")
    model = SentenceTransformer(model_id)
    print(f"  Embedding Aims & Scope...")
    aims_vec = model.encode(AIMS_AND_SCOPE.strip()).reshape(1, -1)
    print(f"  Embedding {len(abstracts)} abstracts...")
    paper_vecs = model.encode(abstracts, show_progress_bar=True)
    scores = cosine_similarity(paper_vecs, aims_vec).flatten()
    results[model_name] = scores
    print(f"  Done! Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

df = pd.DataFrame(results)
df["year"] = [p["year"] for p in papers]
df["title"] = [p["title"] for p in papers]

# Plot 1: Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, model_name in enumerate(models.keys()):
    axes[i].hist(df[model_name], bins=30, color="skyblue", edgecolor="black")
    axes[i].set_title(f"{model_name}\nMean: {df[model_name].mean():.3f}")
    axes[i].set_xlabel("Alignment Score")
    axes[i].set_ylabel("Number of Papers" if i == 0 else "")
plt.suptitle("Alignment Score Distribution by Model", fontsize=13)
plt.tight_layout()
plt.savefig("model_comparison_histograms.png", dpi=200)
plt.close()
print("model_comparison_histograms.png saved")

# Plot 2: Drift
plt.figure(figsize=(12, 5))
colors = ["steelblue", "darkorange", "green"]
for i, model_name in enumerate(models.keys()):
    yearly = df.groupby("year")[model_name].mean().reset_index()
    yearly = yearly.sort_values("year")
    plt.plot(yearly["year"], yearly[model_name],
             marker="o", label=model_name, color=colors[i])
plt.xlabel("Year")
plt.ylabel("Mean Alignment Score")
plt.title("Thematic Drift by Model (2015-2024)")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison_drift.png", dpi=200)
plt.close()
print("model_comparison_drift.png saved")

# Plot 3: Correlation scatter
model_names = list(models.keys())
pairs = [
    (model_names[0], model_names[1]),
    (model_names[0], model_names[2]),
    (model_names[1], model_names[2]),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (m1, m2) in enumerate(pairs):
    correlation = np.corrcoef(df[m1], df[m2])[0, 1]
    axes[i].scatter(df[m1], df[m2], alpha=0.3, s=5, color="steelblue")
    axes[i].set_xlabel(f"{m1} Score")
    axes[i].set_ylabel(f"{m2} Score")
    axes[i].set_title(f"{m1} vs {m2}\nCorrelation: {correlation:.3f}")
plt.suptitle("Model Score Correlations", fontsize=13)
plt.tight_layout()
plt.savefig("model_comparison_correlation.png", dpi=200)
plt.close()
print("model_comparison_correlation.png saved")

# Summary
print("\nMODEL COMPARISON SUMMARY:")
print(f"{'Model':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 50)
for model_name in models.keys():
    scores = df[model_name]
    print(f"{model_name:<15} {scores.mean():>8.4f} {scores.std():>8.4f} {scores.min():>8.4f} {scores.max():>8.4f}")

df.to_csv("model_comparison_results.csv", index=False)
print("\nmodel_comparison_results.csv saved")