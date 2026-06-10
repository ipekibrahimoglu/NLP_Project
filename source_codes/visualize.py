import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
df = pd.DataFrame(papers)
df = df[df["year"] >= 2015].reset_index(drop=True)
embeddings = np.load("paper_embeddings.npy")
embeddings = embeddings[:len(df)]
aims_embedding = np.load("aims_embedding.npy")

print(f"Loaded {len(df)} papers, embeddings shape: {embeddings.shape}")

# Histogram
scores = df["alignment_score"].values
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=30, color="steelblue", edgecolor="black")
plt.xlabel("Alignment Score")
plt.ylabel("Number of Papers")
plt.title("Alignment Score Distribution")
plt.tight_layout()
plt.savefig("histogram.png", dpi=200)
plt.close()
print("histogram.png saved")

# Drift
yearly = df.groupby("year")["alignment_score"].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(yearly["year"], yearly["alignment_score"], marker="o", color="steelblue")
plt.xlabel("Year")
plt.ylabel("Mean Alignment Score")
plt.title("Mean Alignment Score by Year (Thematic Drift)")
plt.tight_layout()
plt.savefig("drift.png", dpi=200)
plt.close()
print("drift.png saved")

# Boxplot
years = sorted(df["year"].unique())
data_by_year = [df[df["year"] == y]["alignment_score"].values for y in years]
plt.figure(figsize=(12, 5))
plt.boxplot(data_by_year, tick_labels=years)
plt.xlabel("Year")
plt.ylabel("Alignment Score")
plt.title("Alignment Score Distribution by Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot.png", dpi=200)
plt.close()
print("boxplot.png saved")

# Outlier rate
threshold = np.percentile(scores, 5)
df["outlier"] = df["alignment_score"] < threshold
outlier_rate = df.groupby("year")["outlier"].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(outlier_rate["year"], outlier_rate["outlier"], color="salmon", edgecolor="black")
plt.xlabel("Year")
plt.ylabel("Outlier Rate")
plt.title("Outlier Rate by Year")
plt.tight_layout()
plt.savefig("outlier_rate.png", dpi=200)
plt.close()
print("outlier_rate.png saved")

# Percentile trend
p10 = df.groupby("year")["alignment_score"].quantile(0.10)
p50 = df.groupby("year")["alignment_score"].quantile(0.50)
p90 = df.groupby("year")["alignment_score"].quantile(0.90)
plt.figure(figsize=(10, 5))
plt.plot(p10.index, p10.values, label="p10", linestyle="--", color="red")
plt.plot(p50.index, p50.values, label="median", color="steelblue")
plt.plot(p90.index, p90.values, label="p90", linestyle="--", color="green")
plt.fill_between(p10.index, p10.values, p90.values, alpha=0.1, color="steelblue")
plt.xlabel("Year")
plt.ylabel("Alignment Score")
plt.title("Yearly Alignment Score Trend (p10 / median / p90)")
plt.legend()
plt.tight_layout()
plt.savefig("trend_p10_median_p90.png", dpi=200)
plt.close()
print("trend_p10_median_p90.png saved")

# UMAP
try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42)
    all_embeddings = np.vstack([embeddings, aims_embedding.reshape(1, -1)])
    reduced = reducer.fit_transform(all_embeddings)
    paper_reduced = reduced[:-1]
    aims_reduced = reduced[-1]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        paper_reduced[:, 0], paper_reduced[:, 1],
        c=df["alignment_score"].values, cmap="RdYlGn",
        alpha=0.7, s=20
    )
    plt.colorbar(scatter, label="Alignment Score")
    plt.scatter(aims_reduced[0], aims_reduced[1], c="black", s=200, marker="*", label="Aims & Scope", zorder=5)
    plt.title("UMAP Projection colored by Alignment Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("umap_alignment.png", dpi=200)
    plt.close()
    print("umap_alignment.png saved")
except Exception as e:
    print("UMAP error:", e)

print("\nAll visualizations saved.")