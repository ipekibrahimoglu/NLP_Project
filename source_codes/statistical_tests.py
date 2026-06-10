import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

# ── Load results ──────────────────────────────────────────────────────────────
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data["papers"])
df = df[df["year"] >= 2015].reset_index(drop=True)

x = df["year"].to_numpy(dtype=float)
y = df["alignment_score"].to_numpy(dtype=float)

years_sorted = sorted(df["year"].unique())
groups       = [df.loc[df["year"] == yr, "alignment_score"].values for yr in years_sorted]

# ── 1. Linear Regression ─────────────────────────────────────────────────────
slope, intercept = np.polyfit(x, y, 1)
y_hat  = slope * x + intercept
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2     = 1.0 - ss_res / ss_tot

print("Linear Regression (year → score):")
print(f"  Slope:     {slope:.6f} per year")
print(f"  Intercept: {intercept:.4f}")
print(f"  R²:        {r2:.4f}")

# ── 2. Spearman Correlation ───────────────────────────────────────────────────
rho, pval = spearmanr(x, y)
print(f"\nSpearman Correlation (year vs score):")
print(f"  rho:     {rho:.4f}")
print(f"  p-value: {pval:.6f}")
if pval < 0.05:
    print("  → Statistically significant drift (p < 0.05)")
else:
    print("  → No statistically significant drift")

# ── 3. Kruskal-Wallis Test ────────────────────────────────────────────────────
stat, p = kruskal(*groups)
print(f"\nKruskal-Wallis test across years:")
print(f"  Statistic: {stat:.4f}")
print(f"  p-value:   {p:.6f}")
if p < 0.05:
    print("  → Score distributions differ significantly across years (p < 0.05)")
else:
    print("  → No significant difference across years")

# ── 4. Bootstrap 95% CI ───────────────────────────────────────────────────────
def bootstrap_ci(arr, n_boot=2000, ci=0.95, seed=42):
    rng   = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))

rows = []
for yr in years_sorted:
    arr    = df.loc[df["year"] == yr, "alignment_score"].values
    lo, hi = bootstrap_ci(arr)
    rows.append({"year": int(yr), "n": len(arr),
                 "mean": float(arr.mean()), "ci95_low": lo, "ci95_high": hi})

ci_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
print(f"\nYearly Bootstrap CI:\n{ci_df.to_string(index=False)}")

# Plot Bootstrap CI
plt.figure(figsize=(10, 5))
plt.plot(ci_df["year"], ci_df["mean"], marker="o", color="steelblue", label="Mean")
plt.fill_between(ci_df["year"], ci_df["ci95_low"], ci_df["ci95_high"],
                 alpha=0.2, color="steelblue", label="95% Bootstrap CI")
slope_y, int_y = np.polyfit(ci_df["year"].values, ci_df["mean"].values, 1)
plt.plot(ci_df["year"], slope_y * ci_df["year"] + int_y,
         linestyle="--", color="red", label=f"Trend (slope={slope_y:.4f}/yr)")
plt.xlabel("Year")
plt.ylabel("Mean Alignment Score")
plt.title("Yearly Mean Alignment with 95% Bootstrap CI")
plt.legend()
plt.tight_layout()
plt.savefig("bootstrap_ci.png", dpi=200)
plt.close()
print("\nSaved: bootstrap_ci.png")

ci_df.to_csv("yearly_bootstrap_ci.csv", index=False)
print("Saved: yearly_bootstrap_ci.csv")

# ── 5. Qualitative Outlier Validation ────────────────────────────────────────
N = 10
outliers_df = df.nsmallest(N, "alignment_score")[["year", "alignment_score", "title", "abstract"]].copy()
outliers_df["abstract_excerpt"] = (
    outliers_df["abstract"].fillna("").str.replace(r"\s+", " ", regex=True).str[:200] + "..."
)
outliers_df = outliers_df.drop(columns=["abstract"])

print(f"\nTOP {N} MOST MISALIGNED PAPERS:")
print("-" * 80)
for _, row in outliers_df.iterrows():
    print(f"[{row['alignment_score']:.3f}] ({int(row['year'])}) {row['title']}")
    print(f"  → {row['abstract_excerpt']}")
    print()

outliers_df.to_csv("outliers_qualitative.csv", index=False)
print("Saved: outliers_qualitative.csv")
