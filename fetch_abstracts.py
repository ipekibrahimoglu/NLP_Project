import requests
import json
import time

INPUT_FILE  = "dois.json"
OUTPUT_FILE = "papers.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dois = json.load(f)

print(f"Total DOIs to process: {len(dois)}")

papers    = []
not_found = 0

for i, item in enumerate(dois):
    doi = item.get("doi", "")
    if not doi:
        continue
    url    = f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
    params = {"fields": "title,abstract,year,authors"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data     = response.json()
            abstract = data.get("abstract", "")
            if abstract:
                papers.append({
                    "doi":      doi,
                    "title":    data.get("title", item.get("title", "")),
                    "abstract": abstract,
                    "year":     data.get("year", item.get("year")),
                    "authors":  [a.get("name", "") for a in data.get("authors", [])]
                })
        else:
            not_found += 1
    except Exception as e:
        print(f"Error on {doi}: {e}")
    if (i + 1) % 50 == 0:
        print(f"Processed: {i+1}/{len(dois)} — Papers with abstract: {len(papers)}")
    time.sleep(1)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"papers": papers, "total": len(papers)}, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(papers)} papers with abstracts saved.")
print(f"Not found / no abstract: {not_found}")

# ── Dataset Summary ───────────────────────────────────────────────────────────
print("\n=== Data Curation Summary ===")
print(f"  DOIs retrieved (Crossref):               {len(dois)}")
print(f"  Papers with abstract (Semantic Scholar): {len(papers)}")
print(f"  Dropout (no abstract found):             {len(dois) - len(papers)}")
print(f"  Retention rate:                          {len(papers)/len(dois)*100:.1f}%")

import pandas as pd
df_summary = pd.DataFrame(papers)
df_summary = df_summary[df_summary["year"] >= 2015]
print(f"  Year range (year >= 2015):               {int(df_summary['year'].min())}–{int(df_summary['year'].max())}")
print(f"\nPapers per year:")
print(df_summary["year"].value_counts().sort_index().rename("n").to_frame().to_string())
