import requests
import json
import time

ISSN = "1351-3249"
START_YEAR = 2015
END_YEAR = 2025

url = f"https://api.crossref.org/journals/{ISSN}/works"
all_dois = []
cursor = "*"

while True:
    params = {
        "filter": f"from-pub-date:{START_YEAR},until-pub-date:{END_YEAR}",
        "select": "DOI,title,published,type",
        "rows": 100,
        "cursor": cursor
    }

    response = requests.get(url, params=params)
    data = response.json()
    items = data["message"]["items"]

    if not items:
        break

    for item in items:
        doi = item.get("DOI", "")
        title = item.get("title", [""])[0]
        year = item.get("published", {}).get("date-parts", [[None]])[0][0]
        all_dois.append({"doi": doi, "title": title, "year": year})

    print(f"Collected: {len(all_dois)}")
    cursor = data["message"].get("next-cursor")
    if not cursor:
        break

    time.sleep(1)

with open("dois.json", "w", encoding="utf-8") as f:
    json.dump(all_dois, f, ensure_ascii=False, indent=2)

print(f"\nDone. Total: {len(all_dois)} DOIs saved to dois.json")