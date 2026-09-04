import requests
import json
import time
import re
from pathlib import Path

API = "https://en.wikisource.org/w/api.php"
OUTPUT_PATH = Path("datasets/wikisource/wikisource_pd_scripts.jsonl")

PD_PATTERNS = [
    r"\{\{\s*PD",
    r"\{\{\s*Public[\s_]+domain",
    r"\[\[\s*Category\s*:\s*Public\s+domain",
]

def mw_api(params):
    base = {
        "format": "json",
        "formatversion": "2"
    }
    base.update(params)
    headers = {
        "User-Agent": "script-supervisor-ai-portfolio-project"
    }
    r = requests.get(API, params=base, headers=headers)
    r.raise_for_status()
    return r.json()

def get_portal_links():
    data = mw_api({
        "action": "parse",
        "page": "Portal:Scripts",
        "prop": "links"
    })

    titles = []
    for link in data.get("parse", {}).get("links", []):
        if link.get("ns") == 0 and "exists" in link:
            titles.append(link["title"])

    return titles

def get_wikitext(title):
    data = mw_api({
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "content",
        "rvslots": "main"
    })

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    revisions = pages[0].get("revisions", [])
    if not revisions:
        return None

    return revisions[0]["slots"]["main"]["content"]

def is_public_domain(text):
    for pattern in PD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def main(max_pages=200):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    titles = get_portal_links()
    print(f"Found {len(titles)} script links")

    kept = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for title in titles[:max_pages]:
            print(f"Checking: {title}")

            wikitext = get_wikitext(title)
            if not wikitext:
                continue

            if not is_public_domain(wikitext):
                continue

            record = {
                "source": "Wikisource",
                "title": title,
                "text": wikitext
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            time.sleep(0.3)

    print(f"\nSaved {kept} public-domain scripts")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main(max_pages=200)
