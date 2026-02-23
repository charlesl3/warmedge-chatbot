import requests
import json
import time
from pathlib import Path

# ===============================
# CONFIG
# ===============================

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WarmEdgeRAGBot/1.0 (contact: your_real_email@example.com)"
}

DATA_ROOT = Path("data")
OUTPUT_DIR = DATA_ROOT / "general_wiki_info"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_NAME = "Category:Figure skating"

FORCE_REFRESH = True
SLEEP_SECONDS = 0.5


# ===============================
# FETCH CATEGORY MEMBERS
# ===============================

def fetch_category_members(category_name):
    print(f"Fetching category members for: {category_name}")

    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_name,
            "cmlimit": "500",
            "format": "json"
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        res = requests.get(WIKI_API, params=params, headers=HEADERS)

        if res.status_code != 200:
            print("Failed to fetch category:", res.status_code)
            break

        data = res.json()
        members.extend(data["query"]["categorymembers"])

        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break

    print(f"Total raw members found: {len(members)}")
    return members


# ===============================
# FETCH PAGE EXTRACT
# ===============================

def fetch_wiki_extract(title):
    params = {
        "action": "query",
        "prop": "extracts|revisions",
        "titles": title,
        "format": "json",
        "explaintext": True,
        "rvprop": "ids|timestamp"
    }

    res = requests.get(WIKI_API, params=params, headers=HEADERS)

    if res.status_code != 200:
        print("Failed:", title, res.status_code)
        return None

    return res.json()


# ===============================
# PROCESS ALL GENERAL PAGES
# ===============================

def process_all_pages():
    members = fetch_category_members(CATEGORY_NAME)

    for member in members:
        title = member["title"]
        namespace = member["ns"]

        # ✅ Only keep real article pages (ns=0)
        if namespace != 0:
            continue

        output_filename = title.replace(" ", "_").replace("/", "_") + ".json"
        output_path = OUTPUT_DIR / output_filename

        if output_path.exists() and not FORCE_REFRESH:
            print("Skipping:", title)
            continue

        print("Fetching:", title)

        wiki_json = fetch_wiki_extract(title)

        if wiki_json:
            wiki_json["_warmedge_meta"] = {
                "category": CATEGORY_NAME,
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "force_refresh": FORCE_REFRESH
            }

            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(wiki_json, out, indent=2, ensure_ascii=False)

        time.sleep(SLEEP_SECONDS)


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("Starting general wiki enrichment...")
    process_all_pages()
    print("Done.")