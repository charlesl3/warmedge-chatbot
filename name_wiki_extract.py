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
OUTPUT_DIR = DATA_ROOT / "skater_wiki_info"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set to True if you want to force re-download
FORCE_REFRESH = True


# ===============================
# WIKI CALL
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
# PROCESS ALL SKATERS
# ===============================

def process_all_skaters():
    discipline_folders = [
        "male_single_skaters",
        "female_single_skaters",
        "male_ice_dancers",
        "female_ice_dancers"
    ]

    for folder in discipline_folders:
        folder_path = DATA_ROOT / folder

        if not folder_path.exists():
            continue

        print(f"\n=== Processing folder: {folder} ===")

        for skater_file in folder_path.glob("*.json"):
            with open(skater_file, "r", encoding="utf-8") as f:
                skater_data = json.load(f)

            title = skater_data["wiki_title"]

            output_filename = title.replace(" ", "_").replace("/", "_") + ".json"
            output_path = OUTPUT_DIR / output_filename

            # Skip only if file exists AND we are not forcing refresh
            if output_path.exists() and not FORCE_REFRESH:
                print("Skipping (already exists):", title)
                continue

            print("Fetching:", title)

            wiki_json = fetch_wiki_extract(title)

            if wiki_json:
                # Optional: attach metadata
                wiki_json["_warmedge_meta"] = {
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "force_refresh": FORCE_REFRESH
                }

                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(wiki_json, out, indent=2, ensure_ascii=False)

            time.sleep(0.5)  # be polite


if __name__ == "__main__":
    print("Starting wiki enrichment...")
    process_all_skaters()
    print("Done.")