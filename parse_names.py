import requests
import time
import json
from pathlib import Path
from urllib.parse import quote

# ===============================
# CONFIG
# ===============================

URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    # Use your real email here to comply with Wikipedia policy
    "User-Agent": "WarmEdgeRAGBot/1.0 (contact: your_real_email@example.com)"
}

BASE_OUTPUT_DIR = Path("data")

ROOT_CATEGORIES = {
    "male_single_skaters": "Category:Male single skaters by nationality",
    "female_single_skaters": "Category:Female single skaters by nationality",
    "male_ice_dancers": "Category:Male ice dancers by nationality",
    "female_ice_dancers": "Category:Female ice dancers by nationality",
}


# ===============================
# CATEGORY HELPERS
# ===============================

def get_subcategories(parent_category):
    subcats = set()

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": parent_category,
        "cmtype": "subcat",
        "cmlimit": "max",
        "format": "json"
    }

    while True:
        res = requests.get(URL, params=params, headers=HEADERS)
        data = res.json()

        for item in data["query"]["categorymembers"]:
            subcats.add(item["title"])

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(0.5)
        else:
            break

    return sorted(subcats)


def get_skaters_from_category(category, discipline_tag):
    skaters = []

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": 0,   # Only article pages
        "cmlimit": "max",
        "format": "json"
    }

    while True:
        res = requests.get(URL, params=params, headers=HEADERS)
        data = res.json()

        for item in data["query"]["categorymembers"]:
            title = item["title"]

            skaters.append({
                "name": title,
                "wiki_title": title,
                "wiki_url": f"https://en.wikipedia.org/wiki/{quote(title)}",
                "nationality_category": category,
                "discipline_group": discipline_tag
            })

        if "continue" in data:
            params.update(data["continue"])
            time.sleep(0.5)
        else:
            break

    return skaters


# ===============================
# EXPORT
# ===============================

def save_skater_json(skater):
    discipline_folder = BASE_OUTPUT_DIR / skater["discipline_group"]
    discipline_folder.mkdir(parents=True, exist_ok=True)

    safe_filename = (
        skater["wiki_title"]
        .replace("/", "_")
        .replace(" ", "_")
    )

    filepath = discipline_folder / f"{safe_filename}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(skater, f, indent=2, ensure_ascii=False)


# ===============================
# MAIN EXTRACTION
# ===============================

def extract_selected_disciplines():
    all_skaters = []
    seen = set()

    for tag, root_category in ROOT_CATEGORIES.items():
        print(f"\n=== Processing {tag} ===")

        subcategories = get_subcategories(root_category)
        print("Found", len(subcategories), "nationality categories")

        for subcat in subcategories:
            print("  ->", subcat)

            skaters = get_skaters_from_category(subcat, tag)

            for skater in skaters:
                if skater["name"] not in seen:
                    all_skaters.append(skater)
                    seen.add(skater["name"])

            time.sleep(0.5)

    return all_skaters


if __name__ == "__main__":
    print("Starting extraction...")

    skaters = extract_selected_disciplines()

    print("\nTotal unique skaters:", len(skaters))

    for skater in skaters:
        save_skater_json(skater)

    print("Done. Files exported into discipline folders under /data.")
