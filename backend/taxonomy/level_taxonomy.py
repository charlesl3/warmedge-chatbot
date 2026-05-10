import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = BASE_DIR / "usfs_levels_2023.json"

with open(TAXONOMY_PATH, "r") as f:
    LEVEL_TAXONOMY = json.load(f)


def detect_legacy_term(query: str):
    query_lower = query.lower()

    result = {
        "legacy_track": None,
        "mapped_track": None,
        "legacy_level": None,
        "mapped_level": None,
        "is_adult": "adult" in query_lower
    }

    # ---- Track rename detection (universal) ----
    for old_track, new_track in LEVEL_TAXONOMY.get("track_renaming", {}).items():
        if old_track in query_lower:
            result["legacy_track"] = old_track
            result["mapped_track"] = new_track

    # ---- Legacy level rename detection (standard only) ----
    for old_level, new_level in LEVEL_TAXONOMY.get(
        "legacy_level_renaming_standard", {}
    ).items():
        if old_level in query_lower:
            result["legacy_level"] = old_level
            result["mapped_level"] = new_level

    # If nothing meaningful detected, return None
    if (
        result["legacy_track"] is None and
        result["legacy_level"] is None and
        not result["is_adult"]
    ):
        return None

    return result
