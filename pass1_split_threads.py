import json
import re
from pathlib import Path


# ------------------------
# Paths
# ------------------------

INPUT_PATH = Path("data/raw/goldenskate_boots_blades.json")
OUTPUT_DIR = Path("data/pass1_threads")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------
# Mechanical cleanup only
# ------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove forum quote artifacts
    text = re.sub(r"Originally posted by.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[quote.*?\].*?\[/quote\]", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove click artifacts
    text = re.sub(r"Click to expand\.\.\.", "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def safe_filename(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ------------------------
# Load input
# ------------------------

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

threads = data["threads_with_posts"]

print(f"Found {len(threads)} threads_with_posts")


# ------------------------
# Process each thread
# ------------------------

for idx, thread in enumerate(threads):
    thread_title = thread["thread_title"]
    thread_url = thread["thread_url"]

    posts = thread["posts"]
    if not posts:
        continue

    # Original author = author of first post
    op_author = posts[0].get("author")

    cleaned_posts = []

    for post in posts:
        cleaned_posts.append({
            "is_op": post.get("author") == op_author,
            "created_at": post.get("datetime_iso"),
            "text": clean_text(post.get("content_text", ""))
        })

    pass1_thread = {
        "thread_title": thread_title,
        "thread_url": thread_url,
        "posts": cleaned_posts
    }

    filename = f"{idx:04d}_{safe_filename(thread_title)}.json"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(pass1_thread, out, ensure_ascii=False, indent=2)


print(f"PASS 1 complete → {OUTPUT_DIR}")
