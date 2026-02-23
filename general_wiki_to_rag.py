import json
from pathlib import Path
import re

# ===============================
# PATHS
# ===============================

INPUT_DIR = Path("data/general_wiki_info")
OUTPUT_DIR = Path("data/general_wiki_rag")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# HELPERS
# ===============================

def extract_title_and_text(wiki_json):
    pages = wiki_json.get("query", {}).get("pages", {})
    if not pages:
        return None, None

    page_data = next(iter(pages.values()))

    title = page_data.get("title")
    extract = page_data.get("extract", "")

    return title, extract


def clean_extract(text):
    if not text:
        return ""

    # Stop at common trailing sections
    stop_markers = [
        "== References ==",
        "== External links ==",
        "== Bibliography ==",
        "== Further reading ==",
        "== See also ==",
        "== Notes ==",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    # Normalize spacing
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def sanitize_filename(title):
    return (
        title.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
    )


# ===============================
# MAIN
# ===============================

def generate_md_units():
    for json_file in INPUT_DIR.glob("*.json"):

        with open(json_file, "r", encoding="utf-8") as f:
            wiki_json = json.load(f)

        title, extract = extract_title_and_text(wiki_json)

        if not title or not extract:
            print("Skipping:", json_file.name)
            continue

        clean_text = clean_extract(extract)

        safe_filename = sanitize_filename(title)
        output_path = OUTPUT_DIR / f"{safe_filename}.md"

        md_content = (
            f"# {title}\n\n"
            f"Source: Wikipedia – General Figure Skating Category\n\n"
            f"{clean_text}"
        )

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(md_content)

        print("Created:", output_path.name)


if __name__ == "__main__":
    print("Generating General Wiki Markdown RAG units...")
    generate_md_units()
    print("Done.")