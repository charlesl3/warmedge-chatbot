import requests
import json
import re
from bs4 import BeautifulSoup

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WarmEdgeRAGBot/1.0 (contact: your_email@example.com)"
}

PAGE_TITLE = "Figure_skating_at_the_2022_Winter_Olympics"


# -------------------------
# Fetch HTML from Wikipedia
# -------------------------
def fetch_page_html(title: str):
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json"
    }

    response = requests.get(
        WIKI_API,
        params=params,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise Exception(data["error"])

    html = data["parse"]["text"]["*"]
    page_title = data["parse"]["title"]

    return page_title, html


# -------------------------
# Clean cell text
# -------------------------
def clean_text(text):
    if not text:
        return ""

    # remove citation markers like [12], [a], etc.
    text = re.sub(r"\[[^\]]*\]", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------
# Extract section name above table
# -------------------------
def find_section_title(table):
    previous = table.find_previous(["h2", "h3"])
    if previous:
        return clean_text(previous.get_text())
    return None


# -------------------------
# Parse tables
# -------------------------
def parse_tables_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # only real data tables
    tables = soup.find_all("table", class_="wikitable")

    structured_tables = []

    for idx, table in enumerate(tables):

        table_data = {
            "index": idx,
            "section": find_section_title(table),
            "caption": None,
            "headers": [],
            "rows": []
        }

        # Caption
        caption = table.find("caption")
        if caption:
            table_data["caption"] = clean_text(caption.get_text())

        # Headers (collect ALL header rows properly)
        header_rows = table.find_all("tr")
        for row in header_rows:
            ths = row.find_all("th")
            if ths:
                headers = [clean_text(th.get_text()) for th in ths]
                table_data["headers"] = headers
                break  # first real header row only

        # Data rows
        for row in table.find_all("tr"):
            tds = row.find_all("td")
            if tds:
                row_data = [clean_text(td.get_text()) for td in tds]
                if any(cell != "" for cell in row_data):
                    table_data["rows"].append(row_data)

        # Skip tiny junk tables
        if len(table_data["rows"]) >= 3:
            structured_tables.append(table_data)

    return structured_tables


# -------------------------
# Main
# -------------------------
def main():
    print("Fetching page...")
    title, html = fetch_page_html(PAGE_TITLE)

    print("Parsing tables...")
    tables = parse_tables_from_html(html)

    output = {
        "title": title,
        "table_count": len(tables),
        "tables": tables
    }

    with open("wiki_clean_tables.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Extracted {len(tables)} clean tables.")
    print("Saved to wiki_clean_tables.json")


if __name__ == "__main__":
    main()