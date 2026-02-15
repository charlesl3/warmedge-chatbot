import pdfplumber
import re
from pathlib import Path


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data/rules/singles/singles_test.pdf"
OUTPUT_PATH = Path("data/singles_rag_units/singles_test_2026.md")

DISCIPLINE = "Singles"
CONTEXT = "Test"
SEASON = "2026"


# --------------------------------------------------
# Extract text from PDF
# --------------------------------------------------

def extract_text(pdf_path: Path) -> str:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    return "\n".join(pages)


# --------------------------------------------------
# Split by Level Headers
# --------------------------------------------------

def split_levels(text: str):
    """
    Split by level headings.
    We detect lines that represent test levels.
    """

    level_pattern = re.compile(
        r"^(Standard.*?|Adult.*?|Adaptive.*?|Pre-Preliminary|Preliminary|Pre-Bronze|Bronze|Pre-Silver|Silver|Pre-Gold|Gold).*?$",
        re.MULTILINE
    )

    matches = list(level_pattern.finditer(text))
    sections = []

    for i, match in enumerate(matches):
        level_name = match.group().strip()
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()
        sections.append((level_name, content))

    return sections


# --------------------------------------------------
# Write Markdown
# --------------------------------------------------

def write_markdown(sections, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        # Metadata
        f.write(f"# Discipline: {DISCIPLINE}\n")
        f.write(f"# Context: {CONTEXT}\n")
        f.write(f"# Season: {SEASON}\n\n")
        f.write("---\n\n")

        for level_name, content in sections:
            f.write(f"# Level: {level_name}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Extracting text...")
    text = extract_text(PDF_PATH)

    print("Splitting into levels...")
    sections = split_levels(text)

    print(f"Detected {len(sections)} level sections.")

    print("Writing markdown...")
    write_markdown(sections, OUTPUT_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
