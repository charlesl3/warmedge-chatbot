import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

PASS2_DIRS = [
    DATA_DIR / "pass2_threads_equipment",
    DATA_DIR / "pass2_threads_general",
]

STORE_DIR = PROJECT_ROOT / "rag_store"
STORE_DIR.mkdir(exist_ok=True)

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Utils
# --------------------------------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------
# Main build logic
# --------------------------------------------------
def main():
    texts = []
    paths = []

    # 1. Read all pass2 markdown files
    for folder in PASS2_DIRS:
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for md_path in sorted(folder.glob("*.md")):
            text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            texts.append(text)
            paths.append(str(md_path.resolve()))

    if not texts:
        raise RuntimeError("No markdown files found for indexing")

    print(f"Loaded {len(texts)} markdown documents")

    # 2. Embed
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    embeddings = np.vstack([l2_normalize(e) for e in embeddings])

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 4. Save index + metadata
    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "paths": paths,
    }
    META_PATH.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(f"FAISS index saved to: {INDEX_PATH}")
    print(f"Metadata saved to:   {META_PATH}")
    print("Build complete.")


if __name__ == "__main__":
    main()
