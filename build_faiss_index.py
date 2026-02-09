import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------
# PATH SETUP
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent  # put scripts in GoldenSkate root
PASS2_MD_DIR = PROJECT_ROOT / "data" / "pass2_threads_md"
STORE_DIR = PROJECT_ROOT / "rag_store"

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Indexing controls
BATCH_SIZE = 64
MAX_CHARS_PER_DOC_FOR_EMBED = 8000  # embed only first N chars (keeps things fast/stable)


# -------------------------
# HELPERS
# -------------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def list_md_files(md_dir: Path) -> list[Path]:
    files = sorted(md_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {md_dir}")
    return files


def read_for_embedding(p: Path) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()
    if len(txt) > MAX_CHARS_PER_DOC_FOR_EMBED:
        txt = txt[:MAX_CHARS_PER_DOC_FOR_EMBED]
    return txt


# -------------------------
# MAIN
# -------------------------
def main():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    md_files = list_md_files(PASS2_MD_DIR)
    ids = [p.stem for p in md_files]  # e.g. "0029_skate_upgrade"
    paths = [str(p) for p in md_files]

    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    # Cosine similarity via Inner Product on L2-normalized vectors
    index = faiss.IndexFlatIP(dim)

    print(f"Found {len(md_files)} markdown files.")
    print(f"Embedding model: {EMBED_MODEL_NAME} (dim={dim})")
    print("Building embeddings and FAISS index...")

    all_vecs = []
    for start in range(0, len(md_files), BATCH_SIZE):
        batch = md_files[start : start + BATCH_SIZE]
        texts = [read_for_embedding(p) for p in batch]

        vecs = model.encode(texts, convert_to_numpy=True).astype("float32")
        vecs = l2_normalize(vecs).astype("float32")
        all_vecs.append(vecs)

        print(f"Embedded {min(start + BATCH_SIZE, len(md_files))}/{len(md_files)}")

    mat = np.vstack(all_vecs).astype("float32")
    index.add(mat)

    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "embed_model": EMBED_MODEL_NAME,
        "index_type": "IndexFlatIP",
        "ids": ids,
        "paths": paths,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved index to: {INDEX_PATH}")
    print(f"Saved meta to:  {META_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
