import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = PROJECT_ROOT / "rag_store"

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS_PER_DOC_IN_CONTEXT = 9000


# --------------------------------------------------
# Utils
# --------------------------------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------
# Load index + metadata
# --------------------------------------------------
def load_index_and_meta():
    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Run build_faiss_index.py first.")
    if not META_PATH.exists():
        raise FileNotFoundError("Meta file not found. Run build_faiss_index.py first.")

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    paths = meta.get("paths")
    if paths is None:
        raise ValueError("Meta file missing 'paths' field")

    if index.ntotal != len(paths):
        raise ValueError("FAISS index size and meta paths length mismatch")

    return index, paths


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def retrieve(query: str, k: int = 6):
    index, paths = load_index_and_meta()

    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Embed query ONLY
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb)

    # Search
    scores, indices = index.search(np.expand_dims(q_emb, axis=0), k)

    results = []

    for idx in indices[0]:
        if idx < 0:
            continue

        md_path = Path(paths[idx])
        if not md_path.exists():
            continue

        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        if len(text) > MAX_CHARS_PER_DOC_IN_CONTEXT:
            text = text[:MAX_CHARS_PER_DOC_IN_CONTEXT] + "\n\n[TRUNCATED]\n"

        results.append(
            {
                "text": text,
                "source_path": str(md_path),
            }
        )

    return results
