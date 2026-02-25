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

INDEX_PATH = STORE_DIR / "warmedge_master_index.faiss"
META_PATH = STORE_DIR / "warmedge_master_meta.json"

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Load once globally
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)


# --------------------------------------------------
# In-process caches
# --------------------------------------------------
_CACHED_INDEX = None
_CACHED_DOCS = None
_CACHED_MODEL = None


# --------------------------------------------------
# Utils
# --------------------------------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


# --------------------------------------------------
# Load index + metadata (cached)
# --------------------------------------------------
def load_index_and_meta():
    global _CACHED_INDEX, _CACHED_DOCS

    if _CACHED_INDEX is not None and _CACHED_DOCS is not None:
        return _CACHED_INDEX, _CACHED_DOCS

    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Run build script first.")

    if not META_PATH.exists():
        raise FileNotFoundError("Meta file not found. Run build script first.")

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    documents = meta.get("documents")
    if documents is None:
        raise ValueError("Meta file missing 'documents'. Rebuild index.")

    if index.ntotal != len(documents):
        raise ValueError("Index size and meta documents mismatch.")

    _CACHED_INDEX = index
    _CACHED_DOCS = documents

    print("Loaded index size:", index.ntotal)

    return index, documents


# --------------------------------------------------
# Load embedding model (cached)
# --------------------------------------------------
def get_embed_model():
    global _CACHED_MODEL

    if _CACHED_MODEL is None:
        print("Loading embedding model:", EMBED_MODEL_NAME)
        _CACHED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

    return _CACHED_MODEL


# --------------------------------------------------
# Track Filtering Logic
# --------------------------------------------------
def track_filter(doc_meta: dict, track: str | None) -> bool:
    """
    Returns True if document should be kept.
    """

    if track is None:
        return True

    # Only filter rules documents by track
    if doc_meta.get("doc_type") != "rules":
        return True

    title = doc_meta.get("title", "").lower()

    if track == "adult":
        return "adult" in title

    if track == "standard":
        return "adult" not in title

    return True


# --------------------------------------------------
# Retrieval
# --------------------------------------------------
def retrieve(query: str, k: int = 6, track: str | None = None):

    q = (query or "").strip()
    if len(q) < 2:
        return {
            "results": [],
            "scores": [],
            "indices": [],
            "top_score": None,
            "sources": [],
        }

    index, documents = load_index_and_meta()
    model = get_embed_model()

    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb)

    # Search wider pool for filtering
    scores, indices = index.search(
        np.expand_dims(q_emb, axis=0),
        k * 5
    )

    idx_list = indices[0].tolist()
    score_list = scores[0].tolist()

    results = []
    sources = []
    filtered_scores = []

    for idx, score in zip(idx_list, score_list):

        if idx < 0:
            continue

        doc_meta = documents[int(idx)]
        text = doc_meta.get("text", "").strip()

        if not text:
            continue

        # Track filtering
        if not track_filter(doc_meta, track):
            continue

        results.append(text)
        sources.append(doc_meta)
        filtered_scores.append(float(score))

        if len(results) >= k:
            break

    top_score = filtered_scores[0] if filtered_scores else None

    return {
        "results": results,
        "scores": filtered_scores,
        "indices": idx_list,
        "top_score": top_score,
        "sources": sources,  # now full metadata
    }