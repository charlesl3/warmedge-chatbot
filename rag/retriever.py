# import json
# from pathlib import Path
#
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
#
# # --------------------------------------------------
# # Paths
# # --------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# STORE_DIR = PROJECT_ROOT / "rag_store"
#
# INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
# META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"
#
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_CHARS_PER_DOC_IN_CONTEXT = 9000
#
#
# # --------------------------------------------------
# # Utils
# # --------------------------------------------------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     return v / (np.linalg.norm(v) + 1e-12)
#
#
# # --------------------------------------------------
# # Load index + metadata
# # --------------------------------------------------
# def load_index_and_meta():
#     if not INDEX_PATH.exists():
#         raise FileNotFoundError("FAISS index not found. Run build_faiss_index.py first.")
#     if not META_PATH.exists():
#         raise FileNotFoundError("Meta file not found. Run build_faiss_index.py first.")
#
#     index = faiss.read_index(str(INDEX_PATH))
#     meta = json.loads(META_PATH.read_text(encoding="utf-8"))
#
#     paths = meta.get("paths")
#     if paths is None:
#         raise ValueError("Meta file missing 'paths' field")
#
#     if index.ntotal != len(paths):
#         raise ValueError("FAISS index size and meta paths length mismatch")
#
#     return index, paths
#
#
# # --------------------------------------------------
# # Retrieval
# # --------------------------------------------------
# def retrieve(query: str, k: int = 6):
#     index, paths = load_index_and_meta()
#
#     model = SentenceTransformer(EMBED_MODEL_NAME)
#
#     # Embed query ONLY
#     q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
#     q_emb = l2_normalize(q_emb)
#
#     # Search
#     scores, indices = index.search(np.expand_dims(q_emb, axis=0), k)
#
#     results = []
#
#     for idx in indices[0]:
#         if idx < 0:
#             continue
#
#         md_path = Path(paths[idx])
#         if not md_path.exists():
#             continue
#
#         text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
#         if not text:
#             continue
#
#         if len(text) > MAX_CHARS_PER_DOC_IN_CONTEXT:
#             text = text[:MAX_CHARS_PER_DOC_IN_CONTEXT] + "\n\n[TRUNCATED]\n"
#
#         results.append(
#             {
#                 "text": text,
#                 "source_path": str(md_path),
#             }
#         )
#
#     return results


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

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"


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
        raise FileNotFoundError(
            "FAISS index not found. Run build_faiss_index.py first."
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            "Meta file not found. Run build_faiss_index.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    print("Loaded FAISS index size:", index.ntotal)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    documents = meta.get("documents")

    if documents is None:
        raise ValueError("Meta file missing 'documents'. Rebuild index.")

    print("Meta documents count:", len(documents))

    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index size and meta documents mismatch."
        )

    _CACHED_INDEX = index
    _CACHED_DOCS = documents

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
# Retrieval
# --------------------------------------------------
def retrieve(query: str, k: int = 6, track: str | None = None):
    """
    track:
        None        -> no filtering
        "adult"     -> only adult rule units
        "standard"  -> only standard rule units
    """

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

    scores, indices = index.search(
        np.expand_dims(q_emb, axis=0),
        k * 4  # search wider pool for filtering
    )

    idx_list = indices[0].tolist()
    score_list = scores[0].tolist()

    results = []
    sources = []
    filtered_scores = []

    for idx, score in zip(idx_list, score_list):
        if idx < 0:
            continue

        doc = documents[int(idx)]
        source_path = doc.get("source_path", "")
        text = doc.get("text", "").strip()

        if not text:
            continue

        # -------- TRACK FILTERING --------
        if track == "adult":
            if "_adult" not in source_path:
                continue

        elif track == "standard":
            if "_standard" not in source_path:
                continue

        results.append({
            "text": text,
            "source_path": source_path
        })

        sources.append(source_path)
        filtered_scores.append(score)

        if len(results) >= k:
            break

    top_score = filtered_scores[0] if filtered_scores else None

    return {
        "results": results,
        "scores": filtered_scores,
        "indices": idx_list,
        "top_score": top_score,
        "sources": sources,
    }

