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

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
def retrieve(query: str, k: int = 6):
    """
    Returns:
    {
        "results": [...],
        "scores": [...],
        "indices": [...],
        "top_score": float | None,
        "sources": [...]
    }
    """

    q = (query or "").strip()

    if len(q) < 2:
        print("Query too short. Skipping retrieval:", repr(q))
        return {
            "results": [],
            "scores": [],
            "indices": [],
            "top_score": None,
            "sources": [],
        }

    index, documents = load_index_and_meta()

    print("\n================ RETRIEVAL DEBUG ================")
    print("Query:", query)
    print("k:", k)

    model = get_embed_model()

    # --------------------------------------------------
    # Embed query
    # --------------------------------------------------
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb)

    # --------------------------------------------------
    # Search deeper pool for better rule recall
    # --------------------------------------------------
    search_k = k * 3

    scores, indices = index.search(
        np.expand_dims(q_emb, axis=0),
        search_k
    )

    idx_list = indices[0].tolist()
    score_list = scores[0].tolist()

    rules_results = []
    thread_results = []

    for idx, score in zip(idx_list, score_list):
        if idx < 0:
            continue

        doc = documents[int(idx)]
        text = doc.get("text", "").strip()
        source_path = doc.get("source_path", "")

        if not text:
            continue

        item = {
            "text": text,
            "source_path": source_path,
            "score": score
        }

        if "rules_rag_units" in source_path:
            rules_results.append(item)
        else:
            thread_results.append(item)

    # --------------------------------------------------
    # PRIORITY MERGE: rules first, then threads
    # --------------------------------------------------
    merged = rules_results + thread_results
    merged = merged[:k]

    results = [
        {"text": r["text"], "source_path": r["source_path"]}
        for r in merged
    ]

    sources = [r["source_path"] for r in merged]
    final_scores = [r["score"] for r in merged]
    final_indices = [
        idx_list[i]
        for i in range(len(merged))
    ]

    top_score = final_scores[0] if final_scores else None

    # --------------------------------------------------
    # Debug output
    # --------------------------------------------------
    print("Top score:", top_score)
    print("Retrieved source paths (after prioritization):")
    for i, src in enumerate(sources):
        print(f"{i+1}. {src}")

    print("==================================================\n")

    return {
        "results": results,
        "scores": final_scores,
        "indices": final_indices,
        "top_score": top_score,
        "sources": sources,
    }
