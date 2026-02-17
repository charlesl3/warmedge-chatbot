# # whole faiss, no chunk
# import json
# from pathlib import Path
#
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
#
# # -------------------------
# # PATHS
# # -------------------------
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATA_DIR = PROJECT_ROOT / "data"
#
# PASS2_EQUIP_DIR = DATA_DIR / "pass2_threads_equipment"
# PASS2_GENERAL_DIR = DATA_DIR / "pass2_threads_general"
# RULES_DIR = DATA_DIR / "rules_rag_units"
# MANUAL_RAG_DIR = DATA_DIR / "manual_rag"
#
# STORE_DIR = PROJECT_ROOT / "rag_store"
# STORE_DIR.mkdir(parents=True, exist_ok=True)
#
# INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
# META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"
#
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_CHARS_PER_DOC = 9000
#
#
# # -------------------------
# # Utils
# # -------------------------
# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     return v / (np.linalg.norm(v) + 1e-12)
#
#
# def collect_md_files():
#     folders = [
#         MANUAL_RAG_DIR,      # manual priority
#         RULES_DIR,
#         PASS2_EQUIP_DIR,
#         PASS2_GENERAL_DIR,
#     ]
#
#     files = []
#
#     for folder in folders:
#         if not folder.exists():
#             print(f"Warning: Missing folder {folder}")
#             continue
#
#         files.extend(sorted(folder.rglob("*.md")))
#
#     return files
#
#
# # -------------------------
# # Main
# # -------------------------
# def main():
#     md_files = collect_md_files()
#
#     if not md_files:
#         raise RuntimeError("No markdown files found.")
#
#     print(f"Found {len(md_files)} markdown files.")
#
#     model = SentenceTransformer(EMBED_MODEL_NAME)
#
#     documents = []
#     embeddings = []
#
#     for md_path in md_files:
#         text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
#
#         if not text:
#             continue
#
#         if len(text) > MAX_CHARS_PER_DOC:
#             text = text[:MAX_CHARS_PER_DOC] + "\n\n[TRUNCATED]\n"
#
#         rel_path = str(md_path.relative_to(PROJECT_ROOT))
#
#         documents.append({
#             "text": text,
#             "source_path": rel_path
#         })
#
#         emb = model.encode(text, convert_to_numpy=True).astype("float32")
#         emb = l2_normalize(emb)
#
#         embeddings.append(emb)
#
#     if not embeddings:
#         raise RuntimeError("No embeddings created.")
#
#     X = np.vstack(embeddings).astype("float32")
#
#     # Cosine similarity via inner product on normalized vectors
#     index = faiss.IndexFlatIP(X.shape[1])
#     index.add(X)
#
#     faiss.write_index(index, str(INDEX_PATH))
#
#     meta = {
#         "embed_model_name": EMBED_MODEL_NAME,
#         "documents": documents
#     }
#
#     META_PATH.write_text(
#         json.dumps(meta, ensure_ascii=False, indent=2),
#         encoding="utf-8"
#     )
#
#     print("Index written to:", INDEX_PATH)
#     print("Meta written to:", META_PATH)
#     print("Index size:", index.ntotal)
#     print("Meta documents:", len(documents))
#
#
# if __name__ == "__main__":
#     main()

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------
# PATHS
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

PASS2_EQUIP_DIR = DATA_DIR / "pass2_threads_equipment"
PASS2_GENERAL_DIR = DATA_DIR / "pass2_threads_general"
RULES_DIR = DATA_DIR / "rules_rag_units"
MANUAL_RAG_DIR = DATA_DIR / "manual_rag"

STORE_DIR = PROJECT_ROOT / "rag_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------
# Utils
# -------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def collect_md_files():
    folders = [
        MANUAL_RAG_DIR,      # manual priority
        RULES_DIR,
        PASS2_EQUIP_DIR,
        PASS2_GENERAL_DIR,
    ]

    files = []

    for folder in folders:
        if not folder.exists():
            print(f"Warning: Missing folder {folder}")
            continue

        files.extend(sorted(folder.rglob("*.md")))

    return files


def chunk_markdown(text: str, max_chars: int = 800):
    """
    Simple paragraph-based chunking.
    Groups paragraphs into chunks up to max_chars.
    """
    chunks = []
    paragraphs = text.split("\n\n")

    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buffer) + len(para) + 2 <= max_chars:
            buffer += para + "\n\n"
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = para + "\n\n"

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


# -------------------------
# Main
# -------------------------
def main():
    md_files = collect_md_files()

    if not md_files:
        raise RuntimeError("No markdown files found.")

    print(f"Found {len(md_files)} markdown files.")

    model = SentenceTransformer(EMBED_MODEL_NAME)

    documents = []
    embeddings = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            continue

        rel_path = str(md_path.relative_to(PROJECT_ROOT))

        chunks = chunk_markdown(text)

        for chunk in chunks:
            documents.append({
                "text": chunk,
                "source_path": rel_path
            })

            emb = model.encode(chunk, convert_to_numpy=True).astype("float32")
            emb = l2_normalize(emb)

            embeddings.append(emb)

    if not embeddings:
        raise RuntimeError("No embeddings created.")

    X = np.vstack(embeddings).astype("float32")

    # Cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "embed_model_name": EMBED_MODEL_NAME,
        "documents": documents
    }

    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Index written to:", INDEX_PATH)
    print("Meta written to:", META_PATH)
    print("Index size:", index.ntotal)
    print("Meta documents:", len(documents))


if __name__ == "__main__":
    main()
