import json
from pathlib import Path

import os
import logging
import warnings

# Silence Hugging Face Hub warnings (best-effort)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*",
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASS2_MD_DIR = PROJECT_ROOT / "data" / "pass2_threads_md"
STORE_DIR = PROJECT_ROOT / "rag_store"

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS_PER_DOC_IN_CONTEXT = 9000


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_index_and_meta():
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta


def retrieve(query: str, k: int = 6):
    index, meta = load_index_and_meta()
    ids = meta["ids"]
    paths = meta.get("paths")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb).astype("float32")

    D, I = index.search(np.expand_dims(q_emb, axis=0), k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue

        doc_id = ids[idx]
        p = Path(paths[idx]) if paths else PASS2_MD_DIR / f"{doc_id}.md"

        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > MAX_CHARS_PER_DOC_IN_CONTEXT:
            text = text[:MAX_CHARS_PER_DOC_IN_CONTEXT] + "\n\n[TRUNCATED]\n"

        results.append({"text": text})

    return results
