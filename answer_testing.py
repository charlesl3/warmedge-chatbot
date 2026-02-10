import json
import subprocess
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------
# PATH SETUP
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
STORE_DIR = PROJECT_ROOT / "rag_store"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "rag_answer.txt"

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

# Must match build_faiss_index.py
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (Ollama)
OLLAMA_MODEL = "llama3.1:8b"

# Retrieval
TOP_K = 6
MAX_CHARS_PER_DOC_IN_CONTEXT = 9000

# Edit this and rerun (PyCharm workflow)
QUESTION = "is figure skating more tough than hockey"


# -------------------------
# HELPERS
# -------------------------
def is_blank_question(q: str) -> bool:
    return q is None or len(q.strip()) == 0


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_index_and_meta():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing index: {INDEX_PATH}. Run build_faiss_index.py first."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing meta: {META_PATH}. Run build_faiss_index.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    paths = meta.get("paths")
    if paths is None:
        raise ValueError("Meta file must contain 'paths'")

    if index.ntotal != len(paths):
        raise ValueError("FAISS index and meta paths length mismatch")

    return index, paths


def retrieve(query: str, k: int = TOP_K):
    index, paths = load_index_and_meta()

    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Embed QUERY ONLY
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb)

    scores, indices = index.search(np.expand_dims(q_emb, axis=0), k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        p = Path(paths[idx])
        if not p.exists():
            continue

        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        if len(text) > MAX_CHARS_PER_DOC_IN_CONTEXT:
            text = text[:MAX_CHARS_PER_DOC_IN_CONTEXT] + "\n\n[TRUNCATED]\n"

        results.append(
            {
                "path": str(p),
                "score": float(score),
                "text": text,
            }
        )

    return results


def build_llm_input(prompt: str, question: str, docs: list[dict]) -> str:
    parts = []

    parts.append(prompt.strip())
    parts.append("")

    parts.append("User question:")
    parts.append(question.strip())
    parts.append("")

    parts.append("GoldenSkate knowledge units:")
    parts.append("")

    for d in docs:
        parts.append("---")
        parts.append(d["text"])
        parts.append("")

    parts.append(
        "Write a natural, helpful answer grounded in the information above.\n"
        "If there is not enough information, say so naturally and ask 1–2 follow-up questions, "
        "starting with: `I do not have enough information from my knowledge set, could you provide more information?`\n"
        "If you relied on any original discussion URLs (thread_url), include a short section at the end starting with:\n"
        "`You may find these information useful:`\n"
        "List ONLY the original thread URLs.\n"
        "Do NOT include filenames, scores, or internal references.\n"
        "If no such URLs exist, do not include this section."
    )

    return "\n".join(parts)


def run_ollama(text: str) -> str:
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


# -------------------------
# MAIN
# -------------------------
def main():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing prompt file: {PROMPT_PATH}\n"
            "Create prompts/rag_answer.txt."
        )

    if is_blank_question(QUESTION):
        print("Answer:\n")
        print("Please ask a valid figure skating related question.")
        return

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    docs = retrieve(QUESTION, k=TOP_K)

    print("Retrieved:")
    for d in docs:
        print(f"- score={d['score']:.3f}  ({d['path']})")
    print("")

    llm_input = build_llm_input(prompt, QUESTION, docs)
    answer = run_ollama(llm_input)

    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
