import json
import subprocess
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
PROMPT_PATH = PROJECT_ROOT / "prompts" / "rag_answer.txt"

INDEX_PATH = STORE_DIR / "goldenskate_pass2.faiss"
META_PATH = STORE_DIR / "goldenskate_pass2_meta.json"

# Must match what you used to build the index
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM (Ollama)
OLLAMA_MODEL = "llama3.1:8b"

# Retrieval
TOP_K = 6
MAX_CHARS_PER_DOC_IN_CONTEXT = 9000  # control prompt size

# Edit this and rerun (PyCharm workflow)
QUESTION = "is figure skating more tough then hockey"


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
    return index, meta


def retrieve(query: str, k: int = TOP_K):
    index, meta = load_index_and_meta()
    ids = meta["ids"]
    paths = meta.get("paths")  # optional but recommended

    model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")[0]
    q_emb = l2_normalize(q_emb).astype("float32")

    D, I = index.search(np.expand_dims(q_emb, axis=0), k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue

        doc_id = ids[idx]
        if paths:
            p = Path(paths[idx])
        else:
            p = PASS2_MD_DIR / f"{doc_id}.md"

        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) > MAX_CHARS_PER_DOC_IN_CONTEXT:
            text = text[:MAX_CHARS_PER_DOC_IN_CONTEXT] + "\n\n[TRUNCATED]\n"

        results.append(
            {
                "id": doc_id,
                "path": str(p),
                "score": float(score),
                "text": text,
            }
        )

    return results


def build_llm_input(prompt: str, question: str, docs: list[dict]) -> str:
    parts = []

    # System-style root prompt
    parts.append(prompt.strip())
    parts.append("")

    # User question
    parts.append("User question:")
    parts.append(question.strip())
    parts.append("")

    # Retrieved context
    parts.append("GoldenSkate knowledge units:")
    parts.append("")

    for d in docs:
        parts.append(f"--- {d['id']} ---")
        parts.append(d["text"])
        parts.append("")

    # Final instruction block
    parts.append(
        "Write a natural, helpful answer grounded in the information above.\n"
        "If there is not enough information, say so naturally and ask 1–2 follow-up questions, "
        "starting with: `I do not have enough information from my knowledge set, could you provide more information?`\n"
        "If you relied on any original discussion URLs (thread_url), include a short section at the end starting with:\n"
        "`You may find these information useful:`\n"
        "List ONLY the original thread URLs.\n"
        "Do NOT include unit IDs, filenames, scores, or internal references.\n"
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
# MAIN (PyCharm-friendly)
# -------------------------
def main():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Missing prompt file: {PROMPT_PATH}\n"
            "Create prompts/rag_answer.txt (a system-style instruction prompt)."
        )

    # 🔒 HARD GUARD: blank question
    if is_blank_question(QUESTION):
        print("Answer:\n")
        print("Please ask a valid figure skating related question.")
        return

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    docs = retrieve(QUESTION, k=TOP_K)

    # Optional debug print
    print("Retrieved:")
    for d in docs:
        print(f"- {d['id']} score={d['score']:.3f}  ({d['path']})")
    print("")

    llm_input = build_llm_input(prompt, QUESTION, docs)
    answer = run_ollama(llm_input)

    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
