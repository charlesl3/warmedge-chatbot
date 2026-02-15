# from pathlib import Path
#
# from rag.retriever import retrieve
# from rag.prompt_builder import build_llm_input
# from rag.llm import run_ollama
# from rag.intents import (
#     is_blank,
#     is_social_message,
#     handle_social_message,
# )
#
# PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"
# TOP_K = 6
#
#
# def answer_question(question: str, history: list[dict]) -> str:
#     if is_blank(question):
#         return "Please ask a valid figure skating related question."
#
#     if is_social_message(question):
#         return handle_social_message(question)
#
#     prompt = PROMPT_PATH.read_text(encoding="utf-8")
#     docs = retrieve(question, k=TOP_K)
#
#     llm_input = build_llm_input(
#         prompt=prompt,
#         question=question,
#         docs=docs,
#         history=history,
#     )
#
#     return run_ollama(llm_input)


from pathlib import Path
from typing import List, Dict

from rag.retriever import retrieve
from rag.prompt_builder import build_llm_input
from rag.llm import run_llm
from rag.intents import (
    is_blank,
    is_social_message,
    handle_social_message,
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"

TOP_K = 6
MIN_TOP_SCORE = 0.15


# --------------------------------------------------
# Legacy Term Normalization (CRITICAL FIX)
# --------------------------------------------------
def normalize_legacy_terms(question: str) -> str:
    q = question.lower()

    replacements = {
        "free skate": "singles",
        "free skating": "singles",
        "moves in the field": "skating skills",
        "mift": "skating skills",
        "mitf": "skating skills",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    return q


# --------------------------------------------------
# Track Inference (Adult vs Standard)
# --------------------------------------------------
def infer_track(question: str, history: List[Dict]) -> str | None:
    q = question.lower()

    if "not adult" in q or "standard" in q or "non-adult" in q:
        return "standard"

    if "adult" in q:
        return "adult"

    # Look back into recent history
    for msg in reversed(history[-6:]):
        if msg["role"] != "user":
            continue

        t = msg["content"].lower()

        if "adult" in t:
            return "adult"

        if "standard" in t or "not adult" in t:
            return "standard"

    return None  # ← important: do NOT default blindly


# --------------------------------------------------
# Main Answer Function
# --------------------------------------------------
def answer_question(question: str, history: List[Dict]) -> str:

    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    # 1️⃣ Normalize legacy naming FIRST
    normalized_question = normalize_legacy_terms(question)

    # 2️⃣ Infer track
    track = infer_track(normalized_question, history)

    # 3️⃣ Retrieval logic

    if track == "adult":
        retrieval = retrieve(normalized_question, k=TOP_K, track="adult")

    elif track == "standard":
        retrieval = retrieve(normalized_question, k=TOP_K, track="standard")

    else:
        # If unspecified → search both tracks
        r1 = retrieve(normalized_question, k=TOP_K, track="adult")
        r2 = retrieve(normalized_question, k=TOP_K, track="standard")

        docs = []
        seen = set()

        for d in r1["results"] + r2["results"]:
            key = d["source_path"]
            if key in seen:
                continue
            seen.add(key)
            docs.append(d)

        retrieval = {
            "results": docs[:TOP_K],
            "top_score": max(
                r1["top_score"] or 0,
                r2["top_score"] or 0,
            ),
        }

    if retrieval["top_score"] is None:
        return "Could you clarify your question?"

    if retrieval["top_score"] < MIN_TOP_SCORE:
        return "Do you mean the Adult track (21+) or the Standard track?"

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    llm_input = build_llm_input(
        prompt=prompt,
        question=normalized_question,
        docs=retrieval["results"],
        history=history,
    )

    return run_llm(llm_input)
