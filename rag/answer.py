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
from typing import List, Dict, Optional

from rag.retriever import retrieve
from rag.prompt_builder import build_llm_input
from rag.llm import run_llm
from rag.intents import (
    is_blank,
    is_social_message,
    handle_social_message,
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"

TOP_K = 4
MIN_TOP_SCORE = 0.15


# --------------------------------------------------
# Track Inference (Adult vs Standard)
# --------------------------------------------------
def infer_track(question: str, history: List[Dict]) -> str:
    q = question.lower()

    # Explicit override
    if "not adult" in q or "standard" in q or "non-adult" in q:
        return "standard"

    if "adult" in q:
        return "adult"

    # Look back into recent history (context carry)
    for msg in reversed(history[-6:]):
        if msg["role"] != "user":
            continue

        t = msg["content"].lower()

        if "adult" in t:
            return "adult"

        if "standard" in t or "not adult" in t:
            return "standard"

    # Default behavior:
    # If nothing specified, assume Standard.
    return "standard"


# --------------------------------------------------
# Main Answer Function
# --------------------------------------------------
def answer_question(question: str, history: List[Dict]) -> str:

    # Basic guards
    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    # 1️⃣ Infer track
    track = infer_track(question, history)

    # 2️⃣ Retrieve ONLY from inferred track
    retrieval = retrieve(question, k=TOP_K, track=track)

    if retrieval["top_score"] is None:
        return "Could you clarify your question?"

    if retrieval["top_score"] < MIN_TOP_SCORE:
        return "Could you clarify which level or track you are referring to?"

    # 3️⃣ Load prompt
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    # 4️⃣ Build LLM input
    llm_input = build_llm_input(
        prompt=prompt,
        question=question,
        docs=retrieval["results"],
        history=history,
    )

    # 5️⃣ Generate
    return run_llm(llm_input)
