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
# Legacy Term Normalization
# --------------------------------------------------
def normalize_legacy_terms(question: str) -> str:
    q = question.lower()

    replacements = {
        # Discipline normalization
        "free skate": "singles",
        "free skating": "singles",
        "moves in the field": "skating skills",
        "mitf": "skating skills",
        "mift": "skating skills",

        # Level normalization (hyphen fixes)
        "pre silver": "pre-silver",
        "pre gold": "pre-gold",
        "pre bronze": "pre-bronze",
        "pre juvenile": "pre-juvenile",
        "pre preliminary": "pre-preliminary",
        "presilver": "pre-silver",
        "pregold": "pre-gold",
        "prebronze": "pre-bronze",
        "prejuvenile": "pre-juvenile",
        "preliminary free skate": "preliminary singles",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    return q



# --------------------------------------------------
# STRICT Track Inference (NO history, NO guessing)
# --------------------------------------------------
def infer_track(question: str) -> str:
    q = question.lower()

    # Explicit override to standard
    if "not adult" in q or "standard" in q or "non-adult" in q:
        return "standard"

    # Explicit adult only if word appears
    if "adult" in q:
        return "adult"

    # 🔥 CRITICAL RULE:
    # If the word "adult" does NOT appear,
    # ALWAYS assume Standard.
    return "standard"


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

    # 2️⃣ STRICT track inference
    track = infer_track(normalized_question)

    # 3️⃣ Retrieve ONLY from that track
    retrieval = retrieve(
        normalized_question,
        k=TOP_K,
        track=track
    )

    if retrieval["top_score"] is None:
        return "Could you clarify your question?"

    if retrieval["top_score"] < MIN_TOP_SCORE:
        return "Could you clarify which level you are referring to?"

    # 4️⃣ Load base prompt
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    # 5️⃣ Build LLM input
    llm_input = build_llm_input(
        prompt=prompt,
        question=normalized_question,
        docs=retrieval["results"],
        history=history,
    )

    # 6️⃣ Generate response
    return run_llm(llm_input)
