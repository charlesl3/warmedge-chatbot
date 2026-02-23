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

TOP_K = 6
MIN_TOP_SCORE = 0.1


# --------------------------------------------------
# Legacy Term Normalization
# --------------------------------------------------
def normalize_legacy_terms(question: str) -> str:
    q = question.lower()

    replacements = {
        "free skate": "singles",
        "free skating": "singles",
        "moves in the field": "skating skills",
        "mitf": "skating skills",
        "mift": "skating skills",
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
# Track Inference
# --------------------------------------------------
def infer_track(question: str) -> str:
    q = question.lower()

    if "not adult" in q or "standard" in q or "non-adult" in q:
        return "standard"

    if "adult" in q:
        return "adult"

    return "standard"


# --------------------------------------------------
# Clarify Decision
# --------------------------------------------------
def weak_retrieval(top_score: Optional[float]) -> bool:
    return top_score is None or top_score < MIN_TOP_SCORE


def clarify_message(track: str) -> str:
    if track == "adult":
        return "Could you clarify your Adult level (for example: Adult Pre-Bronze, Adult Bronze, etc.)?"
    return "Could you clarify which level you are referring to (for example: Pre-Preliminary, Pre-Bronze, Bronze, etc.)?"


# --------------------------------------------------
# Priority Boosting
# --------------------------------------------------
def apply_priority_boost(retrieval: Dict) -> Dict:
    """
    Adjust scores slightly using metadata priority.
    Does not break original ordering if no scores available.
    """

    if "scores" not in retrieval or "sources" not in retrieval:
        return retrieval

    boosted = []

    for score, meta in zip(retrieval["scores"], retrieval["sources"]):
        priority = meta.get("priority", 0)
        adjusted_score = score * (1 + 0.08 * priority)
        boosted.append(adjusted_score)

    # Re-sort based on adjusted scores
    ranked = sorted(
        zip(boosted, retrieval["results"], retrieval["sources"]),
        key=lambda x: x[0],
        reverse=True,
    )

    retrieval["scores"] = [r[0] for r in ranked]
    retrieval["results"] = [r[1] for r in ranked]
    retrieval["sources"] = [r[2] for r in ranked]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None

    return retrieval


# --------------------------------------------------
# Main Answer Function
# --------------------------------------------------
def answer_question(question: str, history: List[Dict]) -> str:

    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    normalized_question = normalize_legacy_terms(question)
    track = infer_track(normalized_question)

    # -------------------------
    # 1️⃣ Primary Retrieval (track filtered)
    # -------------------------
    retrieval = retrieve(
        normalized_question,
        k=TOP_K,
        track=track,
    )

    # Apply metadata-based score boosting
    retrieval = apply_priority_boost(retrieval)

    # -------------------------
    # 2️⃣ Fallback Retrieval (no track filter)
    # -------------------------
    if weak_retrieval(retrieval.get("top_score")):
        fallback = retrieve(
            normalized_question,
            k=TOP_K,
            track=None,
        )
        fallback = apply_priority_boost(fallback)

        if not weak_retrieval(fallback.get("top_score")):
            retrieval = fallback

    # -------------------------
    # 3️⃣ RAG DEBUG DISPLAY
    # -------------------------
    print("\n===== RAG DEBUG =====")
    print("Original Question:", question)
    print("Normalized Question:", normalized_question)
    print("Track Used:", track)
    print("Top Score:", retrieval.get("top_score"))
    print("Results Count:", len(retrieval.get("results", [])))

    for i, meta in enumerate(retrieval.get("sources", []), 1):
        print(f"\nResult {i}")
        print("  Source:", meta.get("source_path"))
        print("  Type:", meta.get("doc_type"))
        print("  Group:", meta.get("source_group"))
        print("  Priority:", meta.get("priority"))

    print("======================\n")

    # -------------------------
    # 4️⃣ If Still Weak → Clarify
    # -------------------------
    if weak_retrieval(retrieval.get("top_score")):
        return clarify_message(track)

    # -------------------------
    # 5️⃣ Build Prompt
    # -------------------------
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    llm_input = build_llm_input(
        prompt=prompt,
        question=normalized_question,
        docs=retrieval["results"],
        history=history,
    )

    # -------------------------
    # 6️⃣ Call LLM
    # -------------------------
    return run_llm(llm_input)