from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import numpy as np

from rag.retriever import retrieve, EMBED_MODEL
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
DEBUG = True

# Similarity thresholds
MERGE_THRESHOLD = 0.60
FALLBACK_SHORT_THRESHOLD = 0.40

# If we merge, make the current message dominate
CURRENT_WEIGHT = 2


# --------------------------------------------------
# Debug Printer
# --------------------------------------------------
def debug_print(*args):
    if DEBUG:
        print(*args)


# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------
# Brand Detection (NEW)
# --------------------------------------------------
BRANDS = [
    "edea",
    "jackson",
    "mk",
    "john wilson",
    "wilson",
    "riedell",
    "risport",
    "graf",
    "aura",
    "eclipse",
    "paramount",
    "wilson",
    "jw"
]


def extract_brand(text: str) -> Optional[str]:
    t = text.lower()
    for b in BRANDS:
        if b in t:
            return b
    return None


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
# Weighted merge builder
# --------------------------------------------------
def build_weighted_merged_query(previous_user: str, current_user: str) -> str:
    current_part = " ".join([current_user] * max(1, CURRENT_WEIGHT))
    return f"{previous_user} {current_part}"


# --------------------------------------------------
# Semantic Retrieval Query Builder (with brand guard)
# --------------------------------------------------
def build_retrieval_query(question: str, history: List[Dict]) -> Tuple[str, bool]:

    user_msgs = [m["content"] for m in history if m["role"] == "user"]

    if len(user_msgs) < 2:
        return question, False

    previous_user = user_msgs[-2]

    # --------------------------------------------------
    # 🔥 BRAND CONFLICT GUARD
    # --------------------------------------------------
    previous_brand = extract_brand(previous_user)
    current_brand = extract_brand(question)

    if previous_brand and current_brand and previous_brand != current_brand:
        if DEBUG:
            debug_print("\n===== MERGE DEBUG =====")
            debug_print("Brand conflict detected")
            debug_print("Previous brand:", previous_brand)
            debug_print("Current brand :", current_brand)
            debug_print("→ NO MERGE\n")
        return question, False

    # --------------------------------------------------
    # Semantic similarity check
    # --------------------------------------------------
    emb_prev = EMBED_MODEL.encode(previous_user, convert_to_numpy=True)
    emb_curr = EMBED_MODEL.encode(question, convert_to_numpy=True)

    merged_plain = previous_user + " " + question
    emb_merged = EMBED_MODEL.encode(merged_plain, convert_to_numpy=True)

    sim_prev_curr = cosine_sim(emb_prev, emb_curr)
    sim_prev_merged = cosine_sim(emb_prev, emb_merged)

    if DEBUG:
        debug_print("\n===== MERGE DEBUG =====")
        debug_print("Previous:", previous_user)
        debug_print("Current :", question)
        debug_print(f"Prev-Curr similarity   = {sim_prev_curr:.4f}")
        debug_print(f"Prev-Merged similarity = {sim_prev_merged:.4f}")
        debug_print("=======================\n")

    # Rule 1
    if sim_prev_merged > MERGE_THRESHOLD:
        return build_weighted_merged_query(previous_user, question), True

    # Rule 2
    if len(question.split()) <= 6 and sim_prev_curr > FALLBACK_SHORT_THRESHOLD:
        return build_weighted_merged_query(previous_user, question), True

    return question, False


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

    if "scores" not in retrieval or "sources" not in retrieval:
        return retrieval

    boosted = []

    for score, meta in zip(retrieval["scores"], retrieval["sources"]):
        priority = meta.get("priority", 0)
        adjusted_score = score + 0.35 * priority
        boosted.append(adjusted_score)

    ranked = sorted(
        zip(boosted, retrieval["results"], retrieval["sources"]),
        key=lambda x: x[0],
        reverse=True,
    )

    manual_chunks = [
        item for item in ranked
        if item[2].get("doc_type") == "manual"
    ]

    if manual_chunks:
        top_manual = max(manual_chunks, key=lambda x: x[0])
        if all(r[2].get("doc_type") != "manual" for r in ranked[:3]):
            ranked.insert(0, top_manual)

    ranked = ranked[:TOP_K]

    retrieval["scores"] = [r[0] for r in ranked]
    retrieval["results"] = [r[1] for r in ranked]
    retrieval["sources"] = [r[2] for r in ranked]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None

    return retrieval


# --------------------------------------------------
# Main Answer Function
# --------------------------------------------------
def answer_question(question: str, history: List[Dict]) -> str:

    total_start = time.time()

    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    normalized_question = normalize_legacy_terms(question)

    retrieval_query, merged_flag = build_retrieval_query(normalized_question, history)
    retrieval_query = normalize_legacy_terms(retrieval_query)

    track = infer_track(normalized_question)

    t_retrieve_start = time.time()

    RAW_K = 30

    retrieval = retrieve(
        retrieval_query,
        k=RAW_K,
        track=track,
    )

    retrieval = apply_priority_boost(retrieval)

    retrieve_time = time.time() - t_retrieve_start

    if DEBUG:
        debug_print("\n===== RAG DEBUG =====")
        debug_print("Original Question:", question)
        debug_print("Merged?         :", merged_flag)
        debug_print("Retrieval Query :", retrieval_query)
        debug_print("Top Score       :", retrieval.get("top_score"))
        debug_print("Results Count   :", len(retrieval.get("results", [])))

        for i, meta in enumerate(retrieval.get("sources", []), 1):
            debug_print(f"\nResult {i}")
            debug_print("  Source Path :", meta.get("source_path"))
            debug_print("  Type        :", meta.get("doc_type"))
            debug_print("  Group       :", meta.get("source_group"))
            debug_print("  Priority    :", meta.get("priority"))

        debug_print("======================\n")

    if weak_retrieval(retrieval.get("top_score")):
        return clarify_message(track)

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    llm_input = build_llm_input(
        prompt=prompt,
        question=normalized_question,
        docs=retrieval["results"],
        history=history,
    )

    t_llm_start = time.time()
    response = run_llm(llm_input)
    llm_time = time.time() - t_llm_start
    total_time = time.time() - total_start

    if DEBUG:
        debug_print("\n===== TIMING DEBUG =====")
        debug_print(f"Retrieve time : {retrieve_time:.4f}s")
        debug_print(f"LLM time      : {llm_time:.4f}s")
        debug_print(f"Total time    : {total_time:.4f}s")
        debug_print("========================\n")

    return response