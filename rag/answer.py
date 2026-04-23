from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import numpy as np
import json

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
MIN_TOP_SCORE = 0.6
DEBUG = True

MERGE_THRESHOLD = 0.60
FALLBACK_SHORT_THRESHOLD = 0.40
CURRENT_WEIGHT = 2

FEEDBACK_PATH = "backend/feedback_memory.json"


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
# Feedback Memory
# --------------------------------------------------
def load_feedback_memory():
    try:
        with open(FEEDBACK_PATH, "r") as f:
            return json.load(f)
    except:
        return []


def boost_with_feedback(query_embedding: np.ndarray, retrieval: Dict) -> Dict:
    memory = load_feedback_memory()

    if not memory or "sources" not in retrieval:
        return retrieval

    doc_boost = {}

    for group in memory:
        doc = group.get("doc")
        examples = group.get("examples", [])

        for ex in examples:
            emb = np.array(ex["embedding"])
            sim = cosine_sim(query_embedding, emb)

            if sim > 0.75:
                doc_boost[doc] = doc_boost.get(doc, 0) + sim

    boosted_scores = []

    for score, meta in zip(retrieval["scores"], retrieval["sources"]):
        doc_id = meta.get("source_path") or meta.get("source_group")
        boost = doc_boost.get(doc_id, 0)
        boosted_scores.append(score + boost)

    ranked = sorted(
        zip(boosted_scores, retrieval["results"], retrieval["sources"]),
        key=lambda x: x[0],
        reverse=True,
    )

    ranked = ranked[:TOP_K]

    retrieval["scores"] = [r[0] for r in ranked]
    retrieval["results"] = [r[1] for r in ranked]
    retrieval["sources"] = [r[2] for r in ranked]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None

    if DEBUG:
        debug_print("\n===== FEEDBACK BOOST DEBUG =====")
        debug_print("Boosted docs:", doc_boost)
        debug_print("===============================\n")

    return retrieval


# --------------------------------------------------
# Brand Detection
# --------------------------------------------------
BRANDS = [
    "edea", "jackson", "mk", "john wilson", "wilson",
    "riedell", "risport", "graf", "aura", "eclipse",
    "paramount", "wilson", "jw", "harlick",
]


def extract_brand(text: str) -> Optional[str]:
    t = text.lower()
    for b in BRANDS:
        if b in t:
            return b
    return None


# --------------------------------------------------
# Normalization
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
# Merge Logic
# --------------------------------------------------
def build_weighted_merged_query(previous_user: str, current_user: str) -> str:
    current_part = " ".join([current_user] * max(1, CURRENT_WEIGHT))
    return f"{previous_user} {current_part}"


def build_retrieval_query(question: str, history: List[Dict]) -> Tuple[str, bool]:
    user_msgs = [m["content"] for m in history if m["role"] == "user"]

    if len(user_msgs) < 2:
        return question, False

    previous_user = user_msgs[-2]

    prev_brand = extract_brand(previous_user)
    curr_brand = extract_brand(question)

    if prev_brand and curr_brand and prev_brand != curr_brand:
        return question, False

    emb_prev = EMBED_MODEL.encode(previous_user, convert_to_numpy=True)
    emb_curr = EMBED_MODEL.encode(question, convert_to_numpy=True)

    merged_plain = previous_user + " " + question
    emb_merged = EMBED_MODEL.encode(merged_plain, convert_to_numpy=True)

    sim_prev_curr = cosine_sim(emb_prev, emb_curr)
    sim_prev_merged = cosine_sim(emb_prev, emb_merged)

    if sim_prev_merged > MERGE_THRESHOLD:
        return build_weighted_merged_query(previous_user, question), True

    if len(question.split()) <= 6 and sim_prev_curr > FALLBACK_SHORT_THRESHOLD:
        return build_weighted_merged_query(previous_user, question), True

    return question, False


# --------------------------------------------------
# Track Inference
# --------------------------------------------------
def infer_track(question: str) -> str:
    q = question.lower()
    if "adult" in q:
        return "adult"
    return "standard"


# --------------------------------------------------
# Retrieval Quality
# --------------------------------------------------
def weak_retrieval(top_score: Optional[float]) -> bool:
    return top_score is None or top_score < MIN_TOP_SCORE


def clarify_message(track: str) -> str:
    return "Could you clarify which level you are referring to?"

def build_fallback_query(question: str, intent: str) -> str:
    q = question.lower().strip()

    if intent == "diagnosis":
        return f"{q} causes figure skating"

    if intent == "how_to":
        return f"{q} drills practice figure skating"

    if intent == "comparison":
        return f"{q} differences figure skating"

    if intent == "experience_lookup":
        return f"{q} common causes figure skating"

    return f"{q} figure skating"


def rewrite_query_by_intent(query: str, intent: str) -> str:
    if intent == "how_to":
        return f"{query} what to try technique steps figure skating"

    if intent == "comparison":
        return f"{query} comparison pros cons figure skating"

    if intent == "diagnosis":
        return f"{query} causes similar skating issues figure skating"

    if intent == "experience_lookup":
        return f"{query} common causes experience figure skating"

    return query


# --------------------------------------------------
# Priority Boost
# --------------------------------------------------
def apply_priority_boost(retrieval: Dict) -> Dict:
    boosted = []

    for score, meta in zip(retrieval["scores"], retrieval["sources"]):
        priority = meta.get("priority", 0)
        boosted.append(score + 0.35 * priority)

    ranked = sorted(
        zip(boosted, retrieval["results"], retrieval["sources"]),
        key=lambda x: x[0],
        reverse=True,
    )[:TOP_K]

    retrieval["scores"] = [r[0] for r in ranked]
    retrieval["results"] = [r[1] for r in ranked]
    retrieval["sources"] = [r[2] for r in ranked]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None

    return retrieval


# --------------------------------------------------
# Extract doc ids
# --------------------------------------------------
def extract_retrieved_doc_ids(retrieval: Dict) -> List[str]:
    ids = []
    seen = set()

    for meta in retrieval.get("sources", []):
        doc_id = meta.get("source_path") or meta.get("source_group")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)

    return ids


# --------------------------------------------------
# Main Answer
# --------------------------------------------------
def answer_question(question, history, intent = "default", k=4):
    if is_blank(question):
        return "Please ask a valid question."

    if is_social_message(question):
        return handle_social_message(question)

    normalized = normalize_legacy_terms(question)

    retrieval_query, _ = build_retrieval_query(normalized, history)
    retrieval_query = normalize_legacy_terms(retrieval_query)
    retrieval_query = rewrite_query_by_intent(retrieval_query, intent)

    track = infer_track(normalized)

    debug_print("[INTENT]", intent)
    debug_print("[RETRIEVAL QUERY]", retrieval_query)

    query_embedding = EMBED_MODEL.encode(retrieval_query, convert_to_numpy=True)

    retrieval = retrieve(retrieval_query, k=30, track=track)

    retrieval = apply_priority_boost(retrieval)


    # 🔥 KEY: learning happens here
    retrieval = boost_with_feedback(query_embedding, retrieval)

    if weak_retrieval(retrieval.get("top_score")):
        fallback_query = build_fallback_query(normalized, intent)
        debug_print("[FALLBACK QUERY]", fallback_query)

        fallback_embedding = EMBED_MODEL.encode(fallback_query, convert_to_numpy=True)
        fallback_retrieval = retrieve(fallback_query, k=30, track=track)
        fallback_retrieval = apply_priority_boost(fallback_retrieval)
        fallback_retrieval = boost_with_feedback(fallback_embedding, fallback_retrieval)

        if weak_retrieval(fallback_retrieval.get("top_score")):
            return clarify_message(track)

        retrieval = fallback_retrieval
        query_embedding = fallback_embedding

    prompt = PROMPT_PATH.read_text()
    # 🔥 FINAL TRIM (only place you trim)
    retrieval["results"] = retrieval["results"][:k]
    retrieval["sources"] = retrieval["sources"][:k]
    retrieval["scores"] = retrieval["scores"][:k]

    print(f"[RAG] final k used = {k}, docs passed = {len(retrieval['results'])}")

    llm_input = build_llm_input(
        prompt=prompt,
        question=normalized,
        docs=retrieval["results"],  # no [:k] anymore
        history=history,
        intent=intent,
    )

    response = run_llm(llm_input)

    retrieved_docs = extract_retrieved_doc_ids(retrieval)[:k]
    return {
        "reply": response,
        "retrieved_docs": retrieved_docs,
        "query_embedding": query_embedding.tolist(),
    }