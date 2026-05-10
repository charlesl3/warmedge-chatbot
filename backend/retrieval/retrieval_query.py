from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from backend.retrieval.query_utils import (
    normalize_legacy_terms,
    extract_brand,
    cosine_sim,
    build_weighted_merged_query,
)
from backend.retrieval.retriever import EMBED_MODEL


MERGE_THRESHOLD = 0.60
FALLBACK_SHORT_THRESHOLD = 0.40


@dataclass(frozen=True)
class RetrievalProfile:
    original_question: str
    normalized_question: str
    primary_query: str
    expanded_queries: List[str]
    intent: str
    topic: str
    track: str
    retrieval_hints: Dict[str, Any] = field(default_factory=dict)
    used_history_merge: bool = False


def infer_track(question: str) -> str:
    q = question.lower()
    if "adult" in q:
        return "adult"
    return "standard"


def maybe_merge_history(question: str, history: list[dict]) -> Tuple[str, bool]:
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


def build_intent_query_suffix(intent: str) -> str:
    if intent == "how_to":
        return "what to try technique steps figure skating"
    if intent == "comparison":
        return "comparison differences pros cons figure skating"
    if intent == "diagnosis":
        return "causes similar skating issues figure skating"
    if intent == "experience_lookup":
        return "common causes experience figure skating"
    return "figure skating"


def build_retrieval_profile(
    question: str,
    history: list[dict],
    intent: str,
    intent_profile: Optional[dict] = None,
    state: Optional[dict] = None,
) -> RetrievalProfile:
    normalized = normalize_legacy_terms(question)
    merged_query, used_merge = maybe_merge_history(normalized, history)

    topic = "unknown"
    if intent_profile:
        topic = intent_profile.get("topic", "unknown")

    suffix = build_intent_query_suffix(intent)

    primary_query = normalize_legacy_terms(f"{merged_query} {suffix}".strip())

    expanded_queries = [
        primary_query,
        normalize_legacy_terms(merged_query),
    ]

    if topic and topic != "unknown":
        expanded_queries.append(f"{merged_query} {topic} figure skating")

    if state:
        if state.get("experience_type") == "adult":
            expanded_queries.append(f"{merged_query} adult figure skating")
        if state.get("jump_level"):
            expanded_queries.append(f"{merged_query} {state['jump_level']} figure skating")

    return RetrievalProfile(
        original_question=question,
        normalized_question=normalized,
        primary_query=primary_query,
        expanded_queries=list(dict.fromkeys(expanded_queries)),
        intent=intent,
        topic=topic,
        track=infer_track(normalized),
        retrieval_hints={
            "state": state or {},
            "intent_profile": intent_profile or {},
        },
        used_history_merge=used_merge,
    )