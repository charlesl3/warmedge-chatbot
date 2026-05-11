from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from backend.retrieval.query_utils import (
    normalize_legacy_terms,
    extract_brand,
    cosine_sim,
    build_weighted_merged_query,
)
from backend.retrieval.retriever import EMBED_MODEL




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

SOFT_MERGE_THRESHOLD = 0.72


JUMP_TERMS = [
    "lutz",
    "flip",
    "salchow",
    "toe loop",
    "loop",
    "axel",
    "waltz jump",
    "single",
    "double",
    "triple",
    "quad",
]


EQUIPMENT_TERMS = [
    "jackson",
    "edea",
    "riedell",
    "risport",
    "graf",
    "mk",
    "john wilson",
    "wilson",
    "blade",
    "boot",
    "boots",
]


PROBLEM_TERMS = [
    "scrape",
    "scraping",
    "scratch",
    "scratches",
    "pull",
    "pulls",
    "inside",
    "outside",
    "unstable",
    "loose",
    "pain",
    "hurt",
    "switch",
    "switched",
    "change",
    "changed",
]


INCOMPLETE_REFERENCES = [
    "it",
    "this",
    "that",
    "those",
    "them",
    "same",
    "one",
    "only",
    "still",
    "right foot",
    "left foot",
]


def is_self_contained_query(question: str) -> bool:

    q = question.lower()
    words = q.split()

    has_jump = any(term in q for term in JUMP_TERMS)

    has_equipment = any(
        term in q
        for term in EQUIPMENT_TERMS
    )

    has_problem = any(
        term in q
        for term in PROBLEM_TERMS
    )

    # Long descriptive skating question
    if (
        len(words) >= 10
        and has_problem
        and (has_jump or has_equipment)
    ):
        return True

    # Explicit jump problem
    if len(words) >= 8 and has_jump:
        return True

    # Explicit equipment issue
    if (
        len(words) >= 8
        and has_equipment
        and has_problem
    ):
        return True

    return False


def is_incomplete_continuation(question: str) -> bool:

    q = question.lower().strip()
    words = q.split()

    # Single-word or tiny continuation
    if len(words) <= 3:
        return True

    # Context-dependent fragments
    if (
        len(words) <= 7
        and any(ref in q for ref in INCOMPLETE_REFERENCES)
    ):
        return True

    # Examples:
    # "only on the right foot"
    # "it still pulls inside"
    if (
        len(words) <= 8
        and any(ref in q for ref in INCOMPLETE_REFERENCES)
    ):
        return True

    return False

def maybe_merge_history(
    question: str,
    history: list[dict],
) -> Tuple[str, bool]:

    user_msgs = [
        m["content"]
        for m in history
        if m["role"] == "user"
    ]

    if len(user_msgs) < 2:
        return question, False

    previous_user = normalize_legacy_terms(
        user_msgs[-2]
    )

    # -------------------------------------------------
    # 1. Current query already self-contained
    # -------------------------------------------------
    # Example:
    # "I switched from Jackson to Edea
    # and now my lutz scratches"
    #
    # Do NOT inherit old conversational context.
    # -------------------------------------------------

    if is_self_contained_query(question):
        return question, False

    # -------------------------------------------------
    # 2. Clearly incomplete continuation
    # -------------------------------------------------
    # Example:
    # "Jackson"
    # "only on the right foot"
    #
    # Merge aggressively.
    # -------------------------------------------------

    if is_incomplete_continuation(question):

        merged = build_weighted_merged_query(
            previous_user,
            question,
        )

        return merged, True

    # -------------------------------------------------
    # 3. Brand conflict protection
    # -------------------------------------------------

    prev_brand = extract_brand(previous_user)
    curr_brand = extract_brand(question)

    if (
        prev_brand
        and curr_brand
        and prev_brand != curr_brand
    ):
        return question, False

    # -------------------------------------------------
    # 4. Soft semantic merge
    # -------------------------------------------------
    # Only for uncertain middle cases.
    # -------------------------------------------------

    emb_prev = EMBED_MODEL.encode(
        previous_user,
        convert_to_numpy=True,
    )

    emb_curr = EMBED_MODEL.encode(
        question,
        convert_to_numpy=True,
    )

    sim_prev_curr = cosine_sim(
        emb_prev,
        emb_curr,
    )

    if (
        sim_prev_curr > SOFT_MERGE_THRESHOLD
        and len(question.split()) <= 10
    ):

        merged = build_weighted_merged_query(
            previous_user,
            question,
        )

        return merged, True

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