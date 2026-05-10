from typing import Dict

from backend.retrieval.retriever import retrieve, EMBED_MODEL
from backend.retrieval.feedback_memory import boost_with_feedback
from backend.retrieval.retrieval_helpers import apply_priority_boost

MIN_TOP_SCORE = 0.6


def execute_retrieval(query: str, track: str, pool_k: int = 30) -> Dict:
    query_embedding = EMBED_MODEL.encode(query, convert_to_numpy=True)

    retrieval = retrieve(query, k=pool_k, track=track)
    retrieval = apply_priority_boost(retrieval)
    retrieval = boost_with_feedback(query_embedding, retrieval)

    retrieval["query_used"] = query
    retrieval["query_embedding"] = query_embedding

    return retrieval


def evaluate_retrieval_quality(retrieval: Dict, desired_k: int) -> Dict:
    scores = retrieval.get("scores", [])
    docs = retrieval.get("results", [])
    top_score = retrieval.get("top_score")

    if not docs or top_score is None:
        return {
            "status": "weak",
            "reason": "no_docs",
            "confidence": "low",
            "should_fallback": True,
        }

    avg_score = sum(scores) / len(scores) if scores else 0.0
    gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0

    if top_score < MIN_TOP_SCORE:
        return {
            "status": "weak",
            "reason": "low_top_score",
            "confidence": "low",
            "top_score": top_score,
            "avg_score": avg_score,
            "should_fallback": True,
        }

    if len(docs) <= 2:
        return {
            "status": "weak",
            "reason": "too_few_docs",
            "confidence": "low",
            "top_score": top_score,
            "avg_score": avg_score,
            "should_fallback": True,
        }

    if avg_score < 0.45 or gap < 0.02:
        return {
            "status": "medium",
            "reason": "ambiguous_scores",
            "confidence": "medium",
            "top_score": top_score,
            "avg_score": avg_score,
            "should_fallback": False,
        }

    return {
        "status": "good",
        "reason": "sufficient",
        "confidence": "high" if avg_score >= 0.65 and len(docs) >= desired_k else "medium",
        "top_score": top_score,
        "avg_score": avg_score,
        "should_fallback": False,
    }


def apply_fallback_if_needed(profile, retrieval: Dict, evaluation: Dict, desired_k: int) -> tuple[Dict, Dict]:
    """
    Fallback changes retrieval strategy, not query semantics.
    It may use prebuilt expanded_queries, but it must not invent a new query.
    """

    if not evaluation.get("should_fallback"):
        return retrieval, {
            "triggered": False,
            "strategy": "none",
            "reason": "not_needed",
            "evaluation_reason": evaluation.get("reason"),
        }

    candidates = []

    # Strategy 1: broader k, same primary query
    candidates.append((
        "increase_pool_k",
        execute_retrieval(profile.primary_query, track=profile.track, pool_k=50)
    ))

    # Strategy 2: use prebuilt query variants only
    for variant in profile.expanded_queries[1:]:
        candidates.append((
            "prebuilt_query_variant",
            execute_retrieval(variant, track=profile.track, pool_k=50)
        ))

    best_strategy, best_retrieval = max(
        candidates,
        key=lambda item: item[1].get("top_score") or 0.0
    )

    best_eval = evaluate_retrieval_quality(best_retrieval, desired_k)

    if best_eval.get("status") == "weak":
        return retrieval, {
            "triggered": True,
            "strategy": best_strategy,
            "reason": "fallback_failed",
            "evaluation": best_eval,
        }

    return best_retrieval, {
        "triggered": True,
        "strategy": best_strategy,
        "reason": "retrieval_recovered",
        "evaluation": best_eval,
    }


def trim_retrieval(retrieval: Dict, k: int) -> Dict:
    retrieval["results"] = retrieval.get("results", [])[:k]
    retrieval["sources"] = retrieval.get("sources", [])[:k]
    retrieval["scores"] = retrieval.get("scores", [])[:k]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None
    return retrieval