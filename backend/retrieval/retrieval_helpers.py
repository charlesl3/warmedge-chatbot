from typing import Dict, List, Optional

MIN_TOP_SCORE = 0.6
TOP_K = 6

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

def infer_answer_confidence(retrieval: Dict, k: int) -> str:
    scores = retrieval.get("scores", [])
    docs = retrieval.get("results", [])
    top_score = retrieval.get("top_score")

    # -------------------------
    # 0. No retrieval
    # -------------------------
    if not docs or top_score is None:
        return "low"

    # -------------------------
    # 1. Very few docs → low
    # -------------------------
    if len(docs) <= 2:
        return "low"

    # -------------------------
    # 2. Score spread (key idea)
    # -------------------------
    # Strong retrieval = top doc clearly better than others
    if len(scores) >= 2:
        gap = scores[0] - scores[1]
    else:
        gap = 0.0

    # -------------------------
    # 3. Average quality
    # -------------------------
    avg_score = sum(scores) / len(scores)

    # -------------------------
    # 4. Heuristics (empirical, not absolute)
    # -------------------------

    # 🔴 LOW confidence
    if avg_score < 0.45:
        return "low"

    if gap < 0.02:
        # top doc not clearly better → ambiguous retrieval
        return "low"

    # 🟡 MEDIUM confidence
    if avg_score < 0.65:
        return "medium"

    if len(docs) < k:
        return "medium"

    # 🟢 HIGH confidence
    return "high"

def clarify_message(track: str) -> str:
    return "Could you clarify which level you are referring to?"

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