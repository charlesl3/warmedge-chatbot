def needs_clarification(query: str):
    q = query.lower()

    vague_patterns = ["this", "that", "good", "which one", "should i"]

    if any(p in q for p in vague_patterns) and len(q.split()) <= 6:
        return True, "vague query"

    if "recommend" in q or "best" in q or "should i" in q:
        if "beginner" not in q and "advanced" not in q and "intermediate" not in q:
            return True, "missing level"

    return False, "sufficient"