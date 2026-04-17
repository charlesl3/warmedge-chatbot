import re


def needs_clarification(query: str):
    q = query.lower()
    tokens = q.split()

    # -------------------------
    # 1. VAGUE SHORT QUERIES
    # -------------------------
    vague_patterns = ["this", "that", "good", "which", "should"]

    if len(tokens) <= 2 and any(p in tokens for p in vague_patterns):
        return True, "vague short query"

    # -------------------------
    # 2. LEVEL SIGNALS
    # -------------------------
    level_keywords = ["beginner", "intermediate", "advanced"]
    has_level = any(k in tokens for k in level_keywords)

    # -------------------------
    # 3. SKILL SIGNALS (ROBUST)
    # -------------------------
    def has_jump_signal(tokens, query):
        # --- keyword-based ---
        jump_keywords = ["axel", "lutz", "flip", "loop", "salchow", "toe"]
        if any(j in tokens for j in jump_keywords):
            return True

        # --- prefix-based (double/triple/quad) ---
        for t in tokens:
            if t.startswith("single") or t.startswith("double") or t.startswith("triple") or t.startswith("quad"):
                return True

        # --- shorthand patterns (1a, 2a, 3f, 4t, etc.) ---
        if re.search(r"\b[1-4][aflst]\b", query):
            return True

        return False

    has_skill_signal = has_jump_signal(tokens, q)

    # -------------------------
    # 4. RECOMMENDATION DETECTION
    # -------------------------
    is_recommendation = (
        "recommend" in q or
        "best" in q or
        "should i" in q
    )

    if is_recommendation:
        # Only clarify if BOTH level and skill signals are missing
        if not has_level and not has_skill_signal:
            return True, "missing level"

    # -------------------------
    # 5. DEFAULT
    # -------------------------
    return False, "sufficient"