import re

def extract_weight(q: str):
    # lb
    match_lb = re.search(r"(\d+)\s?(lb|lbs|pounds?)", q)
    if match_lb:
        return float(match_lb.group(1))

    # kg → convert to lb
    match_kg = re.search(r"(\d+)\s?(kg|kilograms?)", q)
    if match_kg:
        return float(match_kg.group(1)) * 2.20462

    return None

def extract_height(q: str):
    # feet + inches (e.g. 6ft, 6'1)
    match_ft = re.search(r"(\d+)\s?(ft|')\s?(\d+)?", q)
    if match_ft:
        feet = int(match_ft.group(1))
        inches = int(match_ft.group(3) or 0)
        return feet * 12 + inches

    # cm
    match_cm = re.search(r"(\d+)\s?cm", q)
    if match_cm:
        return float(match_cm.group(1)) / 2.54

    return None


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

def build_skater_state(query: str):
    q = query.lower()

    state = {
        "skill_level": "unknown",
        "signals": [],
        "jump_level": None,
        "weight_class": "unknown",
        "height_class": "unknown",
        "experience_type": "unknown",
        "goal": "unknown"
    }

    # ---- skill signals ----
    if any(x in q for x in ["axel", "1a"]):
        state["skill_level"] = "intermediate"
        state["signals"].append("axel")
        state["jump_level"] = "1A"

    if any(x in q for x in ["2t", "2s", "2lo", "2f", "2lz", "double"]):
        state["skill_level"] = "intermediate"
        state["signals"].append("double")

    if any(x in q for x in ["3a", "triple", "quad", "4f"]):
        state["skill_level"] = "advanced"
        state["signals"].append("advanced_jump")

    weight = extract_weight(q)
    height = extract_height(q)

    # ---- weight classification ----
    if weight:
        if weight >= 190:
            state["weight_class"] = "heavy"
        elif weight >= 160:
            state["weight_class"] = "medium"
        else:
            state["weight_class"] = "light"

    # fallback: descriptive words
    if "heavy" in q or "heavier" in q:
        state["weight_class"] = "heavy"

    # ---- height classification ----
    if height:
        if height >= 72:  # 6ft+
            state["height_class"] = "tall"
        elif height <= 64:  # ~5'4
            state["height_class"] = "short"
        else:
            state["height_class"] = "average"

    # fallback
    if "tall" in q:
        state["height_class"] = "tall"
    if "short" in q:
        state["height_class"] = "short"

    if "adult" in q:
        state["experience_type"] = "adult"

    if "recommend" in q or "what skate" in q:
        state["goal"] = "equipment"

    if state["weight_class"] == "heavy" and state["skill_level"] in ["intermediate", "advanced"]:
        state["boot_requirement"] = "high_stiffness"

    return state