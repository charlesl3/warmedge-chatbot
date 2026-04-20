import re


# -------------------------
# SKATER STATE BUILDER
# -------------------------

def extract_weight(q: str):
    match_lb = re.search(r"(\d+)\s?(lb|lbs|pounds?)", q)
    if match_lb:
        return float(match_lb.group(1))

    match_kg = re.search(r"(\d+)\s?(kg|kilograms?)", q)
    if match_kg:
        return float(match_kg.group(1)) * 2.20462

    return None


def extract_height(q: str):
    match_ft = re.search(r"(\d+)\s?(ft|')\s?(\d+)?", q)
    if match_ft:
        feet = int(match_ft.group(1))
        inches = int(match_ft.group(3) or 0)
        return feet * 12 + inches

    match_cm = re.search(r"(\d+)\s?cm", q)
    if match_cm:
        return float(match_cm.group(1)) / 2.54

    return None


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

    if any(x in q for x in ["beginner", "just started"]):
        state["signals"].append("beginner_flag")

    # ---- numeric parsing ----
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
    elif "heavy" in q or "heavier" in q:
        state["weight_class"] = "heavy"

    # ---- height classification ----
    if height:
        if height >= 72:
            state["height_class"] = "tall"
        elif height <= 64:
            state["height_class"] = "short"
        else:
            state["height_class"] = "average"
    elif "tall" in q:
        state["height_class"] = "tall"
    elif "short" in q:
        state["height_class"] = "short"

    # ---- experience ----
    if "adult" in q:
        state["experience_type"] = "adult"

    # ---- goal ----
    if "recommend" in q or "what skate" in q:
        state["goal"] = "equipment"

    return state


# -------------------------
# HELPER: HISTORY CHECK
# -------------------------

def has_prior_skill_signal(history):
    for msg in reversed(history):
        if msg["role"] == "user":
            text = msg["content"].lower()

            if "axel" in text:
                return True

            if re.search(r"\b[1-4][aflst]\b", text):
                return True

            if any(x in text for x in ["double", "triple", "quad"]):
                return True

    return False


# -------------------------
# CLARIFICATION LOGIC
# -------------------------

def needs_clarification(query: str, history: list[dict]):
    q = query.lower()
    tokens = q.split()

    # -------------------------
    # 1. VAGUE SHORT QUERIES
    # -------------------------
    vague_patterns = ["this", "that", "which", "should"]

    if len(tokens) <= 2 and any(p in tokens for p in vague_patterns):
        return True, "vague short query"

    # -------------------------
    # 2. SKILL SIGNALS
    # -------------------------
    def has_jump_signal():
        if "axel" in q:
            return True
        if re.search(r"\b[1-4][aflst]\b", q):
            return True
        if any(x in q for x in ["double", "triple", "quad"]):
            return True
        return False

    has_skill_signal = has_jump_signal()

    # -------------------------
    # 3. RECOMMENDATION DETECTION
    # -------------------------
    is_recommendation = (
        "recommend" in q or
        "best" in q or
        "should i" in q
    )

    # -------------------------
    # 4. HISTORY-AWARE DECISION
    # -------------------------
    if is_recommendation:
        if not has_skill_signal:
            if not has_prior_skill_signal(history):
                return True, "missing level"

    return False, "sufficient"

def classify_query_intent(query: str, history: list[dict]) -> str:
    q = query.lower().strip()

    # 1. Comparison questions
    if any(x in q for x in [" vs ", "versus", "better than", "difference between", "compare"]):
        return "comparison"

    # 2. Action / coaching questions
    if any(x in q for x in ["how to", "how do i", "improve", "fix", "practice", "train"]):
        return "how_to"

    # 3. Symptom / diagnosis questions
    if any(x in q for x in ["feel", "feels", "unstable", "off", "weird", "problem", "issue"]):
        return "diagnosis"

    # 4. General explanation questions
    if any(x in q for x in ["why", "normal", "common", "is it okay", "is it bad"]):
        return "experience_lookup"

    return "default"