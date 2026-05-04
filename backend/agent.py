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


def choose_k(query: str, intent: str, state: dict, history: list[dict]) -> int:
    q = query.lower()
    length = len(q.split())

    # -------------------------
    # 1. Base on intent
    # -------------------------
    if intent == "diagnosis":
        k = 6
    elif intent == "how_to":
        k = 4
    elif intent == "comparison":
        k = 4
    else:
        k = 5

    # -------------------------
    # 2. Short / vague queries → increase k
    # -------------------------
    if length <= 4:
        k += 1

    # -------------------------
    # 3. Strong skill signal → reduce k
    # -------------------------
    if state.get("signals"):
        k -= 1

    # -------------------------
    # 4. Prior context exists → reduce k
    # -------------------------
    has_context = any(
        ("axel" in m["content"].lower() or
         re.search(r"\b[1-4][aflst]\b", m["content"].lower()))
        for m in history if m["role"] == "user"
    )

    if has_context:
        k -= 1

    # -------------------------
    # Clamp range
    # -------------------------
    return max(2, min(k, 7))


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
            "best" in q
    )

    is_how_to = (
            "how to" in q or
            "how do i" in q or
            "how should i" in q or
            "practice" in q or
            "improve" in q or
            "fix" in q
    )

    # -------------------------
    # 4. HISTORY-AWARE DECISION
    # -------------------------
    if is_recommendation:
        if not has_skill_signal:
            if not has_prior_skill_signal(history):
                return True, "missing level"

    # 🚨 NEW: do NOT clarify for how_to
    if is_how_to:
        return False, "how_to_safe"

    return False, "sufficient"


def classify_query_intent(query: str, history: list[dict]) -> str:
    q = query.lower().strip()

    # -------------------------
    # FRONTEND CHIP MODE HINTS
    # -------------------------
    if "mode: drills" in q:
        return "how_to"

    if "mode: diagnose" in q:
        return "diagnosis"

    if "mode: deeper" in q:
        return "experience_lookup"

    if "mode: simplify" in q:
        return "default"

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


def build_answer_plan(
    query: str,
    intent: str,
    state: dict,
    history: list[dict],
    clarify: bool = False,
) -> dict:
    q = query.lower().strip()
    length = len(q.split())

    plan = {
        "mode": "standard",
        "depth": "medium",
        "use_context": False,
        "avoid_repetition": False,
        "structure": ["direct answer", "practical guidance"],
    }

    # -------------------------
    # 1. Clarification overrides everything
    # -------------------------
    if clarify:
        return {
            "mode": "clarification",
            "depth": "short",
            "use_context": False,
            "avoid_repetition": False,
            "structure": ["ask one precise follow-up"],
        }

    # -------------------------
    # 2. Intent -> mode mapping
    # -------------------------
    if intent == "how_to":
        plan["mode"] = "coaching"
        plan["structure"] = ["likely cause", "what to try", "1-2 drills"]

    elif intent == "diagnosis":
        plan["mode"] = "diagnosis"
        plan["structure"] = ["possible causes", "how to check", "what to try"]

    elif intent == "comparison":
        plan["mode"] = "comparison"
        plan["structure"] = ["common ground", "key differences", "recommendation"]

    elif intent == "experience_lookup":
        plan["mode"] = "explanation"
        plan["structure"] = ["direct answer", "why", "practical note"]

    else:
        plan["mode"] = "standard"
        plan["structure"] = ["direct answer", "practical guidance"]

    # -------------------------
    # 3. Short queries -> shorter answers
    # -------------------------
    if length <= 4:
        plan["depth"] = "short"

    # -------------------------
    # 4. Strong skill signals -> more depth
    # -------------------------
    if state.get("signals") and plan["depth"] != "short":
        plan["depth"] = "detailed"

    # -------------------------
    # 5. Prior context -> use it, do not repeat
    # -------------------------
    has_context = any(
        ("axel" in m["content"].lower() or
         re.search(r"\b[1-4][aflst]\b", m["content"].lower()) or
         any(x in m["content"].lower() for x in ["double", "triple", "quad"]))
        for m in history if m["role"] == "user"
    )

    if has_context:
        plan["use_context"] = True
        plan["avoid_repetition"] = True

    return plan

# -------------------------
# SMART FOLLOW-UP LOGIC
# -------------------------

def should_generate_followup(
    query: str,
    intent: str,
    state: dict,
    agent_trace: dict,
    reply: str,
) -> bool:
    q = query.lower().strip()

    # Do not add follow-up to very short social-ish replies
    if len(reply.split()) < 25:
        return False

    # Do not follow up after direct clarification
    if agent_trace.get("clarification", {}).get("triggered"):
        return False

    # Most valuable cases
    if intent in ["how_to", "diagnosis", "comparison"]:
        return True

    # Unknown user level is a useful reason to invite continuation
    if state.get("skill_level") == "unknown":
        return True

    # Weak retrieval / repair means we should invite more detail
    if agent_trace.get("retrieval", {}).get("weak"):
        return True

    if agent_trace.get("repair", {}).get("triggered"):
        return True

    return False


def build_followup_prompt(
    query: str,
    answer: str,
    intent: str,
    state: dict,
    history: list[dict],
) -> str:
    recent_history = history[-6:] if history else []

    history_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in recent_history
    )

    return f"""
You are WarmGPT, a practical figure skating assistant.

Your job is to write ONE short follow-up question that naturally continues the conversation.

User's latest question:
{query}

WarmGPT's answer:
{answer}

Intent:
{intent}

Detected skater state:
- skill_level: {state.get("skill_level")}
- signals: {state.get("signals")}
- jump_level: {state.get("jump_level")}
- body: {state.get("height_class")}, {state.get("weight_class")}
- experience_type: {state.get("experience_type")}
- goal: {state.get("goal")}

Recent conversation:
{history_text}

Rules:
- Output exactly ONE follow-up question.
- Maximum 22 words.
- Make it specific to the user's skating situation.
- Do NOT repeat the answer.
- Do NOT ask "Do you want me to..." or "Would you like me to..."
- Do NOT ask multiple questions.
- Do NOT mention sources, retrieval, confidence, or internal logic.
- If no useful follow-up exists, output exactly: NONE

Good examples (these are just some examples, do not just copy these for all questions):
- Where does it usually break down for you: takeoff, landing, or the setup edge?
- Are you working on this for practice, a test, or competition?
- Does it happen more when you are tired or even at the start of the session?

Output only the follow-up question or NONE.
""".strip()