import re
from backend.generation.llm import run_llm


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


def build_retrieval_strategy(
    query: str,
    intent_profile: dict,
    state: dict,
    history: list[dict],
) -> dict:
    q = query.lower().strip()
    length = len(q.split())

    primary = intent_profile.get("primary_intent", "default")
    secondary = intent_profile.get("secondary_intents", [])
    topic = intent_profile.get("topic", "unknown")

    strategy = {
        "k": 5,
        "reason": [],
        "exploration": "medium",
    }

    # Broad / ambiguous diagnosis needs more evidence
    if primary == "diagnosis":
        strategy["k"] = 6
        strategy["exploration"] = "high"
        strategy["reason"].append("primary_diagnosis")

    # Specific comparison should stay tighter
    elif primary == "comparison":
        strategy["k"] = 4
        strategy["exploration"] = "low"
        strategy["reason"].append("primary_comparison")

    # How-to usually needs enough context but not too broad
    elif primary == "how_to":
        strategy["k"] = 5
        strategy["exploration"] = "medium"
        strategy["reason"].append("primary_how_to")

    # Mixed intent: diagnosis + equipment is usually broader
    if primary == "diagnosis" and "equipment" in secondary:
        strategy["k"] += 1
        strategy["reason"].append("diagnosis_with_equipment")

    # Mixed intent: comparison + equipment should remain precise
    if primary == "comparison" and "equipment" in secondary:
        strategy["k"] -= 1
        strategy["reason"].append("comparison_with_equipment_precision")

    # Short vague queries need more exploration
    if length <= 4:
        strategy["k"] += 1
        strategy["reason"].append("short_query")

    # Strong skill signals make query more specific
    if state.get("signals"):
        strategy["k"] -= 1
        strategy["reason"].append("strong_skill_signal")

    # Topic-specific queries can be tighter
    if topic not in ["unknown", "general"] and length >= 5:
        strategy["k"] -= 1
        strategy["reason"].append("specific_topic")

    # Prior context reduces need for broad retrieval
    has_context = any(
        (
            "axel" in m["content"].lower()
            or re.search(r"\b[1-4][aflst]\b", m["content"].lower())
            or any(x in m["content"].lower() for x in ["double", "triple", "quad"])
        )
        for m in history
        if m["role"] == "user"
    )

    if has_context:
        strategy["k"] -= 1
        strategy["reason"].append("prior_skill_context")

    strategy["k"] = max(2, min(strategy["k"], 8))

    return strategy

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
# THIN LLM CLARIFICATION CONTROLLER
# -------------------------

# -------------------------
# SEMANTIC CLARIFICATION CONTROLLER
# -------------------------

def semantic_clarification_check(query: str, history: list[dict], state: dict | None = None) -> dict:
    """
    Thin LLM controller.
    It does NOT answer the user.
    It only gives a semantic clarification decision.
    Backend remains the final authority.
    """

    recent_history = history[-6:] if history else []
    history_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in recent_history
    )

    prompt = f"""
    You are a backend clarification controller for WarmGPT, a figure skating assistant.

    Do NOT answer the user.
    Only decide whether the latest user message needs clarification before WarmGPT answers.

    Use EXACTLY this format:

    TASK: one short task type
    NEEDS_CLARIFICATION: YES or NO
    ENOUGH_FOR_USEFUL_ANSWER: YES or NO
    REASON: one short reason
    QUESTION: one short clarification question, or NONE

    User message:
    {query}

    Recent conversation:
    {history_text}

    Detected state:
    {state or {} }

    Examples:

    Example 1
    User:
    What skates should I buy?

    Output:
    TASK: equipment_recommendation
    NEEDS_CLARIFICATION: YES
    ENOUGH_FOR_USEFUL_ANSWER: NO
    REASON: skating level missing
    QUESTION: What level are you currently skating at?

    Example 2
    User:
    I am advanced.

    Output:
    TASK: clarification_response
    NEEDS_CLARIFICATION: NO
    ENOUGH_FOR_USEFUL_ANSWER: YES
    REASON: enough information provided
    QUESTION: NONE

    Example 3
    User:
    How do I stop scraping on salchow?

    Output:
    TASK: technique_help
    NEEDS_CLARIFICATION: NO
    ENOUGH_FOR_USEFUL_ANSWER: YES
    REASON: enough information for useful coaching
    QUESTION: NONE

    Example 4
    User:
    Thinking about upgrading my boots.

    Output:
    TASK: equipment_recommendation
    NEEDS_CLARIFICATION: YES
    ENOUGH_FOR_USEFUL_ANSWER: NO
    REASON: skating level missing
    QUESTION: What level are you currently skating at?

    Example 5
    User:
    loop

    Output:
    TASK: ambiguous_term
    NEEDS_CLARIFICATION: YES
    ENOUGH_FOR_USEFUL_ANSWER: NO
    REASON: loop may refer to jump or turn
    QUESTION: Do you mean the loop jump or the loop turn?

    Example 6
    User:
    What skates do you recommend?
    Assistant:
    What level are you currently skating at?
    User:
    Advanced

    Output:
    TASK: equipment_recommendation
    NEEDS_CLARIFICATION: NO
    ENOUGH_FOR_USEFUL_ANSWER: YES
    REASON: enough information for useful recommendation
    QUESTION: NONE

    Rules:
    - Ask clarification only if answering now would likely be useless, misleading, or badly personalized.
    - Do NOT ask clarification merely because more detail could improve the answer.
    - Equipment recommendation questions usually need skating level if no level is known.
    - Technique/how-to questions usually can be answered without clarification.
    - If the user is answering a previous clarification with a short answer, do NOT ask another clarification.
    - If enough information exists for a useful answer, set NEEDS_CLARIFICATION to NO.
    - For ambiguous skating terms like loop, clarify only when the message is too short to infer meaning.
    - Prefer answering over over-clarifying.
    - One clarification is usually enough.
    """.strip()

    try:
        raw = run_llm(prompt).strip()
        upper = raw.upper()

        needs = "NEEDS_CLARIFICATION: YES" in upper
        enough = "ENOUGH_FOR_USEFUL_ANSWER: YES" in upper

        reason_match = re.search(r"REASON:\s*(.*)", raw, re.IGNORECASE)
        question_match = re.search(r"QUESTION:\s*(.*)", raw, re.IGNORECASE)

        reason = reason_match.group(1).strip() if reason_match else "semantic_controller"
        question = question_match.group(1).strip() if question_match else ""

        if question.upper() == "NONE":
            question = ""

        return {
            "needs_clarification": needs,
            "enough_for_useful_answer": enough,
            "reason": reason,
            "clarification_question": question,
            "raw": raw,
        }

    except Exception as e:
        print("[CLARIFICATION CONTROLLER] skipped:", str(e))
        return {
            "needs_clarification": False,
            "enough_for_useful_answer": True,
            "reason": "semantic_controller_error",
            "clarification_question": "",
            "raw": "",
        }


def semantic_clarification_attachment_check(
    original_query: str,
    clarification_question: str,
    user_reply: str,
) -> dict:
    prompt = f"""
You are a backend clarification attachment checker for WarmGPT.

Do NOT answer the user.

Decide whether the latest user reply is answering the previous clarification question.

Use EXACTLY this format:

IS_ANSWERING: YES or NO
CONFIDENCE: high, medium, or low
RESOLVED_QUERY: one sentence combining the original query and the user's clarification answer
REASON: one short reason

Original user query:
{original_query}

Clarification question asked by assistant:
{clarification_question}

Latest user reply:
{user_reply}

Rules:
- If the reply gives level, skill, jump ability, test level, body detail, boot preference, budget, goal, or any missing detail requested by the clarification, answer YES.
- The reply can be indirect. For example, "I can do double salchow" answers "What level are you?"
- Do NOT treat the reply as a new standalone question unless it clearly asks a new unrelated question.
- Preserve the original query as the main topic.
- The resolved query should keep the original user goal and add the clarification answer as context.
""".strip()

    try:
        raw = run_llm(prompt).strip()
        upper = raw.upper()

        is_answering = "IS_ANSWERING: YES" in upper

        confidence_match = re.search(r"CONFIDENCE:\s*(.*)", raw, re.IGNORECASE)
        resolved_match = re.search(r"RESOLVED_QUERY:\s*(.*)", raw, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.*)", raw, re.IGNORECASE)

        confidence = confidence_match.group(1).strip().lower() if confidence_match else "low"
        resolved_query = resolved_match.group(1).strip() if resolved_match else original_query
        reason = reason_match.group(1).strip() if reason_match else "semantic_attachment"

        return {
            "is_answering": is_answering,
            "confidence": confidence,
            "resolved_query": resolved_query,
            "reason": reason,
            "raw": raw,
        }

    except Exception as e:
        print("[CLARIFICATION ATTACHMENT] skipped:", str(e))
        return {
            "is_answering": False,
            "confidence": "low",
            "resolved_query": original_query,
            "reason": "attachment_checker_error",
            "raw": "",
        }
# -------------------------
# CLARIFICATION LOGIC
# -------------------------

def needs_clarification(query: str, history: list[dict], state: dict | None = None):
    q = query.lower()
    tokens = q.split()

    # -------------------------
    # 0. If user is answering a previous clarification, do NOT clarify again
    # -------------------------
    last_assistant = next(
        (m for m in reversed(history) if m.get("role") == "assistant"),
        None
    )

    if last_assistant:
        last_text = last_assistant.get("content", "").lower()

        looks_like_clarification = (
            "could you tell me" in last_text
            or "what level" in last_text
            or "which one" in last_text
            or last_text.endswith("?")
        )

        if looks_like_clarification and len(tokens) <= 4:
            return False, "answering_previous_clarification", ""

    # -------------------------
    # 1. Hard-coded high-confidence domain ambiguity
    # -------------------------
    if q.strip() == "loop":
        return True, "ambiguous_term_loop", "Do you mean the loop jump or the loop turn?"

    # -------------------------
    # 2. Vague short query
    # -------------------------
    vague_patterns = ["this", "that", "which", "should"]

    if len(tokens) <= 2 and any(p in tokens for p in vague_patterns):
        return True, "vague short query", "What are you referring to?"

    # -------------------------
    # 3. Semantic LLM controller
    # -------------------------
    semantic = semantic_clarification_check(query, history, state)

    if (
        semantic.get("needs_clarification")
        and not semantic.get("enough_for_useful_answer")
    ):
        question = semantic.get("clarification_question") or (
            "Could you share one more detail so I can answer accurately?"
        )

        return True, semantic.get("reason", "semantic_clarification"), question

    # -------------------------
    # 4. Deterministic fallback rules
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

    is_recommendation = (
        "recommend" in q
        or "best" in q
        or "buy" in q
        or "upgrade" in q
        or "new boots" in q
        or "new skates" in q
        or "what skates" in q
        or "what boots" in q
        or "shopping" in q
        or "setup" in q
    )

    is_how_to = (
        "how to" in q
        or "how do i" in q
        or "how should i" in q
        or "practice" in q
        or "improve" in q
        or "fix" in q
    )

    if is_how_to:
        return False, "how_to_safe", ""

    if is_recommendation:
        if not has_skill_signal:
            if not has_prior_skill_signal(history):
                return True, "missing level", "What level are you currently skating at?"

    return False, "sufficient", ""

def classify_query_intent(query: str, history: list[dict]) -> str:
    q = query.lower().strip()

    # -------------------------
    # FRONTEND CHIP MODE HINTS
    # -------------------------

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

def build_intent_profile(query: str, history: list[dict], state: dict | None = None) -> dict:
    """
    Intent v2:
    Uses a thin semantic controller to produce a richer intent profile.
    Keeps primary_intent compatible with the old intent system.
    """

    fallback_intent = classify_query_intent(query, history)

    recent_history = history[-6:] if history else []
    history_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in recent_history
    )

    prompt = f"""
You are a backend intent controller for WarmGPT, a figure skating assistant.

Do NOT answer the user.
Classify the user's latest message for routing and answer planning.

Use EXACTLY this format:

PRIMARY_INTENT: one of how_to, diagnosis, comparison, experience_lookup, default
SECONDARY_INTENTS: comma-separated list or NONE
TOPIC: short topic label
FOCUS_TERMS: comma-separated list or NONE
REASON: short reason

User message:
{query}

Recent conversation:
{history_text}

Detected state:
{state or {}}

Examples:

Examples:

User:
How do I stop scraping on salchow?

Output:
PRIMARY_INTENT: how_to
SECONDARY_INTENTS: diagnosis
TOPIC: salchow_takeoff
FOCUS_TERMS: salchow, scraping, takeoff
REASON: user wants a practical fix for a skating issue


User:
My spin got worse after changing blades.

Output:
PRIMARY_INTENT: diagnosis
SECONDARY_INTENTS: equipment, how_to
TOPIC: blade_change_spin_issue
FOCUS_TERMS: spin, blades, blade change
REASON: user describes a problem likely related to equipment transition and technique adaptation


User:
Edea Chorus vs Ice Fly?

Output:
PRIMARY_INTENT: comparison
SECONDARY_INTENTS: equipment
TOPIC: boot_comparison
FOCUS_TERMS: edea chorus, ice fly, boots
REASON: user is comparing two boot models


User:
Why do my skates feel unstable?

Output:
PRIMARY_INTENT: diagnosis
SECONDARY_INTENTS: experience_lookup
TOPIC: skate_instability
FOCUS_TERMS: skates, unstable
REASON: user asks about a symptom and possible causes

Rules:
- Choose only one PRIMARY_INTENT.
- Use SECONDARY_INTENTS when the query mixes multiple needs.
- Prefer how_to when the user wants steps, drills, or improvement.
- Prefer diagnosis when the user describes a problem, symptom, or something feeling wrong.
- Prefer comparison when the user asks vs, better, difference, or compare.
- Prefer experience_lookup when the user asks why, whether something is normal, common, okay, or bad.
- Use default only if none of the above clearly fit.
- FOCUS_TERMS should contain only semantically important skating concepts.
- Ignore filler words and generic verbs.
- Prefer multi-word skating phrases when appropriate.
- Include skater type or level if important.
- Maximum 5 focus terms.
""".strip()

    try:
        raw = run_llm(prompt).strip()

        primary_match = re.search(r"PRIMARY_INTENT:\s*(.*)", raw, re.IGNORECASE)
        secondary_match = re.search(r"SECONDARY_INTENTS:\s*(.*)", raw, re.IGNORECASE)
        topic_match = re.search(r"TOPIC:\s*(.*)", raw, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.*)", raw, re.IGNORECASE)
        focus_match = re.search(
            r"FOCUS_TERMS:\s*(.*)",
            raw,
            re.IGNORECASE
        )


        primary = primary_match.group(1).strip() if primary_match else fallback_intent
        primary = primary.lower()

        allowed = {"how_to", "diagnosis", "comparison", "experience_lookup", "default"}
        if primary not in allowed:
            primary = fallback_intent

        secondary_raw = secondary_match.group(1).strip() if secondary_match else "NONE"

        if secondary_raw.upper() == "NONE":
            secondary = []
        else:
            secondary = [
                x.strip().lower()
                for x in secondary_raw.split(",")
                if x.strip()
            ]

        focus_raw = (
            focus_match.group(1).strip()
            if focus_match else "NONE"
        )

        if focus_raw.upper() == "NONE":
            focus_terms = []
        else:
            focus_terms = [
                x.strip().lower()
                for x in focus_raw.split(",")
                if x.strip()
            ]

        topic = topic_match.group(1).strip() if topic_match else "unknown"
        reason = reason_match.group(1).strip() if reason_match else "semantic_intent"

        return {
            "primary_intent": primary,
            "secondary_intents": secondary,
            "topic": topic,
            "focus_terms": focus_terms,
            "reason": reason,
            "raw": raw,
            "fallback_intent": fallback_intent,
        }

    except Exception as e:
        print("[INTENT CONTROLLER] skipped:", str(e))
        return {
            "primary_intent": fallback_intent,
            "secondary_intents": [],
            "topic": "unknown",
            "reason": "intent_controller_error",
            "raw": "",
            "fallback_intent": fallback_intent,
        }


def build_answer_plan(
    query: str,
    intent: str,
    state: dict,
    history: list[dict],
    clarify: bool = False,
    intent_profile: dict | None = None,
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

    # -------------------------
    # 6. Intent v2 enrichment
    # -------------------------
    if intent_profile:
        secondary = intent_profile.get("secondary_intents", [])
        topic = intent_profile.get("topic", "unknown")

        plan["intent_profile"] = intent_profile
        plan["topic"] = topic

        if "equipment" in secondary:
            plan["structure"].append("equipment note")

        if "how_to" in secondary and "what to try" not in plan["structure"]:
            plan["structure"].append("what to try")

        if "diagnosis" in secondary and "possible causes" not in plan["structure"]:
            plan["structure"].insert(0, "possible causes")

    return plan

# -------------------------
# SMART FOLLOW-UP LOGIC
# -------------------------

def build_followup_decision(
    query: str,
    intent: str,
    state: dict,
    agent_trace: dict,
    reply: str,
) -> dict:
    """
    Decide whether to generate a follow-up and why.
    Follow-up is now state-aware, not only intent-based.
    """

    # 1. Too short: no follow-up
    if len(reply.split()) < 25:
        return {
            "generate": False,
            "reason": "reply_too_short",
            "type": "none",
        }

    clarification = agent_trace.get("clarification", {})
    clarification_state = agent_trace.get("clarification_state", {})

    # 2. Clarification and follow-up are mutually exclusive
    if (
        clarification.get("triggered")
        or clarification_state.get("force_answer")
        or clarification_state.get("count", 0) > 0
    ):
        return {
            "generate": False,
            "reason": "clarification_active",
            "type": "none",
        }

    retrieval = agent_trace.get("retrieval", {})
    fallback = agent_trace.get("fallback", {})
    repair = agent_trace.get("repair", {})
    query_profile = agent_trace.get("query_profile", {})

    retrieval_status = retrieval.get("status")
    confidence = retrieval.get("confidence")
    retrieval_reason = retrieval.get("reason")

    # 3. Suppress when conversation is already resolved
    if (
        retrieval_status == "good"
        and confidence == "high"
        and not repair.get("triggered")
        and not fallback.get("triggered")
    ):
        return {
            "generate": False,
            "reason": "resolved_high_confidence",
            "type": "none",
        }

    # 4. Retrieval ambiguity
    if retrieval_reason == "ambiguous_scores" or confidence == "medium":
        return {
            "generate": True,
            "reason": "retrieval_ambiguity",
            "type": "diagnostic_narrowing",
        }

    # 5. Fallback recovery
    if fallback.get("triggered"):
        return {
            "generate": True,
            "reason": "fallback_recovery",
            "type": "retrieval_recovery",
        }

    # 6. Repair recovery
    if repair.get("triggered"):
        return {
            "generate": True,
            "reason": repair.get("reason", "repair_recovery"),
            "type": "repair_recovery",
        }

    # 7. Context continuation
    if query_profile.get("used_history_merge"):
        return {
            "generate": True,
            "reason": "used_history_merge",
            "type": "context_continuation",
        }

    # 8. Missing user state
    if state.get("skill_level") == "unknown" and intent in ["how_to", "diagnosis", "comparison"]:
        return {
            "generate": True,
            "reason": "missing_user_state",
            "type": "state_collection",
        }

    # 9. Default useful coaching continuation
    if intent in ["how_to", "diagnosis", "comparison"]:
        return {
            "generate": True,
            "reason": "coaching_continuation",
            "type": "progression_coaching",
        }

    return {
        "generate": False,
        "reason": "not_needed",
        "type": "none",
    }


def build_followup_prompt(
    query: str,
    answer: str,
    intent: str,
    state: dict,
    history: list[dict],
    followup_decision: dict | None = None,
    agent_trace: dict | None = None,
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

Follow-up decision:
- reason: {(followup_decision or {}).get("reason")}
- type: {(followup_decision or {}).get("type")}

System state:
- retrieval_status: {(agent_trace or {}).get("retrieval", {}).get("status")}
- retrieval_reason: {(agent_trace or {}).get("retrieval", {}).get("reason")}
- confidence: {(agent_trace or {}).get("retrieval", {}).get("confidence")}
- fallback_triggered: {(agent_trace or {}).get("fallback", {}).get("triggered")}
- repair_triggered: {(agent_trace or {}).get("repair", {}).get("triggered")}
- repair_reason: {(agent_trace or {}).get("repair", {}).get("reason")}
- used_history_merge: {(agent_trace or {}).get("query_profile", {}).get("used_history_merge")}

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
Follow-up strategy rules:
- If type is diagnostic_narrowing, ask about the most useful missing symptom/detail.
- If type is retrieval_recovery, ask for the missing detail that would make the answer more specific.
- If type is repair_recovery, ask about the dimension that remains most uncertain.
- If type is context_continuation, continue the same thread instead of restarting the topic.
- If type is state_collection, ask for skating level, equipment, or goal only if genuinely useful.
- If type is progression_coaching, ask what part breaks down next.

Good examples (these are just some examples, do not just copy these for all questions):
- Where does it usually break down for you: takeoff, landing, or the setup edge?
- Are you working on this for practice, a test, or competition?
- Does it happen more when you are tired or even at the start of the session?

Output only the follow-up question or NONE.
""".strip()