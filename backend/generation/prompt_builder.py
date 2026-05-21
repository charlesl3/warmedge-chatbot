from backend.taxonomy.level_taxonomy import detect_legacy_term
from backend.agents.agent import build_skater_state

# def get_intent_instruction(intent: str) -> str:
#     if intent == "how_to":
#         return (
#             "Task mode: actionable coaching.\n"
#             "Give concrete steps the user can try.\n"
#             "Prioritize practical technique advice."
#         )
#
#     if intent == "comparison":
#         return (
#             "Task mode: comparison.\n"
#             "Compare the main options or viewpoints.\n"
#             "Use this structure:\n"
#             "1. Common ground\n"
#             "2. Key differences\n"
#             "3. Recommendation"
#         )
#
#     if intent == "diagnosis":
#         return (
#             "Task mode: diagnosis.\n"
#             "Analyze the user's issue and structure the answer as follows:\n\n"
#             "Likely causes:\n"
#             "- List 2–3 plausible causes\n\n"
#             "What to try:\n"
#             "- Give concrete, practical fixes\n\n"
#             "Notes:\n"
#             "- Mention uncertainty or when it may differ\n"
#         )
#
#     if intent == "experience_lookup":
#         return (
#             "Task mode: explanation.\n"
#             "Explain what commonly causes this issue or situation.\n"
#             "Summarize patterns clearly."
#         )
#
#     return "Task mode: standard skating answer."


def get_answer_plan_instruction(answer_plan: dict | None) -> str:
    if not answer_plan:
        return "Answer clearly and practically."

    mode = answer_plan.get("mode", "standard")
    depth = answer_plan.get("depth", "medium")
    use_context = answer_plan.get("use_context", False)
    avoid_repetition = answer_plan.get("avoid_repetition", False)
    structure = answer_plan.get("structure", [])

    lines = []

    if mode == "clarification":
        lines.append("Do not answer the full question yet.")
        lines.append("Ask one short, precise clarifying follow-up.")
        return "\n".join(lines)

    if mode == "coaching":
        lines.append("Treat this as a practical coaching question.")
        lines.append("Start with the most likely cause or key issue.")
        lines.append("Then give concrete steps the user can try.")
        lines.append("Prioritize useful correction cues and practical drills.")
        lines.append("Stay organized and practical.")

    elif mode == "diagnosis":
        lines.append("Treat this as a diagnosis-style question.")
        lines.append("List 2-4 plausible causes.")
        lines.append("For each cause, mention how the user can check it.")
        lines.append("Then suggest what to try.")
        lines.append("Do not pretend certainty if multiple causes are possible.")

    elif mode == "comparison":
        lines.append("Treat this as a comparison question.")
        lines.append("Use a structured comparison.")
        lines.append("Cover common ground, key differences, and recommendation.")

    elif mode == "explanation":
        lines.append("Treat this as an explanation question.")
        lines.append("Answer directly first.")
        lines.append("Then explain why.")
        lines.append("End with a practical skating takeaway.")

    else:
        lines.append("Answer directly and practically.")

    if depth == "short":
        lines.append("Give a brief but complete answer.")

    elif depth == "detailed":
        lines.append(
            "Give a well-developed, well-organized, and long answer with practical nuance, reasoning, and useful examples where appropriate."
        )

    else:
        lines.append("Give a thoughtful, well-organized, and moderately developed answer.")

    if use_context:
        lines.append("Use the earlier conversation context when it is relevant.")

    if avoid_repetition:
        lines.append("Do not repeat basics that were already discussed earlier.")

    if structure:
        lines.append("Preferred structure: " + " -> ".join(structure))

    return "\n".join(lines)

def get_confidence_instruction(confidence: str | None) -> str:
    if confidence == "high":
        return (
            "Confidence behavior: high.\n"
            "Answer directly and confidently.\n"
            "Do not mention confidence."
        )

    if confidence == "medium":
        return (
            "Confidence behavior: medium.\n"
            "Give the most likely answer, but acknowledge that there may be more than one cause or interpretation.\n"
            "Do not mention confidence."
        )

    if confidence == "low":
        return (
            "Confidence behavior: low.\n"
            "Avoid sounding overly certain.\n"
            "Give safe, general guidance first.\n"
            "If the question depends on missing details, ask one precise follow-up question.\n"
            "Do not mention confidence."
        )

    return (
        "Confidence behavior: normal.\n"
        "Answer naturally.\n"
        "Do not mention confidence."
    )

def build_llm_input(
    prompt: str,
    question: str,
    docs: list[str],
    history: list[dict],
    intent: str = "default",
    answer_plan: dict | None = None,
    confidence: str | None = None,
user_profile: dict | None = None,
) -> str:

    parts = []

    # Base system prompt
    parts.append(prompt.strip())
    parts.append("")

    normalization = detect_legacy_term(question)

    if normalization:

        # ---- Track rename enforcement ----
        if normalization.get("legacy_track"):
            parts.append(
                f'Important: If the user refers to the legacy test name '
                f'"{normalization["legacy_track"]}" '
                f'(including minor spelling variations), '
                f'clarify that it was renamed to '
                f'"{normalization["mapped_track"]}" in July 2023 '
                f'and use the current name in your response.'
            )
            parts.append("")

        # ---- Level rename enforcement ----
        if normalization.get("legacy_level"):
            parts.append(
                f'Important: If the user refers to the legacy level '
                f'"{normalization["legacy_level"]}" '
                f'(including minor spelling variations), '
                f'clarify that it was renamed to '
                f'"{normalization["mapped_level"]}" in July 2023 '
                f'and use the current level name in your response.'
            )
            parts.append("")

        # ---- Adult context ----
        if normalization.get("is_adult"):
            parts.append(
                "Note: This question concerns the Adult test track. "
                "Interpret levels using the Adult structure."
            )
            parts.append("")

    # Conversation history
    if history:
        parts.append("Conversation so far:")
        for m in history:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("")

    # User question
    parts.append("User question:")
    parts.append(question.strip())
    parts.append("")

    # Retrieved documents
    parts.append("Relevant skating knowledge:")
    parts.append("")

    for text in docs:
        parts.append(text)
        parts.append("")

    # ---- Inject inferred state ----
    state = build_skater_state(question)

    parts.append("Current query-derived skating context:")
    parts.append(f"- Query-implied level: {state.get('skill_level')}")
    parts.append(f"- Signals: {state.get('signals')}")
    parts.append(f"- Jump level: {state.get('jump_level')}")
    parts.append(f"- Body: {state.get('height_class')}, {state.get('weight_class')}")
    parts.append(f"- Experience: {state.get('experience_type')}")
    parts.append("")

    parts.append(
        "Important skating assumptions:\n"
        "- Axel or above is not beginner level.\n"
        "- Equipment recommendations should match realistic support needs.\n"
    )
    parts.append("")

    # parts.append(get_intent_instruction(intent))
    parts.append("")

    parts.append("Answer plan:")
    parts.append(get_answer_plan_instruction(answer_plan))
    # -------------------------
    parts.append(
        "Formatting rules:\n"
        "- Prefer clean Markdown headings and bullet points.\n"
        "- Avoid Markdown tables unless explicitly requested.\n"
        "- Avoid raw HTML tags like <br>.\n"
        "- Use short sections instead of large tables.\n"
        "- Keep formatting stable for chat-style rendering.\n"
    )
    parts.append("")

    # -------------------------
    # QUICK DRILL (NEW)
    # -------------------------
    drill_lines = []

    if intent in ["how_to", "diagnosis"]:
        drill_lines.append(
            "Include practical drills or correction cues when they genuinely help."
        )

    parts.append("Drill:")
    parts.append("\n".join(drill_lines))
    parts.append("")

    parts.append("Confidence behavior:")
    parts.append(get_confidence_instruction(confidence))
    parts.append("")

    # -------------------------
    # PROFILE PERSONALIZATION
    # -------------------------

    if user_profile:

        skater_level = (
            user_profile.get("skater_level")
            or "beginner"
        )

        first_name = (
            user_profile.get("first_name")
            or "Skater"
        )

        parts.append("Persistent user profile for response style only:")
        highest_jump = (
                user_profile.get("highest_jump")
                or "unknown"
        )

        highest_test_level = (
                user_profile.get("highest_test_level")
                or "unknown"
        )

        parts.append(f"- Highest jump: {highest_jump}")
        parts.append(f"- Highest test level: {highest_test_level}")
        parts.append(
            "- Use this profile to tune explanation depth and tone, not to override the current query context.")
        parts.append("")
        parts.append(
            "Use highest jump and test level as supporting context "
            "for the user's likely skating background and technical familiarity."
        )

        parts.append(
            "Do not rigidly assume ability from these fields alone."
        )

        parts.append("")

        if skater_level == "beginner":

            parts.append(
                "Explanation style:\n"
                "- Explain skating terminology clearly.\n"
                "- Avoid assuming advanced skating knowledge.\n"
                "- Prioritize safety and basic understanding.\n"
                "- You may smartly encourage the skater to enjoy their figure skating journey as a beginner, when it is good to do so\n"
            )

        elif skater_level == "intermediate":

            parts.append(
                "Explanation style:\n"
                "- Use moderate skating terminology naturally.\n"
                "- Balance explanation and technical detail.\n"
                "- You may smartly acknowledge that the skater is an intermediate-level skater and encourage them to be confident towards advanced levels, when it is good to do so\n"
            )

        elif skater_level == "advanced":

            parts.append(
                "Explanation style:\n"
                "- User is an advanced skater.\n"
                "- Use technical skating terminology naturally.\n"
                "- Avoid oversimplifying mechanics.\n"
                "- Include nuanced technique reasoning when useful.\n"
                "- You may smartly acknowledge or praise that the skater is an advanced-level skater, when it is good to do so\n"
            )

        elif skater_level == "non_skater":

            parts.append(
                "Explanation style:\n"
                "- Do not assume skating background.\n"
                "- Explain tests, jumps, and skating structure clearly.\n"
                "- Use beginner-friendly language.\n"
                "- You may smartly thank or encourage them to know more about figure skating, when it is good to do so\n"
            )

        parts.append("")

    # Final instruction
    parts.append(
        "Profile/context distinction:\n"
        "- Persistent user profile controls response style and assumed explanation depth.\n"
        "- Current query-derived context controls the specific skating situation being answered.\n"
        "- If they differ, respect both: an advanced skater may ask a basic drill question, and a beginner may ask about advanced topics.\n"
    )
    parts.append("")

    parts.append(
        "Reply naturally as an experienced figure skater.\n"
        "Do not mention sources, forums, or internal processing."
    )

    return "\n".join(parts)