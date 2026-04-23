from rag.level_taxonomy import detect_legacy_term
from backend.agent import build_skater_state


def get_intent_instruction(intent: str) -> str:
    if intent == "how_to":
        return (
            "Task mode: actionable coaching.\n"
            "Give concrete steps the user can try.\n"
            "Prioritize practical technique advice."
        )

    if intent == "comparison":
        return (
            "Task mode: comparison.\n"
            "Compare the main options or viewpoints.\n"
            "Use this structure:\n"
            "1. Common ground\n"
            "2. Key differences\n"
            "3. Recommendation"
        )

    if intent == "diagnosis":
        return (
            "Task mode: diagnosis.\n"
            "Analyze the user's issue and structure the answer as follows:\n\n"
            "Likely causes:\n"
            "- List 2–3 plausible causes\n\n"
            "What to try:\n"
            "- Give concrete, practical fixes\n\n"
            "Notes:\n"
            "- Mention uncertainty or when it may differ\n"
        )

    if intent == "experience_lookup":
        return (
            "Task mode: explanation.\n"
            "Explain what commonly causes this issue or situation.\n"
            "Summarize patterns clearly."
        )

    return "Task mode: standard skating answer."


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
        lines.append("Include at most 1-2 drills.")
        lines.append("Avoid turning this into a long lecture.")

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
        lines.append("Keep the answer brief.")
        lines.append("Prefer one short paragraph or a very short structured answer.")

    elif depth == "detailed":
        lines.append("You may give a fuller answer, but stay organized and practical.")

    else:
        lines.append("Keep the answer moderately detailed.")

    if use_context:
        lines.append("Use the earlier conversation context when it is relevant.")

    if avoid_repetition:
        lines.append("Do not repeat basics that were already discussed earlier.")

    if structure:
        lines.append("Preferred structure: " + " -> ".join(structure))

    return "\n".join(lines)


def build_llm_input(
    prompt: str,
    question: str,
    docs: list[str],
    history: list[dict],
    intent: str = "default",
    answer_plan: dict | None = None,
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

    parts.append("User skating profile:")
    parts.append(f"- Skill level: {state.get('skill_level')}")
    parts.append(f"- Signals: {state.get('signals')}")
    parts.append(f"- Jump level: {state.get('jump_level')}")
    parts.append(f"- Body: {state.get('height_class')}, {state.get('weight_class')}")
    parts.append(f"- Experience: {state.get('experience_type')}")
    parts.append("")

    parts.append(
        "Important rules:\n"
        "- Do NOT treat Axel or above as beginner.\n"
        "- Recommend appropriate stiffness for heavier skaters.\n"
        "- Avoid beginner equipment unless clearly beginner.\n"
    )
    parts.append("")

    parts.append(get_intent_instruction(intent))
    parts.append("")

    parts.append("Answer plan:")
    parts.append(get_answer_plan_instruction(answer_plan))
    parts.append("")

    # Final instruction
    parts.append(
        "Reply naturally as an experienced figure skater.\n"
        "Do not mention sources, forums, or internal processing."
    )

    return "\n".join(parts)