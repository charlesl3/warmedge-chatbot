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
            "Infer likely causes from the user's described symptoms.\n"
            "Do not overstate certainty.\n"
            "List likely causes and what to try."
        )

    if intent == "experience_lookup":
        return (
            "Task mode: explanation.\n"
            "Explain what commonly causes this issue or situation.\n"
            "Summarize patterns clearly."
        )

    return "Task mode: standard skating answer."

def build_llm_input(
    prompt: str,
    question: str,
    docs: list[str],
    history: list[dict],
    intent: str = "default",
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
    state = build_skater_state(question)  # you'll pass or compute this

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

    # Final instruction
    parts.append(
        "Reply naturally as an experienced figure skater.\n"
        "Do not mention sources, forums, or internal processing."
    )

    return "\n".join(parts)