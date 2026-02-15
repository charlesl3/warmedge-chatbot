from rag.level_taxonomy import detect_legacy_term


def build_llm_input(prompt: str, question: str, docs: list[dict], history: list[dict]) -> str:
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
                f'(including minor spelling variations or slightly different phrasing), '
                f'you must clarify that it was renamed to '
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
                f'you must clarify that it was renamed to '
                f'"{normalization["mapped_level"]}" in July 2023 '
                f'and use the current level name in your response.'
            )
            parts.append("")

        # ---- Adult context ----
        if normalization.get("is_adult"):
            parts.append(
                "Note: This question concerns the Adult test track. "
                "Interpret levels using the Adult level structure."
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
    parts.append("Relevant skating experiences:")
    parts.append("")

    for d in docs:
        parts.append(d["text"])
        parts.append("")

    # Final instruction
    parts.append(
        "Reply naturally as an experienced figure skater.\n"
        "Do not mention sources, forums, or internal processing."
    )

    return "\n".join(parts)
