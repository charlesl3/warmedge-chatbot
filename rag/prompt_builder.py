def build_llm_input(prompt: str, question: str, docs: list[dict], history: list[dict]) -> str:
    parts = []

    parts.append(prompt.strip())
    parts.append("")

    if history:
        parts.append("Conversation so far:")
        for m in history:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("")

    parts.append("User question:")
    parts.append(question.strip())
    parts.append("")

    parts.append("Relevant skating experiences:")
    parts.append("")

    for d in docs:
        parts.append(d["text"])
        parts.append("")

    parts.append(
        "Reply naturally as an experienced figure skater.\n"
        "Do not mention sources, forums, or internal processing."
    )

    return "\n".join(parts)
