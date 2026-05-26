from backend.generation.llm import run_llm


def evaluate_answer_quality(
    answer: str,
    docs: list[str],
    intent: str,
    answer_plan: dict | None = None,
    intent_profile: dict | None = None,
) -> dict:

    weak_phrases = [
        "it depends",
        "not sure",
        "generally",
        "in some cases",
    ]

    lower = answer.lower()
    focus_terms = (
            intent_profile or {}
    ).get("focus_terms", [])
    has_weak_phrase = any(p in lower for p in weak_phrases)


    if has_weak_phrase:
        return {
            "status": "weak",
            "reason": "vague_language",
            "should_repair": True,
        }
    if focus_terms:
        matched_terms = [
            term for term in focus_terms
            if term.lower() in lower
        ]

        coverage = len(matched_terms) / max(len(focus_terms), 1)

        if coverage < 0.25:
            return {
                "status": "weak",
                "reason": "missing_query_focus",
                "missing_focus_terms": [
                    t for t in focus_terms
                    if t not in matched_terms
                ],
                "should_repair": True,
            }

    return {
        "status": "good",
        "reason": "sufficient",
        "should_repair": False,
    }


def repair_answer(
    original_answer: str,
    question: str,
    docs: list[str],
    intent: str,
    answer_plan: dict | None,
    intent_profile: dict | None,
    reason: str,
) -> str:
    """
    Repair modifies generation behavior only.
    It does not modify query.
    It does not rerun retrieval.
    """

    context = "\n\n".join(docs)
    focus_terms = (
            intent_profile or {}
    ).get("focus_terms", [])

    prompt = f"""
You are WarmGPT, a practical figure skating assistant.

The previous answer could be improved in this area:
{reason}

Rewrite the answer using the SAME retrieved context.
Do not request new retrieval.
Do not mention sources or internal logic.

User question:
{question}

Retrieved context:
{context}

Previous answer:
{original_answer}

Intent:
{intent}

Answer plan:
{answer_plan or {}}

Important semantic focus terms:
{focus_terms}

Improvement goals:
- Make the answer more useful and complete.
- Stay grounded in the retrieved skating context.
- Expand practical reasoning when helpful.
- Keep the answer natural and conversational.
- Avoid vague filler or generic statements.
- Preserve important skating-specific details.
""".strip()

    return run_llm(prompt)