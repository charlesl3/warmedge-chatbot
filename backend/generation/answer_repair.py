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

    if len(answer.split()) < 80 and intent in ["how_to", "diagnosis", "comparison"]:
        return {
            "status": "weak",
            "reason": "too_short",
            "should_repair": True,
        }

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

        if coverage < 0.5:
            return {
                "status": "weak",
                "reason": "missing_query_focus",
                "missing_focus_terms": [
                    t for t in focus_terms
                    if t not in matched_terms
                ],
                "should_repair": True,
            }

    if intent == "diagnosis":
        expected = ["cause", "try"]
        missing = [x for x in expected if x not in lower]
        if missing:
            return {
                "status": "weak",
                "reason": "missing_diagnosis_structure",
                "missing": missing,
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

The previous answer was weak for this reason:
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

Repair rules:
- Be more specific and practical.
- Keep the answer grounded in the retrieved context.
- If this is diagnosis, clearly separate likely causes and what to try.
- If this is how-to, give concrete steps.
- Avoid vague filler like "it depends" unless you explain what it depends on.
- Preserve the important semantic focus terms from the original query.
""".strip()

    return run_llm(prompt)