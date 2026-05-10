from backend.generation.llm import run_llm


def evaluate_answer_quality(answer: str, docs: list[str], intent: str, answer_plan: dict | None = None) -> dict:
    weak_phrases = [
        "it depends",
        "not sure",
        "generally",
        "in some cases",
    ]

    lower = answer.lower()
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
    reason: str,
) -> str:
    """
    Repair modifies generation behavior only.
    It does not modify query.
    It does not rerun retrieval.
    """

    context = "\n\n".join(docs)

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

Repair rules:
- Be more specific and practical.
- Keep the answer grounded in the retrieved context.
- If this is diagnosis, clearly separate likely causes and what to try.
- If this is how-to, give concrete steps.
- Avoid vague filler like "it depends" unless you explain what it depends on.
""".strip()

    return run_llm(prompt)