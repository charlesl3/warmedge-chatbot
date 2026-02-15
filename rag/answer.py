# from pathlib import Path
#
# from rag.retriever import retrieve
# from rag.prompt_builder import build_llm_input
# from rag.llm import run_ollama
# from rag.intents import (
#     is_blank,
#     is_social_message,
#     handle_social_message,
# )
#
# PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"
# TOP_K = 6
#
#
# def answer_question(question: str, history: list[dict]) -> str:
#     if is_blank(question):
#         return "Please ask a valid figure skating related question."
#
#     if is_social_message(question):
#         return handle_social_message(question)
#
#     prompt = PROMPT_PATH.read_text(encoding="utf-8")
#     docs = retrieve(question, k=TOP_K)
#
#     llm_input = build_llm_input(
#         prompt=prompt,
#         question=question,
#         docs=docs,
#         history=history,
#     )
#
#     return run_ollama(llm_input)


from pathlib import Path

from rag.retriever import retrieve
from rag.prompt_builder import build_llm_input
from rag.llm import run_llm
from rag.intents import (
    is_blank,
    is_social_message,
    handle_social_message,
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"
TOP_K = 6
MIN_TOP_SCORE = 0.15


# --------------------------------------------------
# Context Inference (Lightweight Session Logic)
# --------------------------------------------------
def infer_missing_context(question: str, history: list[dict]) -> str:
    q_lower = question.lower()

    # If question already specifies track or discipline, do nothing
    explicit_tokens = ["adult", "standard", "non-adult", "singles", "skating skills"]
    if any(token in q_lower for token in explicit_tokens):
        return question

    # Find last user message
    previous_user_msg = None
    for msg in reversed(history):
        if msg.get("role") == "user":
            previous_user_msg = msg.get("content", "").lower()
            break

    if not previous_user_msg:
        return question

    inferred_tokens = []

    if "adult" in previous_user_msg:
        inferred_tokens.append("adult")
    elif "standard" in previous_user_msg or "non-adult" in previous_user_msg:
        inferred_tokens.append("standard")

    if "singles" in previous_user_msg:
        inferred_tokens.append("singles")
    elif "skating skills" in previous_user_msg:
        inferred_tokens.append("skating skills")

    if inferred_tokens:
        return " ".join(inferred_tokens) + " " + question

    return question


def answer_question(question: str, history: list[dict]) -> str:
    # --------------------------------------------------
    # 1. Basic validation
    # --------------------------------------------------
    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    # --------------------------------------------------
    # 2. Infer Missing Context
    # --------------------------------------------------
    question = infer_missing_context(question, history)

    # --------------------------------------------------
    # 3. Retrieval
    # --------------------------------------------------
    retrieval = retrieve(question, k=TOP_K)

    if retrieval["top_score"] is None:
        return "Could you clarify your question a bit more?"

    if retrieval["top_score"] < MIN_TOP_SCORE:
        return (
            "I am not confident I found reliable information for that question. "
            "Could you clarify what you mean?"
        )

    docs = retrieval["results"]

    # --------------------------------------------------
    # 4. Track Disambiguation (Adult vs Standard)
    # --------------------------------------------------
    q_lower = question.lower()

    if "adult" in q_lower:
        filtered_docs = [
            d for d in docs
            if "_adult" in d.get("source_path", "")
        ]

    elif "standard" in q_lower or "non-adult" in q_lower:
        filtered_docs = [
            d for d in docs
            if "_standard" in d.get("source_path", "")
        ]

    else:
        # Track not specified — allow both
        filtered_docs = [
            d for d in docs
            if "_adult" in d.get("source_path", "")
            or "_standard" in d.get("source_path", "")
        ]

    # Fallback safety
    if not filtered_docs:
        filtered_docs = docs

    docs = filtered_docs

    # --------------------------------------------------
    # 5. Load base prompt
    # --------------------------------------------------
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    STYLE_HINT = (
        "Answer concisely (3–6 sentences). "
        "Do not assume emotion. "
        "Do not assume the user's blade, boot, or skate model unless explicitly stated. "
        "If unsure, ask one focused clarification question instead of guessing."
    )

    prompt = STYLE_HINT + "\n\n" + prompt

    # --------------------------------------------------
    # 6. Build final LLM input
    # --------------------------------------------------
    llm_input = build_llm_input(
        prompt=prompt,
        question=question,
        docs=docs,
        history=history,
    )

    # --------------------------------------------------
    # 7. Call LLM
    # --------------------------------------------------
    response = run_llm(llm_input)

    return response
