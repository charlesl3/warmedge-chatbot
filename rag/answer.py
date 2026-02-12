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
MIN_TOP_SCORE = 0.30  # tune later if needed


def answer_question(question: str, history: list[dict]) -> str:
    # --------------------------------------------------
    # 1. Basic validation
    # --------------------------------------------------
    if is_blank(question):
        return "Please ask a valid figure skating related question."

    if is_social_message(question):
        return handle_social_message(question)

    # --------------------------------------------------
    # 2. Retrieval
    # --------------------------------------------------
    retrieval = retrieve(question, k=TOP_K)

    # Low-information query (blocked inside retriever)
    if retrieval["top_score"] is None:
        return "Could you clarify your question a bit more?"

    # Weak retrieval confidence
    if retrieval["top_score"] < MIN_TOP_SCORE:
        return (
            "I am not confident I found reliable information for that question. "
            "Could you clarify what you mean?"
        )

    docs = retrieval["results"]

    # --------------------------------------------------
    # 3. Load base prompt
    # --------------------------------------------------
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    # --------------------------------------------------
    # 4. Inject style control layer (concise + no assumptions)
    # --------------------------------------------------
    STYLE_HINT = (
        "Answer concisely (3–6 sentences). "
        "Do not assume emotion. "
        "Do not assume the user's blade, boot, or skate model unless explicitly stated. "
        "If unsure, ask one focused clarification question instead of guessing."
    )

    prompt = STYLE_HINT + "\n\n" + prompt

    # --------------------------------------------------
    # 5. Build final LLM input
    # --------------------------------------------------
    llm_input = build_llm_input(
        prompt=prompt,
        question=question,
        docs=docs,
        history=history,
    )

    # --------------------------------------------------
    # 6. Call LLM
    # --------------------------------------------------
    response = run_llm(llm_input)

    return response