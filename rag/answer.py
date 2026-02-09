from pathlib import Path

from rag.retriever import retrieve
from rag.prompt_builder import build_llm_input
from rag.llm import run_ollama
from rag.intents import (
    is_blank,
    is_social_message,
    handle_social_message,
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"


def answer_question(question: str, history: list[dict]) -> str:
    # Blank input guard
    if is_blank(question):
        return "Please ask a valid figure skating related question."

    # Social / small talk (NO RAG)
    if is_social_message(question):
        return handle_social_message(question)

    # Real figure skating question → RAG
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    docs = retrieve(question)

    llm_input = build_llm_input(
        prompt=prompt,
        question=question,
        docs=docs,
        history=history,
    )

    return run_ollama(llm_input)
