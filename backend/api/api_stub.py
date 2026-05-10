from typing import List, Dict

from backend.generation.answer import answer_question
from backend.taxonomy.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

def chat_api(
    message: str,
    history: List[Dict[str, str]],
) -> Dict[str, object]:
    """
    Web-ready chat interface.
    Acts as the controller layer for web usage.
    """

    # 1️⃣ Blank input
    if is_blank(message):
        reply = "Please ask a valid figure skating related question."
        return {
            "reply": reply,
            "history": history,
            "end": False,
        }

    # 2️⃣ Social / small talk (NO RAG)
    if is_social_message(message):
        reply = handle_social_message(message)
        return {
            "reply": reply,
            "history": history,
            "end": is_farewell(message),
        }

    # 3️⃣ Real skating question → RAG brain
    result = answer_question(
        question=message,
        history=history,
    )

    reply = result.get("reply", "") if isinstance(result, dict) else result

    return {
        "reply": reply,
        "history": history
        + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
        "end": False,
    }
