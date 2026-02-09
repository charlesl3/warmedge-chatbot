from typing import List, Dict
from rag.answer import answer_question


def chat_api(
    message: str,
    history: List[Dict[str, str]],
) -> Dict[str, object]:
    """
    Web-ready chat interface.
    This mimics an HTTP POST handler without running a server.
    """

    reply = answer_question(
        question=message,
        history=history,
    )

    return {
        "reply": reply,
        "history": history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ],
    }
