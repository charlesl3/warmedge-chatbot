import json
from pathlib import Path
from threading import Lock

CHAT_FILE = Path("data/chat_history.json")

# Prevent concurrent writes
CHAT_LOCK = Lock()


def ensure_chat_file():
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_FILE.exists():
        CHAT_FILE.write_text("{}", encoding="utf-8")


def load_chats():
    ensure_chat_file()

    try:
        with CHAT_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Recover if file becomes corrupted
        return {}


def save_chats(chats):
    ensure_chat_file()

    # Prevent two requests writing simultaneously
    with CHAT_LOCK:
        with CHAT_FILE.open("w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)


# NEW: ensure session exists and migrate old format
def ensure_chat_session(chats, session_id, first_user_message="New chat"):
    """
    Make sure the chat session exists and follows the new structure:

    {
        session_id: {
            "title": "...",
            "messages": [...]
        }
    }
    """

    if session_id not in chats:
        chats[session_id] = {
            "title": first_user_message[:40],
            "messages": []
        }

    # Handle migration from old format (list of messages)
    elif isinstance(chats[session_id], list):

        old_messages = chats[session_id]

        title = "New chat"
        for msg in old_messages:
            if msg.get("role") == "user":
                title = msg.get("content", "New chat")[:40]
                break

        chats[session_id] = {
            "title": title,
            "messages": old_messages
        }

    return chats