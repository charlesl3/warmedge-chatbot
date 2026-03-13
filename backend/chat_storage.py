import json
from pathlib import Path

CHAT_FILE = Path("data/chat_history.json")


def ensure_chat_file():
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_FILE.exists():
        CHAT_FILE.write_text("{}", encoding="utf-8")


def load_chats():
    ensure_chat_file()
    with CHAT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_chats(chats):
    ensure_chat_file()
    with CHAT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)