import re


def is_blank(text: str) -> bool:
    return text is None or len(text.strip()) == 0


# -------------------------
# GREETING / SMALL TALK
# -------------------------

_GREETING_PATTERNS = [
    r"^hi$",
    r"^hello$",
    r"^hey$",
    r"^hiya$",
    r"^hi there$",
    r"^heya$",
    r"^hola$",
    r"^good (morning|afternoon|evening)$",
    r"^how are you$",
    r"^how's it going$",
    r"^hows it going$",
    r"^what's up$",
    r"^whats up$",
    r"^nice to meet you$",
]


_THANKS_PATTERNS = [
    r"^thanks$",
    r"^thank you$",
    r"^thanks a lot$",
    r"^thx$",
    r"^much appreciated$",
    r"^appreciate it$",
]


_FAREWELL_PATTERNS = [
    r"^bye$",
    r"^byebye$",
    r"^goodbye$",
    r"^see you$",
    r"^see ya$",
    r"^talk to you later$",
    r"^ttyl$",
    r"^catch you later$",
]


def _match_any(patterns: list[str], text: str) -> bool:
    for p in patterns:
        if re.fullmatch(p, text):
            return True
    return False


def is_social_message(text: str) -> bool:
    """
    Returns True if the input is clearly small talk / politeness
    and should NOT go through RAG.
    """
    if not text:
        return False

    t = text.strip().lower()

    # Avoid swallowing real questions
    if len(t.split()) > 6:
        return False

    return (
        _match_any(_GREETING_PATTERNS, t)
        or _match_any(_THANKS_PATTERNS, t)
        or _match_any(_FAREWELL_PATTERNS, t)
    )


def is_farewell(text: str) -> bool:
    if not text:
        return False
    return _match_any(_FAREWELL_PATTERNS, text.strip().lower())


# -------------------------
# RESPONSE HANDLING
# -------------------------

def handle_social_message(text: str) -> str:
    t = text.strip().lower()

    if _match_any(_GREETING_PATTERNS, t):
        return (
            "Hi! Nice to meet you. "
            "Feel free to ask me any figure skating question."
        )

    if _match_any(_THANKS_PATTERNS, t):
        return "You are welcome! Happy skating."

    if _match_any(_FAREWELL_PATTERNS, t):
        return "Goodbye! Wishing you good skating sessions."

    return "Hello! How can I help with figure skating today?"
