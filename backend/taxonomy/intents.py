import re


# -------------------------
# BASIC CHECKS
# -------------------------

def is_blank(text: str) -> bool:
    return text is None or len(text.strip()) == 0


# -------------------------
# PATTERNS
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


# -------------------------
# INTERNAL HELPERS
# -------------------------

def _match_any(patterns: list[str], text: str) -> bool:
    for p in patterns:
        if re.fullmatch(p, text):
            return True
    return False


def _contains_any(patterns: list[str], text: str) -> bool:
    """
    Detects whether a social intent appears as a standalone
    word or phrase inside a short message.

    This is what allows:
    - "thanks bye"
    - "thank you bye"
    """
    for p in patterns:
        core = p.strip("^$")
        if re.search(rf"\b{core}\b", text):
            return True
    return False


def _split_phrases(text: str) -> list[str]:
    """
    Split short social messages into phrases.

    Examples:
    - "thank you, bye" -> ["thank you", "bye"]
    - "thanks and goodbye" -> ["thanks", "goodbye"]
    """
    parts = re.split(r"[,\s]+and\s+|,|;", text)
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# INTENT DETECTION
# -------------------------

def is_social_message(text: str) -> bool:
    """
    Returns True if the message is purely social / polite
    and should NOT go through RAG.
    """
    if not text:
        return False

    t = text.strip().lower()

    # Prevent swallowing real questions
    if len(t.split()) > 8:
        return False

    phrases = _split_phrases(t)

    return all(
        _match_any(_GREETING_PATTERNS, p)
        or _match_any(_THANKS_PATTERNS, p)
        or _match_any(_FAREWELL_PATTERNS, p)
        or _contains_any(_THANKS_PATTERNS + _FAREWELL_PATTERNS, p)
        for p in phrases
    )


def is_farewell(text: str) -> bool:
    """
    Returns True if ANY farewell intent appears.
    """
    if not text:
        return False

    t = text.strip().lower()

    return (
        _contains_any(_FAREWELL_PATTERNS, t)
    )


# -------------------------
# RESPONSE HANDLING
# -------------------------

def handle_social_message(text: str) -> str:
    t = text.strip().lower()
    phrases = _split_phrases(t)

    has_greeting = (
        any(_match_any(_GREETING_PATTERNS, p) for p in phrases)
        or _contains_any(_GREETING_PATTERNS, t)
    )
    has_thanks = (
        any(_match_any(_THANKS_PATTERNS, p) for p in phrases)
        or _contains_any(_THANKS_PATTERNS, t)
    )
    has_farewell = (
        any(_match_any(_FAREWELL_PATTERNS, p) for p in phrases)
        or _contains_any(_FAREWELL_PATTERNS, t)
    )

    if has_greeting:
        return (
            "Hi! Nice to meet you. "
            "Feel free to ask me any figure skating question."
        )

    if has_thanks and has_farewell:
        return "You are welcome! Goodbye, and happy skating."

    if has_thanks:
        return "You are welcome! Happy skating."

    if has_farewell:
        return "Goodbye! Wishing you good skating sessions."

    return "Hello! How can I help with figure skating today?"
