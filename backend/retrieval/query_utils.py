from typing import Optional
import numpy as np


MERGE_THRESHOLD = 0.60
FALLBACK_SHORT_THRESHOLD = 0.40
CURRENT_WEIGHT = 2

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

BRANDS = [
    "edea", "jackson", "mk", "john wilson", "wilson",
    "riedell", "risport", "graf", "aura", "eclipse",
    "paramount", "wilson", "jw", "harlick",
]


def extract_brand(text: str) -> Optional[str]:
    t = text.lower()
    for b in BRANDS:
        if b in t:
            return b
    return None


# --------------------------------------------------
# Normalization
# --------------------------------------------------
def normalize_legacy_terms(question: str) -> str:
    q = question.lower()

    replacements = {
        "free skate": "singles",
        "free skating": "singles",
        "moves in the field": "skating skills",
        "mitf": "skating skills",
        "mift": "skating skills",
        "pre silver": "pre-silver",
        "pre gold": "pre-gold",
        "pre bronze": "pre-bronze",
        "pre juvenile": "pre-juvenile",
        "pre preliminary": "pre-preliminary",
        "presilver": "pre-silver",
        "pregold": "pre-gold",
        "prebronze": "pre-bronze",
        "prejuvenile": "pre-juvenile",
        "preliminary free skate": "preliminary singles",
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    return q


# --------------------------------------------------
# Merge Logic
# --------------------------------------------------
def build_weighted_merged_query(previous_user: str, current_user: str) -> str:
    current_part = " ".join([current_user] * max(1, CURRENT_WEIGHT))
    return f"{previous_user} {current_part}"
