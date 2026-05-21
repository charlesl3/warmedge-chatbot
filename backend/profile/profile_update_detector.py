import re

from backend.generation.llm import run_llm


# -------------------------
# ORDERED JUMP HIERARCHY
# -------------------------

JUMP_ORDER = [
    "waltz",
    "1T",
    "1S",
    "1Lo",
    "1F",
    "1Lz",
    "1A",
    "2T",
    "2S",
    "2Lo",
    "2F",
    "2Lz",
    "2A",
    "3T",
    "3S",
    "3Lo",
    "3F",
    "3Lz",
    "3A",
]


# -------------------------
# CANONICAL MAP
# -------------------------

JUMP_CANONICAL_MAP = {

    # -----------------
    # Singles
    # -----------------

    "single toe": "1T",
    "single toe loop": "1T",
    "1 toe": "1T",
    "1t": "1T",
    "1 toe loop": "1T",
    "single toejump": "1T",
    "1toeloop": "1T",

    "single salchow": "1S",
    "single sal": "1S",
    "1 salchow": "1S",
    "1 sal": "1S",
    "1s": "1S",

    "single loop": "1Lo",
    "1 loop": "1Lo",
    "1lo": "1Lo",
    "lo op": "1Lo",

    "single flip": "1F",
    "1 flip": "1F",
    "1f": "1F",
    "1fl": "1F",
    "1flip": "1F",
    "single flp": "1F",
    "single flap": "1F",
    "1 flap": "1F",

    "single lutz": "1Lz",
    "1 lutz": "1Lz",
    "1lz": "1Lz",
    "lu-z": "1Lz",

    "axel": "1A",
    "single axel": "1A",
    "1 axel": "1A",
    "1a": "1A",
    "ax-el": "1A",

    # -----------------
    # Doubles
    # -----------------

    "double toe": "2T",
    "double toe loop": "2T",
    "2 toe": "2T",
    "2t": "2T",
    "2 toe loop": "2T",
    "double toejump": "2T",

    "double salchow": "2S",
    "double sal": "2S",
    "2 salchow": "2S",
    "2 sal": "2S",
    "2s": "2S",

    "double loop": "2Lo",
    "2 loop": "2Lo",
    "2lo": "2Lo",

    "double flip": "2F",
    "2 flip": "2F",
    "2f": "2F",
    "2fl": "2F",

    "double lutz": "2Lz",
    "2 lutz": "2Lz",
    "2lz": "2Lz",

    "double axel": "2A",
    "2 axel": "2A",
    "2a": "2A",

    # -----------------
    # Triples
    # -----------------

    "triple toe": "3T",
    "3 toe": "3T",
    "3t": "3T",

    "triple salchow": "3S",
    "triple sal": "3S",
    "3 salchow": "3S",
    "3 sal": "3S",
    "3s": "3S",

    "triple loop": "3Lo",
    "3 loop": "3Lo",
    "3lo": "3Lo",

    "triple flip": "3F",
    "3 flip": "3F",
    "3f": "3F",

    "triple lutz": "3Lz",
    "3 lutz": "3Lz",
    "3lz": "3Lz",

    "triple axel": "3A",
    "3 axel": "3A",
    "3a": "3A",
}


TEST_ORDER = [
    "Pre-Bronze",
    "Bronze",
    "Silver",
    "Gold",
    "Juvenile",
    "Intermediate",
    "Novice",
    "Junior",
    "Senior",
]


# -------------------------
# HELPERS
# -------------------------

def jump_rank(jump: str | None):

    if not jump:
        return -1

    try:
        return JUMP_ORDER.index(jump)

    except ValueError:
        return -1


def normalize_jump_name(value: str):

    if not value:
        return None

    normalized = (
        value
        .strip()
        .lower()
    )

    # -------------------------
    # LIGHT CLEANING
    # -------------------------

    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    # -------------------------
    # COMMON VARIANTS
    # -------------------------

    replacements = {

        "toe loop": "toe",
        "toejump": "toe",

        "sal": "salchow",

        "flip jump": "flip",
        "flap": "flip",
        "flp": "flip",

        "lutz jump": "lutz",

        "axel jump": "axel",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    result = JUMP_CANONICAL_MAP.get(normalized)

    print(
        "[NORMALIZED JUMP]",
        value,
        "->",
        result,
    )

    return result


def test_rank(level: str | None):

    if not level:
        return -1

    normalized = level.strip().title()

    try:
        return TEST_ORDER.index(normalized)

    except ValueError:
        return -1


# -------------------------
# STRICT DETECTOR
# -------------------------

def detect_profile_update_candidate(
    query: str,
    user_profile: dict | None,
):

    if not user_profile:
        return None

    current_jump = (
        user_profile.get("highest_jump")
        or ""
    )

    current_test = (
        user_profile.get("highest_test_level")
        or ""
    )

    # ---------------------------------
    # STRICT HARD FILTERS
    # ---------------------------------

    q = query.lower()

    vague_patterns = [
        "want to learn",
        "thinking about",
        "maybe",
        "someday",
        "trying to learn",
        "hope to",
    ]

    if any(x in q for x in vague_patterns):
        return None

    # ---------------------------------
    # LLM SEMANTIC DETECTOR
    # ---------------------------------

    prompt = f"""
You are a STRICT profile update detector for a figure skating assistant.

Your job:
Determine whether the user's message contains VERY EXPLICIT evidence
that their persistent skating profile should be updated.

You should be EXTREMELY conservative.

ONLY output an update when:
- the user clearly states current skating ability
- the statement strongly implies real progression
- confidence is HIGH

NEVER infer from:
- goals
- wishes
- plans
- hypotheticals
- future intentions

Current profile:
highest_jump: {current_jump}
highest_test_level: {current_test}

User message:
{query}

Output EXACTLY in this format:

SHOULD_UPDATE: YES or NO
FIELD: highest_jump or highest_test_level or NONE
NEW_VALUE: value or NONE
CONFIDENCE: high medium low
REASON: short reason

NEW_VALUE RULES:

Output ONLY canonical skating notation.

Allowed jump outputs:

1T
1S
1Lo
1F
1Lz
1A
2T
2S
2Lo
2F
2Lz
2A
3T
3S
3Lo
3F
3Lz
3A

Interpret common skating:
- typos
- abbreviations
- spacing mistakes
- shorthand
- punctuation noise

Examples:

"1 flip"
→ 1F

"1fl"
→ 1F

"single flp"
→ 1F

"lu-z"
→ 1Lz

"ax-el"
→ 1A

"2 sal"
→ 2S

"lo op"
→ 1Lo

Even with imperfect spelling,
output the correct canonical notation when confidence is high.

If uncertain:
output NONE.

Examples:

User:
I landed my axel yesterday.

Output:
SHOULD_UPDATE: YES
FIELD: highest_jump
NEW_VALUE: 1A
CONFIDENCE: high
REASON: user explicitly landed axel

User:
I want to learn axel.

Output:
SHOULD_UPDATE: NO
FIELD: NONE
NEW_VALUE: NONE
CONFIDENCE: low
REASON: future goal only

User:
I am doing lutz now.

Output:
SHOULD_UPDATE: YES
FIELD: highest_jump
NEW_VALUE: 1Lz
CONFIDENCE: high
REASON: user explicitly states current jump ability
""".strip()

    try:

        raw = run_llm(prompt)

        print("[PROFILE UPDATE RAW]")
        print(raw)

        upper = raw.upper()

        should_update = (
            "SHOULD_UPDATE: YES" in upper
        )

        if not should_update:
            return None

        field_match = re.search(
            r"FIELD:\s*(.*)",
            raw,
            re.IGNORECASE,
        )

        value_match = re.search(
            r"NEW_VALUE:\s*(.*)",
            raw,
            re.IGNORECASE,
        )

        confidence_match = re.search(
            r"CONFIDENCE:\s*(.*)",
            raw,
            re.IGNORECASE,
        )

        reason_match = re.search(
            r"REASON:\s*(.*)",
            raw,
            re.IGNORECASE,
        )

        field = (
            field_match.group(1).strip()
            if field_match else "NONE"
        )

        raw_value = (
            value_match.group(1).strip()
            if value_match else "NONE"
        )

        new_value = normalize_jump_name(raw_value)

        if field == "highest_jump" and not new_value:
            return None

        confidence = (
            confidence_match.group(1).strip().lower()
            if confidence_match else "low"
        )

        reason = (
            reason_match.group(1).strip()
            if reason_match else ""
        )

        # ---------------------------------
        # STRICT CONFIDENCE RULE
        # ---------------------------------

        if confidence != "high":
            return None

        # ---------------------------------
        # MONOTONIC PROGRESSION ONLY
        # ---------------------------------

        if field == "highest_jump":

            if (
                jump_rank(new_value)
                <= jump_rank(current_jump)
            ):
                return None

        if field == "highest_test_level":

            if (
                test_rank(new_value)
                <= test_rank(current_test)
            ):
                return None

        return {
            "field": field,

            "old_value":
                current_jump
                if field == "highest_jump"
                else current_test,

            "new_value": new_value,

            "confidence": confidence,

            "reason": reason,

            "raw": raw,
        }

    except Exception as e:

        print(
            "[PROFILE UPDATE DETECTOR ERROR]",
            str(e),
        )

        return None