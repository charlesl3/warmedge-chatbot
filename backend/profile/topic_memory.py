from datetime import datetime

from backend.generation.llm import run_llm


# -------------------------
# ALLOWED TOPICS
# -------------------------

ALLOWED_TOPICS = {

    # jumps
    "axel",
    "lutz",
    "flip",
    "loop",
    "salchow",
    "toe_loop",

    # broader skating areas
    "double_jumps",
    "triple_jumps",
    "spins",
    "edges",

    # equipment
    "boots",
    "sharpening",

    # skating life / training
    "off_ice",
    "competition",
    "choreography",
}


# -------------------------
# THIN LLM TOPIC EXTRACTOR
# -------------------------

def extract_topics(query: str):

    prompt = f"""
You are a skating topic memory extractor.

Your job:
Extract ONLY broad, stable skating interest areas that are useful for long-term user memory.

IMPORTANT:
The topics should represent recurring skating domains the user cares about,
NOT temporary mechanics, symptoms, or sub-details.

GOOD TOPICS:
- jumps
- axel
- lutz
- flip
- loop
- salchow
- toe_loop
- spins
- boots
- sharpening
- edges
- competition
- choreography
- off_ice
- double_jumps
- triple_jumps

BAD TOPICS:
- rotation
- underrotation
- leaning
- timing
- fear
- confidence
- edge collapse
- posture
- takeoff

STRICT RULES:
- Return ONLY comma-separated topic tags
- Maximum 1-3 topics
- Use ONLY normalized tags
- No explanation
- No sentences
- No temporary skating issues
- No mechanics details

GOOD OUTPUT:
axel
boots, double_jumps
sharpening

BAD OUTPUT:
axel, rotation
lutz, edge collapse
boots, confidence

User query:
{query}
""".strip()

    try:

        raw = run_llm(prompt).strip().lower()

        print("\n[TOPIC RAW]")
        print(raw)

        raw = raw.replace("\n", ",")

        topics = [
            x.strip()
            for x in raw.split(",")
            if x.strip()
        ]

        # deduplicate
        topics = list(dict.fromkeys(topics))

        # safety limit
        topics = topics[:3]

        # whitelist filter
        topics = [
            t for t in topics
            if t in ALLOWED_TOPICS
        ]

        print("[TOPIC PARSED]")
        print(topics)

        return topics

    except Exception as e:

        print("[TOPIC EXTRACT ERROR]", str(e))
        return []


# -------------------------
# ASYNC MEMORY UPDATE
# -------------------------

def update_topic_memory(
    supabase,
    user_id: str,
    query: str,
):

    try:

        print("\n[TOPIC MEMORY ASYNC START]")
        print("query:", query)

        topics = extract_topics(query)

        print("[TOPIC MEMORY TOPICS]")
        print(topics)

        if not topics:

            print("[TOPIC MEMORY] no valid topics")
            return

        for topic in topics:

            existing = (
                supabase
                .table("user_topic_memory")
                .select("*")
                .eq("user_id", user_id)
                .eq("topic", topic)
                .execute()
            )

            rows = existing.data

            # -------------------------
            # EXISTING TOPIC
            # -------------------------

            if rows:

                row = rows[0]

                new_score = row["score"] + 1

                (
                    supabase
                    .table("user_topic_memory")
                    .update({
                        "score": new_score,
                        "last_seen": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                    })
                    .eq("id", row["id"])
                    .execute()
                )

                print(
                    f"[TOPIC MEMORY UPDATED] {topic} -> {new_score}"
                )

            # -------------------------
            # NEW TOPIC
            # -------------------------

            else:

                (
                    supabase
                    .table("user_topic_memory")
                    .insert({
                        "user_id": user_id,
                        "topic": topic,
                        "score": 1,
                    })
                    .execute()
                )

                print(
                    f"[TOPIC MEMORY INSERTED] {topic}"
                )

        print("[TOPIC MEMORY ASYNC DONE]")

    except Exception as e:

        print("[TOPIC MEMORY ERROR]", str(e))