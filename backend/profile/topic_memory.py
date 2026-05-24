from datetime import datetime
import json

from backend.generation.llm import run_llm


MIN_TOPIC_SCORE_TO_KEEP = 2
MAX_TOPICS_PER_QUERY = 3
MAX_DISTILLED_TOPICS_PER_USER = 30
MAX_TOPIC_SCORE = 50


def normalize_topic(topic: str) -> str:

    topic = topic.strip().lower()

    topic = topic.replace("-", "_")
    topic = topic.replace(" ", "_")

    topic = "".join(
        ch for ch in topic
        if ch.isalnum() or ch == "_"
    )

    return topic


# -------------------------
# TOPIC EXTRACTION
# -------------------------

def extract_topics(query: str):

    prompt = f"""
You are a figure skating topic memory extractor.

Extract broad, stable skating topics from the user's current query.

Rules:
- Return ONLY comma-separated topic tags.
- Maximum 1-3 topics.
- Use short noun phrases only.
- Prefer pure nouns, not symptoms or temporary feelings.
- Topics are NOT limited to a fixed whitelist.
- Normalize topics with underscores.
- Do not include vague words like issue, problem, thing, question, help, advice.
- Do not include overly temporary details like fear, confidence, leaning, timing, posture unless they are clearly the main recurring topic.
- Keep correct wiring to the user's present query.
- Light fuzziness is okay, but do not invent unrelated topics.

Good:
axel
lutz, edge_work
blade_sharpening
competition, choreography
spin_entries

Bad:
problem
help
jump_issue
confidence
general_skating

User query:
{query}
""".strip()

    try:

        raw = run_llm(prompt).strip().lower()

        print("\n[TOPIC RAW]")
        print(raw)

        raw = raw.replace("\n", ",")

        topics = [
            normalize_topic(x)
            for x in raw.split(",")
            if normalize_topic(x)
        ]

        blocked = {
            "problem",
            "issue",
            "thing",
            "question",
            "help",
            "advice",
            "general",
            "general_skating",
            "skating",
            "technique",
            "tips",
            "practice",
        }

        topics = [
            t for t in topics
            if t not in blocked
            and len(t) >= 3
        ]

        topics = list(dict.fromkeys(topics))

        topics = topics[:MAX_TOPICS_PER_QUERY]

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

        now = datetime.utcnow().isoformat()

        for topic in topics:

            existing = (
                supabase
                .table("user_topic_memory")
                .select("*")
                .eq("user_id", user_id)
                .eq("topic", topic)
                .execute()
            )

            rows = existing.data or []

            # -------------------------
            # EXISTING TOPIC
            # -------------------------

            if rows:

                row = rows[0]

                new_score = min(
                    MAX_TOPIC_SCORE,
                    int(row.get("score", 0)) + 1
                )

                (
                    supabase
                    .table("user_topic_memory")
                    .update({
                        "score": new_score,
                        "last_seen": now,
                        "updated_at": now,
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
                        "last_seen": now,
                        "created_at": now,
                        "updated_at": now,
                    })
                    .execute()
                )

                print(
                    f"[TOPIC MEMORY INSERTED] {topic}"
                )

        print("[TOPIC MEMORY ASYNC DONE]")

    except Exception as e:

        print("[TOPIC MEMORY ERROR]", str(e))


# -------------------------
# WEEKLY TOPIC DISTILLATION
# -------------------------

def distill_user_topic_memory(
    supabase,
    user_id: str,
):

    try:

        print("\n[TOPIC DISTILL START]")
        print("user_id:", user_id)

        response = (
            supabase
            .table("user_topic_memory")
            .select("topic, score, last_seen")
            .eq("user_id", user_id)
            .execute()
        )

        rows = response.data or []

        if not rows:

            return {
                "success": True,
                "topics": [],
            }

        raw_topics = [
            {
                "topic": row["topic"],
                "score": int(row.get("score", 0)),
                "last_seen": row.get("last_seen"),
            }
            for row in rows
        ]

        prompt = f"""
You are cleaning a user's figure skating topic memory.

Goal:
Create a cleaner canonical topic memory for long-term personalization.

Tasks:
1. Remove vague topics.
2. Remove low-value topics.
3. Merge synonyms.
4. Prefer pure noun topics.
5. Keep topics useful for future skating answers.
6. Preserve semantic meaning.
7. Keep correct user-interest wiring.

Input topics:
{raw_topics}

Return ONLY valid JSON.

Schema:
{{
  "keep": [
    {{
      "topic": "axel",
      "score": 5
    }}
  ]
}}

Rules:
- Use snake_case topic names.
- Keep at most {MAX_DISTILLED_TOPICS_PER_USER} topics.
- Usually remove topics with score below {MIN_TOPIC_SCORE_TO_KEEP}.
- Merge synonyms like:
  blade_sharpening + sharpening -> sharpening
- No explanations.
- No markdown.
- JSON only.
""".strip()

        raw = run_llm(prompt).strip()

        print("[TOPIC DISTILL RAW]")
        print(raw)

        # -------------------------
        # JSON CLEANING
        # -------------------------

        raw = raw.replace("```json", "")
        raw = raw.replace("```", "")
        raw = raw.strip()

        try:

            data = json.loads(raw)

        except Exception:

            print("[TOPIC DISTILL JSON RECOVERY]")

            repair_prompt = f"""
Fix this invalid JSON.

Return ONLY valid JSON.

Invalid JSON:
{raw}
""".strip()

            repaired = run_llm(repair_prompt).strip()

            repaired = repaired.replace("```json", "")
            repaired = repaired.replace("```", "")
            repaired = repaired.strip()

            data = json.loads(repaired)

        keep_items = data.get("keep", [])

        # -------------------------
        # BUILD FINAL TOPIC SET
        # -------------------------

        final_topics = []

        for item in keep_items:

            topic = normalize_topic(
                item.get("topic", "")
            )

            score = int(item.get("score", 1))

            score = max(
                1,
                min(score, MAX_TOPIC_SCORE)
            )

            if not topic:
                continue

            final_topics.append({
                "topic": topic,
                "score": score,
            })

        # -------------------------
        # DEDUPLICATE
        # -------------------------

        seen = set()

        deduped = []

        for item in final_topics:

            if item["topic"] in seen:
                continue

            seen.add(item["topic"])

            deduped.append(item)

        final_topics = deduped

        # -------------------------
        # SAFETY
        # -------------------------

        if len(final_topics) == 0:

            print("[TOPIC DISTILL ABORT] zero topics")

            return {
                "success": False,
                "error": "No distilled topics generated",
            }

        # -------------------------
        # FULL OVERWRITE
        # -------------------------

        (
            supabase
            .table("user_topic_memory")
            .delete()
            .eq("user_id", user_id)
            .execute()
        )

        now = datetime.utcnow().isoformat()

        insert_rows = [
            {
                "user_id": user_id,
                "topic": item["topic"],
                "score": item["score"],
                "last_seen": now,
                "created_at": now,
                "updated_at": now,
            }
            for item in final_topics
        ]

        (
            supabase
            .table("user_topic_memory")
            .insert(insert_rows)
            .execute()
        )

        print("[TOPIC DISTILL DONE]")
        print(final_topics)

        return {
            "success": True,
            "topics": final_topics,
        }

    except Exception as e:

        print("[TOPIC DISTILL ERROR]", str(e))

        return {
            "success": False,
            "error": str(e),
        }

# -------------------------
# DISTILL ALL USERS
# -------------------------

def distill_all_users(
    supabase,
):

    try:

        print("\n[GLOBAL TOPIC DISTILL START]")

        response = (
            supabase
            .table("profiles")
            .select("id")
            .execute()
        )

        users = response.data or []

        print(
            f"[GLOBAL TOPIC DISTILL] users={len(users)}"
        )

        success_count = 0
        failed_count = 0

        for user in users:

            user_id = user["id"]

            try:

                print(
                    f"\n[DISTILL USER] {user_id}"
                )

                result = distill_user_topic_memory(
                    supabase=supabase,
                    user_id=user_id,
                )

                if result.get("success"):

                    success_count += 1

                else:

                    failed_count += 1

                    print(
                        "[DISTILL FAILED]",
                        result,
                    )

            except Exception as e:

                failed_count += 1

                print(
                    f"[DISTILL USER ERROR] {user_id}"
                )

                print(str(e))

        print("\n[GLOBAL TOPIC DISTILL DONE]")

        return {
            "success": True,
            "users": len(users),
            "success_count": success_count,
            "failed_count": failed_count,
        }

    except Exception as e:

        print("[GLOBAL TOPIC DISTILL ERROR]", str(e))

        return {
            "success": False,
            "error": str(e),
        }