def load_user_topics(
    supabase,
    user_id: str,
    limit: int = 5,
):

    try:

        response = (
            supabase
            .table("user_topic_memory")
            .select("topic, score")
            .eq("user_id", user_id)
            .order("score", desc=True)
            .limit(limit)
            .execute()
        )

        rows = response.data or []

        topics = [
            row["topic"]
            for row in rows
            if row.get("score", 0) >= 2
        ]

        print("\n[USER TOPIC MEMORY]")
        print("topics:", topics)

        return topics

    except Exception as e:

        print("[USER TOPIC LOAD ERROR]", str(e))
        return []