from datetime import date


def _today_str():
    return date.today().isoformat()


def ensure_tracker_settings(supabase, user_id: str):
    existing = (
        supabase
        .table("blade_tracker_settings")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    rows = existing.data or []

    if rows:
        return rows[0]

    inserted = (
        supabase
        .table("blade_tracker_settings")
        .insert({
            "user_id": user_id,
            "threshold_hours": 40,
        })
        .execute()
    )

    print("[BLADE TRACKER] created default settings")

    return inserted.data[0]


def ensure_active_cycle(supabase, user_id: str):
    existing = (
        supabase
        .table("blade_sharpen_cycles")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )

    rows = existing.data or []

    if rows:
        return rows[0]

    inserted = (
        supabase
        .table("blade_sharpen_cycles")
        .insert({
            "user_id": user_id,
            "sharpened_at": None,
            "is_active": True,
            "total_hours": 0,
        })
        .execute()
    )

    print("[BLADE TRACKER] created active cycle")

    return inserted.data[0]


def get_tracker_state(supabase, user_id: str):
    settings = ensure_tracker_settings(supabase, user_id)
    active_cycle = ensure_active_cycle(supabase, user_id)

    sessions = (
        supabase
        .table("skating_sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("cycle_id", active_cycle["id"])
        .order("session_date", desc=True)
        .execute()
    )

    session_rows = sessions.data or []

    total_hours = sum(float(row.get("hours") or 0) for row in session_rows)
    threshold = float(settings.get("threshold_hours") or 40)

    should_sharpen = total_hours >= threshold

    print("\n[BLADE TRACKER STATE]")
    print("user_id:", user_id)
    print("cycle_id:", active_cycle["id"])
    print("hours:", total_hours)
    print("threshold:", threshold)
    print("should_sharpen:", should_sharpen)

    return {
        "threshold_hours": threshold,
        "hours_since_sharpening": total_hours,
        "should_sharpen": should_sharpen,

        "last_sharpened_at": active_cycle.get(
            "sharpened_at"
        ),

        "active_cycle": active_cycle,

        "sessions": session_rows[:20],
    }


def log_skating_session(
    supabase,
    user_id: str,
    hours: float,
    session_date: str | None = None,
    note: str | None = None,
):
    if hours <= 0:
        raise ValueError("Hours must be positive.")

    active_cycle = ensure_active_cycle(supabase, user_id)

    target_date = session_date or _today_str()

    existing = (
        supabase
        .table("skating_sessions")
        .select("*")
        .eq("user_id", user_id)
        .eq("cycle_id", active_cycle["id"])
        .eq("session_date", target_date)
        .execute()
    )

    rows = existing.data or []

    if rows:

        keep_id = rows[0]["id"]

        (
            supabase
            .table("skating_sessions")
            .update({
                "hours": hours,
                "note": note,
            })
            .eq("id", keep_id)
            .execute()
        )


        # remove older duplicate rows for same date
        for duplicate in rows[1:]:
            (
                supabase
                .table("skating_sessions")
                .delete()
                .eq("id", duplicate["id"])
                .execute()
            )

        print("\n[BLADE TRACKER] updated existing session")

    else:

        (
            supabase
            .table("skating_sessions")
            .insert({
                "user_id": user_id,
                "cycle_id": active_cycle["id"],
                "session_date": target_date,
                "hours": hours,
                "note": note,
            })
            .execute()
        )

        print("\n[BLADE TRACKER] created new session")

    print("\n[BLADE TRACKER] logged session")
    print("hours:", hours)
    print("date:", session_date or _today_str())

    return get_tracker_state(supabase, user_id)


def mark_blades_sharpened(
    supabase,
    user_id: str,
    sharpened_at: str | None = None,
):
    active_cycle = ensure_active_cycle(supabase, user_id)

    sessions = (
        supabase
        .table("skating_sessions")
        .select("hours")
        .eq("user_id", user_id)
        .eq("cycle_id", active_cycle["id"])
        .execute()
    )

    rows = sessions.data or []
    total_hours = sum(float(row.get("hours") or 0) for row in rows)

    (
        supabase
        .table("blade_sharpen_cycles")
        .update({
            "is_active": False,
            "ended_at": sharpened_at or _today_str(),
            "total_hours": total_hours,
            "updated_at": "now()",
        })
        .eq("id", active_cycle["id"])
        .execute()
    )

    (
        supabase
        .table("blade_sharpen_cycles")
        .insert({
            "user_id": user_id,
            "sharpened_at": sharpened_at or _today_str(),
            "is_active": True,
            "total_hours": 0,
        })
        .execute()
    )

    print("\n[BLADE TRACKER] marked sharpened")
    print("previous_cycle_hours:", total_hours)
    print("new_cycle_date:", sharpened_at or _today_str())

    return get_tracker_state(supabase, user_id)


def update_threshold(
    supabase,
    user_id: str,
    threshold_hours: float,
):
    if threshold_hours <= 0:
        raise ValueError("Threshold must be positive.")

    ensure_tracker_settings(supabase, user_id)

    (
        supabase
        .table("blade_tracker_settings")
        .update({
            "threshold_hours": threshold_hours,
            "updated_at": "now()",
        })
        .eq("user_id", user_id)
        .execute()
    )

    print("\n[BLADE TRACKER] updated threshold")
    print("threshold_hours:", threshold_hours)

    return get_tracker_state(supabase, user_id)

def delete_skating_session(
    supabase,
    user_id: str,
    session_date: str,
):
    active_cycle = ensure_active_cycle(supabase, user_id)

    (
        supabase
        .table("skating_sessions")
        .delete()
        .eq("user_id", user_id)
        .eq("cycle_id", active_cycle["id"])
        .eq("session_date", session_date)
        .execute()
    )

    print("\n[BLADE TRACKER] deleted session")
    print("date:", session_date)

    return get_tracker_state(supabase, user_id)