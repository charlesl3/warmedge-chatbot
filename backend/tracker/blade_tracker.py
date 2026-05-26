from datetime import date
from datetime import datetime


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
    user_id,
    hours,
    session_date=None,
    note=None,
):


    active_cycle = ensure_active_cycle(supabase, user_id)

    target_date = session_date or _today_str()
    # normalize empty notes
    if note:
        note = note.strip()

    if note == "":
        note = None

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

    # -------------------------
    # 0 HOURS = DELETE SESSION
    # -------------------------

    if hours <= 0:

        if rows:
            (
                supabase
                .table("skating_sessions")
                .delete()
                .eq("id", rows[0]["id"])
                .execute()
            )

            print("\n[BLADE TRACKER] deleted session via 0 hours")

        return get_tracker_state(supabase, user_id)

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

# -------------------------
# TRACKER REASONING CONTEXT
# -------------------------

def build_tracker_reasoning_context(
    tracker_state: dict,
    subtle: bool = False,
):

    hours = float(
        tracker_state.get(
            "hours_since_sharpening"
        ) or 0
    )

    threshold = float(
        tracker_state.get(
            "threshold_hours"
        ) or 40
    )

    last_sharpened = (
        tracker_state.get(
            "last_sharpened_at"
        )
    )

    ratio = hours / threshold if threshold > 0 else 0

    # -------------------------
    # LOW HOURS
    # -------------------------

    if ratio <= 0.35:
        explanation = (
            "The skater is still relatively early in the "
            "current sharpening cycle."
            if subtle else
            "Blade dullness is less likely to be the "
            "primary cause unless there are sharpening "
            "quality or mounting issues."
        )

        return (
            f"The skater has only logged "
            f"{hours:g} skating hours since the last sharpening "
            f"(threshold: {threshold:g} hours). "
            f"{explanation}"
        )

    # -------------------------
    # MID HOURS
    # -------------------------

    if ratio <= 0.75:

        return (
            f"The skater has logged "
            f"{hours:g} hours since the last sharpening "
            f"(threshold: {threshold:g} hours). "
            f"Moderate edge wear could plausibly contribute "
            f"to skating inconsistencies."
        )

    # -------------------------
    # HIGH HOURS
    # -------------------------

    return (
        f"The skater has logged "
        f"{hours:g} skating hours since the last sharpening "
        f"(threshold: {threshold:g} hours). "
        f"Blade wear or dullness is now a plausible contributing factor."
    )