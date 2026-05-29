from fastapi import FastAPI, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from supabase import create_client
from dotenv import load_dotenv
from backend.profile.profile_update_detector import (
    detect_profile_update_candidate,
)

from backend.profile.topic_memory import (
    update_topic_memory,
    distill_all_users,
)

from backend.profile.topic_retrieval import (
    load_user_topics,
)

from backend.tracker.blade_tracker import (
    get_tracker_state,
    log_skating_session,
    mark_blades_sharpened,
    update_threshold,
    delete_skating_session,
    build_tracker_reasoning_context,
)

import traceback
import re
import uuid
import os
import json
import math
import time

from backend.agents.agent import (
    needs_clarification,
    build_skater_state,
    build_intent_profile,
    build_retrieval_strategy,
    build_answer_plan,
    build_followup_decision,
    build_followup_prompt,
    semantic_clarification_attachment_check,
    detect_interaction_sentiment,
)

from backend.generation.answer import answer_question

from backend.taxonomy.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

from backend.retrieval.feedback_memory import (
    load_feedback_memory,
    save_feedback_memory,
    add_feedback_example,
)

from backend.retrieval.retriever import load_index_and_meta, get_embed_model
from backend.generation.llm import run_llm
from backend.memory.chat_storage import load_chats, save_chats, ensure_chat_session
load_dotenv()


app = FastAPI()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
)


def strip_thinking(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()

# -------------------------
# PRELOAD RAG AT STARTUP
# -------------------------
@app.on_event("startup")
def preload_rag():
    print("Preloading RAG system...")
    load_index_and_meta()
    get_embed_model()
    print("RAG preloaded.")


# -------------------------
# LATENCY TIMER
# -------------------------

def start_timer():
    return time.perf_counter()

def end_timer(start):
    return round(
        time.perf_counter() - start,
        3,
    )

# -------------------------
# BAD RUN DETECTOR
# -------------------------
def detect_bad_run(trace: dict) -> str | None:
    try:
        retrieval = trace.get("retrieval", {})
        repair = trace.get("repair", {})
        intent = trace.get("intent", {})

        status = retrieval.get("status")
        repair_failed = (
                repair.get("triggered")
                and repair.get("status") == "failed"
        )
        fallback_intent = intent.get("is_fallback", False)

        if status == "weak" and repair_failed:
            return "low retrieval + repair failed"

        if status == "weak":
            return "low retrieval"

        if fallback_intent:
            return "intent fallback"

        return None

    except Exception:
        return None

def print_compact_trace(trace: dict):

    input_ = trace.get("input", {})
    state = trace.get("state", {})
    profile_info = trace.get("profile", {})
    intent = trace.get("intent", {})
    clarification = trace.get("clarification", {})
    plan = trace.get("plan", {})
    retrieval = trace.get("retrieval", {})
    repair = trace.get("repair", {})
    query_profile = trace.get("query_profile", {})
    fallback = trace.get("fallback", {})
    output = trace.get("output", {})
    followup = trace.get("followup", {})
    retrieval_strategy = trace.get("retrieval_strategy", {})


    # -------------------------
    # INPUT
    # -------------------------
    print("[INPUT]")
    print(f"query        : {input_.get('query')}")
    print(f"history_len  : {input_.get('history_len')}")

    # -------------------------
    # PROFILE
    # -------------------------
    print("\n[PROFILE]")
    print(f"name         : {profile_info.get('name')}")
    print(f"highest_jump : {profile_info.get('highest_jump')}")
    print(f"test_level   : {profile_info.get('highest_test_level')}")
    # -------------------------
    # STATE
    # -------------------------
    print("\n[STATE]")
    print(f"query_level  : {state.get('skill_level')}  # transient, inferred from current message")
    print(f"signals      : {state.get('signals')}")
    print(f"jump_level   : {state.get('jump_level')}")
    print(f"experience   : {state.get('experience_type')}")
    print(f"goal         : {state.get('goal')}")

    # -------------------------
    # AGENT
    # -------------------------
    print("\n[AGENT]")
    print(f"intent       : {intent.get('label')}")
    profile = intent.get("profile", {})
    print(f"secondary    : {profile.get('secondary_intents')}")
    print(f"topic        : {profile.get('topic')}")
    print(f"focus_terms  : {profile.get('focus_terms')}")
    print(f"fallback     : {intent.get('is_fallback')}")
    print(f"clarify      : {clarification.get('triggered')}")
    print(f"clarify_q    : {clarification.get('question')}")
    print(f"clarify_candidate_reason  : {clarification.get('reason')}")
    print(f"clarify_cnt  : {trace.get('clarification_state', {}).get('count')}")
    print(f"force_answer : {trace.get('clarification_state', {}).get('force_answer')}")
    print(f"mode         : {plan.get('mode')}")
    print(f"depth        : {plan.get('depth')}")


    # -------------------------
    # QUERY PROFILE
    # -------------------------
    print("\n[QUERY PROFILE]")
    print(f"primary_query: {query_profile.get('primary_query')}")
    print(f"topic        : {query_profile.get('topic')}")
    print(f"track        : {query_profile.get('track')}")
    print(f"merged       : {query_profile.get('used_history_merge')}")


    # -------------------------
    # RAG
    # -------------------------
    print("\n[RETRIEVAL]")
    print(f"k            : {retrieval.get('k')}")
    print(f"docs         : {retrieval.get('docs_returned')}")
    print(f"status       : {retrieval.get('status')}")
    print(f"top_score    : {retrieval.get('top_score')}")
    print(f"confidence   : {retrieval.get('confidence')}")
    print(f"reason       : {retrieval.get('reason')}")

    print(f"strategy_k   : {retrieval_strategy.get('k')}")
    print(f"strategy_exp : {retrieval_strategy.get('exploration')}")
    print(f"strategy_why : {retrieval_strategy.get('reason')}")

    # -------------------------
    # FALLBACK
    # -------------------------
    print("\n[FALLBACK]")
    print(f"triggered    : {fallback.get('triggered')}")
    print(f"strategy     : {fallback.get('strategy')}")
    print(f"reason       : {fallback.get('reason')}")



    # -------------------------
    # REPAIR
    # -------------------------
    print("\n[REPAIR]")
    print(f"triggered    : {repair.get('triggered')}")
    print(f"reason       : {repair.get('reason')}")



    # -------------------------
    # PROFILE UPDATE
    # -------------------------

    profile_update = trace.get(
        "profile_update_candidate"
    )

    print("\n[PROFILE UPDATE]")

    if profile_update:

        print(
            f"field        : {profile_update.get('field')}"
        )

        print(
            f"old_value    : {profile_update.get('old_value')}"
        )

        print(
            f"new_value    : {profile_update.get('new_value')}"
        )

        print(
            f"confidence   : {profile_update.get('confidence')}"
        )

        print(
            f"reason       : {profile_update.get('reason')}"
        )

    else:
        print("candidate    : none")

    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n[OUTPUT]")
    print(f"length       : {output.get('length')}")
    print(f"followup     : {followup.get('triggered')}")
    print(f"followup_reason : {followup.get('decision', {}).get('reason')}")
    print(f"followup_type   : {followup.get('decision', {}).get('type')}")
# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://warmedge.org",
        "https://www.warmedge.org",
        "https://warmedge.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# RAM session memory
# -------------------------
SESSIONS = {}
LAST_RAG_CONTEXT = {}
CLARIFICATION_STATE = {}

MAX_TURNS = 4
MAX_CLARIFICATIONS = 1
MAX_RAG_CONTEXT_PER_SESSION = 50


# -------------------------
# Feedback memory storage
# -------------------------
FEEDBACK_PATH = "backend/memory/feedback_memory.json"

MAX_EXAMPLES_PER_DOC = 5
MAX_DOC_GROUPS = 100
SIMILARITY_DUP_THRESHOLD = 0.90



def strip_thinking(text: str) -> str:

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()

# -------------------------
# Output cleaning
# -------------------------
def clean_output(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text

def remove_markdown_tables(text: str) -> str:

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        stripped = line.strip()

        # remove markdown table rows
        if (
            "|" in stripped
            and stripped.count("|") >= 2
        ):
            continue

        # remove markdown alignment rows
        if re.match(
            r"^\s*[:\-| ]+\s*$",
            stripped
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def detect_transform_mode(message: str) -> str | None:
    q = message.lower()

    if "mode: simplify" in q:
        return "simplify"

    if "mode: deeper" in q:
        return "deeper"

    return None


def get_last_assistant_answer(history):
    for m in reversed(history):
        if m.get("role") == "assistant":
            return m.get("content")
    return None


# -------------------------
# TRACKER QUERY CLASSIFIER
# -------------------------

def classify_tracker_query(message: str) -> dict:

    prompt = f"""
You are a STRICT skating tracker query classifier.

Your ONLY job:
Detect whether the user is EXPLICITLY asking
to retrieve sharpening tracker information.

Allowed tracker retrievals:
- last sharpening date
- hours since sharpening
Only trigger if the user's question can be answered DIRECTLY and FULLY using these tracker fields alone.

Do NOT trigger for:
- comparative timeline questions
- reasoning questions
- boot-change chronology
- equipment history
- advice
- diagnostics
- indirect sharpening discussion

Examples that SHOULD trigger:

- When was my last sharpening?
- When did I last sharpen?
- How many hours since my last sharpening?
- 我上次磨刀是什么时候？
- Depuis combien d’heures ai-je patiné depuis mon dernier aiguisage ?

Examples that should NOT trigger:

- Should I sharpen my blades?
- My sharpening feels weird.
- How often should I sharpen?
- My edges feel dull.
- I hate my sharpening lately.
- Is this a sharpening issue?
- Wait, have I even sharpened since switching boots?

Return EXACTLY one of these:

TRACKER_QUERY: YES
TYPE: sharpening_lookup

OR

TRACKER_QUERY: NO
TYPE: none

User message:
{message}
""".strip()

    try:

        raw = run_llm(prompt).strip()


        normalized = raw.upper()

        if (
            "TRACKER_QUERY: YES" in normalized
            and "SHARPENING_LOOKUP" in normalized
        ):
            return {
                "tracker_query": True,
                "type": "sharpening_lookup",
            }

        return {
            "tracker_query": False,
            "type": "none",
        }

    except Exception as e:

        print("[TRACKER CLASSIFIER ERROR]", str(e))

        return {
            "tracker_query": False,
            "type": "none",
        }


def build_sharpening_tracker_reply(tracker: dict) -> str:
    last_sharpened_at = tracker.get("last_sharpened_at")
    hours = float(
        tracker.get("hours_since_sharpening") or 0
    )

    if not last_sharpened_at:
        return (
            "I do not have a recorded sharpening date yet. "
            f"You have logged {hours:g} skating hours in the current blade cycle."
        )

    return (
        f"Your last recorded sharpening was on {last_sharpened_at}. "
        f"You have skated {hours:g} hours since then."
    )

# -------------------------
# SOFT TRACKER CONTEXT
# -------------------------

def should_inject_sharpening_context(
    query: str,
) -> bool:

    prompt = f"""
You are a STRICT skating sharpening relevance classifier.

Your ONLY job:
Decide whether blade sharpness or edge wear could plausibly
be relevant to the user's skating issue.

Return ONLY:

RELEVANT: YES

OR

RELEVANT: NO

Trigger ONLY for things like:
- slipping edges
- scratchy turns
- unstable edges
- rocker weirdness
- edge inconsistency
- blade feel problems
- scraping takeoffs
- edge grip problems

Do NOT trigger for:
- choreography
- music
- competition stress
- mental blocks
- costumes
- general jump advice
- unrelated technique discussions

User query:
{query}
""".strip()

    try:

        raw = run_llm(prompt).strip().upper()

        print(
            f"[TRACKER] "
            f"relevance={'yes' if 'RELEVANT: YES' in raw else 'no'}"
        )

        return "RELEVANT: YES" in raw

    except Exception as e:

        print("[SHARPENING RELEVANCE ERROR]", str(e))

        return False


# -------------------------
# Auth helper
# -------------------------
def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    return authorization.replace("Bearer ", "").strip()

def get_authenticated_user(token: str):

    try:
        user_response = supabase.auth.get_user(token)

        if user_response.user is None:
            return None

        return user_response.user

    except Exception as e:
        print("[AUTH ERROR]", str(e))
        return None

def load_user_profile(user_id: str):

    try:

        response = (
            supabase
            .table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )

        return response.data

    except Exception as e:
        print("[PROFILE LOAD ERROR]", str(e))
        return None

def perform_profile_update(user_id: str, field: str, new_value: str):
    try:
        print("[ASYNC PROFILE UPDATE START]")
        print("[ASYNC PROFILE UPDATE] user_id:", user_id)
        print("[ASYNC PROFILE UPDATE] field:", field)
        print("[ASYNC PROFILE UPDATE] new_value:", new_value)

        (
            supabase
            .table("profiles")
            .update({
                field: new_value,
                "updated_at": "now()",
            })
            .eq("id", user_id)
            .execute()
        )

        print("[ASYNC PROFILE UPDATE DONE]")

    except Exception as e:
        print("[ASYNC PROFILE UPDATE ERROR]", str(e))

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ProfileUpdateRequest(BaseModel):
    field: str
    new_value: str

class HelpfulFeedbackRequest(BaseModel):
    session_id: str
    message_id: str

class SkatingSessionRequest(BaseModel):
    hours: float
    session_date: str | None = None
    note: str | None = None
    practice_focus: list[str] | None = None


class SharpenedRequest(BaseModel):
    sharpened_at: str | None = None


class BladeThresholdRequest(BaseModel):
    threshold_hours: float

class DeleteSkatingSessionRequest(BaseModel):
    session_date: str

class SkaterSummaryRequest(BaseModel):
    pass
class SkaterIdentityRequest(BaseModel):
    pass

# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}

def require_authenticated_user(authorization: str | None):
    token = extract_bearer_token(authorization)

    if not token:
        return None

    return get_authenticated_user(token)

def parse_identity_labels(raw: str) -> list[str]:
    raw = strip_thinking(raw)
    raw = clean_output(raw)

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            labels = parsed.get("labels", [])
        elif isinstance(parsed, list):
            labels = parsed
        else:
            labels = []

    except Exception:
        labels = [
            line.strip("-•0123456789. ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    cleaned = []

    for label in labels:
        if not isinstance(label, str):
            continue

        label = re.sub(r"[^A-Za-z0-9 '\-]", "", label).strip()
        label = re.sub(r"\s+", " ", label)

        if not label:
            continue

        if len(label.split()) > 4:
            continue

        if label.lower() in [x.lower() for x in cleaned]:
            continue

        cleaned.append(label)

    if not cleaned:
        cleaned = ["Emerging Skater", "Curious Observer", "Future Pattern Finder"]

    return cleaned[:5]

@app.post("/skater-summary")
def generate_skater_summary(
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        tracker = get_tracker_state(
            supabase,
            user.id,
        )

        topics = load_user_topics(
            supabase,
            user.id,
            limit=8,
        )

        profile_res = (
            supabase
            .table("profiles")
            .select("first_name, skater_level, highest_jump, highest_test_level")
            .eq("id", user.id)
            .single()
            .execute()
        )

        profile = profile_res.data or {}

        recent_sessions = tracker.get("sessions", [])[:20]
        focus_statistics = tracker.get("focus_statistics", {})

        prompt = f"""
You are WarmGPT's lightweight skater summary module.

This is NOT a RAG answer.
Do not retrieve external skating knowledge.
Summarize the user's own skating pattern based only on the provided data.

User profile:
- First name: {profile.get("first_name") or "Skater"}
- Skater level: {profile.get("skater_level") or "unknown"}
- Highest jump: {profile.get("highest_jump") or "unknown"}
- Highest test level: {profile.get("highest_test_level") or "unknown"}

Frequent skating topics:
{topics}

Practice focus statistics:
{focus_statistics}

Recent skating sessions:
{recent_sessions}

Write a concise, warm, useful skating summary.

Rules:
- max 150 words.
- max 2 paragraphs
- Mention the user's dominant practice patterns.
- Mention one possible imbalance or neglected area if visible.
- Mention frequent asked topics only if they are available.
- Do not overclaim.
- Do not sound medical.
- Do not use markdown tables.
- Be kind, be nice, be encouraging
- Use short paragraphs or bullets.
- End with one gentle next-step suggestion.
""".strip()

        summary = run_llm(prompt)
        summary = strip_thinking(summary)

        return {
            "success": True,
            "summary": summary,
            "topics": topics,
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/skater-identity")
def generate_skater_identity(
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        tracker = get_tracker_state(
            supabase,
            user.id,
        )

        topics = load_user_topics(
            supabase,
            user.id,
            limit=12,
        )

        profile_res = (
            supabase
            .table("profiles")
            .select("first_name, skater_level, highest_jump, highest_test_level")
            .eq("id", user.id)
            .single()
            .execute()
        )

        profile = profile_res.data or {}

        recent_sessions = tracker.get("sessions", [])[:30]
        focus_statistics = tracker.get("focus_statistics", {})

        prompt = f"""
You are WarmGPT's skater identity label generator.

Your task:
Generate 3 to 5 short identity labels (noun) for this skater.

These labels will be rendered as chic name cards.
They must NOT be paragraphs.
They must NOT include explanations.

Data source:
Use only the profile, skating logs, practice focus statistics, and recurring question topics below.

User profile:
- First name: {profile.get("first_name") or "Skater"}
- Skater level: {profile.get("skater_level") or "unknown"}
- Highest jump: {profile.get("highest_jump") or "unknown"}
- Highest test level: {profile.get("highest_test_level") or "unknown"}

Recurring skating question topics:
{topics}

Practice focus statistics:
{focus_statistics}

Recent skating sessions:
{recent_sessions}

Style rules:
- Return ONLY valid JSON.
- Format exactly: {{"labels": ["Label One", "Label Two", "Label Three"]}}
- 3 to 5 labels.
- Be super creative! 
- Each label must be 1 to 4 words.
- No emojis.
- No markdown.
- No explanations.
- No percentages.
- No medical language.
- Make the labels elegant, slightly unusual, warm, and memorable.
- Avoid corporate-sounding labels like Dedicated Learner unless clearly deserved.
- It is okay if a label feels a little oddly specific but still flattering and related.
- If the user has little data, still give positive early-stage labels.
- Do not shame low activity.
- Do not mention missing data.

Good label style (not limited to the above labels, be creative!):
- Pattern Collector
- Quiet Grinder
- Edge Detective
- Technical Dreamer
- Ice Theorist
- Methodical Skater
- Future Pattern Finder
- Curious Blade Mind
- Detail Chaser
- Soft Power Skater


Bad label style (not limited to the above labels):
- Good Skater
- Beginner Skater
- Data Missing
- No Sessions Logged
- Lazy Skater
- User Profile
""".strip()

        raw = run_llm(prompt)
        labels = parse_identity_labels(raw)

        return {
            "success": True,
            "labels": labels,
        }

    except Exception as e:
        print("[SKATER IDENTITY ERROR]", str(e))
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/blade-tracker")
def get_blade_tracker(
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        print("\n[BLADE TRACKER API] GET /blade-tracker")
        print("user_id:", user.id)

        data = get_tracker_state(
            supabase,
            user.id,
        )

        return {
            "success": True,
            "tracker": data,
        }

    except Exception as e:
        print("[BLADE TRACKER API ERROR]", str(e))
        return {"success": False, "error": str(e)}


@app.post("/blade-tracker/session")
def add_skating_session(
    req: SkatingSessionRequest,
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        print("\n[BLADE TRACKER API] POST /blade-tracker/session")
        print("user_id:", user.id)
        print("hours:", req.hours)

        data = log_skating_session(
            supabase=supabase,
            user_id=user.id,
            hours=req.hours,
            session_date=req.session_date,
            note=req.note,
            practice_focus=req.practice_focus,
        )

        return {
            "success": True,
            "tracker": data,
        }


    except Exception as e:

        print("\n===== BLADE TRACKER SESSION ERROR =====")

        traceback.print_exc()

        print("=======================================\n")

        return {

            "success": False,

            "error": str(e),

        }


@app.delete("/blade-tracker/session")
def delete_session(
    req: DeleteSkatingSessionRequest,
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        data = delete_skating_session(
            supabase=supabase,
            user_id=user.id,
            session_date=req.session_date,
        )

        return {
            "success": True,
            "tracker": data,
        }

    except Exception as e:
        print("[BLADE TRACKER DELETE ERROR]", str(e))
        return {"success": False, "error": str(e)}

@app.post("/blade-tracker/sharpened")
def sharpen_blades(
    req: SharpenedRequest,
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        print("\n[BLADE TRACKER API] POST /blade-tracker/sharpened")
        print("user_id:", user.id)
        print("sharpened_at:", req.sharpened_at)

        data = mark_blades_sharpened(
            supabase=supabase,
            user_id=user.id,
            sharpened_at=req.sharpened_at,
        )

        print("[SHARPEN RESULT]")
        print(data)

        return {
            "success": True,
            "tracker": data,
        }

    except Exception as e:
        print("[BLADE TRACKER API ERROR]", str(e))
        return {"success": False, "error": str(e)}


@app.patch("/blade-tracker/threshold")
def set_blade_threshold(
    req: BladeThresholdRequest,
    authorization: str | None = Header(default=None),
):
    try:
        user = require_authenticated_user(authorization)

        if not user:
            return {"success": False, "error": "Unauthorized"}

        print("\n[BLADE TRACKER API] PATCH /blade-tracker/threshold")
        print("user_id:", user.id)
        print("threshold_hours:", req.threshold_hours)

        data = update_threshold(
            supabase=supabase,
            user_id=user.id,
            threshold_hours=req.threshold_hours,
        )

        return {
            "success": True,
            "tracker": data,
        }

    except Exception as e:
        print("[BLADE TRACKER API ERROR]", str(e))
        return {"success": False, "error": str(e)}


# -------------------------
# Feedback endpoints
# -------------------------
@app.post("/feedback/helpful")
def mark_helpful(req: HelpfulFeedbackRequest):
    session_id = req.session_id
    message_id = req.message_id

    if session_id not in LAST_RAG_CONTEXT:
        return {
            "status": "ignored",
            "reason": "No RAG context found."
        }

    contexts = LAST_RAG_CONTEXT[session_id]

    ctx = next(
        (item for item in contexts if item.get("message_id") == message_id),
        None
    )

    if ctx is None:
        return {
            "status": "ignored",
            "reason": "Matching message_id not found."
        }

    query = ctx.get("query")
    embedding = ctx.get("query_embedding")
    docs = ctx.get("retrieved_docs", [])

    if not query or not embedding or not docs:
        return {
            "status": "ignored",
            "reason": "Missing query / embedding / docs."
        }

    memory = load_feedback_memory()
    memory = add_feedback_example(memory, query, embedding, docs)
    save_feedback_memory(memory)

    return {
        "status": "ok",
        "stored_docs": docs
    }


@app.get("/feedback-memory")
def get_feedback_memory():
    return load_feedback_memory()


@app.get("/download-feedback")
def download_feedback():
    return FileResponse(FEEDBACK_PATH)


# -------------------------
# STEP 2: Chat list endpoint
# -------------------------
@app.get("/chats")
def get_chats():

    chats = load_chats()
    chat_list = []

    for session_id, chat_data in chats.items():

        if isinstance(chat_data, dict):
            title = chat_data.get("title", "New chat")

        elif isinstance(chat_data, list):
            title = "New chat"
            for msg in chat_data:
                if msg.get("role") == "user":
                    title = msg.get("content", "New chat")[:40]
                    break

        chat_list.append({
            "session_id": session_id,
            "title": title
        })

    return list(reversed(chat_list))


# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    request_timer = start_timer()
    print("\n━━━━━━━━━━ NEW REQUEST ━━━━━━━━━━")
    print(f"[QUERY] {req.message}")
    try:
        token = extract_bearer_token(authorization)

        user_profile = None
        authenticated_user = None
        tracker_reasoning_context = None
        user_topic_memory = []

        if token:

            authenticated_user = get_authenticated_user(token)
            user_id = None
            user_email = None

            if authenticated_user:
                user_id = authenticated_user.id
                user_email = authenticated_user.email

            if authenticated_user:
                print("[AUTH] verified user")
                print("[AUTH] user id:", authenticated_user.id)
                print("[AUTH] email:", authenticated_user.email)
                user_profile = load_user_profile(
                    authenticated_user.id
                )

                print("[PROFILE]", user_profile)

                if user_profile:
                    print(
                        "[PROFILE] name:",
                        user_profile.get("first_name"),
                    )

                    print(
                        "[PROFILE] skater_level:",
                        user_profile.get("skater_level"),
                    )
                    print(
                        "[PROFILE] highest_jump:",
                        user_profile.get("highest_jump"),
                    )

                    print(
                        "[PROFILE] highest_test_level:",
                        user_profile.get("highest_test_level"),
                    )
                    user_topic_memory = load_user_topics(
                        supabase,
                        authenticated_user.id,
                    )


            else:
                print("[AUTH] invalid token")

        else:
            print("[AUTH] no bearer token")
            user_profile = None

        if not authenticated_user:
            user_topic_memory = []
        message = req.message.strip()
        # -------------------------
        # OPTIONAL TRACKER CONTEXT
        # -------------------------


        try:

            blade_keywords = [

                "blade",
                "blades",
                "edge",
                "edges",
                "sharpen",
                "sharpend",
                "sharpened",
                "sharpening",
                "sharp",
                "hollow",
                "rocker",
                "dull",
                "flat",
                "slipping",
                "slippery",

            ]

            possible_blade_query = any(
                x in message.lower()
                for x in blade_keywords
            )
            if authenticated_user and possible_blade_query:


                sharpening_timer = start_timer()

                relevance = (
                    should_inject_sharpening_context(
                        message
                    )
                )

                print(
                    f"[LATENCY] sharpening_relevance: "
                    f"{end_timer(sharpening_timer)}s"
                )

                if relevance:
                    tracker_state = get_tracker_state(
                        supabase=supabase,
                        user_id=authenticated_user.id,
                    )

                    subtle_tracker_context = (
                            "should" not in message.lower()
                            and "sharpen" not in message.lower()
                    )

                    tracker_reasoning_context = (
                        build_tracker_reasoning_context(
                            tracker_state,
                            subtle=subtle_tracker_context,
                        )
                    )

                    print("\n[TRACKER REASONING CONTEXT]")
                    print(tracker_reasoning_context)

        except Exception as e:

            print(
                "[TRACKER REASONING ERROR]",
                str(e),
            )

        # -------------------------
        # STRICT TRACKER TOOL ROUTE
        # -------------------------



        tracker_timer = start_timer()

        tracker_route = classify_tracker_query(message)

        print(
            f"[LATENCY] tracker_classifier: "
            f"{end_timer(tracker_timer)}s"
        )

        if tracker_route["tracker_query"]:

            if not authenticated_user:
                return {
                    "reply": "Please log in so I can check your blade tracker.",
                    "session_id": req.session_id or str(uuid.uuid4()),
                    "end": False,
                }

            tracker = get_tracker_state(
                supabase=supabase,
                user_id=authenticated_user.id,
            )

            reply = build_sharpening_tracker_reply(tracker)

            session_id = req.session_id or str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())
            user_message_id = str(uuid.uuid4())

            history = SESSIONS.get(session_id, [])

            history.append({
                "role": "user",
                "content": message,
            })

            history.append({
                "role": "assistant",
                "content": reply,
            })

            chats = load_chats()
            chats = ensure_chat_session(chats, session_id, message)

            chats[session_id]["messages"].append({
                "id": user_message_id,
                "role": "user",
                "content": message,
            })

            chats[session_id]["messages"].append({
                "id": assistant_message_id,
                "role": "assistant",
                "content": reply,
            })

            save_chats(chats)

            SESSIONS[session_id] = history[-MAX_TURNS * 2:]

            print("\n[TRACKER TOOL ROUTE]")
            print("type: sharpening_lookup")
            print("query:", message)
            print("reply:", reply)

            return {
                "reply": reply,
                "session_id": session_id,
                "message_id": assistant_message_id,
                "end": False,
            }

        session_id = req.session_id or str(uuid.uuid4())
        transform_mode = detect_transform_mode(message)

        if session_id not in SESSIONS:
            SESSIONS[session_id] = []
        if session_id not in CLARIFICATION_STATE:
            CLARIFICATION_STATE[session_id] = {
                "count": 0,
                "force_answer": False,
                "last_reason": None,
                "active": False,
                "original_query": None,
                "question": None,
                "resolved_query": None,
            }

        history = list(SESSIONS[session_id])
        interaction_sentiment = None

        # -------------------------
        # TRANSFORM MODES
        # -------------------------

        if transform_mode == "simplify":

            last_answer = get_last_assistant_answer(history)

            if not last_answer:
                return {
                    "reply": "There is nothing to simplify yet.",
                    "session_id": session_id,
                    "end": False,
                }

            prompt = f"""
        Rewrite the following answer into a MUCH shorter version.

        STRICT RULES:
        - Reduce the length by at least 70%
        - Keep only the core answer
        - Remove examples, nuance, caveats, and extended explanation
        - Maximum 3-5 sentences
        - Use very simple language
        - Do NOT add new information

        Original answer:
        {last_answer}
        """.strip()

            reply = clean_output(run_llm(prompt))

            assistant_message_id = str(uuid.uuid4())

            history.append({
                "role": "assistant",
                "content": reply
            })

            chats = load_chats()
            chats = ensure_chat_session(chats, session_id, message)

            chats[session_id]["messages"].append({
                "id": assistant_message_id,
                "role": "assistant",
                "content": reply
            })

            save_chats(chats)

            SESSIONS[session_id] = history[-MAX_TURNS * 2:]

            return {
                "reply": reply,
                "session_id": session_id,
                "message_id": assistant_message_id,
                "end": False,
            }

        if transform_mode == "deeper":

            last_answer = get_last_assistant_answer(history)

            if not last_answer:
                return {
                    "reply": "There is no earlier answer to expand yet.",
                    "session_id": session_id,
                    "end": False,
                }

            prompt = f"""
        Expand the following skating answer into a significantly deeper explanation.

        STRICT RULES:
        - Make the answer substantially longer
        - Add mechanics, reasoning, edge behavior, timing, and common mistakes
        - Explain WHY things happen
        - Add practical skating nuance and realistic examples
        - Stay organized and practical
        - Do NOT repeat the same points unnecessarily

        Original answer:
        {last_answer}
        """.strip()

            reply = clean_output(
                run_llm(prompt)
            )

            reply = remove_markdown_tables(reply)

            assistant_message_id = str(uuid.uuid4())

            history.append({
                "role": "assistant",
                "content": reply
            })

            chats = load_chats()
            chats = ensure_chat_session(chats, session_id, message)

            chats[session_id]["messages"].append({
                "id": assistant_message_id,
                "role": "assistant",
                "content": reply
            })

            save_chats(chats)

            SESSIONS[session_id] = history[-MAX_TURNS * 2:]

            return {
                "reply": reply,
                "session_id": session_id,
                "message_id": assistant_message_id,
                "end": False,
            }

        clarification_state = CLARIFICATION_STATE[session_id]
        agent_trace = {
            "input": {
                "query": message,
                "history_len": len(history),
            },
            "profile": {
    "name": user_profile.get("first_name") if user_profile else None,
    "skater_level": user_profile.get("skater_level") if user_profile else None,
    "highest_jump": user_profile.get("highest_jump") if user_profile else None,
    "highest_test_level": user_profile.get("highest_test_level") if user_profile else None,
},
        }

        # -------------------------
        # Blank input
        # -------------------------
        if is_blank(message):
            return {
                "reply": "Please ask a valid figure skating related question.",
                "session_id": session_id,
                "end": False,
            }

        # -------------------------
        # Social message
        # -------------------------
        if is_social_message(message):

            reply = clean_output(handle_social_message(message))
            assistant_message_id = str(uuid.uuid4())
            user_message_id = str(uuid.uuid4())

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})

            chats = load_chats()
            chats = ensure_chat_session(chats, session_id, message)

            chats[session_id]["messages"].append({
                "id": user_message_id,
                "role": "user",
                "content": message
            })
            chats[session_id]["messages"].append({
                "id": assistant_message_id,
                "role": "assistant",
                "content": reply
            })

            save_chats(chats)
            SESSIONS[session_id] = history[-MAX_TURNS * 2:]

            return {
                "reply": reply,
                "session_id": session_id,
                "message_id": assistant_message_id,
                "end": is_farewell(message),
            }

        # -------------------------
        # BUILD STATE FIRST
        # -------------------------


        # -------------------------
        # AGENT DECISION (NOW HISTORY-AWARE)
        # -------------------------
        # -------------------------

        effective_message = message
        clarification_attachment = {
            "active": clarification_state.get("active", False),
            "matched": False,
            "reason": None,
            "resolved_query": None,
        }

        if clarification_state.get("active"):
            attachment = semantic_clarification_attachment_check(
                original_query=clarification_state.get("original_query") or "",
                clarification_question=clarification_state.get("question") or "",
                user_reply=message,
            )

            clarification_attachment.update({
                "matched": attachment.get("is_answering", False),
                "reason": attachment.get("reason"),
                "resolved_query": attachment.get("resolved_query"),
                "confidence": attachment.get("confidence"),
            })

            if (
                    attachment.get("is_answering")
                    and attachment.get("confidence") in ["high", "medium"]
            ):
                effective_message = attachment.get("resolved_query") or message
                clarification_state["force_answer"] = True
                clarification_state["resolved_query"] = effective_message

        state = build_skater_state(effective_message)

        interaction_sentiment = detect_interaction_sentiment(effective_message)

        print("\n[INTERACTION SENTIMENT]")
        print(f"label      : {interaction_sentiment['label']}")
        print(f"compound   : {interaction_sentiment['compound']}")
        agent_trace["interaction_sentiment"] = interaction_sentiment

        agent_trace["state"] = state
        agent_trace["clarification_attachment"] = clarification_attachment

        profile_update_timer = start_timer()

        profile_update_candidate = detect_profile_update_candidate(
            query=effective_message,
            user_profile=user_profile,
        )

        print("\n[PROFILE UPDATE CHECK]")
        print(
            f"[LATENCY] profile_update: "
            f"{end_timer(profile_update_timer)}s"
        )

        agent_trace["profile_update_candidate"] = profile_update_candidate

        intent_profile = build_intent_profile(
            effective_message,
            history,
            state=state,
        )

        intent = intent_profile["primary_intent"]

        agent_trace["intent"] = {
            "label": intent,
            "is_fallback": (intent == "default"),
            "profile": intent_profile,
        }


        retrieval_strategy = build_retrieval_strategy(
            query=effective_message,
            intent_profile=intent_profile,
            state=state,
            history=history,
        )

        agent_trace["retrieval_strategy"] = retrieval_strategy

        k = retrieval_strategy["k"]

        clarify, reason, clarification_question = needs_clarification(
            effective_message,
            history,
            state=state,
        )

        # ------------------------------------------------
        # Clarification convergence logic
        # ------------------------------------------------

        if clarification_state["force_answer"]:
            clarify = False
            reason = "force_answer_mode"
            clarification_question = ""

        if clarification_state["count"] >= MAX_CLARIFICATIONS:
            clarify = False
            reason = "clarification_budget_exceeded"
            clarification_question = ""

        agent_trace["clarification"] = {
            "triggered": clarify,
            "reason": reason,
            "question": clarification_question,
        }

        agent_trace["clarification_state"] = {
            "count": clarification_state["count"],
            "force_answer": clarification_state["force_answer"],
        }

        answer_plan = build_answer_plan(
            query=effective_message,
            intent=intent,
            state=state,
            history=history,
            clarify=clarify,
            intent_profile=intent_profile,
        )

        agent_trace["plan"] = answer_plan

        working_history = history + [{"role": "user", "content": message}]

        chats = load_chats()
        chats = ensure_chat_session(chats, session_id, message)

        user_message_id = str(uuid.uuid4())
        chats[session_id]["messages"].append({
            "id": user_message_id,
            "role": "user",
            "content": message
        })
        save_chats(chats)

        # -------------------------
        # SMART CLARIFICATION LOGIC
        # -------------------------
        if clarify:
            # 🚨 NEW: check if prior context exists
            has_prior_context = any(
                ("axel" in m["content"].lower() or
                 re.search(r"\b[1-4][aflst]\b", m["content"].lower()))
                for m in history if m["role"] == "user"
            )

            if not has_prior_context:
                reply = clarification_question or (
                    "Could you share one more detail so I can answer accurately?"
                )

                assistant_message_id = str(uuid.uuid4())


                working_history.append({
                    "role": "assistant",
                    "content": reply
                })

                chats[session_id]["messages"].append({
                    "id": assistant_message_id,
                    "role": "assistant",
                    "content": reply
                })
                save_chats(chats)

                SESSIONS[session_id] = working_history[-MAX_TURNS * 2:]
                clarification_state["active"] = True
                clarification_state["original_query"] = message
                clarification_state["question"] = reply
                clarification_state["count"] += 1
                clarification_state["last_reason"] = reason

                return {
                    "reply": reply,
                    "session_id": session_id,
                    "message_id": assistant_message_id,
                    "end": False,
                }


        # -------------------------
        # RAG PATH (UNCHANGED)
        # -------------------------
        rag_result = answer_question(
            question=message,
            history=history,
            intent=intent,
            k=k,
            answer_plan=answer_plan,
            intent_profile=intent_profile,
            state=state,
            user_profile=user_profile,
            user_topic_memory=user_topic_memory,
            tracker_reasoning_context=tracker_reasoning_context,
            interaction_sentiment=interaction_sentiment,
        )

        # -------------------------
        # SIMPLIFY: NO RAG
        # -------------------------



        retrieved_docs = []
        query_embedding = None
        profile_update_candidate_response = None

        if isinstance(rag_result, dict):

            reply = rag_result.get("reply", "")

            retrieved_docs = rag_result.get(
                "retrieved_docs",
                [],
            )

            query_embedding = rag_result.get(
                "query_embedding"
            )

            profile_update_candidate_response = (
                rag_result.get(
                    "profile_update_candidate"
                )
            )
        else:
            reply = rag_result

        reply = clean_output(reply)
        reply = remove_markdown_tables(reply)
        reply = strip_thinking(reply)

        fallback_trace = {}
        repair_trace = {}
        retrieval_eval = {}
        retrieval_profile = {}

        if isinstance(rag_result, dict):
            fallback_trace = rag_result.get("fallback", {})
            repair_trace = rag_result.get("repair", {})
            retrieval_eval = rag_result.get("retrieval_eval", {})
            retrieval_profile = rag_result.get("retrieval_profile", {})

        agent_trace["query_profile"] = retrieval_profile

        agent_trace["retrieval"] = {
            "k": k,
            "docs_returned": len(retrieved_docs),
            "status": retrieval_eval.get("status"),
            "confidence": retrieval_eval.get("confidence"),
            "reason": retrieval_eval.get("reason"),
            "top_score": retrieval_eval.get("top_score"),
        }

        agent_trace["fallback"] = fallback_trace

        agent_trace["repair"] = repair_trace

        agent_trace["output"] = {
            "length": len(reply.split())
        }


        # -------------------------
        # SMART LLM FOLLOW-UP
        # -------------------------
        followup = None

        followup_decision = build_followup_decision(
            query=effective_message,
            intent=intent,
            state=state,
            agent_trace=agent_trace,
            reply=reply,
        )

        if followup_decision.get("generate"):
            try:
                followup_prompt = build_followup_prompt(
                    query=effective_message,
                    answer=reply,
                    intent=intent,
                    state=state,
                    history=working_history,
                    followup_decision=followup_decision,
                    agent_trace=agent_trace,
                )

                followup_timer = start_timer()

                raw_followup = clean_output(
                    run_llm(followup_prompt)
                ).strip()

                print("\n[FOLLOWUP]")
                print(
                    f"[LATENCY] followup_generation: "
                    f"{end_timer(followup_timer)}s"
                )

                if raw_followup and raw_followup.upper() != "NONE":
                    # Safety: keep only one short question
                    raw_followup = raw_followup.split("\n")[0].strip()

                    if "?" in raw_followup:
                        raw_followup = raw_followup[: raw_followup.find("?") + 1]

                    if raw_followup.endswith("?") and len(raw_followup.split()) <= 25:
                        followup = raw_followup
                        reply = reply + "\n\n" + followup

            except Exception as e:
                print("[FOLLOWUP] skipped due to error:", str(e))

        agent_trace["followup"] = {
            "triggered": followup is not None,
            "text": followup,
            "decision": followup_decision,
        }

        assistant_message_id = str(uuid.uuid4())

        # ------------------------------------------------
        # Successful real answer resets clarification state
        # ------------------------------------------------
        clarification_state["count"] = 0
        clarification_state["force_answer"] = False
        clarification_state["last_reason"] = None
        clarification_state["active"] = False
        clarification_state["original_query"] = None
        clarification_state["question"] = None
        clarification_state["resolved_query"] = None

        working_history.append({
            "role": "assistant",
            "content": reply
        })

        if session_id not in LAST_RAG_CONTEXT:
            LAST_RAG_CONTEXT[session_id] = []

        LAST_RAG_CONTEXT[session_id].append({
            "message_id": assistant_message_id,
            "query": message,
            "query_embedding": query_embedding,
            "retrieved_docs": retrieved_docs
        })

        LAST_RAG_CONTEXT[session_id] = LAST_RAG_CONTEXT[session_id][-MAX_RAG_CONTEXT_PER_SESSION:]

        chats = load_chats()
        chats = ensure_chat_session(chats, session_id, message)

        chats[session_id]["messages"].append({
            "id": assistant_message_id,
            "role": "assistant",
            "content": reply
        })

        save_chats(chats)

        SESSIONS[session_id] = working_history[-MAX_TURNS * 2:]


        bad_reason = detect_bad_run(agent_trace)

        if bad_reason:
            print(
                f"\n⚠️ BAD RUN DETECTED: "
                f"{bad_reason}"
            )

            print_compact_trace(agent_trace)

        issue = detect_bad_run(agent_trace)


        # -------------------------
        # ASYNC TOPIC MEMORY
        # -------------------------

        if authenticated_user:
            background_tasks.add_task(
                update_topic_memory,
                supabase,
                authenticated_user.id,
                message,
            )
        print(
            f"[LATENCY] total_request: "
            
            f"{end_timer(request_timer)}s"
        )
        print("━━━━━━━━━━ REQUEST END ━━━━━━━━━━\n")
        return {
            "reply": reply,
            "session_id": session_id,
            "message_id": assistant_message_id,
            "sources": retrieved_docs[:2],
            "repaired": repair_trace.get("triggered", False),

            "profile_update_candidate":
                profile_update_candidate,

            "end": False,
        }

    except Exception:
        traceback.print_exc()
        return {
            "reply": "Something went wrong.",
            "end": False,
        }


# -------------------------
# Get single chat
# -------------------------
@app.get("/chat/{session_id}")
def get_chat(session_id: str):

    chats = load_chats()

    if session_id not in chats:
        return {"error": "Chat not found"}

    chat_data = chats[session_id]

    if isinstance(chat_data, dict):
        return {
            "session_id": session_id,
            "title": chat_data.get("title", "New chat"),
            "messages": chat_data.get("messages", [])
        }
    else:
        return {
            "session_id": session_id,
            "title": "New chat",
            "messages": chat_data
        }


# -------------------------
# Clear all chats
# -------------------------
@app.delete("/chats")
def clear_all_chats():

    save_chats({})

    global SESSIONS, LAST_RAG_CONTEXT
    SESSIONS = {}
    LAST_RAG_CONTEXT = {}

    return {"status": "all chats cleared"}

@app.post("/internal/topics/distill-all")
def distill_all_topics():

    try:

        result = distill_all_users(
            supabase=supabase,
        )

        return result

    except Exception as e:

        print(
            "[GLOBAL TOPIC DISTILL API ERROR]",
            str(e),
        )

        return {
            "success": False,
            "error": str(e),
        }

@app.post("/profile/update")
def update_profile(
    req: ProfileUpdateRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
):
    try:
        token = extract_bearer_token(authorization)

        if not token:
            return {"success": False, "error": "Missing auth token"}

        authenticated_user = get_authenticated_user(token)

        if not authenticated_user:
            return {"success": False, "error": "Invalid auth token"}

        user_profile = load_user_profile(authenticated_user.id)

        if not user_profile:
            return {"success": False, "error": "Profile not found"}

        if req.field != "highest_jump":
            return {"success": False, "error": "Unsupported profile field"}

        current_jump = user_profile.get("highest_jump")

        # reuse your detector/ranking logic indirectly by allowing only backend-approved values
        allowed_jumps = [
            "waltz",
            "1T", "1S", "1Lo", "1F", "1Lz", "1A",
            "2T", "2S", "2Lo", "2F", "2Lz", "2A",
            "3T", "3S", "3Lo", "3F", "3Lz", "3A",
        ]

        if req.new_value not in allowed_jumps:
            return {"success": False, "error": "Invalid jump value"}

        def rank(jump):
            if not jump:
                return -1
            try:
                return allowed_jumps.index(jump)
            except ValueError:
                return -1

        if rank(req.new_value) <= rank(current_jump):
            return {"success": False, "error": "Update is not upward progression"}

        background_tasks.add_task(
            perform_profile_update,
            authenticated_user.id,
            req.field,
            req.new_value,
        )

        return {
            "success": True,
            "queued": True,
            "field": req.field,
            "new_value": req.new_value,
        }

    except Exception as e:
        print("[PROFILE UPDATE ERROR]", str(e))
        return {"success": False, "error": str(e)}