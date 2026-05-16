from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

import traceback
import re
import uuid
import os
import json
import math

from backend.agents.agent import (
    needs_clarification,
    build_skater_state,
    build_intent_profile,
    build_retrieval_strategy,
    build_answer_plan,
    build_followup_decision,
    build_followup_prompt,
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


app = FastAPI()

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

    print("\n==================================================")

    # -------------------------
    # INPUT
    # -------------------------
    print("[INPUT]")
    print(f"query        : {input_.get('query')}")
    print(f"history_len  : {input_.get('history_len')}")

    # -------------------------
    # STATE
    # -------------------------
    print("\n[STATE]")
    print(f"skill_level  : {state.get('skill_level')}")
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
    print(f"eval_reason  : {fallback.get('evaluation_reason')}")



    # -------------------------
    # REPAIR
    # -------------------------
    print("\n[REPAIR]")
    print(f"triggered    : {repair.get('triggered')}")
    print(f"reason       : {repair.get('reason')}")
    print(f"status       : {repair.get('status')}")
    print(f"focus_cov    : {repair.get('focus_coverage')}")
    print(f"missing_focus: {repair.get('missing_focus_terms')}")

    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n[OUTPUT]")
    print(f"length       : {output.get('length')}")
    print(f"followup     : {followup.get('triggered')}")
    print(f"followup_reason : {followup.get('decision', {}).get('reason')}")
    print(f"followup_type   : {followup.get('decision', {}).get('type')}")
    print("==================================================\n")
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


def load_feedback_memory():
    if not os.path.exists(FEEDBACK_PATH):
        return []

    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []



def cosine_similarity(a, b):
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
# -------------------------
# Output cleaning
# -------------------------
def clean_output(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text

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
# Request schemas
# -------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class HelpfulFeedbackRequest(BaseModel):
    session_id: str
    message_id: str


# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}


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
def chat(req: ChatRequest):

    try:
        message = req.message.strip()
        session_id = req.session_id or str(uuid.uuid4())

        if session_id not in SESSIONS:
            SESSIONS[session_id] = []
        if session_id not in CLARIFICATION_STATE:
            CLARIFICATION_STATE[session_id] = {
                "count": 0,
                "force_answer": False,
                "last_reason": None,
            }

        history = list(SESSIONS[session_id])
        clarification_state = CLARIFICATION_STATE[session_id]
        agent_trace = {
            "input": {
                "query": message,
                "history_len": len(history),
            }
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
        state = build_skater_state(message)
        agent_trace["state"] = state

        # -------------------------
        # AGENT DECISION (NOW HISTORY-AWARE)
        # -------------------------
        # -------------------------
        # CONTEXTUAL INTENT FIX (NEW)
        # -------------------------
        def looks_like_question(text: str):
            text = text.lower()
            return (
                    "?" in text or
                    text.startswith(("which", "what", "where", "when", "how", "is", "are", "do", "does", "can"))
            )

        def is_short_answer(msg: str):
            return len(msg.split()) <= 3

        last_assistant = next(
            (m for m in reversed(history) if m["role"] == "assistant"),
            None
        )

        is_answering_clarification = (
                last_assistant and
                looks_like_question(last_assistant["content"]) and
                is_short_answer(message)
        )

        if is_answering_clarification:

            clarification_state["force_answer"] = True

            previous_user_query = next(
                (
                    m["content"]
                    for m in reversed(history)
                    if m["role"] == "user"
                ),
                ""
            )

            previous_profile = build_intent_profile(
                previous_user_query,
                history,
                state=state,
            )

            intent_profile = previous_profile
            intent = previous_profile["primary_intent"]
        else:
            intent_profile = build_intent_profile(
                message,
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
            query=message,
            intent_profile=intent_profile,
            state=state,
            history=history,
        )

        agent_trace["retrieval_strategy"] = retrieval_strategy

        k = retrieval_strategy["k"]

        clarify, reason, clarification_question = needs_clarification(
            message,
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
            query=message,
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
                clarification_state["count"] += 1
                clarification_state["last_reason"] = reason

                return {
                    "reply": reply,
                    "session_id": session_id,
                    "message_id": assistant_message_id,
                    "end": False,
                }

        # -------------------------
        # topic classifier
        # -------------------------



        # -------------------------
        # RAG PATH (UNCHANGED)
        # -------------------------
        rag_result = answer_question(
            question=message,
            history=working_history,
            intent=intent,
            k=k,
            answer_plan=answer_plan,
            intent_profile=intent_profile,
            state=state,
        )

        transform_mode = detect_transform_mode(message)

        # -------------------------
        # SIMPLIFY: NO RAG
        # -------------------------
        if transform_mode == "simplify":
            last_answer = get_last_assistant_answer(history)

            if last_answer:
                prompt = f"""
        Rewrite the following answer more simply and concisely.
        Keep the meaning but reduce complexity and length.

        Answer:
        {last_answer}
        """.strip()

                reply = clean_output(run_llm(prompt))
                assistant_message_id = str(uuid.uuid4())

                working_history.append({
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
                SESSIONS[session_id] = working_history[-MAX_TURNS * 2:]

                return {
                    "reply": reply,
                    "session_id": session_id,
                    "message_id": assistant_message_id,
                    "end": False,
                }
            # -------------------------
            # DEEPER: RAG + MERGE
            # -------------------------
        if transform_mode == "deeper":
            last_answer = get_last_assistant_answer(history)

            if last_answer:
                rag_result = answer_question(
                    question=message,
                    history=working_history,
                    intent="experience_lookup",
                    k=max(k, 5),
                    answer_plan=answer_plan,
                )

                rag_text = rag_result.get("reply", "") if isinstance(rag_result, dict) else rag_result

                merge_prompt = f"""
You are expanding an existing skating answer.

Base answer:
{last_answer}

Additional information:
{rag_text}

Write ONE deeper, unified answer.

Rules:
- Keep useful parts of the base answer
- Add technical reasoning, mechanics, and nuance
- Avoid repetition
- Do NOT mention sources or retrieval
""".strip()

                reply = clean_output(run_llm(merge_prompt))
                assistant_message_id = str(uuid.uuid4())

                working_history.append({
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
                SESSIONS[session_id] = working_history[-MAX_TURNS * 2:]

                return {
                    "reply": reply,
                    "session_id": session_id,
                    "message_id": assistant_message_id,
                    "end": False,
                }


        retrieved_docs = []
        query_embedding = None

        if isinstance(rag_result, dict):
            reply = rag_result.get("reply", "")
            retrieved_docs = rag_result.get("retrieved_docs", [])
            query_embedding = rag_result.get("query_embedding")
        else:
            reply = rag_result

        reply = clean_output(reply)
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
            query=message,
            intent=intent,
            state=state,
            agent_trace=agent_trace,
            reply=reply,
        )

        if followup_decision.get("generate"):
            try:
                followup_prompt = build_followup_prompt(
                    query=message,
                    answer=reply,
                    intent=intent,
                    state=state,
                    history=working_history,
                    followup_decision=followup_decision,
                    agent_trace=agent_trace,
                )

                raw_followup = clean_output(run_llm(followup_prompt)).strip()

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

        print_compact_trace(agent_trace)

        issue = detect_bad_run(agent_trace)

        if issue:
            print(f"⚠️ BAD RUN DETECTED: {issue}")

        return {
            "reply": reply,
            "session_id": session_id,
            "message_id": assistant_message_id,
            "sources": retrieved_docs[:2],
            "repaired": repair_trace.get("triggered", False),
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