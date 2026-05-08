from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from backend.agent import (
    needs_clarification,
    build_skater_state,
    classify_query_intent,
    choose_k,
    build_answer_plan,
    should_generate_followup,
    build_followup_prompt,
)

import traceback
import re
import uuid
import os
import json
import math

from rag.answer import answer_question
from rag.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

from rag.retriever import load_index_and_meta, get_embed_model
from rag.llm import run_llm
from backend.chat_storage import load_chats, save_chats, ensure_chat_session


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

        weak = retrieval.get("weak", False)
        repair_failed = repair.get("triggered") and not repair.get("improved")
        fallback_intent = intent.get("is_fallback", False)

        if weak and repair_failed:
            return "low retrieval + repair failed"

        if weak:
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
    output = trace.get("output", {})
    followup = trace.get("followup", {})

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
    print(f"fallback     : {intent.get('is_fallback')}")
    print(f"clarify      : {clarification.get('triggered')}")
    print(f"clarify_q    : {clarification.get('question')}")
    print(f"clarify_why  : {clarification.get('reason')}")
    print(f"clarify_cnt  : {trace.get('clarification_state', {}).get('count')}")
    print(f"force_answer : {trace.get('clarification_state', {}).get('force_answer')}")
    print(f"mode         : {plan.get('mode')}")
    print(f"depth        : {plan.get('depth')}")

    # -------------------------
    # RAG
    # -------------------------
    print("\n[RAG]")
    print(f"k            : {retrieval.get('k')}")
    print(f"docs_initial : {retrieval.get('docs_initial')}")
    print(f"docs_final   : {retrieval.get('docs_returned')}")
    print(f"weak         : {retrieval.get('weak')}")

    # -------------------------
    # REPAIR
    # -------------------------
    print("\n[REPAIR]")
    print(f"triggered    : {repair.get('triggered')}")
    print(f"improved     : {repair.get('improved')}")
    print(f"reason       : {repair.get('reason')}")

    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n[OUTPUT]")
    print(f"length       : {output.get('length')}")
    print(f"followup     : {followup.get('triggered')}")

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
FEEDBACK_PATH = "backend/feedback_memory.json"

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


def save_feedback_memory(memory):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


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


def add_feedback_example(memory, query, embedding, docs):
    if not query or not embedding or not docs:
        return memory

    doc_set = set(docs)

    for doc_id in doc_set:
        group = next((g for g in memory if g.get("doc") == doc_id), None)

        if group is None:
            memory.append({
                "doc": doc_id,
                "examples": [
                    {
                        "query": query,
                        "embedding": embedding
                    }
                ]
            })
            continue

        examples = group.get("examples", [])

        max_sim = 0.0
        for ex in examples:
            sim = cosine_similarity(embedding, ex.get("embedding", []))
            if sim > max_sim:
                max_sim = sim

        if max_sim < SIMILARITY_DUP_THRESHOLD:
            examples.append({
                "query": query,
                "embedding": embedding
            })

        group["examples"] = examples[-MAX_EXAMPLES_PER_DOC:]

    return memory[-MAX_DOC_GROUPS:]


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
# Self-repair logic
# -------------------------
def should_repair(answer, docs, query):

    weak_phrases = [
        "it depends",
        "not sure",
        "generally",
        "in some cases"
    ]

    has_weak_phrase = any(p in answer.lower() for p in weak_phrases)

    if len(docs) <= 2:
        return True

    if has_weak_phrase:
        return True

    if len(answer.split()) < 80:  # <-- increase threshold
        return True

    return False


def repair_query(query: str, intent: str) -> str:
    if intent == "diagnosis":
        return query + " figure skating problems caused by edge quality, physical strength, or mental mindsets"

    if intent == "how_to":
        return query + " step by step technique practice figure skating"

    if intent == "comparison":
        return query + " differences pros cons figure skating"

    return query + " detailed explanation figure skating"


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

            # ------------------------------------------------
            # IMPORTANT:
            # User already responded to clarification.
            # We now FORCE ANSWER MODE to prevent
            # endless clarification recursion.
            # ------------------------------------------------
            clarification_state["force_answer"] = True

            # inherit previous intent
            intent = "diagnosis"

        else:
            intent = classify_query_intent(message, history)


        agent_trace["intent"] = {
            "label": intent,
            "is_fallback": (intent == "default")
        }

        k = choose_k(message, intent, state, history)
        agent_trace["retrieval"] = {
            "k": k,
            "weak": False
        }

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
        repaired = False
        agent_trace["retrieval"]["docs_returned"] = len(retrieved_docs)
        agent_trace["retrieval"]["weak"] = (len(retrieved_docs) <= 2)
        agent_trace["retrieval"]["docs_initial"] = len(retrieved_docs)

        # -------------------------
        # SELF-REPAIR (NEW)
        # -------------------------
        repair_triggered = False

        if should_repair(reply, retrieved_docs, message):
            repair_triggered = True
            repaired_query = repair_query(message, intent)

            repaired_result = answer_question(
                question=repaired_query,
                history=working_history,
                intent=intent,
                k=k,
                answer_plan=answer_plan,
            )

            if isinstance(repaired_result, dict):
                new_reply = clean_output(repaired_result.get("reply", ""))
                new_docs = repaired_result.get("retrieved_docs", [])

                # simple improvement check
                if len(new_docs) > len(retrieved_docs):
                    reply = new_reply
                    retrieved_docs = new_docs
                    repaired = True

        agent_trace["repair"] = {
            "triggered": repair_triggered,
            "improved": repaired,
            "docs_after": len(retrieved_docs),
            "reason": (
                "none"
                if not repair_triggered
                else (
                    "low_docs"
                    if len(retrieved_docs) <= 2
                    else "weak_answer"
                )
            )
        }

        agent_trace["output"] = {
            "length": len(reply.split())
        }

        # -------------------------
        # SMART LLM FOLLOW-UP
        # -------------------------
        followup = None

        if should_generate_followup(
                query=message,
                intent=intent,
                state=state,
                agent_trace=agent_trace,
                reply=reply,
        ):
            try:
                followup_prompt = build_followup_prompt(
                    query=message,
                    answer=reply,
                    intent=intent,
                    state=state,
                    history=working_history,
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
            "repaired": repaired,
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