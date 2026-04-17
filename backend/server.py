from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from backend.agent import needs_clarification  # NEW
from backend.agent import build_skater_state




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

MAX_TURNS = 4
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

        history = list(SESSIONS[session_id])

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
        # AGENT DECISION LAYER
        # -------------------------
        clarify, reason = needs_clarification(message)
        print(f"[AGENT] clarify={clarify}, reason={reason}")

        state = build_skater_state(message)
        print("[STATE]", state)

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

        if clarify:
            reply = (
                "Could you tell me your level and what you want to use it for? "
                "I can give a more precise answer."
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

            return {
                "reply": reply,
                "session_id": session_id,
                "message_id": assistant_message_id,
                "end": False,
            }

        # -------------------------
        # RAG path (unchanged)
        # -------------------------
        rag_result = answer_question(
            question=message,
            history=working_history,
        )

        retrieved_docs = []
        query_embedding = None

        if isinstance(rag_result, dict):
            reply = rag_result.get("reply", "")
            retrieved_docs = rag_result.get("retrieved_docs", [])
            query_embedding = rag_result.get("query_embedding")
        else:
            reply = rag_result

        reply = clean_output(reply)

        assistant_message_id = str(uuid.uuid4())

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

        return {
            "reply": reply,
            "session_id": session_id,
            "message_id": assistant_message_id,
            "sources": retrieved_docs[:2],
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