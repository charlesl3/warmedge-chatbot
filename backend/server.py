from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import re
import uuid

from rag.answer import answer_question
from rag.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

from rag.retriever import load_index_and_meta, get_embed_model

# NEW: persistent storage
from backend.chat_storage import load_chats, save_chats


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
# Per-session history storage (RAM)
# -------------------------
SESSIONS = {}

MAX_TURNS = 4


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
# Request schema
# -------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# -------------------------
# Health check
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}


# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        message = req.message.strip()

        # Generate or reuse session
        session_id = req.session_id or str(uuid.uuid4())

        # =========================
        # DEBUG: Session Tracking
        # =========================
        existing_history = SESSIONS.get(session_id, [])

        print("\n===== SESSION DEBUG =====")
        print("Incoming session_id:", req.session_id)
        print("Using session_id:", session_id)
        print("Session exists:", session_id in SESSIONS)
        print("History length BEFORE:", len(existing_history))
        if existing_history:
            print("Last message in history:", existing_history[-1])
        print("=========================\n")

        if session_id not in SESSIONS:
            SESSIONS[session_id] = []

        # Always work on a copy first
        history = list(SESSIONS[session_id])

        # -------------------------
        # 1️⃣ Blank
        # -------------------------
        if is_blank(message):
            return {
                "reply": "Please ask a valid figure skating related question.",
                "session_id": session_id,
                "end": False,
            }

        # -------------------------
        # 2️⃣ Social
        # -------------------------
        if is_social_message(message):
            reply = clean_output(handle_social_message(message))

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})

            # Save to persistent storage
            chats = load_chats()
            if session_id not in chats:
                chats[session_id] = []
            chats[session_id].append({"role": "user", "content": message})
            chats[session_id].append({"role": "assistant", "content": reply})
            save_chats(chats)

            # Commit trimmed history
            SESSIONS[session_id] = history[-MAX_TURNS * 2 :]

            return {
                "reply": reply,
                "session_id": session_id,
                "end": is_farewell(message),
            }

        # -------------------------
        # 3️⃣ RAG Path
        # -------------------------

        # Append user message to working copy only
        working_history = history + [{"role": "user", "content": message}]

        # Save user message to disk
        chats = load_chats()
        if session_id not in chats:
            chats[session_id] = []
        chats[session_id].append({"role": "user", "content": message})
        save_chats(chats)


        reply = answer_question(
            question=message,
            history=working_history,
        )

        reply = clean_output(reply)

        working_history.append({"role": "assistant", "content": reply})

        # Save assistant reply to disk
        chats = load_chats()
        chats[session_id].append({"role": "assistant", "content": reply})
        save_chats(chats)
        print("CHAT STORAGE:", chats)

        # Commit trimmed RAM history
        SESSIONS[session_id] = working_history[-MAX_TURNS * 2 :]

        return {
            "reply": reply,
            "session_id": session_id,
            "end": False,
        }

    except Exception:
        traceback.print_exc()
        return {
            "reply": "Something went wrong.",
            "end": False,
        }