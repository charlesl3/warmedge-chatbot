# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import traceback
# import os
#
# # ---- imports from your existing code ----
# from rag.answer import answer_question
# from rag.intents import (
#     is_blank,
#     is_social_message,
#     is_farewell,
#     handle_social_message,
# )
#
# app = FastAPI()
#
# # -------------------------
# # CORS (REQUIRED for browser)
# # -------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://warmedge.org",
#         "https://www.warmedge.org",
#         "https://warmedge.vercel.app",
#         "http://localhost:3000",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # -------------------------
# # Request schema
# # -------------------------
# class ChatRequest(BaseModel):
#     message: str
#     history: list = []
#
# # -------------------------
# # Health check (optional)
# # -------------------------
# @app.get("/")
# def root():
#     return {"status": "ok"}
#
# # -------------------------
# # Chat endpoint
# # -------------------------
# @app.post("/chat")
# def chat(req: ChatRequest):
#
#     # 🔎 DEBUG ENV CHECK
#     print("========== ENV DEBUG ==========")
#     print("HF TOKEN VALUE:", os.getenv("HF_TOKEN"))
#     print("ENV KEYS:", list(os.environ.keys()))
#     print("================================")
#
#     message = req.message.strip()
#     history = req.history or []
#
#     try:
#         # 1) Blank input
#         if is_blank(message):
#             return {
#                 "reply": "Please ask a valid figure skating related question.",
#                 "history": history,
#                 "end": False,
#             }
#
#         # 2) Social / preset path (NO LLM)
#         if is_social_message(message):
#             reply = handle_social_message(message)
#             return {
#                 "reply": reply,
#                 "history": history,
#                 "end": is_farewell(message),
#             }
#
#         # 3) REAL LLM / RAG PATH
#         reply = answer_question(
#             question=message,
#             history=history,
#         )
#
#         return {
#             "reply": reply,
#             "history": history
#             + [
#                 {"role": "user", "content": message},
#                 {"role": "assistant", "content": reply},
#             ],
#             "end": False,
#         }
#
#     except Exception as e:
#         print("=== LLM ERROR ===")
#         traceback.print_exc()
#         print("=================")
#
#         return {
#             "reply": "Something went wrong.",
#             "history": history,
#             "end": False,
#         }


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

app = FastAPI()

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
# Per-session history storage
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

        reply = answer_question(
            question=message,
            history=working_history,
        )

        reply = clean_output(reply)

        # Only commit if generation succeeded
        working_history.append({"role": "assistant", "content": reply})
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
