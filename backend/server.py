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
# Backend-owned history
# -------------------------
GLOBAL_HISTORY = []

# How many full turns to keep
MAX_TURNS = 4  # 4 user + 4 assistant messages


# -------------------------
# Output Cleaning Layer
# -------------------------
def clean_output(text: str) -> str:
    # Remove bold markdown (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)

    # Remove italic markdown (*text*)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove stray backticks
    text = re.sub(r'`+', '', text)

    # Remove markdown headers (# Heading)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    return text


# -------------------------
# Request schema
# -------------------------
class ChatRequest(BaseModel):
    message: str


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
    global GLOBAL_HISTORY

    message = req.message.strip()

    try:
        # 1️⃣ Blank input
        if is_blank(message):
            return {
                "reply": "Please ask a valid figure skating related question.",
                "end": False,
            }

        # 2️⃣ Social / small talk (NO RAG)
        if is_social_message(message):
            reply = handle_social_message(message)

            # 🔥 Clean social reply too
            reply = clean_output(reply)

            GLOBAL_HISTORY.append({"role": "user", "content": message})
            GLOBAL_HISTORY.append({"role": "assistant", "content": reply})

            GLOBAL_HISTORY = GLOBAL_HISTORY[-MAX_TURNS * 2 :]

            return {
                "reply": reply,
                "end": is_farewell(message),
            }

        # 3️⃣ REAL RAG PATH

        # Append user
        GLOBAL_HISTORY.append({"role": "user", "content": message})

        # Generate answer
        reply = answer_question(
            question=message,
            history=GLOBAL_HISTORY,
        )

        # 🔥 Clean model output BEFORE storing & returning
        reply = clean_output(reply)

        # Append assistant
        GLOBAL_HISTORY.append({"role": "assistant", "content": reply})

        # Trim history
        GLOBAL_HISTORY = GLOBAL_HISTORY[-MAX_TURNS * 2 :]

        return {
            "reply": reply,
            "end": False,
        }

    except Exception:
        print("=== LLM ERROR ===")
        traceback.print_exc()
        print("=================")

        return {
            "reply": "Something went wrong.",
            "end": False,
        }
