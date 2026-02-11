from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import os

# ---- imports from your existing code ----
from rag.answer import answer_question
from rag.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

app = FastAPI()

# -------------------------
# CORS (REQUIRED for browser)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://warmedge.org",
        "https://www.warmedge.org",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request schema
# -------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

# -------------------------
# Health check (optional)
# -------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    history = req.history or []

    try:
        # 1) Blank input
        if is_blank(message):
            return {
                "reply": "Please ask a valid figure skating related question.",
                "history": history,
                "end": False,
            }

        # 2) Social / preset path (NO LLM)
        if is_social_message(message):
            reply = handle_social_message(message)
            return {
                "reply": reply,
                "history": history,
                "end": is_farewell(message),
            }

        # 3) REAL LLM / RAG PATH
        reply = answer_question(
            question=message,
            history=history,
        )

        return {
            "reply": reply,
            "history": history
            + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ],
            "end": False,
        }

    except Exception as e:
        # 🔥 THIS IS THE KEY PART 🔥
        print("=== LLM ERROR ===")
        traceback.print_exc()
        print("=================")

        return {
            "reply": "Something went wrong.",
            "history": history,
            "end": False,
        }
