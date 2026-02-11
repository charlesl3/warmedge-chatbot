from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from backend.api_stub import chat_api

app = FastAPI()

# ---- Allow browser access (CORS) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local testing only
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


@app.post("/chat")
def chat(req: ChatRequest):
    return chat_api(
        message=req.message,
        history=req.history,
    )
