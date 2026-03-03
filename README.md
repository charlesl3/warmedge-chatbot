# WarmEdge Chatbot

A local-first, production-ready figure skating chatbot built with:

-   FAISS vector search\
-   SentenceTransformers embeddings\
-   LLM-based answer generation\
-   Intent routing for conversational control\
-   FastAPI backend

This repository contains the backend service only.\
The frontend (Next.js / Vercel) consumes this API via HTTP.

------------------------------------------------------------------------

## How to run?

The best way to run WarmGPT without risk of errors is via the public frontend.  [Click here](https://warmedge.vercel.app/chat)

------------------------------------------------------------------------

## Architecture Overview

User Question\
→ Intent Router\
→ Retrieval (FAISS)\
→ Prompt Builder\
→ LLM\
→ Response

The system is designed to be:

-   Deterministic at retrieval layer
-   Modular at LLM layer
-   Replaceable across model providers
-   Deployable on Railway

------------------------------------------------------------------------

## Project Structure

backend/ server.py \# FastAPI entrypoint (API server) api_stub.py \#
Local test endpoint test_api_local.py \# Manual local API test

chat/ chat_loop.py \# CLI testing loop (local development)

data/ pass1\_\* \# Raw cleaned thread batches pass2\_\* \# Structured
knowledge units raw/ \# Original scraped data

prompts/ rag_answer.txt \# Main RAG answer system prompt

rag/ answer.py \# Orchestrates retrieval + LLM retriever.py \# FAISS
search logic prompt_builder.py \# Formats final LLM prompt intents.py \#
Intent classification llm.py \# LLM API call (HF / other)

rag_store/ *.faiss \# Vector index *\_meta.json \# Metadata for indexed
documents

------------------------------------------------------------------------

## Core Concepts

### 1. Retrieval (FAISS)

-   Embeddings generated using SentenceTransformers
-   Stored in FAISS index
-   Metadata stored separately
-   Top-k relevant documents retrieved per query

### 2. Intent Routing

Simple rule-based intent detection: - greeting - general question -
knowledge lookup

Used to avoid unnecessary LLM calls.

### 3. LLM Layer

LLM is called only after: - Retrieval - Prompt assembly - Intent
filtering

Currently uses HuggingFace Router API.

------------------------------------------------------------------------

## Environment Variables

Required:

HF_API_TOKEN=your_token_here


------------------------------------------------------------------------

## Local Development

### 1. Create virtual environment

conda activate warmedge-chatbot

or

python -m venv venv source venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run local API server

uvicorn backend.server:app --reload

Test:

curl -X POST http://localhost:8000/chat -H "Content-Type:
application/json" -d '{"message":"hi","history":\[\]}'

### 4. Run CLI testing loop

python -m chat.chat_loop

------------------------------------------------------------------------

## Deployment (Railway)

1.  Push code to GitHub
2.  Connect repository to Railway
3.  Add environment variable: HF_API_TOKEN
4.  Ensure start command is:

uvicorn backend.server:app --host 0.0.0.0 --port 8080

5.  Wait for deployment
6.  Test:

https://warmedge-chatbot-production.up.railway.app

------------------------------------------------------------------------

## Frontend Integration (Vercel)

Frontend sends:

POST /chat\
Body:

{ "message": "user input", "history": \[\] }

Backend returns:

{ "reply": "...", "history": \[...\], "end": false }

Set in Vercel:

NEXT_PUBLIC_CHAT_API_URL= https://warmedge.vercel.app/chat

------------------------------------------------------------------------

## Rebuilding FAISS Index

If new knowledge units are added:

1.  Regenerate embeddings
2.  Rebuild FAISS index
3.  Replace files in rag_store/
4.  Redeploy backend

------------------------------------------------------------------------

## Status

-   Retrieval: stable
-   Intent routing: working
-   LLM layer: modular
-   Production deployment: active
-   Frontend integration: complete

------------------------------------------------------------------------


# Additional – Build Log

## DONE

- RAG works (FAISS + embeddings + markdown knowledge units)
- Prompt builder stable
- Conversation memory working
- Reddit + GoldenSkate distilled into structured essays
- Test renaming layer added (2023 USFS update)
- Moves in the Field → Skating Skills handled
- Free Skate → Singles handled
- Juvenile / Intermediate / etc. → mapped to new level names
- Automatic clarification of legacy terms in responses
- Adult vs Standard separation enforced
- Adult level ladder explicitly defined
- Standard level ladder explicitly defined
- All official test tracks listed in system prompt
- Adult tracks listed (Skills, Singles, Pattern Dance, Free Dance, Solo Free Dance, Pairs)
- Misspellings tolerated (LLM handles fuzzy input)
- Prevent silent blending of old + new systems
- Free Skate ≠ Freestyle rule enforced
- test contents added: Singles, MITF for standard and adults
- Model's provider was removed, because of hugging face, so I changed to openai's model
- more history appending logics
- added a few good manual units
- fixed memory issue - it was because the front end did not send back session id
- added wiki pages for all skaters! - huge!
- test contents: singles standard split
- add a feature for users to clean the screen

## TO DO

- competition rag building

