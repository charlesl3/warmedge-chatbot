# WarmGPT Backend (WarmEdge Chatbot)

Production-ready Retrieval-Augmented Generation (RAG) system for figure
skating knowledge.

## Overview

WarmGPT converts unstructured skating discussions (Reddit, GoldenSkate,
Wiki) into structured knowledge and delivers grounded answers using a
retrieval-first pipeline.

**Stack** - FAISS (vector search) - SentenceTransformers (embeddings) -
Intent routing (rule-based) - Modular LLM layer - FastAPI backend -
Railway deployment

Frontend (Next.js / Vercel) consumes this API via HTTP.

------------------------------------------------------------------------

## Architecture

User → Intent Router → FAISS Retrieval → Prompt Builder → LLM → Response

Design principles: - Retrieval-first (no blind LLM calls) -
Deterministic search layer - Modular model abstraction - Clear
separation of frontend and backend

------------------------------------------------------------------------

## Key Components

### Retrieval

-   Embeddings stored in FAISS
-   Top-k similarity search
-   Metadata stored separately

### Intent Routing

-   Greeting
-   Social message
-   Knowledge lookup Reduces unnecessary LLM calls.

### LLM Layer

-   Called only after retrieval
-   Provider abstracted and replaceable

------------------------------------------------------------------------

## How to Run

The best bug-free way is to run from our website: [Click here](https://warmedge.vercel.app/chat)

------------------------------------------------------------------------

## Run Locally (optional)

``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.server:app --reload
```

Test:

``` bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"hi","history":[]}'
```

------------------------------------------------------------------------

## Deployment (Railway)

-   Push to GitHub
-   Connect repo to Railway
-   Set environment variable: HF_API_TOKEN=your_token_here
-   Start command: uvicorn backend.server:app --host 0.0.0.0 --port 8080

------------------------------------------------------------------------

## My Contribution

-   Designed full RAG architecture
-   Built knowledge distillation pipeline
-   Implemented FAISS retrieval + intent routing
-   Structured prompt builder + LLM abstraction
-   Deployed production backend (Railway)

------------------------------------------------------------------------

## Engineering Logs

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