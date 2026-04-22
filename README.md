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

------------------------------------------------------------------------

## Agent Logic (Current)

### 1. Clarification (gatekeeping)

- Function: `needs_clarification(query, history)`
- Purpose: decide whether the system has enough information to answer  

**Behavior:**
- detects vague or underspecified queries  
- checks for missing skill level in recommendation questions  
- uses history to avoid unnecessary clarification  

**Execution update:**
- runs after intent classification  
- only triggers when `intent == "default"`  

**If triggered:**
- stops pipeline and asks a follow-up question  

---

### 2. State Construction (user understanding)

- Function: `build_skater_state(query)`
- Purpose: extract structured user information  

**Includes:**
- skill level (beginner / intermediate / advanced)  
- jump signals (axel, double, etc.)  
- body info (height, weight → categories)  
- experience type (adult vs standard)  
- goal (e.g., equipment recommendation)  

**Usage:**
- used later in prompt to condition answers  
- provides implicit context when user input is incomplete  

---

### 3. Intent Classification (routing)

- Function: `classify_query_intent(query, history)`
- Purpose: decide how to process the query  

**Current intents:**
- `how_to` → actionable improvement  
- `comparison` → compare options  
- `diagnosis` → infer causes from symptoms  
- `experience_lookup` → general explanation  
- `default` → fallback  

**Role:**
- executed before clarification  
- controls retrieval rewriting and response behavior  

---

### 4. Diagnosis Mode (structured reasoning)

- Trigger: `intent == "diagnosis"`  

**Behavior:**
- enforces structured output:
  - Likely causes  
  - What to try  
  - Notes  

**Purpose:**
- improves consistency and actionability of responses  
- aligns output with underlying knowledge structure  

---

### 5. Retrieval Fallback (retry strategy)

- Location: `answer_question()`  

**Behavior:**
- if initial retrieval is weak:
  - generate fallback query  
  - perform second retrieval  
- if still weak:
  - trigger clarification  

**Flow:**
- retrieve → weak → retry → (success → answer) / (fail → clarify)  

**Purpose:**
- reduces premature clarification  
- improves robustness for vague but valid queries  

### 6. Self-Repair Loop (answer-level retry)

- Location: `/chat` endpoint in `server.py`

**Behavior:**
- after generating the first answer:
  - evaluate answer quality (length, vagueness, retrieval strength)
- if answer is weak:
  - trigger a second pass (reuse same query, full RAG pipeline)
- if second pass improves retrieval:
  - replace original answer
- otherwise:
  - keep original answer

**Flow:**
- retrieve → answer → judge →
  (strong → return) / (weak → retry → compare → return best)

**Purpose:**
- fixes weak first-pass answers without user intervention
- complements retrieval fallback (operates at answer level, not retrieval level)
- improves robustness when:
  - answer is too short
  - retrieval was borderline but not failed
- preserves UX simplicity (user only sees final answer)