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

**Execution:**
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
- controls downstream behavior and response type  

---

### 4. Answer Planning (response control)

- Function: `build_answer_plan(query, intent, state, history, clarify)`
- Purpose: determine how the answer should be constructed  

**Behavior:**
- assigns:
  - mode (coaching / diagnosis / explanation / comparison / standard / clarification)  
  - depth (short / medium / detailed)  
  - structure (e.g., causes → what to try → drills)  
  - context usage flags (use_context, avoid_repetition)  

**Execution:**
- runs after intent and clarification  
- output is passed into `answer_question(...)`  

**Role:**
- controls response style independently of retrieval  
- ensures different query types produce different answer structures  

---

### 5. Dynamic Retrieval Depth (adaptive k)

- Function: `choose_k(query, intent, state, history)`
- Purpose: adjust retrieval depth based on context  

**Behavior:**
- increases k for short or vague queries  
- decreases k when strong skill signals exist  
- decreases k when prior context is present  
- varies k by intent  

**Execution:**
- runs before retrieval  
- determines number of documents retrieved  

---

### 6. Retrieval Fallback (retry strategy)

- Location: `answer_question()`  

**Behavior:**
- if initial retrieval is weak:
  - generate fallback query  
  - perform second retrieval  
- if still weak:
  - trigger clarification  

**Flow:**
- retrieve → weak → retry → (success → answer) / (fail → clarify)  

---

### 7. Self-Repair Loop (answer-level retry)

- Location: `/chat` endpoint in `server.py`

**Behavior:**
- after generating the first answer:
  - evaluate answer quality (length, vagueness, retrieval strength)  
- if answer is weak:
  - trigger a second pass through full RAG pipeline  
- if second pass improves retrieval:
  - replace original answer  
- otherwise:
  - keep original answer  

**Flow:**
- retrieve → answer → judge →  
  (strong → return) / (weak → retry → compare → return best)

### 8. Smart Follow-up Layer (LLM-powered continuation)

- Location: `/chat` in `server.py` + helpers in `agent.py`

**Behavior:**
- after final answer (post self-repair):
  - decide if a follow-up is useful based on:
    - intent (how_to / diagnosis / comparison)  
    - missing user context (e.g., skill level)  
    - weak or repaired retrieval  
- if triggered:
  - build a structured prompt (query + answer + intent + state + recent history)  
  - call LLM to generate **one short, specific follow-up question**  
  - append to the answer  
- otherwise:
  - return answer as-is  

**Constraints:**
- at most **one** question  
- short (≤ ~20 words)  
- context-aware and skating-specific  
- no generic phrasing  
- does not modify original answer  

**Flow:**
- retrieve → answer → repair →  
  followup_decision → (no → return) / (yes → LLM_generate → append → return)