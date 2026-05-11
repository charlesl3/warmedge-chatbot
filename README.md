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
## Additional Features

### 1. Chip Mode Routing (light transformation layer)

- Functions: `detect_transform_mode(message)`, `classify_query_intent(...)`
- Purpose: allow user to control **how the answer is generated**, not just what is asked  

**Behavior:**
- frontend injects `Mode: simplify / deeper / drills / diagnose`
- backend detects mode and routes execution accordingly  
- distinguishes:
  - transformation tasks (simplify, deeper)  
  - normal RAG queries (drills, diagnose)

**Execution:**
- runs before `answer_question(...)` in `/chat`
- branches into:
  - simplify → rewrite path (no RAG)
  - deeper → hybrid path (RAG + merge)
  - others → unchanged RAG pipeline, just a new regular query  

**If triggered:**
- simplify:
  - rewrites last assistant answer only  
- deeper:
  - combines last answer + new retrieval, then merges  
- drills / diagnose:
  - mapped to `how_to` / `diagnosis` intent  

**Effect:**
- introduces a lightweight “agent-like” routing layer  
- improves consistency (no unnecessary re-retrieval)  
- enables controlled depth expansion without breaking RAG  

## Agent Logic v1

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

## Agent Logic v2.0

### 1. Clarification (gatekeeping)

- Purpose:
  decide whether the system has enough information to generate a useful answer before running full RAG.

- Main functions:
  - `semantic_clarification_check()`
    - thin semantic LLM controller
  - `needs_clarification()`
    - backend clarification gatekeeper

- v2 improvements:
  - moved from keyword-based clarification
    - `"recommend" in query`
  - to semantic clarification understanding
    - `thinking about upgrading my boots`

- Trigger examples:
  - missing skating level for equipment recommendations
  - ambiguous short domain terms (`loop`)
  - vague references (`which one?`)

- Usually does NOT clarify:
  - technique/how-to questions
  - factual questions
  - short replies to previous clarification

- Convergence logic:
  - uses:
    - `MAX_CLARIFICATIONS`
    - `force_answer`
  - prevents endless clarification loops
  - forces answering after enough clarification

- Conversation-state awareness:
  - detects when user is replying to a clarification question
  - activates:
    - `force_answer = True`

- Philosophy:
  - clarify only when answer would become:
    - misleading
    - useless
    - badly personalized
  - prefer useful answers over excessive questioning


## 2. Answer Plan Builder (generation planning)

- Purpose:
  build a lightweight answer-generation strategy before LLM generation.

- Main functions:
  - `build_answer_plan()`

- Responsibilities:
  - determine:
    - response structure
    - coaching depth
    - diagnosis framing
    - practical-action emphasis
    - tone softening

- v2 improvements:
  - moved from static prompt behavior
  - to dynamic answer planning
  - answer structure now adapts to:
    - intent
    - secondary intents
    - user state
    - topic

- Example plan signals:
  - `"what_to_try"`
  - `"likely_causes"`
  - `"equipment_note"`
  - `"confidence_softening"`

- Philosophy:
  - generation should not use:
    - one-prompt-fits-all
  - build an answer strategy first
  - then let the LLM execute the plan


---

## 3. Retrieval Query Construction + Fallback + Repair

### Retrieval Query Construction

- Purpose:
  build the complete immutable retrieval profile before retrieval.

- Main functions:
  - `build_retrieval_profile()`
  - `maybe_merge_history()`

- Responsibilities:
  - semantic query construction
  - terminology normalization
  - history-aware merging
  - intent-aware enrichment
  - topic-aware enrichment

- v2 improvements:
  - semantic query mutation now happens ONLY here
  - removed:
    - hidden fallback rewriting
    - repair-time query rewriting
    - scattered intent injection

- Philosophy:
  - retrieval semantics should freeze before retrieval
  - avoid hidden query mutation later in pipeline


### Retrieval Evaluator + Fallback

- Purpose:
  evaluate retrieval quality before generation and recover weak retrieval mechanically.

- Main functions:
  - `evaluate_retrieval_quality()`
  - `apply_fallback_if_needed()`

- Retrieval checks:
  - top score
  - score ambiguity
  - document count
  - retrieval confidence

- v2 improvements:
  - fallback separated completely from repair
  - fallback now modifies:
    - retrieval strategy only
  - fallback does NOT modify:
    - semantic query meaning

- Example fallback actions:
  - increase retrieval pool
  - use alternate prebuilt query variants
  - broaden retrieval depth

- Philosophy:
  - weak retrieval is a retrieval problem
  - recover retrieval before generation


### Answer Evaluator + Repair

- Purpose:
  evaluate generated answers and repair weak answers after generation.

- Main functions:
  - `evaluate_answer_quality()`
  - `repair_answer()`

- Evaluation checks:
  - vague language
  - weak structure
  - missing practical guidance
  - unsupported reasoning

- v2 improvements:
  - removed:
    - repair-time retrieval reruns
    - repair-time query rewriting
  - repair now modifies:
    - generation behavior only

- Philosophy:
  - repair should improve answer quality
  - not secretly rerun retrieval

### 4. Retrieval Query Construction + Conversational Merge

- Purpose:
  build a stable retrieval query before retrieval execution.

- Main functions:
  - `build_retrieval_profile()`
  - `maybe_merge_history()`
  - `is_self_contained_query()`
  - `is_incomplete_continuation()`

- Responsibilities:
  - query normalization
  - intent-aware enrichment
  - topic-aware enrichment
  - conversational context reconstruction

- v2 improvements:
  - moved from blind history concatenation
  - to semantic merge gating
  - semantic query mutation now happens ONLY here

- Merge philosophy:
  - merge only when current query depends on previous context
  - avoid unrelated conversational contamination

- Usually merges:
  - short incomplete replies
    - `Jackson`
    - `only on the right foot`
    - `that edge`
  - context-dependent continuations

- Usually does NOT merge:
  - self-contained descriptive questions
  - clear topic pivots
  - fully specified equipment/technique questions

- Previous issue:
  - semantically related but unrelated skating topics could merge
    - `loop turn`
    - `lutz scratching`
  - caused retrieval noise and ambiguous retrieval scores

- Current approach:
  - first checks:
    - semantic completeness
  - then checks:
    - conversational dependency
  - embeddings now act only as:
    - soft merge fallback

- Philosophy:
  - retrieval should reconstruct conversational intent
  - not blindly concatenate recent messages
  - prioritize retrieval precision over excessive memory inheritance