from pathlib import Path

from backend.generation.prompt_builder import build_llm_input
from backend.generation.llm import run_llm

from backend.taxonomy.intents import (
    is_blank,
    is_social_message,
    handle_social_message,
)

import time

from backend.retrieval.retrieval_query import build_retrieval_profile
from backend.retrieval.retrieval_pipeline import (
    execute_retrieval,
    evaluate_retrieval_quality,
    apply_fallback_if_needed,
    trim_retrieval,
)
from backend.retrieval.retrieval_helpers import extract_retrieved_doc_ids

from backend.generation.answer_repair import (
    evaluate_answer_quality,
    repair_answer,
)

PROMPT_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "rag_answer.txt"
)

# -------------------------
# LOCAL LATENCY TIMER
# -------------------------

def start_timer():
    return time.perf_counter()


def end_timer(start):
    return round(
        time.perf_counter() - start,
        3,
    )


def answer_question(
    question,
    history,
    intent="default",
    k=4,
    answer_plan=None,
    intent_profile=None,
    state=None,
    user_profile=None,
    user_topic_memory=None,
    profile_update_candidate=None,
    tracker_reasoning_context=None,
):

    # ---------------------------------
    # basic filtering
    # ---------------------------------

    if is_blank(question):
        return "Please ask a valid question."

    if is_social_message(question):
        return handle_social_message(question)

    # ---------------------------------
    # query construction
    # ---------------------------------

    profile = build_retrieval_profile(
        question=question,
        history=history,
        intent=intent,
        intent_profile=intent_profile,
        state=state,
    )

    # ---------------------------------
    # retrieval execution
    # ---------------------------------

    retrieval = execute_retrieval(
        query=profile.primary_query,
        track=profile.track,
        pool_k=30,
    )

    # ---------------------------------
    # DEBUG: retrieved topics
    # ---------------------------------

    print("\n[RETRIEVED DOCS]")
    print(f"query: {profile.primary_query}")

    for i, meta in enumerate(retrieval.get("sources", [])[:8], start=1):

        title = (
                meta.get("title")
                or meta.get("source_group")
                or meta.get("source_path")
                or "unknown"
        )

        score = retrieval.get("scores", [None])[i - 1]

        if isinstance(score, float):
            score = round(score, 3)

        print(
            f"{i}. "
            f"{score} | "
            f"{title[:55]}"
        )

    # ---------------------------------
    # retrieval evaluation
    # ---------------------------------

    retrieval_eval = evaluate_retrieval_quality(
        retrieval,
        desired_k=k,
    )

    # ---------------------------------
    # fallback
    # ---------------------------------

    retrieval, fallback_trace = apply_fallback_if_needed(
        profile=profile,
        retrieval=retrieval,
        evaluation=retrieval_eval,
        desired_k=k,
    )

    retrieval = trim_retrieval(retrieval, k)

    retrieval_eval_final = evaluate_retrieval_quality(
        retrieval,
        desired_k=k,
    )

    confidence = retrieval_eval_final.get(
        "confidence",
        "medium",
    )

    # ---------------------------------
    # generation
    # ---------------------------------

    prompt = PROMPT_PATH.read_text()

    llm_input = build_llm_input(
        prompt=prompt,
        question=profile.normalized_question,
        docs=retrieval["results"],
        history=history,
        intent=intent,
        answer_plan=answer_plan,
        confidence=confidence,
        user_profile=user_profile,
        user_topic_memory=user_topic_memory,
        tracker_reasoning_context=
        tracker_reasoning_context,
    )

    generation_timer = start_timer()
    response = run_llm(llm_input)
    print(
        f"[LATENCY] main_generation: "
        f"{end_timer(generation_timer)}s"
    )

    # ---------------------------------
    # answer evaluation
    # ---------------------------------

    answer_eval = evaluate_answer_quality(
        answer=response,
        docs=retrieval["results"],
        intent=intent,
        answer_plan=answer_plan,
        intent_profile=intent_profile,
    )

    repair_trace = {
        "triggered": False,
        "reason": answer_eval.get("reason"),
        "status": "not_needed",
    }

    # ---------------------------------
    # repair
    # ---------------------------------

    if (
            answer_eval.get("should_repair")
            and confidence != "high"
    ):
        repair_timer = start_timer()
        response = repair_answer(
            original_answer=response,
            question=profile.normalized_question,
            docs=retrieval["results"],
            intent=intent,
            answer_plan=answer_plan,
            intent_profile=intent_profile,
            reason=answer_eval.get("reason"),
        )
        print(
            f"[LATENCY] repair_generation: "
            f"{end_timer(repair_timer)}s"
        )

        repair_trace = {
            "triggered": True,
            "reason": answer_eval.get("reason"),
            "status": "repaired",
        }

    # ---------------------------------
    # final packaging
    # ---------------------------------

    retrieved_docs = extract_retrieved_doc_ids(retrieval)

    query_embedding = retrieval.get("query_embedding")

    return {
        "reply": response,

        "retrieved_docs": retrieved_docs[:k],

        "query_embedding":
            query_embedding.tolist()
            if query_embedding is not None
            else None,

        "retrieval_profile": {
            "primary_query": profile.primary_query,
            "expanded_queries": profile.expanded_queries,
            "intent": profile.intent,
            "topic": profile.topic,
            "track": profile.track,
            "used_history_merge":
                profile.used_history_merge,
        },

        "retrieval_eval":
            retrieval_eval_final,

        "fallback":
            fallback_trace,

        "repair":
            repair_trace,
        "profile_update_candidate":
            profile_update_candidate,
    }