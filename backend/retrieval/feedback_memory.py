import json
from pathlib import Path
from typing import Dict

import numpy as np
import os

from backend.retrieval.query_utils import cosine_sim


TOP_K = 6
DEBUG = True
MAX_EXAMPLES_PER_DOC = 5
MAX_DOC_GROUPS = 100
SIMILARITY_DUP_THRESHOLD = 0.90


FEEDBACK_PATH = (
    Path(__file__).resolve().parents[1]
    / "memory"
    / "feedback_memory.json"
)


def debug_print(*args):
    if DEBUG:
        print(*args)


def load_feedback_memory():
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def boost_with_feedback(query_embedding: np.ndarray, retrieval: Dict) -> Dict:
    memory = load_feedback_memory()

    if not memory or "sources" not in retrieval:
        return retrieval

    doc_boost = {}

    for group in memory:
        doc = group.get("doc")
        examples = group.get("examples", [])

        for ex in examples:
            emb = np.array(ex["embedding"])
            sim = cosine_sim(query_embedding, emb)

            if sim > 0.75:
                doc_boost[doc] = doc_boost.get(doc, 0) + sim

    boosted_scores = []

    for score, meta in zip(retrieval["scores"], retrieval["sources"]):
        doc_id = meta.get("source_path") or meta.get("source_group")
        boost = doc_boost.get(doc_id, 0)
        boosted_scores.append(score + boost)

    ranked = sorted(
        zip(boosted_scores, retrieval["results"], retrieval["sources"]),
        key=lambda x: x[0],
        reverse=True,
    )[:TOP_K]

    retrieval["scores"] = [r[0] for r in ranked]
    retrieval["results"] = [r[1] for r in ranked]
    retrieval["sources"] = [r[2] for r in ranked]
    retrieval["top_score"] = retrieval["scores"][0] if retrieval["scores"] else None

    if doc_boost:
        debug_print(f"[FEEDBACK] boosted_docs={len(doc_boost)}")

    return retrieval

def save_feedback_memory(memory):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def add_feedback_example(memory, query, embedding, docs):
    if (
            not query
            or embedding is None
            or len(embedding) == 0
            or not docs
    ):
        return memory

    doc_set = set(docs)

    for doc_id in doc_set:
        group = next((g for g in memory if g.get("doc") == doc_id), None)

        if group is None:
            memory.append({
                "doc": doc_id,
                "examples": [
                    {
                        "query": query,
                        "embedding": embedding
                    }
                ]
            })
            continue

        examples = group.get("examples", [])

        max_sim = 0.0
        for ex in examples:
            sim = cosine_sim(embedding, ex.get("embedding", []))
            if sim > max_sim:
                max_sim = sim

        if max_sim < SIMILARITY_DUP_THRESHOLD:
            examples.append({
                "query": query,
                "embedding": embedding
            })

        group["examples"] = examples[-MAX_EXAMPLES_PER_DOC:]

    return memory[-MAX_DOC_GROUPS:]

