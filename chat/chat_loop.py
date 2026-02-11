import os
import sys
import time
import threading

# -------------------------
# Silence transformers noise
# -------------------------
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()

# -------------------------
# RAG imports
# -------------------------
from rag.answer import answer_question
from rag.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)

# -------------------------
# Spinner utility
# -------------------------
def spinner(message, stop_event):
    symbols = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {symbols[i % len(symbols)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")


# -------------------------
# Force model warm-up
# -------------------------
def warm_up_rag():
    # Trigger model + embeddings load once
    answer_question(
        question="warm up",
        history=[]
    )


# -------------------------
# Main chat loop
# -------------------------
def main():
    history = []

    print("WarmEdge Skating Chatbot")

    # ---- Startup loading animation ----
    stop_event = threading.Event()
    t = threading.Thread(
        target=spinner,
        args=("Loading skating knowledge", stop_event),
        daemon=True,
    )
    t.start()

    warm_up_rag()

    stop_event.set()
    t.join()

    print("Ready. Ask me figure skating questions. Type 'exit' to quit.\n")

    # ---- Chat loop ----
    while True:
        user_input = input("You: ").strip()

        # Hard exit
        if user_input.lower() in {"exit", "quit"}:
            print("\nAssistant: Goodbye! Wishing you good skating sessions.\n")
            break

        # Blank input
        if is_blank(user_input):
            print("\nAssistant: Please ask a valid figure skating related question.\n")
            continue

        # Social / small talk (NO RAG)
        if is_social_message(user_input):
            reply = handle_social_message(user_input)
            print(f"\nAssistant: {reply}\n")

            if is_farewell(user_input):
                break

            continue

        # ---- Real skating question → RAG ----
        history.append({"role": "user", "content": user_input})

        # Thinking spinner
        stop_event = threading.Event()
        t = threading.Thread(
            target=spinner,
            args=("Thinking", stop_event),
            daemon=True,
        )
        t.start()

        reply = answer_question(
            question=user_input,
            history=history,
        )

        stop_event.set()
        t.join()

        print(f"\nAssistant: {reply}\n")

        history.append({"role": "assistant", "content": reply})


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
