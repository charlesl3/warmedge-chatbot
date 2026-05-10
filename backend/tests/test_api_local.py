import os
import sys
import time
import threading

# -------------------------
# Silence transformers noise (CLI only)
# -------------------------
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from transformers import logging
logging.set_verbosity_error()

# -------------------------
# API import
# -------------------------
from backend.api_stub import chat_api

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
# CLI test loop
# -------------------------
def main():
    history = []

    print("Local API test (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        # Manual escape hatch
        if user_input.lower() in {"exit", "quit"}:
            break

        # ---- Thinking spinner ----
        stop_event = threading.Event()
        t = threading.Thread(
            target=spinner,
            args=("Thinking", stop_event),
            daemon=True,
        )
        t.start()

        response = chat_api(
            message=user_input,
            history=history,
        )

        stop_event.set()
        t.join()

        print("\nAssistant:", response["reply"], "\n")

        history = response["history"]

        # Respect API-controlled termination
        if response.get("end", False):
            break


if __name__ == "__main__":
    main()
