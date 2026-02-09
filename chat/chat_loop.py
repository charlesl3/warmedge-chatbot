from rag.answer import answer_question
from rag.intents import (
    is_blank,
    is_social_message,
    is_farewell,
    handle_social_message,
)


def main():
    history = []

    print("WarmEdge Skating Chatbot")
    print("Ask me figure skating questions. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        # Hard exit commands (manual override)
        if user_input.lower() in {"exit", "quit"}:
            print("\nAssistant: Goodbye! Wishing you good skating sessions.\n")
            break

        # Blank input → gentle reminder
        if is_blank(user_input):
            print("\nAssistant: Please ask a valid figure skating related question.\n")
            continue

        # Social / small talk (NO RAG)
        if is_social_message(user_input):
            reply = handle_social_message(user_input)
            print(f"\nAssistant: {reply}\n")

            # Farewell → end chatbot process
            if is_farewell(user_input):
                break

            # Otherwise continue the conversation
            continue

        # -------------------------
        # Real skating question → RAG
        # -------------------------
        history.append({"role": "user", "content": user_input})

        reply = answer_question(
            question=user_input,
            history=history,
        )

        print(f"\nAssistant: {reply}\n")

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
