from backend.api_stub import chat_api


def main():
    history = []

    print("Local API test (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        # Manual escape hatch
        if user_input.lower() in {"exit", "quit"}:
            break

        response = chat_api(
            message=user_input,
            history=history,
        )

        print("\nAssistant:", response["reply"], "\n")

        history = response["history"]

        # Respect API-controlled termination
        if response.get("end", False):
            break


if __name__ == "__main__":
    main()
