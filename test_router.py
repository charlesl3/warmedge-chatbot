import os
import requests
import json

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-20b"



def main():
    token = os.getenv("HF_TOKEN")

    print("=== ENV CHECK ===")
    if not token:
        print("HF_TOKEN not set.")
        return
    else:
        print("HF_TOKEN detected.")
        print("Token prefix:", token[:10], "...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello briefly."}
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    print("\n=== SENDING REQUEST ===")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except Exception as e:
        print("Network error:", e)
        return

    print("\n=== STATUS ===")
    print(response.status_code)

    try:
        data = response.json()
        print("\n=== RESPONSE JSON ===")
        print(json.dumps(data, indent=2))
    except Exception:
        print("\n=== RAW RESPONSE ===")
        print(response.text)

    if response.status_code == 200:
        print("\nRouter + model working.")
    elif response.status_code == 400:
        print("\nModel not supported or not chat-compatible.")
    elif response.status_code == 401:
        print("\nAuthentication failed.")
    elif response.status_code == 403:
        print("\nAccess or billing issue.")
    else:
        print("\nUnexpected status.")

if __name__ == "__main__":
    main()
