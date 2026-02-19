# import subprocess
#
# OLLAMA_MODEL = "llama3.1:8b"
#
#
# def run_ollama(text: str) -> str:
#     result = subprocess.run(
#         ["ollama", "run", OLLAMA_MODEL],
#         input=text,
#         text=True,
#         capture_output=True,
#         check=True,
#     )
#     return result.stdout.strip()



import os
import requests

MODEL_NAME = "openai/gpt-oss-20b"
API_URL = "https://router.huggingface.co/v1/chat/completions"


def run_llm(prompt: str) -> str:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN not set.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise, practical skating assistant. Follow all instructions carefully."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("BODY:", response.text)
        raise RuntimeError(response.text)

    data = response.json()

    if "choices" not in data or len(data["choices"]) == 0:
        raise RuntimeError(f"Unexpected response format: {data}")

    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()

    # 🚨 Guard against blank model output
    if not content:
        print("WARNING: Model returned empty content. Retrying once...")
        retry = requests.post(API_URL, headers=headers, json=payload)

        if retry.status_code == 200:
            retry_data = retry.json()
            retry_message = retry_data["choices"][0]["message"]
            retry_content = (retry_message.get("content") or "").strip()

            if retry_content:
                return retry_content

        return "I need a moment — could you repeat that question?"

    return content
