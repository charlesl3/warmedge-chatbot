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

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = "https://router.huggingface.co/v1/chat/completions"


def run_llm(prompt: str) -> str:
    token = os.getenv("HF_TOKEN")

    if not token:
        return "HF_TOKEN not set."

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
        "max_tokens": 220,   # reduced from 700
        "temperature": 0.25,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("BODY:", response.text)
        raise RuntimeError(response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]
