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
API_URL = f"https://router.huggingface.co/v1/chat/completions"

HF_TOKEN = os.getenv("HF_API_TOKEN")


def run_llm(prompt: str) -> str:
    if not HF_TOKEN:
        return "HF_API_TOKEN not set."

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("BODY:", response.text)
        raise RuntimeError(response.text)

    data = response.json()
    return data["choices"][0]["message"]["content"]
