import subprocess

OLLAMA_MODEL = "llama3.1:8b"


def run_ollama(text: str) -> str:
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()
