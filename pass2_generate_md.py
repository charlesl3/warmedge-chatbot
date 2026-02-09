import json
import subprocess
from pathlib import Path

# ========================
# Paths
# ========================

DATA_DIR = Path("data")

PASS1_DIR = DATA_DIR / "pass1_threads_general"
PASS2_DIR = DATA_DIR / "pass2_threads_general"

GOOD_IN_1 = DATA_DIR / "pass2_good_input1.json"
GOOD_OUT_1 = DATA_DIR / "pass2_good_output1.md"
GOOD_IN_2 = DATA_DIR / "pass2_good_input2.json"
GOOD_OUT_2 = DATA_DIR / "pass2_good_output2.md"

PASS2_DIR.mkdir(parents=True, exist_ok=True)


# ========================
# LLM call (Ollama)
# ========================

def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


# ========================
# Few-shot builder
# ========================

def build_few_shot():
    examples = []

    for inp_path, out_path in [
        (GOOD_IN_1, GOOD_OUT_1),
        (GOOD_IN_2, GOOD_OUT_2),
    ]:
        with open(inp_path, "r", encoding="utf-8") as f:
            inp = json.load(f)

        with open(out_path, "r", encoding="utf-8") as f:
            out_md = f.read()

        posts_min = [
            {
                "is_op": p["is_op"],
                "text": p["text"]
            }
            for p in inp["posts"]
        ]

        example = f"""
### EXAMPLE INPUT
Thread title:
{inp["thread_title"]}

Thread URL:
{inp["thread_url"]}

Posts:
{json.dumps(posts_min, ensure_ascii=False, indent=2)}

### EXAMPLE OUTPUT
{out_md}
"""
        examples.append(example)

    return "\n\n".join(examples)


# ========================
# System prompt
# ========================

SYSTEM_PROMPT = """
You are transforming a figure skating forum discussion into ONE reusable knowledge unit.

This is a PARAPHRASING + INTEGRATION task.
You are rewriting real skaters’ experiences into a clear, readable reference, 
in the format of ONE Markdown knowledge unit.




GENERAL RULES:
- Treat the entire forum thread as ONE shared skating experience
- Use ONLY information explicitly present in the thread
- Do NOT remove, generalize, or simplify specific details such as:
  - brand names
  - boot or blade models
  - body metrics (height, weight, age)
  - skill level labels
  - measurements
  - locations
- Do NOT produce generic paraphrases of the original post or replies
- You MUST include all concrete examples mentioned in the thread
- Do NOT quote users verbatim or reference usernames
- Do NOT invent facts or add outside knowledge
- Do NOT give medical, biomechanical, or professional advice
- Combine repeated or overlapping ideas into a single coherent explanation
- Preserve uncertainty, disagreement, and multiple viewpoints when they appear
- Write for figure skaters who will read this later as experiential guidance, not instruction


OUTPUT FORMAT:
- Markdown only
- Use the SAME structure and style as the provided examples
- Natural language paragraphs (lists are allowed but not required)
- Clear section headers: Experience → Original situation → What skaters commonly suggest → What to try (non-medical) → Uncertainty / caveats
- Do NOT add or remove sections

YOU MUST USE EXACTLY THESE SECTIONS AND NAMES:

## Experience: <concise descriptive title>

### Original situation
- Describe the skater’s situation and main concern
- Base this primarily on posts where "is_op": true
- Include relevant background (equipment, symptoms, skill level, context)

### What skaters commonly suggest
- Summarize patterns from replies where "is_op": false
- Focus on shared observations, not individuals
- Include multiple viewpoints if present

### What to try (non-medical)
- Describe actions or approaches skaters commonly mention
- Present them as possibilities, not guarantees
- Do NOT frame anything as instruction or advice

### Uncertainty / caveats
- Clearly state where experiences differ
- Note limitations, disagreements, or individual dependence

STYLE NOTES:
- Write as a synthesized experience, not a checklist
- Avoid absolute language
- Prefer phrasing like:
  “Many skaters report…”
  “Some people find…”
  “Others note…”

Do NOT include explanations or text outside the Markdown.
"""



# ========================
# Pass 2 runner
# ========================

def run_pass2():
    few_shot = build_few_shot()
    files = sorted(PASS1_DIR.glob("*.json"))

    print(f"PASS 2: processing {len(files)} threads")

    for f in files:
        with open(f, "r", encoding="utf-8") as infile:
            thread = json.load(infile)

        posts_min = [
            {
                "is_op": p["is_op"],
                "text": p["text"]
            }
            for p in thread["posts"]
        ]

        user_prompt = f"""
### INPUT
Thread title:
{thread["thread_title"]}

Thread URL:
{thread["thread_url"]}

Posts:
{json.dumps(posts_min, ensure_ascii=False, indent=2)}

### TASK
Write a knowledge unit in the same style as the examples above.
"""

        full_prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + few_shot
            + "\n\n"
            + user_prompt
        )

        try:
            output_md = call_ollama(full_prompt)
        except Exception as e:
            print(f"✗ FAIL {f.name}: {e}")
            continue

        out_path = PASS2_DIR / f.with_suffix(".md").name
        with open(out_path, "w", encoding="utf-8") as outfile:
            outfile.write(output_md)

        print(f"✓ {out_path.name}")

    print("PASS 2 complete.")


# ========================
# Entry
# ========================

if __name__ == "__main__":
    run_pass2()
