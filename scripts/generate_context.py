import ollama
import json
import os
import sys
import time
import re
from tqdm import tqdm

sys.path.append(
    "/home/samir_bstb/proyectos/pytorch/nlp"
)

from prompts import build_prompt

MODEL_NAME = "qwen2.5:7b"
BASE_INPUT = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/original"
BASE_OUTPUT = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/llm_raw"
os.makedirs(BASE_OUTPUT, exist_ok=True)

INPUT_FILES = [
    "msnbc.jsonl"
]

MAX_SAMPLES = 656

def clean_generation(text, mention):
    if text is None:
        return mention
    
    # REMOVE EXTRA SPACES
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # REMOVE PREFIXES
    text = re.sub(
        r"^(Answer:|Output:|Sentence:)",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    # REMOVE QUOTES
    text = text.strip("\"' ")

    # AVOID EMPTY
    if len(text) == 0:
        return f"{mention} is a named entity."

    # AVOID TRIVIAL OUTPUT
    normalized = text.lower().strip()

    mention_norm = mention.lower().strip()

    if normalized == mention_norm:
        return f"{mention} is a notable entity mentioned in the text."

    # FORCE MENTION
    if mention_norm not in normalized:

        text = f"{mention} {text}"

    # LIMIT LENGTH
    words = text.split()

    if len(words) > 25:
        text = " ".join(words[:25])

    # FORCE PERIOD
    if not text.endswith("."):
        text += "."

    return text

# GENERATE
for filename in INPUT_FILES:
    input_path = os.path.join(
        BASE_INPUT,
        filename
    )

    output_name = filename.replace(
        ".jsonl",
        "_qwen.jsonl"
    )

    output_path = os.path.join(
        BASE_OUTPUT,
        output_name
    )

    print(f"\nProcessing {filename}")

    data = []

    with open(input_path, "r") as infile:

        for idx, line in enumerate(infile):

            if idx >= MAX_SAMPLES:
                break

            data.append(json.loads(line))

    # GENERATE
    with open(output_path, "w") as outfile:
        for sample in tqdm(data):
            mention = sample["mention"]
            prompt = build_prompt(sample)
            success = False
            backoff_delay = 5

            while not success:
                try:
                    # OLLAMA
                    response = ollama.chat(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    text = response["message"]["content"]

                    llm_context = clean_generation(
                        text,
                        mention
                    )

                    # SAVE
                    output = {
                        "mention": mention,
                        "llm_context": llm_context
                    }

                    json.dump(output, outfile)
                    outfile.write("\n")
                    success = True

                except Exception as e:
                    print(
                        f"\nGeneration error: {e}"
                    )

                    print(
                        f"Retrying in {backoff_delay}s..."
                    )

                    time.sleep(backoff_delay)

                    backoff_delay = min(
                        backoff_delay * 2,
                        60
                    )

            time.sleep(0.5)

print("\nALL DONE")