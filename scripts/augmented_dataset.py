import json
import re
import os
from tqdm import tqdm

INPUT_ORIGINAL = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/original/msnbc.jsonl"
INPUT_LLM = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/llm_raw/msnbc_qwen.jsonl"
OUTPUT_DIR = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/augmented"

OUTPUT = os.path.join(
    OUTPUT_DIR,
    "msnbc_join4.jsonl"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


JOIN_STRATEGY = 4
MAX_LENGTH = 192

def augment():
    with open(INPUT_ORIGINAL, "r") as f1, \
         open(INPUT_LLM, "r") as f2, \
         open(OUTPUT, "w") as out:

        for line_o, line_llm in tqdm(zip(f1, f2)):
            # LOAD
            original = json.loads(line_o)
            llm = json.loads(line_llm)

            mention = original["mention"]
            left = original["context_left"]
            right = original["context_right"]
            llm_context = llm["llm_context"]

            # CLEAN
            llm_context = re.sub(r"<[^>]+>", "", llm_context)
            llm_context = re.sub(r"\s+", " ", llm_context).strip()

            # FALLBACK
            if len(llm_context) < 5:
                llm_context = (f"{mention} is a named entity.")
 
            #################################################
            # JOIN STRATEGY 4
            # Original + LLM
            #################################################

            half = MAX_LENGTH // 2
            tokens_left = left.split()
            tokens_left = tokens_left[-half:]

            tokens_right_orig = right.split()
            tokens_right_orig = tokens_right_orig[:half]

            new_right = " ".join(tokens_right_orig) + "\n" + llm_context

            # SAVE
            original["context_left"] = " ".join(tokens_left)
            original["context_right"] = new_right

            json.dump(original, out)
            out.write("\n")

    print("\nDONE")

if __name__ == "__main__":
    augment()