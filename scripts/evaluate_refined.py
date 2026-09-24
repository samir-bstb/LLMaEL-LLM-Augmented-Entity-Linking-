import json
from refined.inference.processor import Refined
from refined.data_types.base_types import Span   
import re

# MODEL
refined = Refined.from_pretrained(
    model_name="wikipedia_model",
    entity_set="wikipedia"
)

DATASET_PATH = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/augmented/msnbc_join4.jsonl"
#DATASET_PATH = "/home/samir_bstb/proyectos/pytorch/nlp/dataset/original/msnbc.jsonl"
MAX_SAMPLES = 656

def normalize(text):
    if text is None:
        return ""

    text = text.lower()

    # underscores -> spaces
    text = text.replace("_", " ")

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # strip spaces
    text = text.strip()

    return text

def best_span(spans, mention):
    mention_l = mention.lower().strip()

    # 1. exact match
    for span in spans:
        if span.text.lower().strip() == mention_l:
            return span

    # 2. mention is in the content or viceversa
    for span in spans:
        t = span.text.lower().strip()
        if mention_l in t or t in mention_l:
            return span

    return None

# METRICS
correct = 0
total = 0
no_prediction = 0

# EVALUATION

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= MAX_SAMPLES:
            break

        sample = json.loads(line)

        mention = sample["mention"]
        left = sample["context_left"]
        right = sample["context_right"]

        text = left + " " + mention + " " + right

        mention_start = len(left) + 1
        mention_end = mention_start + len(mention)

        try:
            from refined.data_types.base_types import Span

            span_obj = Span(
                text=mention,
                start=mention_start,
                ln=len(mention)
            )

            spans = refined.process_text(
                text,
                spans=[span_obj]
            )

            prediction = None

            if spans and spans[0].predicted_entity is not None:
                prediction = (
                    spans[0].predicted_entity.wikipedia_entity_title
                )

            if prediction is None:
                no_prediction += 1
            else:
                label = sample.get("label") or sample.get("Wikipedia_title", "")
                if normalize(prediction) == normalize(label):
                    correct += 1

            print("\n================")
            print("MENTION:", mention)
            print("LABEL  :", label)
            print("PRED   :", prediction)

            total += 1

        except Exception as e:
            print("ERROR:", e)

# RESULTS

accuracy = correct / total if total > 0 else 0

print("\n==============================")
print("RESULTS")
print("==============================")
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"No prediction: {no_prediction}")
print(f"Accuracy: {accuracy:.4f}")