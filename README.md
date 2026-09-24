<div style="font-family: Arial, Helvetica, sans-serif;">

# LLMaEL: LLM-Augmented Entity Linking Experimentation

This repository contains the implementation and experimental replication of the **LLMaEL (Large Language Models as Context Augmenters for Entity Linking)** framework, based on research by THU-KEG ([Paper on Hugging Face](https://huggingface.co/papers/2407.04020) | [Official Repository](https://github.com/THU-KEG/LLMAEL)).

In this experiment, we adapted the context augmentation pipeline using a local language model (**Qwen 2.5:7b via Ollama**) to enhance predictions made by the specialized Entity Linking model **ReFinED** on the **MSNBC** dataset.

## Background and Problem Statement

* **Entity Linking (EL):** The task of identifying ambiguous mentions in text (e.g., *"Apple"*) and linking them to a unique entity in a knowledge base (e.g., *Apple Inc.* vs. *the fruit*).

* **The Challenge:**
  * Specialized EL models (such as BLINK, GENRE, or ReFinED) excel at format mapping but struggle with rare or long-tail entities due to limited training data.
  * Large Language Models (LLMs) possess vast parametric knowledge regarding rare entities, but directly querying them for EL tasks often leads to hallucinations or incorrect Knowledge Base entity formatting.

* **The LLMaEL Approach:** Utilize the LLM not to perform entity linking directly, but as a **context augmenter**. The LLM generates descriptive background context, which is then fed into the specialized EL model.

## Experimental Methodology

### 1. Context Augmentation ($c'$)

Given an original text $c$ containing a mention $m_i$, we pass the following prompt to the LLM:

$$
\text{Prompt: Consider the following text. Text: } [CONTEXT] \text{ Please provide me more descriptive information about } [MENTION] \text{ from the text above. Make sure to include } [MENTION] \text{ in your description.}
$$

The LLM produces augmented context $c' = LLM(p, c, m_i)$.

### 2. Context Joining Strategy (Strategy 4)

From the various concatenation strategies evaluated in the original paper, we implemented **Strategy 4 (Original + LLM with mention in original)**:

* **Original Text:** *"Jordan scored 30 points in the finals."* (Offset points to mention in original text: *"Jordan"*)
* **Generated Text:** *"Michael Jordan is a legendary NBA basketball player."*
* **Final Concatenated Text:** `[Original Context] + [Generated Context]`, instructing the EL model that the target mention offset remains within the original text segment.

## Experimental Setup

| Parameter | Configuration |
| --- | --- |
| **Dataset** | MSNBC (656 samples) |
| **EL Model** | ReFinED |
| **Local LLM** | Ollama (`qwen2.5:7b`) |
| **Concatenation Strategy** | Strategy 4 (`Original + LLM`, offset in Original) |

## Project Structure

```
.
├── dataset/
│   ├── msnbc.jsonl             # Original MSNBC dataset
│   ├── raw/                    # Raw LLM outputs
│   └── augmented/              # Concatenated datasets (Strategy 4)
├── scripts/
│   ├── generate_context.py     # Script to query Ollama and generate context
│   ├── augmented_dataset.py    # Script applying Strategy 4 concatenation
│   ├── evaluate_refined.py     # Evaluation script for ReFinED (Original vs Augmented)
│   └── prompts.py              # Prompt template definitions
├── requirements.txt            # Project dependencies
└── README.md
```

## Results

Evaluation conducted across **656 samples** of the MSNBC dataset:

| Model / Configuration | Correct Samples | No Prediction | Accuracy (%) |
| --- | --- | --- | --- |
| **ReFinED (Baseline / Original)** | 543 | 47 | **82.77%** |
| **ReFinED + LLMaEL (Strategy 4)** | **550** | **33** | **83.84%** |
| **Difference / Improvement** | **+7** | **-14** | **+1.07%** |

### Key Takeaways

1. **Accuracy Increase:** Incorporating generated context from `qwen2.5:7b` improved overall accuracy by **+1.07%**.
2. **Error Reduction:** Unpredicted cases ("No prediction") dropped from 47 to 33. This confirms that additional descriptive context helped ReFinED disambiguate cases where original text lacked sufficient information.

## How to Run

### 1. Requirements and Setup

Ensure Python 3.10+ and [Ollama](https://ollama.com/) are installed on your system.

```bash
# Pull the LLM model in Ollama
ollama pull qwen2.5:7b

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Context with the LLM

```bash
python scripts/generate_context.py
```

### 3. Create Augmented Dataset (Strategy 4)

```bash
python scripts/augmented_dataset.py
```

### 4. Evaluate with ReFinED

```bash
python scripts/evaluate_refined.py
```

## Resources and References

* **Video Walkthrough:** [Watch on YouTube](https://youtu.be/gbH7Hb9nQuY)
* **Original Paper:** *LLMaEL: Large Language Models are Good Context Augmenters for Entity Linking* ([Hugging Face](https://huggingface.co/papers/2407.04020))
* **Official LLMaEL Repository:** [THU-KEG/LLMAEL](https://github.com/THU-KEG/LLMAEL)

</div>
