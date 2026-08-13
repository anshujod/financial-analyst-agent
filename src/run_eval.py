"""Run the eval set through the base model and the base+LoRA-adapter model, saving both
answers alongside the reference answer.

NOTE: generation on the local M3 (MPS, fp16, unquantized) benchmarks at roughly 14 minutes
per answer (see Task 3.4 / src/load_adapter.py). Running the full eval set (2 models x all
eval rows) locally would take on the order of half a day. Use LIMIT below for a quick local
smoke test; for the full run, use the equivalent cells in colab/qlora_finetune.ipynb on a
T4 GPU, where each generation takes seconds instead of minutes.
"""

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = ROOT / "adapter"
EVAL_PATH = ROOT / "data" / "eval.jsonl"
OUTPUT_PATH = ROOT / "data" / "eval_results.jsonl"

MAX_NEW_TOKENS = 200
LIMIT = 2  # smoke-test size for local runs; pass --full to run the entire eval set


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_eval_rows(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def split_messages(messages: list[dict]) -> tuple[list[dict], str]:
    """Return (prompt_messages, reference_answer) from a formatted eval row."""
    assert messages[-1]["role"] == "assistant"
    return messages[:-1], messages[-1]["content"]


def generate(model, tokenizer, device: str, prompt_messages: list[dict]) -> str:
    inputs = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)


def main():
    limit = None if "--full" in sys.argv else LIMIT
    rows = load_eval_rows(limit)
    print(f"Evaluating {len(rows)} row(s){' (full set)' if limit is None else ' (smoke test — pass --full for the entire eval set)'}")

    device, dtype = pick_device_and_dtype()
    print(f"Loading base model + adapter on '{device}' with dtype {dtype} ...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    # Load once: PeftModel.disable_adapter() below gives us the base-only forward pass
    # without loading the base weights into memory a second time.
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    results = []
    for i, row in enumerate(rows):
        prompt_messages, reference_answer = split_messages(row["messages"])
        question = prompt_messages[-1]["content"].split("Question: ")[-1]

        print(f"\n[{i + 1}/{len(rows)}] {question[:80]}")

        start = time.time()
        with model.disable_adapter():
            base_answer = generate(model, tokenizer, device, prompt_messages)
        print(f"  base done in {time.time() - start:.0f}s")

        start = time.time()
        finetuned_answer = generate(model, tokenizer, device, prompt_messages)
        print(f"  finetuned done in {time.time() - start:.0f}s")

        results.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "base_answer": base_answer,
                "finetuned_answer": finetuned_answer,
            }
        )

    with OUTPUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} result(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
