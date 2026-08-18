import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Unzip the adapter downloaded from Colab (Task 3.4) into this folder before running.
ADAPTER_DIR = ROOT / "adapter"

TEST_QUESTION = "What was NVIDIA's total revenue for fiscal year 2024?"
TEST_CONTEXT = (
    "Revenue for fiscal year 2024 was $60,922 million, up 126% from $26,974 million "
    "in fiscal year 2023, driven primarily by Data Center revenue growth."
)


def adapter_size_mb(adapter_dir: Path) -> float:
    total_bytes = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
    return total_bytes / 1e6


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    # bitsandbytes 4-bit quantization is CUDA-only and does not run on Apple Silicon,
    # so local inference here uses the unquantized base model in fp16/bf16 instead of
    # replicating the Colab 4-bit setup.
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def main():
    if not ADAPTER_DIR.exists():
        print(
            f"No adapter found at {ADAPTER_DIR}.\n"
            "Unzip the adapter downloaded from the Colab notebook (Task 3.4) into that "
            "folder before running this script."
        )
        sys.exit(1)

    size_mb = adapter_size_mb(ADAPTER_DIR)
    print(f"Adapter size: {size_mb:.1f} MB")
    if size_mb > 500:
        print("WARNING: this is much larger than a typical LoRA adapter — check that only "
              "adapter weights (not the merged base model) were saved.")

    device, dtype = pick_device_and_dtype()
    print(f"Loading base model on '{device}' with dtype {dtype} ...")

    try:
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        ).to(device)
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)
        load_time = time.time() - start
        print(f"Loaded base model + adapter in {load_time:.1f}s")
    except torch.cuda.OutOfMemoryError:
        print("Out of memory while loading the model. Try closing other apps, or run this "
              "on Colab instead of locally.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model/adapter: {e}")
        sys.exit(1)

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous financial analyst assistant. Answer questions about "
            "NVIDIA's fiscal year 2024 Form 10-K using only the provided filing excerpt. Cite "
            "figures precisely, use correct financial terminology, and hedge claims to their "
            "source (e.g. 'per the FY24 filing', 'as reported in Item 7').",
        },
        {
            "role": "user",
            "content": f"Filing excerpt (Item 7. Management's Discussion and Analysis of Financial "
            f"Condition and Results of Operations):\n{TEST_CONTEXT}\n\nQuestion: {TEST_QUESTION}",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    start = time.time()
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    gen_time = time.time() - start

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    print(f"\nGeneration time: {gen_time:.1f}s")
    print(f"\nQ: {TEST_QUESTION}")
    print(f"A: {response}")


if __name__ == "__main__":
    main()
