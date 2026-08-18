import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_qa_pairs.jsonl"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "eval.jsonl"

EVAL_FRACTION = 0.15
# Keep prompts comfortably under the notebook's max_length=4096 (measured ~2.9k tokens
# for 10k chars + system prompt + chat template). Excerpts longer than this get cut.
CONTEXT_CHAR_LIMIT = 10000

SYSTEM_PROMPT = (
    "You are a meticulous financial analyst assistant. Answer questions about NVIDIA's "
    "fiscal year 2024 Form 10-K using only the provided filing excerpt. Cite figures "
    "precisely, use correct financial terminology, and hedge claims to their source "
    "(e.g. 'per the FY24 filing', 'as reported in Item 7')."
)


def load_pairs(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def to_chat_format(pair: dict) -> dict:
    context = pair["context"][:CONTEXT_CHAR_LIMIT]
    user_content = f"Filing excerpt ({pair['node_title']}):\n{context}\n\nQuestion: {pair['question']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": pair["answer"]},
        ],
        "node_id": pair["node_id"],
        "node_title": pair["node_title"],
        "context": context,
    }


def excerpt_split(rows: list[dict], eval_fraction: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by *excerpt* (the retrieved context text), not by section, holding out
    entire excerpts for eval. Questions derived from the same excerpt must not appear
    in both train and eval — otherwise the eval measures memorization of seen text
    rather than grounding on unseen excerpts (what the agent actually faces at
    inference, when the retriever returns passages the model never saw)."""
    rng = random.Random(seed)
    by_excerpt = defaultdict(list)
    for row in rows:
        by_excerpt[row["context"]].append(row)

    excerpts = list(by_excerpt)
    rng.shuffle(excerpts)

    n_eval_rows = round(len(rows) * eval_fraction)
    eval_excerpts: set[str] = set()
    eval_rows = 0
    for ex in excerpts:
        if eval_rows >= n_eval_rows:
            break
        eval_excerpts.add(ex)
        eval_rows += len(by_excerpt[ex])

    train = [r for ex, rs in by_excerpt.items() if ex not in eval_excerpts for r in rs]
    eval_ = [r for ex in eval_excerpts for r in by_excerpt[ex]]
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def check_truncation(rows: list[dict], max_tokens: int = 4096) -> None:
    """Warn if any formatted row would be truncated at the notebook's max_length.
    Requires the Qwen tokenizer; skipped if transformers isn't available."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ROOT / "adapter")
    except Exception:
        print(f"[check_truncation] tokenizer unavailable — skipping (run with the venv that has transformers)")
        return

    over = []
    for i, row in enumerate(rows):
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        n = len(tok(text, truncation=True, max_length=max_tokens)["input_ids"])
        if n >= max_tokens:
            over.append((i, n))
    if over:
        print(f"[check_truncation] WARNING: {len(over)}/{len(rows)} rows hit max_length={max_tokens} "
              f"(e.g. row {over[0][0]} = {over[0][1]} tokens). Raise max_length or shrink CONTEXT_CHAR_LIMIT.")
    else:
        print(f"[check_truncation] OK: all {len(rows)} rows fit under {max_tokens} tokens")


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    pairs = load_pairs(INPUT_PATH)
    formatted = [to_chat_format(p) for p in pairs]

    train, eval_ = excerpt_split(formatted, EVAL_FRACTION)

    train_excerpts = {r["context"] for r in train}
    leaked = sum(1 for r in eval_ if r["context"] in train_excerpts)
    print(f"Total examples: {len(formatted)}")
    print(f"Train: {len(train)} ({TRAIN_PATH})")
    print(f"Eval:  {len(eval_)} ({EVAL_PATH})")
    print(f"Eval rows sharing an excerpt with train (must be 0): {leaked}")

    write_jsonl(train, TRAIN_PATH)
    write_jsonl(eval_, EVAL_PATH)

    check_truncation(train)
    check_truncation(eval_)

    print("\n--- 3 example formatted rows ---")
    for row in train[:3]:
        print(f"\nnode: {row['node_title']}")
        print(json.dumps(row["messages"], indent=2)[:1500])


if __name__ == "__main__":
    main()
