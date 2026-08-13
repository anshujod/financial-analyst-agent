import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_qa_pairs.jsonl"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "eval.jsonl"

EVAL_FRACTION = 0.15
CONTEXT_CHAR_LIMIT = 12000  # matches the slice actually shown to the LLM during QA generation

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
    }


def stratified_split(rows: list[dict], eval_fraction: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split ~eval_fraction of each section's rows into eval, keeping the rest in train.
    Sections with only one row go entirely to train (a single example can't be split)."""
    rng = random.Random(seed)
    by_section = defaultdict(list)
    for row in rows:
        by_section[row["node_id"]].append(row)

    train, eval_ = [], []
    for section_rows in by_section.values():
        rng.shuffle(section_rows)
        n_eval = round(len(section_rows) * eval_fraction) if len(section_rows) >= 2 else 0
        eval_.extend(section_rows[:n_eval])
        train.extend(section_rows[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    pairs = load_pairs(INPUT_PATH)
    formatted = [to_chat_format(p) for p in pairs]

    train, eval_ = stratified_split(formatted, EVAL_FRACTION)
    write_jsonl(train, TRAIN_PATH)
    write_jsonl(eval_, EVAL_PATH)

    train_sections = {r["node_id"] for r in train}
    eval_sections = {r["node_id"] for r in eval_}
    both = train_sections & eval_sections

    print(f"Total examples: {len(formatted)}")
    print(f"Train: {len(train)} ({TRAIN_PATH})")
    print(f"Eval:  {len(eval_)} ({EVAL_PATH})")
    print(f"Sections represented in both splits: {len(both)}/{len(train_sections | eval_sections)}")

    print("\n--- 3 example formatted rows ---")
    for row in train[:3]:
        print(f"\nnode: {row['node_title']}")
        print(json.dumps(row["messages"], indent=2)[:1500])


if __name__ == "__main__":
    main()
