import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw_qa_pairs.jsonl"
OUTPUT_PATH = ROOT / "data" / "reviewed_qa_pairs.jsonl"

SAMPLE_FRACTION = 0.2
CONTEXT_SNIPPET_LEN = 300


def load_pairs(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def print_pair(index: int, total: int, pair: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{index}/{total}] Section: {pair['node_title']}")
    print(f"{'=' * 60}")
    print(f"Q: {pair['question']}")
    print(f"\nContext (snippet): {pair['context'][:CONTEXT_SNIPPET_LEN].strip()}...")
    print(f"\nA: {pair['answer']}")


def review_pair(pair: dict) -> dict | None:
    while True:
        choice = input("\nkeep / edit / drop / quit? ").strip().lower()
        if choice in ("keep", "k"):
            return pair
        if choice in ("drop", "d"):
            return None
        if choice in ("edit", "e"):
            new_question = input(f"New question [{pair['question']}]: ").strip()
            new_answer = input("New answer (leave blank to keep current):\n").strip()
            edited = dict(pair)
            if new_question:
                edited["question"] = new_question
            if new_answer:
                edited["answer"] = new_answer
            return edited
        if choice in ("quit", "q"):
            raise KeyboardInterrupt
        print("Please type 'keep', 'edit', 'drop', or 'quit'.")


def main():
    all_pairs = load_pairs(INPUT_PATH)
    sample_size = max(1, round(len(all_pairs) * SAMPLE_FRACTION))
    sampled = random.sample(all_pairs, sample_size)

    print(f"Reviewing {sample_size} of {len(all_pairs)} pairs ({SAMPLE_FRACTION:.0%} sample).")
    print("For each pair: type 'keep', 'edit', 'drop', or 'quit' to stop early and save progress.")

    reviewed = []
    try:
        for i, pair in enumerate(sampled, start=1):
            print_pair(i, sample_size, pair)
            result = review_pair(pair)
            if result is not None:
                reviewed.append(result)
    except KeyboardInterrupt:
        print("\n\nStopping early, saving progress so far.")

    with OUTPUT_PATH.open("w") as f:
        for pair in reviewed:
            f.write(json.dumps(pair) + "\n")

    print(f"\nSaved {len(reviewed)} reviewed pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
