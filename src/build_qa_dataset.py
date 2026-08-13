import json
import time
from collections import defaultdict
from pathlib import Path

from generate_qa import generate_qa_for_node

ROOT = Path(__file__).resolve().parent.parent
TREE_PATH = ROOT / "data" / "financial_tree.json"
OUTPUT_PATH = ROOT / "data" / "raw_qa_pairs.jsonl"

MAX_RETRIES = 5
BASE_DELAY = 2  # seconds


def generate_with_retry(node: dict) -> list[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            return generate_qa_for_node(node)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  FAILED after {MAX_RETRIES} attempts on '{node['title']}': {e}")
                return []
            delay = BASE_DELAY * (2**attempt)
            print(f"  retry {attempt + 1}/{MAX_RETRIES} for '{node['title']}' after {e} "
                  f"(sleeping {delay}s)")
            time.sleep(delay)
    return []


def main():
    nodes = json.loads(TREE_PATH.read_text())
    all_pairs = []

    for i, node in enumerate(nodes):
        pairs = generate_with_retry(node)
        all_pairs.extend(pairs)
        print(f"[{i + 1}/{len(nodes)}] {node['title']}: {len(pairs)} pairs")

    with OUTPUT_PATH.open("w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    per_node_counts = defaultdict(int)
    total_answer_len = 0
    for pair in all_pairs:
        per_node_counts[pair["node_title"]] += 1
        total_answer_len += len(pair["answer"])

    avg_answer_len = total_answer_len / len(all_pairs) if all_pairs else 0

    print(f"\nSaved {len(all_pairs)} pairs to {OUTPUT_PATH}")
    print(f"Average answer length: {avg_answer_len:.0f} characters")
    print("\nPairs per section:")
    for title, count in per_node_counts.items():
        print(f"  {count:3d}  {title}")


if __name__ == "__main__":
    main()
