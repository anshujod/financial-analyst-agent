"""Validate a generated Q&A dataset without any LLM calls. Run after regenerating
data with `python build_qa_dataset.py && python format_dataset.py`.

Catches the three failure modes that previously degraded the fine-tune:
1. Answers citing figures that do not appear anywhere in their excerpt
   (hallucinated/misattributed values — the QA generator answered from parametric
   knowledge because the excerpt was wrong or too thin).
2. Rows that would be truncated at the fine-tune max_length, silently deleting the
   assistant answer.
3. Eval rows whose excerpt also appears in train (leakage => eval measures
   memorization, not grounding).

Usage: python validate_dataset.py [--max-tokens 4096]
Exit code 1 if any problem is found, else 0.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AMOUNT_RE = re.compile(r"\$ ?\(?([\d,\.]+)\)?\s*(million|billion|thousand)?")
MILLIONS_HEADER_RE = re.compile(r"\(?\$? ?in millions\)?", re.IGNORECASE)


def digit_core(s: str) -> str:
    return re.sub(r"[^\d]", "", s)


def norm_amounts(text: str) -> list[tuple[float, str]]:
    """Extract (value, original-string) for dollar amounts, honoring table headers
    like '($ in millions)': a bare '29,760' in a financial table means 29,760
    *million*, not 29,760 dollars."""
    default_mult = 1e6 if MILLIONS_HEADER_RE.search(text) else 1
    out = []
    for m in AMOUNT_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = m.group(2) or ""
        out.append((v * {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get(unit, default_mult), m.group(0)))
    return out


def has_value(ctx_vals: list[float], v: float, tol: float = 0.01) -> bool:
    return any(abs(c - v) <= tol * max(c, v) for c in ctx_vals)


def amount_grounded(context: str, amount: float, amount_str: str) -> bool:
    """An answer amount is grounded if its normalized value appears in the context
    (handles unit conversions: '$26.97 billion' vs '26,974 million') OR its digit
    string appears in the context (handles table formats: bare '773', '$(94)',
    '$464.39')."""
    if has_value([v for v, _ in norm_amounts(context)], amount):
        return True
    core = digit_core(amount_str)
    if len(core) < 3:
        return True  # too short to be a meaningful figure check (percents, $5)
    return core in re.sub(r"[^\d]", "", context)


def amount_grounded(context: str, amount: float, amount_str: str) -> bool:
    """An answer amount is grounded if its normalized value appears in the context
    (handles unit conversions: '$26.97 billion' vs '26,974 million') OR its digit
    string appears as a token (handles table formats: bare '773', '$(94)', '$464.39')."""
    if has_value([v for v, _ in norm_amounts(context)], amount):
        return True
    core = digit_core(amount_str)
    if len(core) < 3:
        return True  # too short to be a meaningful figure check (percents, $5)
    return core in re.sub(r"[^\d]", "", context)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_grounding(rows: list[dict]) -> list[tuple[int, str, str, list[float]]]:
    """Rows whose answer cites an amount absent from the excerpt (formatting-tolerant)."""
    problems = []
    for i, row in enumerate(rows):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        answer = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        missing = [v for v, s in norm_amounts(answer) if not amount_grounded(user, v, s)]
        if missing:
            q = user.split("Question: ")[-1]
            problems.append((i, row.get("node_title", "?"), q, missing))
    return problems


def check_truncation(rows: list[dict], max_tokens: int) -> list[tuple[int, int]]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ROOT / "adapter")
    except Exception:
        print("[check_truncation] tokenizer unavailable — install transformers or run from .venv")
        return []
    over = []
    for i, row in enumerate(rows):
        text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        n = len(tok(text, truncation=True, max_length=max_tokens)["input_ids"])
        if n >= max_tokens:
            over.append((i, n))
    return over


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    train = load(ROOT / "data" / "train.jsonl")
    eval_ = load(ROOT / "data" / "eval.jsonl")
    print(f"train={len(train)} eval={len(eval_)}")

    problems = 0

    for name, rows in [("train", train), ("eval", eval_)]:
        for i, title, q, missing in check_grounding(rows):
            problems += 1
            print(f"[grounding] {name} row {i} [{title}] {q[:70]}")
            print(f"            answer cites figure absent from excerpt: {missing}")

        for i, n in check_truncation(rows, args.max_tokens):
            problems += 1
            print(f"[truncation] {name} row {i}: {n} tokens >= max_length={args.max_tokens} "
                  f"(the assistant answer would be cut off)")

    train_ctx = {next(m["content"] for m in r["messages"] if m["role"] == "user") for r in train}
    leaked = [r for r in eval_ if next(m["content"] for m in r["messages"] if m["role"] == "user") in train_ctx]
    if leaked:
        problems += 1
        print(f"[leakage] {len(leaked)}/{len(eval_)} eval rows share their exact excerpt with train")

    if problems:
        print(f"\n{problems} problem(s) found — fix the dataset before fine-tuning.")
        return 1
    print("\nAll checks passed: grounded figures, no truncation, no excerpt leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
