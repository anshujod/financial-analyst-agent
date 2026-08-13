import json
from pathlib import Path

from config import client

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "eval_results.jsonl"
OUTPUT_PATH = ROOT / "data" / "judged_results.jsonl"

MODEL = "openai/gpt-4o-mini"

JUDGE_PROMPT = """You are grading two candidate answers to a financial-analyst question against
a reference answer. Score each candidate on two dimensions, 1-5 (5 is best):

- tone_terminology: precise financial-analyst voice — cites figures exactly, uses correct
  terminology (YoY, gross margin, liquidity, EBITDA, etc.) where relevant, and hedges claims
  to their source (e.g. "per the FY24 filing").
- factual_grounding: factual accuracy and consistency with the reference answer — no invented
  figures, no contradictions with the reference.

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

CANDIDATE A: {answer_a}

CANDIDATE B: {answer_b}

Respond ONLY with JSON in this exact format:
{{
  "candidate_a": {{"tone_terminology": <1-5>, "factual_grounding": <1-5>}},
  "candidate_b": {{"tone_terminology": <1-5>, "factual_grounding": <1-5>}}
}}"""


def judge_row(row: dict, model: str = MODEL) -> dict:
    """Score base_answer and finetuned_answer against reference_answer. Candidates are sent
    as blind "A"/"B" (A=base, B=finetuned here) to reduce order/label bias in the judge."""
    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        reference_answer=row["reference_answer"],
        answer_a=row["base_answer"],
        answer_b=row["finetuned_answer"],
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)

    return {
        **row,
        "base_scores": scores["candidate_a"],
        "finetuned_scores": scores["candidate_b"],
    }


def main():
    rows = [json.loads(line) for line in INPUT_PATH.read_text().splitlines() if line.strip()]

    judged = []
    for i, row in enumerate(rows):
        result = judge_row(row)
        judged.append(result)
        print(f"[{i + 1}/{len(rows)}] {row['question'][:70]}")
        print(f"  base:      {result['base_scores']}")
        print(f"  finetuned: {result['finetuned_scores']}")

    with OUTPUT_PATH.open("w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")

    dims = ["tone_terminology", "factual_grounding"]
    for label, key in [("Base", "base_scores"), ("Fine-tuned", "finetuned_scores")]:
        avgs = {d: sum(r[key][d] for r in judged) / len(judged) for d in dims}
        print(f"\n{label} averages: " + ", ".join(f"{d}={v:.2f}" for d, v in avgs.items()))

    print(f"\nSaved {len(judged)} judged rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
