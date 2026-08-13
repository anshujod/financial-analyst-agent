import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "judged_results.jsonl"
SUMMARY_CSV_PATH = ROOT / "data" / "eval_summary.csv"

DIMENSIONS = ["tone_terminology", "factual_grounding"]
TOP_N_STANDOUTS = 4


def load_judged(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_summary_table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for label, key in [("base", "base_scores"), ("finetuned", "finetuned_scores")]:
        record = {"model": label}
        for dim in DIMENSIONS:
            record[dim] = sum(r[key][dim] for r in rows) / len(rows)
        records.append(record)
    return pd.DataFrame(records).set_index("model")


def compute_improvement(row: dict) -> int:
    """Total point improvement of finetuned over base, summed across dimensions."""
    return sum(
        row["finetuned_scores"][dim] - row["base_scores"][dim] for dim in DIMENSIONS
    )


def print_standout(row: dict) -> None:
    delta = compute_improvement(row)
    print(f"\n{'=' * 60}")
    print(f"Question: {row['question']}")
    print(f"Improvement: {delta:+d} points ({row['base_scores']} -> {row['finetuned_scores']})")
    print(f"\nReference: {row['reference_answer']}")
    print(f"\nBase:      {row['base_answer']}")
    print(f"\nFine-tuned: {row['finetuned_answer']}")


def main():
    rows = load_judged(INPUT_PATH)

    summary = build_summary_table(rows)
    summary.to_csv(SUMMARY_CSV_PATH)
    print("Average scores (base vs. fine-tuned):")
    print(summary.round(2))
    print(f"\nSaved summary table to {SUMMARY_CSV_PATH}")

    for dim in DIMENSIONS:
        delta = summary.loc["finetuned", dim] - summary.loc["base", dim]
        if delta <= 0:
            print(
                f"\nFLAG: fine-tuning showed no improvement (or regressed) on '{dim}' "
                f"(base={summary.loc['base', dim]:.2f}, finetuned={summary.loc['finetuned', dim]:.2f}). "
                "Investigate before proceeding."
            )

    ranked = sorted(rows, key=compute_improvement, reverse=True)
    print(f"\n\n{'#' * 60}")
    print(f"# Top {TOP_N_STANDOUTS} standout examples (largest improvement)")
    print(f"{'#' * 60}")
    for row in ranked[:TOP_N_STANDOUTS]:
        print_standout(row)


if __name__ == "__main__":
    main()
