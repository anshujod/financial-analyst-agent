import json
from pathlib import Path

from config import client

ROOT = Path(__file__).resolve().parent.parent
MODEL = "openai/gpt-4o-mini"

QA_GEN_PROMPT = """You are generating fine-tuning data for a financial analyst assistant. Given
the text of one section of a 10-K filing, write {n_pairs} question/answer pairs that a financial
analyst studying this filing might ask and answer. Cover a genuinely diverse range of angles
(figures, drivers, risks, definitions, comparisons) rather than rephrasing the same fact —
if the section doesn't support that many distinct, non-redundant questions, return fewer rather
than padding with near-duplicates.

Write answers in a precise financial-analyst voice:
- Cite concrete figures and named items exactly as they appear in the text (dollar amounts,
  percentages, product/segment names).
- Use appropriate hedging that ties claims to their source, e.g. "per the FY24 filing," "as
  reported in Item 7," "according to the Consolidated Statements of Income."
- Use correct financial terminology where relevant (YoY, gross margin, liquidity, EBITDA,
  operating margin, etc.) — but only when the section actually supports that framing.
- Keep answers grounded strictly in the provided text. Do not invent figures.

Example style (from a different filing, for tone reference only):
Q: What drove the year-over-year change in gross margin?
A: Per the FY24 filing, gross margin expanded to 72.7% from 56.9% YoY, driven primarily by a
richer product mix within the Compute & Networking segment and lower per-unit costs on the
Hopper GPU platform.

Q: What is the company's current liquidity position?
A: As reported in Item 7, the company held $25.98 billion in cash, cash equivalents, and
marketable securities as of fiscal year-end, which management states is sufficient to meet
liquidity requirements for at least the next twelve months.

Now generate up to {n_pairs} Q&A pairs for this section.

Section title: {title}

Section text:
{text}

Respond ONLY with JSON in this exact format:
{{"qa_pairs": [{{"question": "...", "answer": "..."}}, ...]}}
"""


def target_pair_count(text: str) -> int:
    """Scale the requested pair count with section length so short/boilerplate sections
    (e.g. a 1-page "Item 6. [Reserved]") aren't forced into padding with near-duplicates,
    while long, information-dense sections (Risk Factors, Item 15) can yield more."""
    length = len(text)
    if length < 2000:
        return 4
    if length < 6000:
        return 7
    if length < 20000:
        return 10
    return 16


def generate_qa_for_node(node: dict, model: str = MODEL) -> list[dict]:
    """Generate Q&A pairs for one tree node's text in financial-analyst voice. Pair count
    scales with section length (see target_pair_count)."""
    n_pairs = target_pair_count(node["text"])
    prompt = QA_GEN_PROMPT.format(title=node["title"], text=node["text"][:12000], n_pairs=n_pairs)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    return [
        {
            "question": pair["question"],
            "context": node["text"],
            "answer": pair["answer"],
            "node_id": node["node_id"],
            "node_title": node["title"],
        }
        for pair in result["qa_pairs"]
    ]


def main():
    nodes = json.loads((ROOT / "data" / "financial_tree.json").read_text())
    node = next(n for n in nodes if n["node_id"] == "item-1a-risk-factors")
    pairs = generate_qa_for_node(node)
    for pair in pairs:
        print(f"\nQ: {pair['question']}")
        print(f"A: {pair['answer']}")


if __name__ == "__main__":
    main()
