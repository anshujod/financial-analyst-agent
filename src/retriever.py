import json
from pathlib import Path

from config import client

ROOT = Path(__file__).resolve().parent.parent
TREE_PATH = ROOT / "data" / "financial_tree.json"

MODEL = "openai/gpt-4o-mini"

TREE_SEARCH_PROMPT = """You are navigating a 10-K filing's section index to answer a financial
analyst's question. Below is a list of sections, each with a title and summary. Decide which
section(s) most likely contain the answer to the question. You may pick more than one section if
the answer could span sections, but prefer being precise.

QUESTION: {question}

SECTIONS:
{section_list}

Respond ONLY with JSON in this exact format, copying node ids exactly as they appear above
(do not invent or guess ids — copy them verbatim from the SECTIONS list):
{{
  "reasoning": "<your step-by-step reasoning about which sections are relevant and why>",
  "selected_node_ids": ["item-1a-risk-factors"]
}}"""


def load_nodes(path: Path = TREE_PATH) -> list[dict]:
    return json.loads(path.read_text())


def tree_search(question: str, nodes: list[dict], model: str = MODEL) -> dict:
    """Show the LLM only node titles + summaries; return its reasoning and selected node ids."""
    section_list = "\n".join(
        f'- {n["node_id"]}: {n["title"]}\n  Summary: {n["summary"]}' for n in nodes
    )
    prompt = TREE_SEARCH_PROMPT.format(question=question, section_list=section_list)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def retrieve(question: str, nodes: list[dict], model: str = MODEL) -> dict:
    """Run tree_search, then fetch the full text of the selected node(s)."""
    search_result = tree_search(question, nodes, model=model)
    selected_ids = set(search_result["selected_node_ids"])
    selected_nodes = [n for n in nodes if n["node_id"] in selected_ids]

    return {
        "question": question,
        "reasoning": search_result["reasoning"],
        "selected_node_ids": search_result["selected_node_ids"],
        "selected_titles": [n["title"] for n in selected_nodes],
        "context": "\n\n---\n\n".join(f'## {n["title"]}\n{n["text"]}' for n in selected_nodes),
    }


def main():
    nodes = load_nodes()
    questions = [
        "What was NVIDIA's total revenue and how did it change year over year?",
        "What are the main risk factors related to NVIDIA's reliance on third-party suppliers?",
        "What was NVIDIA's gross margin for the fiscal year?",
        "Who are NVIDIA's executive officers and directors?",
        "What legal proceedings is NVIDIA involved in?",
    ]

    for q in questions:
        result = retrieve(q, nodes)
        print(f"\nQ: {q}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Selected: {result['selected_node_ids']} -> {result['selected_titles']}")


if __name__ == "__main__":
    main()
