import json
import re
from pathlib import Path

from pypdf import PdfReader

from config import client

ROOT = Path(__file__).resolve().parent.parent
FILING_PATH = ROOT / "data" / "filing.pdf"
TREE_PATH = ROOT / "data" / "financial_tree.json"

MODEL = "openai/gpt-4o-mini"
SCAN_PAGES = 15

ITEM_HEADING_RE = re.compile(r"^(Item\s*\d+[A-C]?\.)", re.IGNORECASE)
TOC_SKIP_PAGES = 4  # cover page + table of contents; real headings never recur here


def slugify(title: str, used: set[str], max_len: int = 40) -> str:
    """Semantic id like 'item-1a-risk-factors' — avoids numeric ids that an LLM
    could confuse with the filing's own "Item N" numbering (e.g. mistaking node
    id '0008' for "Item 8" when it actually refers to a different section)."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rsplit("-", 1)[0]
    base, candidate, n = slug, slug, 2
    while candidate in used:
        candidate = f"{base}-{n}"
        n += 1
    used.add(candidate)
    return candidate


def load_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() for page in reader.pages]


def propose_sections(pages: list[str]) -> list[dict]:
    """Scan the first SCAN_PAGES pages and ask the LLM to propose section boundaries."""
    scan_text = "\n".join(
        f"--- PDF PAGE {i + 1} ---\n{text}" for i, text in enumerate(pages[:SCAN_PAGES])
    )

    prompt = f"""You are analyzing the opening pages of a 10-K annual report PDF, including its
table of contents. Propose a flat list of top-level sections: each numbered "Item" exactly as it
appears in the table of contents (e.g. "Item 1. Business", "Item 1A. Risk Factors"), plus
substantial front matter such as the cover page, with the PDF page each section starts on.

Do NOT include the "Part I" / "Part II" / "Part III" / "Part IV" divider headings as their own
sections — those are just group labels over multiple Items, not standalone sections, and would
create overlapping page ranges with the Items themselves. Also include the final "Signatures"
section.

Use the "--- PDF PAGE N ---" markers below to determine actual PDF page numbers. Do not use any
page numbers printed inside the document text (e.g. footer page numbers) — only the PDF PAGE
markers are authoritative, since footer numbers can be offset from actual PDF pages.

Return strict JSON: {{"sections": [{{"title": str, "start_page": int}}, ...]}}, ordered by
start_page ascending.

Opening pages:
{scan_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result["sections"]


def find_true_start_pages(pages: list[str]) -> dict[str, int]:
    """Locate each Item's and Signatures' actual first occurrence outside the TOC.

    The LLM only sees the first SCAN_PAGES pages, so for any section beyond that
    window it can only copy the page numbers *printed* in the table of contents —
    which reflect the filing's own internal pagination (e.g. separate "F-page"
    numbering for financial statement notes), not actual PDF page indices. Those
    two numbering schemes diverge well before page 15 in practice, so we verify
    every heading's real PDF page by searching the full extracted text instead of
    trusting the LLM's guess.
    """
    found: dict[str, int] = {}
    for i, text in enumerate(pages):
        if i < TOC_SKIP_PAGES:
            continue
        for line in text.split("\n"):
            stripped = line.strip()
            match = ITEM_HEADING_RE.match(stripped)
            if match:
                key = re.sub(r"\s+", " ", match.group(1)).strip().lower()
                found.setdefault(key, i + 1)
            elif stripped.lower() == "signatures":
                found.setdefault("signatures", i + 1)
    return found


def correct_start_pages(sections: list[dict], pages: list[str]) -> list[dict]:
    true_pages = find_true_start_pages(pages)
    corrected = []
    for section in sections:
        title = section["title"]
        match = ITEM_HEADING_RE.match(title)
        key = re.sub(r"\s+", " ", match.group(1)).strip().lower() if match else None
        if key is None and "signature" in title.lower():
            key = "signatures"

        start_page = section["start_page"]
        if key and key in true_pages:
            start_page = true_pages[key]

        corrected.append({"title": title, "start_page": start_page})
    return corrected


def summarize_section(title: str, text: str) -> str:
    prompt = f"""Summarize the following 10-K section in 2-3 sentences. Focus on concrete facts,
figures, and named items (dollar amounts, percentages, product names, segment names) rather than
generic description. This summary will be shown to a retriever LLM deciding whether this section
is relevant to a financial question, so make it information-dense.

Section title: {title}

Section text:
{text[:12000]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def build_tree(pdf_path: Path = FILING_PATH) -> list[dict]:
    pages = load_pages(pdf_path)
    total_pages = len(pages)

    sections = propose_sections(pages)
    sections = correct_start_pages(sections, pages)
    sections.sort(key=lambda s: s["start_page"])

    nodes = []
    used_ids: set[str] = set()
    for i, section in enumerate(sections):
        start_page = section["start_page"]
        end_page = (
            sections[i + 1]["start_page"] - 1 if i + 1 < len(sections) else total_pages
        )
        end_page = max(end_page, start_page)

        text = "\n".join(pages[start_page - 1:end_page])
        summary = summarize_section(section["title"], text)

        nodes.append(
            {
                "node_id": slugify(section["title"], used_ids),
                "title": section["title"],
                "start_page": start_page,
                "end_page": end_page,
                "text": text,
                "summary": summary,
            }
        )
        print(f"[{i + 1}/{len(sections)}] {section['title']} (pages {start_page}-{end_page})")

    return nodes


def main():
    nodes = build_tree()
    TREE_PATH.write_text(json.dumps(nodes, indent=2))
    print(f"\nSaved {len(nodes)} sections to {TREE_PATH}")


if __name__ == "__main__":
    main()
