import json
import re
import sys
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


def normalize_text(text: str) -> str:
    """Normalize PDF-extraction artifacts: non-breaking spaces, ligatures (ﬁ/ﬂ/ﬀ),
    typographic apostrophes, and collapsed whitespace. Heading matching must be
    robust to these (e.g. 'Item\xa015.' and 'ﬁscal' appear in the raw extraction)."""
    text = text.replace("\xa0", " ")
    for lig, repl in [("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb00", "ff")]:
        text = text.replace(lig, repl)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return text


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
        for line in normalize_text(text).split("\n"):
            stripped = line.strip()
            match = ITEM_HEADING_RE.match(stripped)
            if match:
                key = re.sub(r"\s+", " ", match.group(1)).strip().lower()
                found.setdefault(key, i + 1)
            elif stripped.lower() == "signatures":
                found.setdefault("signatures", i + 1)
    return found


CONTENTLESS_LINE_RE = re.compile(r"^(none|not applicable|reserved)\.?$", re.IGNORECASE)
POINTER_PHRASES = (
    "please see",
    "see note",
    "refer to note",
    "set forth",
    "included in",
    "incorporated by reference",
    "will be contained in",
    "is hereby incorporated",
)
INCORPORATED_BY_REFERENCE = "incorporated by reference"
# a short segment with multiple pointer phrases is a pure proxy/filing pointer
POINTER_MAX_CHARS = 4000


def own_segment(text: str) -> str:
    """The part of a node's text from its own heading up to the next 'Item N.'
    heading (or end of text). Neighboring sections sharing the same page (e.g. page
    50 holds Items 2-5) must not bleed into this section's content check."""
    lines = normalize_text(text).split("\n")
    seg = [lines[0]]
    for line in lines[1:]:
        if re.match(r"^item\s*\d+[a-c]?\.", line.strip(), re.I):
            break
        seg.append(line)
    return "\n".join(seg)


def is_contentless_section(title: str, text: str) -> bool:
    """True for sections whose *own* disclosure is a one-line placeholder ('None.',
    'Not applicable.', '[Reserved]') or a pure pointer (Items 3, 8, 10-14, 16 point
    at the proxy statement or at the financial statements instead of disclosing
    anything). These generate garbage Q&A pairs: the QA generator answers from
    parametric knowledge because the excerpt has no content, which teaches the
    fine-tuned model to hallucinate."""
    if "[reserved]" in title.lower():
        return True
    segment = own_segment(text)
    lines = [l.strip() for l in segment.split("\n") if l.strip()]
    body = lines[1:] if lines and re.match(r"^item\s*\d+", lines[0], re.I) else lines
    body_text = " ".join(body).strip().lower()
    if body and CONTENTLESS_LINE_RE.match(body[0]):
        return True
    # very short body that just points elsewhere ("Please see Note 13...",
    # "The information required by this Item is set forth in ...")
    if len(body_text) < 500 and any(p in body_text for p in POINTER_PHRASES):
        return True
    # multi-paragraph pointer to the proxy statement (Items 10-14)
    if (
        len(segment) < POINTER_MAX_CHARS
        and body_text.count(INCORPORATED_BY_REFERENCE) >= 2
    ):
        return True
    return False


def trim_to_heading(text: str, title: str) -> str | None:
    """Cut a node's page text so it starts at its own heading, dropping the previous
    section's tail that shares the page (e.g. page 50 holds Items 2-5; each node must
    start at its own heading, not at 'Item 2. Properties'). Headings may wrap across
    extracted lines or contain double spaces, so match with flexible whitespace.
    Returns None when the heading is not found (e.g. the cover page)."""
    needle = re.sub(r"\s+", " ", normalize_text(title)).strip()
    pattern = re.escape(needle).replace(r"\ ", r"\s+")
    m = re.search(pattern, normalize_text(text), re.IGNORECASE)
    if m:
        return text[m.start():]
    # fallback: first ~40 chars of the title (unlikely to wrap). Must be long enough
    # and word-bounded — a bare 'cover' would match inside 'recovery'.
    short = needle[:40].rsplit(" ", 1)[0]
    if len(short) >= 12:
        m = re.search(
            r"\b" + re.escape(short).replace(r"\ ", r"\s+") + r"\b",
            normalize_text(text),
            re.IGNORECASE,
        )
        if m:
            return text[m.start():]
    return None


def trim_to_next_heading(text: str) -> str:
    """Cut a node's text at the next 'Item N.' heading so page-shared neighbors
    (Item 5's page also holds 'Item 6. [Reserved]', page 50 holds Items 2-5) do not
    leak into this section's excerpt."""
    lines = text.split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if re.match(r"^item\s*\d+[a-c]?\.", line.strip(), re.I):
            break
        out.append(line)
    return "\n".join(out)


def sanitize_tree(nodes: list[dict], pages: list[str], total_pages: int) -> list[dict]:
    """Post-process the raw page-range nodes:
    1. Trim each node's text to its own heading -> next heading (page-sharing
       sections, e.g. Items 2-5 on page 50, must not bleed into each other).
    2. Mark content-less placeholder sections (None / Not applicable / [Reserved] /
       incorporated-by-reference) with skip_qa so no Q&A pairs are generated from
       them (they teach the model to hallucinate).
    3. A placeholder whose heading sits at the *bottom* of a page (Item 6 [Reserved]
       at the bottom of Item 5's stock-performance page) is absorbed: the node is
       removed and its page flows back to the real section that owns it.
    """
    pages = [normalize_text(p) for p in pages]
    nodes = sorted(nodes, key=lambda n: n["start_page"])

    def next_start(idx: int) -> int:
        return nodes[idx + 1]["start_page"] if idx + 1 < len(nodes) else total_pages + 1

    processed = []
    for i, node in enumerate(nodes):
        end = max(node["start_page"], next_start(i) - 1)
        text_pages = pages[node["start_page"] - 1 : end]
        text = trim_to_heading(text_pages[0], node["title"]) if text_pages else None
        if text is None:
            # no heading found (e.g. cover page): keep the full page range as-is
            text = "\n".join(text_pages)
        else:
            for extra in text_pages[1:]:
                text += "\n" + extra
            text = trim_to_next_heading(text)
        skip = is_contentless_section(node["title"], text)

        absorb = False
        if skip and node["start_page"] < total_pages:
            page_text = pages[node["start_page"] - 1]
            pos = page_text.lower().find(normalize_text(node["title"]).lower())
            if pos > 0 and pos / max(len(page_text), 1) > 0.6:
                absorb = True
        processed.append({**node, "end_page": end, "text": text, "skip_qa": skip, "_absorb": absorb})

    # recompute end pages over surviving (non-absorbed) nodes so absorbed pages
    # flow back to the section that actually owns them (Item 5 gains page 52)
    survivors = [p for p in processed if not p["_absorb"]]
    for i, node in enumerate(survivors):
        end = (
            survivors[i + 1]["start_page"] - 1 if i + 1 < len(survivors) else total_pages
        )
        end = max(end, node["start_page"])
        text_pages = pages[node["start_page"] - 1 : end]
        text = trim_to_heading(text_pages[0], node["title"]) if text_pages else None
        if text is None:
            text = "\n".join(text_pages)
        else:
            for extra in text_pages[1:]:
                text += "\n" + extra
            text = trim_to_next_heading(text)
        node["end_page"] = end
        node["text"] = text

    result = []
    for p in processed:
        if p["_absorb"]:
            print(f"  ABSORB {p['node_id']}: bottom-of-page placeholder -> removed")
            continue
        if p["skip_qa"]:
            print(f"  SKIP  {p['node_id']}: content-less placeholder (kept, no Q&A)")
        result.append(p)
    return result


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

    nodes = sanitize_tree(nodes, pages, total_pages)

    return nodes


def rebuild_from_existing(pages: list[str], existing_nodes: list[dict]) -> list[dict]:
    """Rebuild node texts from the PDF using the *existing* node titles and start
    pages, without calling the LLM (no propose_sections/summarize_section). Used to
    fix a broken tree in place: trims page-shared text to each node's own heading,
    drops content-less placeholders, and recomputes end pages.

    Usage: python tree_builder.py --rebuild
    """
    total_pages = len(pages)
    base = []
    used_ids: set[str] = set()
    for n in sorted(existing_nodes, key=lambda n: n["start_page"]):
        base.append(
            {
                "node_id": slugify(n["title"], used_ids),
                "title": n["title"],
                "start_page": n["start_page"],
                "end_page": n["end_page"],
                "text": n["text"],
                "summary": n["summary"],
            }
        )
    return sanitize_tree(base, pages, total_pages)


def main():
    if "--rebuild" in sys.argv:
        # Fix the existing tree in place without LLM calls.
        pages = load_pages(FILING_PATH)
        existing = json.loads(TREE_PATH.read_text())
        print(f"Rebuilding {len(existing)} nodes from {len(pages)} pages (no LLM calls)...")
        nodes = rebuild_from_existing(pages, existing)
        TREE_PATH.write_text(json.dumps(nodes, indent=2))
        print(f"\nSaved {len(nodes)} sections to {TREE_PATH}")
        for n in nodes:
            print(f"  {n['node_id']:45s} pages {n['start_page']:>3}-{n['end_page']:>3}  "
                  f"text {len(n['text']):>6} chars | {n['text'][:60].strip()!r}")
        return

    nodes = build_tree()
    TREE_PATH.write_text(json.dumps(nodes, indent=2))
    print(f"\nSaved {len(nodes)} sections to {TREE_PATH}")


if __name__ == "__main__":
    main()
