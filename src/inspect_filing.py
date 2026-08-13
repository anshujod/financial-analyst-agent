from pathlib import Path

from pypdf import PdfReader

FILING_PATH = Path(__file__).resolve().parent.parent / "data" / "filing.pdf"


def main():
    reader = PdfReader(FILING_PATH)

    print(f"Page count: {len(reader.pages)}")

    outline = reader.outline
    if outline:
        print(f"\nEmbedded outline/bookmarks: YES ({len(outline)} top-level entries)")
    else:
        print("\nEmbedded outline/bookmarks: NO")

    sample_pages = [0, len(reader.pages) // 2, len(reader.pages) - 1]
    for page_num in sample_pages:
        text = reader.pages[page_num].extract_text()
        print(f"\n{'=' * 20} Page {page_num + 1} {'=' * 20}")
        print(text[:1500])


if __name__ == "__main__":
    main()
