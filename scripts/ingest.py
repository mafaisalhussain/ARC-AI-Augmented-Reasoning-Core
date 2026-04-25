"""
One-shot ingestion: scrape Maryland sources, chunk, embed, persist to ChromaDB.

Usage:
    python -m scripts.ingest
"""
from __future__ import annotations

import sys
from tqdm import tqdm

from app.chunker import chunk_page
from app.scraper import save_raw, scrape_all
from app.vectorstore import VectorStore


def main() -> int:
    print("step 1/3 — scraping Maryland housing sources …")
    pages = scrape_all()
    if not pages:
        print("No pages scraped. Check network / seed URLs.")
        return 1
    raw_path = save_raw(pages)
    print(f"  scraped {len(pages)} pages → {raw_path}")

    print("\nstep 2/3 — chunking …")
    all_chunks = []
    for p in tqdm(pages):
        all_chunks.extend(
            chunk_page(
                doc_id=p.doc_id,
                source=p.source,
                url=p.url,
                title=p.title,
                text=p.text,
            )
        )
    print(f"  produced {len(all_chunks)} chunks")

    print("\nstep 3/3 — embedding + storing in ChromaDB …")
    store = VectorStore()
    written = store.add_chunks(all_chunks)
    print(f"  wrote {written} chunks. collection size: {store.count()}")

    print("\ndone. try:  python -m app.chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
