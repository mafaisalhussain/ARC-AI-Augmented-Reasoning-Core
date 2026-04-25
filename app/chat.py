"""
CLI REPL for ARC AI. Phase 1 entry point.

Usage:
    python -m app.chat

Run `python -m scripts.ingest` first to populate ChromaDB.
"""
from __future__ import annotations

import sys

from app.llm import stream_chat
from app.vectorstore import VectorStore


BANNER = r"""
  _   ___  ___    _   ___ 
 /_\ | _ \/ __|  /_\ |_ _|
/ _ \|   / (__  / _ \ | | 
/_/ \_\_|_\\___|/_/ \_\___|
Maryland Housing & Rental Law Assistant  —  Phase 1 (CLI)
Type your question. Ctrl-C or "exit" to quit.
"""


def main() -> int:
    store = VectorStore()
    n = store.count()
    if n == 0:
        print("Vector store is empty. Run: python -m scripts.ingest")
        return 1

    print(BANNER)
    print(f"Loaded {n} chunks.\n")

    try:
        while True:
            try:
                query = input("you › ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                break

            hits = store.query(query)
            print("\narc ›", end=" ", flush=True)
            for token in stream_chat(query, hits):
                print(token, end="", flush=True)
            print("\n")

            if hits:
                print("sources:")
                seen = set()
                for i, h in enumerate(hits, start=1):
                    key = h.url
                    if key in seen:
                        continue
                    seen.add(key)
                    print(f"  [S{i}] {h.title or h.url}  —  {h.url}")
                print()
    except KeyboardInterrupt:
        pass

    print("\nbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
