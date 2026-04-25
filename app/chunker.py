"""
Token-aware chunker. Splits cleaned page text into overlapping windows
suitable for embedding. Uses tiktoken's cl100k_base as a reasonable
default tokenizer (close enough for MiniLM budgeting).
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from app.config import settings

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str           # f"{doc_id}-{i}"
    doc_id: str
    source: str
    url: str
    title: str
    text: str
    position: int


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    tokens = _ENC.encode(text)
    if not tokens:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start:start + size]
        if not window:
            break
        out.append(_ENC.decode(window))
        if start + size >= len(tokens):
            break
    return out


def chunk_page(
    *,
    doc_id: str,
    source: str,
    url: str,
    title: str,
    text: str,
) -> list[Chunk]:
    pieces = _split_tokens(text, settings.chunk_size, settings.chunk_overlap)
    return [
        Chunk(
            chunk_id=f"{doc_id}-{i:03d}",
            doc_id=doc_id,
            source=source,
            url=url,
            title=title,
            text=piece,
            position=i,
        )
        for i, piece in enumerate(pieces)
    ]
