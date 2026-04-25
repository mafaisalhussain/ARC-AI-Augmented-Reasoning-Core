"""
Ollama client. Supports a per-request model override for the multi-model switcher.
"""
from __future__ import annotations

from typing import Iterator

import ollama

from app.config import settings
from app.vectorstore import RetrievalHit


SYSTEM_PROMPT = """You are ARC AI, an assistant that answers questions about \
Maryland housing and rental law. You use ONLY the provided context passages \
from Maryland state and county sources (AG, DHCD, Montgomery County, \
Baltimore County, Baltimore City, Prince George's County, and the People's \
Law Library).

Rules:
- If the answer isn't in the context, say so plainly. Do not invent citations.
- Cite sources inline as [S1], [S2], etc. matching the numbered context below.
- Keep answers concrete and practical. Prefer short paragraphs over legalese.
- You are not a lawyer. End complex answers with a one-line disclaimer.
"""


def _format_context(hits: list[RetrievalHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[S{i}] ({h.source}) {h.title}\nURL: {h.url}\n{h.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(
    query: str,
    hits: list[RetrievalHit],
    history: list[dict] | None = None,
) -> list[dict]:
    context = _format_context(hits) if hits else "(no relevant passages found)"
    user = (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\nUser question: {query}\n\n"
        f"Answer using only the context above. Cite sources as [S1], [S2], etc."
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def stream_chat(
    query: str,
    hits: list[RetrievalHit],
    history: list[dict] | None = None,
    model: str | None = None,
) -> Iterator[str]:
    client = ollama.Client(host=settings.ollama_host)
    messages = build_messages(query, hits, history)
    for chunk in client.chat(
        model=model or settings.ollama_model,
        messages=messages,
        stream=True,
        options={"temperature": 0.2},
    ):
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            yield piece


def complete(
    query: str,
    hits: list[RetrievalHit],
    history: list[dict] | None = None,
    model: str | None = None,
) -> str:
    return "".join(stream_chat(query, hits, history, model))


def list_available_models() -> list[str]:
    try:
        client = ollama.Client(host=settings.ollama_host)
        resp = client.list()
        models = resp.get("models", [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        return sorted(names)
    except Exception as e:
        print(f"[llm] failed to list models: {e}")
        return [settings.ollama_model]