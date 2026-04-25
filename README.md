# ARC AI — Maryland Housing & Rental Law Assistant

Local-first RAG assistant. Phase 1 is a CLI chat. Phase 2 wraps it in FastAPI.
Phase 3 adds a frontend. Phase 4 adds NLP features.

## Stack

- **LLM**: Llama 3.1 8B via Ollama (local)
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector store**: ChromaDB (persisted to `./data/processed/chroma`)
- **Scraper**: `requests` + `beautifulsoup4`
- **Backend (Phase 2)**: FastAPI

## Data sources (scraped once, stored locally)

1. Maryland Code — `mgaleg.maryland.gov` (Real Property, Title 8)
2. Maryland Courts — tenant/landlord self-help pages
3. Maryland Attorney General — tenant consumer guides
4. People's Law Library of Maryland — `peoples-law.org`

## Setup (VS Code)

```bash
# 1. Clone / open the folder in VS Code

# 2. Create venv (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Install + start Ollama, pull the model
#    https://ollama.com/download
ollama pull llama3.1:8b
ollama serve                     # run in another terminal if not auto-started

# 5. Copy env template
cp .env.example .env
```

## Run Phase 1

```bash
# one-time: scrape + embed + store
python -m scripts.ingest

# chat
python -m app.chat
```

Type a question. Example: *"How much notice must a landlord give before raising rent in Maryland?"*

## Layout

```
app/
  config.py        settings
  scraper.py       pulls pages from 4 MD sources
  chunker.py       token-aware chunking
  vectorstore.py   ChromaDB wrapper
  llm.py           Ollama client + grounded prompt
  chat.py          CLI REPL  ← Phase 1 entry
scripts/
  ingest.py        one-shot pipeline
data/
  raw/             scraped JSONL
  processed/       chroma DB
```

## Phases

- [x] **Phase 1** — CLI REPL (this)
- [ ] **Phase 2** — FastAPI (`POST /chat`, streaming)
- [ ] **Phase 3** — Web frontend
- [ ] **Phase 4** — NLP: intent classification, legal NER, extractive QA fallback, citation grounding

## Disclaimer

ARC AI is not a lawyer. Answers are informational only.
