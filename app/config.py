"""Central configuration. Reads from .env, falls back to defaults."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"

    # Storage
    chroma_dir: str = "./data/processed/chroma"
    raw_dir: str = "./data/raw"

    # Retrieval
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 75

    # Scraper
    user_agent: str = "ARC-AI-Research-Bot/0.1 (educational use)"
    request_delay: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()

# Ensure dirs exist
Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
Path(settings.raw_dir).mkdir(parents=True, exist_ok=True)

# Collection name in Chroma
COLLECTION_NAME = "maryland_housing_law"
