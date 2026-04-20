from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_llm_model: str = "gemma4:31b-cloud"

    # Endee
    endee_url: str = "http://localhost:8080"
    endee_index_name: str = "cnc_hybrid_vdb"
    dense_dim: int = 768
    space_type: str = "cosine"

    # Sparse
    sparse_model_path: str = "endee/bm25"

    # RAG
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    doc_server_url: str = "http://localhost:8003"
    nltk_data_path: str = str(Path.home() / "nltk_data")

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
