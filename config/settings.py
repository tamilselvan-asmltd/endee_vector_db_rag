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
    endee_m: int = 32
    endee_ef_con: int = 256
    endee_precision: str = "float32"


    # Sparse
    sparse_model_path: str = "endee/bm25"

    # RAG
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    doc_server_url: str = "http://localhost:8003"
    nltk_data_path: str = str(Path.home() / "nltk_data")

    # Reranker
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_path: str = str(Path(__file__).parent.parent / "models" / "reranker")
    rerank_top_k: int = 15

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
