from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_llm_model: str = "gpt-oss:120b-cloud"
    llm_temperature: float = 0.0

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
    history_window_size: int = 5

    # Reranker
    use_reranker: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_path: str = str(Path(__file__).parent.parent / "models" / "reranker")
    rerank_top_k: int = 15

    # Redis Semantic Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    semantic_cache_threshold: float = 0.85
    semantic_cache_ttl: int = 3600  # 1 hour
    semantic_cache_index_name: str = "semantic_cache_idx"
    semantic_cache_prefix: str = "sem_cache:"

    # Redis Retriever Cache (Semantic Chunk Caching)
    retriever_cache_enabled: bool = True
    retriever_cache_threshold: float = 0.80
    retriever_cache_ttl: int = 3600 # 1 hour
    retriever_cache_index_name: str = "retriever_cache_idx"
    retriever_cache_prefix: str = "ret_cache:"

    # Redis Chat History
    redis_chat_history_prefix: str = "chat:"
    max_history_messages: int = 10

    # Redis Embedding Cache
    embed_cache_prefix: str = "embed_cache:"
    embed_cache_ttl: int = 86400 * 7  # 7 days

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
