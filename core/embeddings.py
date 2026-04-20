import requests
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor
from endee_model import SparseModel
from config.settings import settings

class EmbeddingService:
    """Handles generation of dense and sparse embeddings."""
    _sparse_model = None

    def __init__(self):
        self.ollama_url = f"{settings.ollama_url}/api/embeddings"
        self.ollama_model = settings.ollama_embed_model
        
        # Singleton pattern for SparseModel to keep it in memory
        if EmbeddingService._sparse_model is None:
            print("[*] Loading SparseModel into memory...")
            EmbeddingService._sparse_model = SparseModel(settings.sparse_model_path)
            
        self.sparse_model = EmbeddingService._sparse_model
        self.dense_dim = settings.dense_dim

    def get_dense_embedding(self, text: str) -> List[float]:
        """Fetches dense embedding from Ollama."""
        print(f"[*] Generating dense embedding for: {text[:50]}...")
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model, 
                "prompt": text,
                "keep_alive": "5m"
            },
            timeout=120,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        
        if len(embedding) != self.dense_dim:
            raise ValueError(f"Expected dimension {self.dense_dim}, but got {len(embedding)}")
        
        return embedding

    def get_dense_embeddings_batch(self, texts: List[str], max_workers: int = 5) -> List[List[float]]:
        """Fetches dense embeddings in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.get_dense_embedding, texts))

    def get_sparse_embedding(self, text: str, is_query: bool = False):
        """Fetches sparse embedding using local SparseModel."""
        type_str = "query" if is_query else "document"
        print(f"[*] Generating sparse {type_str} embedding...")
        if is_query:
            return next(self.sparse_model.query_embed(text))
        return next(self.sparse_model.embed([text]))
