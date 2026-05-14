import requests
import os
import redis
import json
import hashlib
from typing import List
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
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
        
        # Redis client for embedding cache
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=False # Vectors stay as bytes or we use JSON
        )
        self.embed_cache_prefix = settings.embed_cache_prefix
        self.embed_cache_ttl = settings.embed_cache_ttl

    @lru_cache(maxsize=128)
    def get_dense_embedding(self, text: str) -> List[float]:
        """Fetches dense embedding from Redis cache or Ollama."""
        # 1. Check Redis Cache
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"{self.embed_cache_prefix}{text_hash}"
        
        try:
            cached_val = self.redis_client.get(cache_key)
            if cached_val:
                print(f"[*] Redis Embedding Cache HIT for: {text[:30]}...")
                return json.loads(cached_val)
        except Exception as e:
            print(f"[!] Error checking embedding cache: {e}")

        # 2. Generate if not cached
        print(f"[*] Generating dense embedding via Ollama for: {text[:50]}...")
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
        
        # 3. Store in Redis Cache
        try:
            self.redis_client.setex(
                cache_key,
                self.embed_cache_ttl,
                json.dumps(embedding)
            )
            print(f"[*] Stored embedding in Redis cache (TTL: {self.embed_cache_ttl}s)")
        except Exception as e:
            print(f"[!] Error storing in embedding cache: {e}")
            
        return embedding

    def clear_all(self):
        """Clears ALL cached embeddings from Redis."""
        cursor = 0
        pattern = f"{self.embed_cache_prefix}*"
        count = 0
        while True:
            cursor, keys = self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self.redis_client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break
        print(f"[*] Cleared {count} cached embeddings.")
        return count

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
