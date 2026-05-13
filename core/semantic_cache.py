import time
import json
import numpy as np
import redis
import re
import string
from redis.commands.search.field import VectorField, TagField, TextField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from typing import Optional, Dict, Any, List
from config.settings import settings

class RedisSemanticCache:
    """
    Implements a semantic cache using Redis Stack Vector Search.
    Supports HNSW index, cosine similarity, TTL, and multi-tenant isolation.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=False  # Vectors need bytes
        )
        self.index_name = settings.semantic_cache_index_name
        self.threshold = settings.semantic_cache_threshold
        self.ttl = settings.semantic_cache_ttl
        self.vector_dim = settings.dense_dim
        
        # Load stopwords for normalization
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            
        self._initialize_index()

    def normalize_query(self, query: str) -> str:
        """
        Normalizes the query for better cache matching:
        - Lowercase
        - Remove punctuation
        - Strip filler words (stopwords)
        """
        # 1. Lowercase and remove punctuation
        query = query.lower()
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        # 2. Tokenize and remove stopwords
        words = query.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        
        # 3. Join back
        normalized = " ".join(filtered_words).strip()
        
        # If normalization results in empty string, fallback to original query lowercased
        return normalized if normalized else query.strip()

    def _initialize_index(self):
        """Creates the HNSW vector index if it doesn't exist."""
        try:
            self.client.ft(self.index_name).info()
            print(f"[*] Redis Semantic Cache index '{self.index_name}' already exists.")
        except redis.exceptions.ResponseError:
            print(f"[*] Creating Redis Semantic Cache index: {self.index_name}")
            
            schema = (
                TextField("query"),
                TextField("response"),
                TagField("tenant_id"),
                TagField("doc_version"),
                TagField("prompt_hash"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                )
            )
            
            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            )

    def search(
        self, 
        query_embedding: List[float], 
        tenant_id: str = "default", 
        doc_version: str = "1.0",
        prompt_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Searches the cache for a semantically similar query.
        Returns the cached response if similarity > threshold.
        """
        # Convert list to float32 numpy array then to bytes
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build filter string
        filter_str = f"@tenant_id:{{{tenant_id}}} @doc_version:{{{doc_version}}}"
        if prompt_hash:
            filter_str += f" @prompt_hash:{{{prompt_hash}}}"
            
        # Redis Vector Search Query
        # We use [* => { ... }] for k-NN search
        # distance = 1 - cosine_similarity (for COSINE metric in Redis)
        # So we want distance < (1 - threshold)
        q = Query(f"({filter_str})=>[KNN 1 @embedding $vec AS score]") \
            .sort_by("score") \
            .return_fields("query", "response", "tenant_id", "doc_version", "score") \
            .dialect(2)
        
        params = {"vec": query_vector}
        
        start_time = time.perf_counter()
        results = self.client.ft(self.index_name).search(q, query_params=params)
        search_time = time.perf_counter() - start_time
        
        if results.docs:
            doc = results.docs[0]
            # Redis 'score' in COSINE is 1 - similarity. 
            # Lower score means more similar.
            score = float(doc.score)
            similarity = 1 - score
            
            print(f"[*] Cache search took {search_time:.4f}s. Best similarity: {similarity:.4f}")
            
            if similarity >= self.threshold:
                print(f"[+] Cache HIT! Similarity {similarity:.4f} exceeds threshold {self.threshold}")
                return {
                    "query": doc.query,
                    "response": doc.response,
                    "similarity": similarity,
                    "search_time": search_time,
                    "hit": True
                }
            else:
                print(f"[*] Cache MISS. Similarity {similarity:.4f} below threshold {self.threshold}")
                return {
                    "similarity": similarity,
                    "search_time": search_time,
                    "hit": False
                }
        else:
            print(f"[*] Cache MISS. No results found.")
            return {
                "similarity": 0.0,
                "search_time": search_time,
                "hit": False
            }

    def store(
        self, 
        query: str, 
        embedding: List[float], 
        response: str, 
        tenant_id: str = "default", 
        doc_version: str = "1.0",
        prompt_hash: Optional[str] = None
    ):
        """Stores a new entry in the semantic cache."""
        key = f"cache:{tenant_id}:{int(time.time() * 1000)}"
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        
        data = {
            "query": query,
            "response": response,
            "embedding": query_vector,
            "tenant_id": tenant_id,
            "doc_version": doc_version,
            "created_at": int(time.time()),
            "prompt_hash": prompt_hash or "none"
        }
        
        self.client.hset(key, mapping=data)
        if self.ttl > 0:
            self.client.expire(key, self.ttl)
            
        print(f"[*] Stored response in semantic cache with key: {key} (TTL: {self.ttl}s)")

    def invalidate_version(self, doc_version: str):
        """Deletes all cache entries for a specific document version."""
        # This is a bit slow as it needs to scan or use FT.SEARCH and then delete
        q = Query(f"@doc_version:{{{doc_version}}}").return_fields("id").dialect(2)
        results = self.client.ft(self.index_name).search(q)
        
        if results.docs:
            keys = [doc.id for doc in results.docs]
            self.client.delete(*keys)
            print(f"[*] Invalidated {len(keys)} entries for version: {doc_version}")

    def clear_tenant(self, tenant_id: str):
        """Clears all cache entries for a specific tenant."""
        q = Query(f"@tenant_id:{{{tenant_id}}}").return_fields("id").dialect(2)
        results = self.client.ft(self.index_name).search(q)
        
        if results.docs:
            keys = [doc.id for doc in results.docs]
            self.client.delete(*keys)
            print(f"[*] Cleared {len(keys)} entries for tenant: {tenant_id}")
