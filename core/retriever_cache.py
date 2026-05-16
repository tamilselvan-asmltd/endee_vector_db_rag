import time
import json
import numpy as np
import redis
import string
import hashlib
from redis.commands.search.field import VectorField, TagField, TextField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from config.settings import settings

class RetrieverSemanticCache:
    """
    Implements a semantic cache for retrieved document chunks using Redis Stack Vector Search.
    Caches the final reranked results (List[Document]) for a given query.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=False  # Vectors need bytes
        )
        self.index_name = settings.retriever_cache_index_name
        self.threshold = settings.retriever_cache_threshold
        self.ttl = settings.retriever_cache_ttl
        self.vector_dim = settings.dense_dim
        
        # Load stopwords for normalization if available
        try:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            
        self._initialize_index()

    def normalize_query(self, query: str) -> str:
        """Normalizes the query for better cache matching."""
        query = query.lower().translate(str.maketrans('', '', string.punctuation))
        words = query.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        normalized = " ".join(filtered_words).strip()
        return normalized if normalized else query.strip()

    def _initialize_index(self):
        """Creates the HNSW vector index if it doesn't exist."""
        try:
            info = self.client.ft(self.index_name).info()
            print(f"[*] Redis Retriever Cache index '{self.index_name}' already exists.")
        except redis.exceptions.ResponseError as e:
            if "Unknown index name" in str(e) or "Index not found" in str(e):
                print(f"[*] Creating Redis Retriever Cache index: {self.index_name}")
                
                schema = (
                    TextField("query"),
                    TextField("chunks_json"), # Serialized List[Document]
                    TagField("tenant_id"),
                    TagField("doc_version"),
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
                
                try:
                    self.client.ft(self.index_name).create_index(
                        fields=schema,
                        definition=IndexDefinition(prefix=[settings.retriever_cache_prefix], index_type=IndexType.HASH)
                    )
                    print(f"[+] Redis Retriever Cache index '{self.index_name}' created successfully.")
                except Exception as create_e:
                    print(f"[!] Critical Error creating Redis Retriever Cache index: {create_e}")
            else:
                print(f"[!] Redis error during FT.INFO for {self.index_name}: {e}")
        except Exception as e:
            print(f"[!] Unexpected error during Retriever Cache init: {e}")

    def search(
        self, 
        query_embedding: List[float], 
        tenant_id: str = "default", 
        doc_version: str = "1.0"
    ) -> Optional[List[Document]]:
        """
        Searches the cache for semantically similar previous retrievals.
        Returns the list of LangChain Documents if hit.
        """
        if not settings.retriever_cache_enabled:
            return None

        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        # Build filters
        filters = [f"@doc_version:{{{doc_version}}}"]
        if tenant_id and tenant_id != "all":
            filters.append(f"@tenant_id:{{{tenant_id}}}")
            
        filter_str = " ".join(filters)
            
        # Redis Vector Search Query
        q = Query(f"({filter_str})=>[KNN 1 @embedding $vec AS score]") \
            .sort_by("score") \
            .return_fields("query", "chunks_json", "score") \
            .dialect(2)
        
        params = {"vec": query_vector}
        
        try:
            results = self.client.ft(self.index_name).search(q, query_params=params)
            
            if results.docs:
                doc = results.docs[0]
                score = float(doc.score)
                similarity = 1 - score
                
                if similarity >= self.threshold:
                    print(f"[+] Retriever Cache HIT! Similarity {similarity:.4f} >= {self.threshold}")
                    
                    chunks_json = doc.chunks_json
                    if isinstance(chunks_json, bytes):
                        chunks_json = chunks_json.decode('utf-8')
                    
                    data = json.loads(chunks_json)
                    
                    # Reconstruct LangChain Documents
                    return [
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in data
                    ]
        except Exception as e:
            print(f"[!] Error searching retriever cache: {e}")
            
        return None

    def store(
        self, 
        query: str, 
        embedding: List[float], 
        documents: List[Document], 
        tenant_id: str = "default", 
        doc_version: str = "1.0"
    ):
        """Stores the retrieved chunks in the cache."""
        if not settings.retriever_cache_enabled:
            return

        query_hash = hashlib.sha256(query.encode()).hexdigest()
        key = f"{settings.retriever_cache_prefix}{tenant_id}:{query_hash}"
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        
        # Serialize Documents
        chunks_data = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in documents
        ]
        
        data = {
            "query": query,
            "chunks_json": json.dumps(chunks_data),
            "embedding": query_vector,
            "tenant_id": tenant_id,
            "doc_version": doc_version,
            "created_at": int(time.time())
        }
        
        try:
            self.client.hset(key, mapping=data)
            if self.ttl > 0:
                self.client.expire(key, self.ttl)
            print(f"[*] Stored {len(documents)} chunks in retriever cache. Key: {key}")
        except Exception as e:
            print(f"[!] Error storing in retriever cache: {e}")

    def clear_all(self):
        """Clears all retriever cache entries."""
        cursor = 0
        pattern = f"{settings.retriever_cache_prefix}*"
        count = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                self.client.delete(*keys)
                count += len(keys)
            if cursor == 0:
                break
        print(f"[*] Cleared {count} retriever cache entries.")
        return count
