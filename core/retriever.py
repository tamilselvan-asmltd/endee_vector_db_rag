import os
import time

# Silence noisy transformers logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from typing import List, Any
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config.settings import settings

# Global cache for the reranker model to prevent reloading across re-initializations
_RERANKER_INSTANCE = None

class HybridEndeeRetriever(BaseRetriever):
    """
    Custom LangChain retriever for Endee Hybrid Search with CrossEncoder reranking.
    """
    index: Any
    embedding_service: Any
    top_k: int = Field(default=settings.top_k)
    base_filter: List[dict] = Field(default_factory=list)
    last_retrieval_time: float = Field(default=0.0)
    reranker: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        global _RERANKER_INSTANCE
        
        # Load reranker if enabled and not already loaded in global cache
        if settings.use_reranker:
            if _RERANKER_INSTANCE is None:
                import os
                model_path = settings.reranker_model_path
                model_name = settings.reranker_model_name
                
                # Check if local path exists
                if os.path.exists(model_path):
                    print(f"[*] Loading Reranker from LOCAL path: {model_path}")
                    try:
                        _RERANKER_INSTANCE = CrossEncoder(model_path)
                    except Exception as e:
                        print(f"[!] Error loading from local path, attempting fallback: {e}")
                
                # Fallback to name (will download and then we save locally)
                if _RERANKER_INSTANCE is None:
                    print(f"[*] Initializing Reranker (Cold Start/Download): {model_name}")
                    try:
                        _RERANKER_INSTANCE = CrossEncoder(model_name)
                        # Save locally for future offline use
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        print(f"[*] Saving Reranker to local path for offline use: {model_path}")
                        _RERANKER_INSTANCE.save(model_path)
                    except Exception as e:
                        print(f"[!] Error initializing reranker: {e}")
            
            self.reranker = _RERANKER_INSTANCE

    def _get_relevant_documents(self, query: str) -> List[Document]:
        start_time = time.perf_counter()
        
        # 1) Generate embeddings
        q_dense = self.embedding_service.get_dense_embedding(query)
        q_sparse = self.embedding_service.get_sparse_embedding(query, is_query=True)

        # 2) Perform hybrid query
        # Fetch more candidates if reranking is enabled
        fetch_k = settings.rerank_top_k if settings.use_reranker else self.top_k
        
        print(f"[*] Querying with filter: {self.base_filter} (fetch_k={fetch_k})")
        hits = self.index.query(
            vector=q_dense,
            sparse_indices=q_sparse.indices.tolist(),
            sparse_values=q_sparse.values.tolist(),
            top_k=fetch_k,
            filter=self.base_filter if self.base_filter else None
        )
        print(f"[*] Hybrid Search complete. Found {len(hits)} relevant chunks.")

        # 3) Convert hits to LangChain Documents
        docs = []
        for h in hits:
            meta = h.get("meta", {})
            text = meta.pop("text", "")
            
            # Inject document link if filename exists
            filename = meta.get("filename")
            if filename:
                import os
                import urllib.parse
                basename = os.path.basename(filename)
                # URL-encode filename to handle spaces in Markdown links
                encoded_name = urllib.parse.quote(basename)
                meta["link"] = f"{settings.doc_server_url}/{encoded_name}"
                
            docs.append(
                Document(
                    page_content=text,
                    metadata=meta
                )
            )
        
        # 4) Rerank if enabled and model is loaded
        if settings.use_reranker and self.reranker and docs:
            print(f"[*] Reranking {len(docs)} documents using {settings.reranker_model_name}...")
            pairs = [[query, d.page_content] for d in docs]
            scores = self.reranker.predict(pairs)
            
            # Update metadata with rerank scores and sort
            for i, score in enumerate(scores):
                docs[i].metadata["rerank_score"] = float(score)
            
            docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
            
            # Filter to final top_k
            docs = docs[:self.top_k]
            print(f"[*] Reranking complete. Returning top {len(docs)} results.")

        self.last_retrieval_time = time.perf_counter() - start_time
        return docs
