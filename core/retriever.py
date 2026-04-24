import time
from typing import List, Any
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from config.settings import settings

class HybridEndeeRetriever(BaseRetriever):
    """
    Custom LangChain retriever for Endee Hybrid Search.
    """
    index: Any
    embedding_service: Any
    top_k: int = Field(default=settings.top_k)
    base_filter: List[dict] = Field(default_factory=list)
    last_retrieval_time: float = Field(default=0.0)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        start_time = time.perf_counter()
        
        # 1) Generate embeddings
        q_dense = self.embedding_service.get_dense_embedding(query)
        q_sparse = self.embedding_service.get_sparse_embedding(query, is_query=True)

        # 2) Perform hybrid query
        print(f"[*] Querying with filter: {self.base_filter}")
        hits = self.index.query(
            vector=q_dense,
            sparse_indices=q_sparse.indices.tolist(),
            sparse_values=q_sparse.values.tolist(),
            top_k=self.top_k,
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
        
        self.last_retrieval_time = time.perf_counter() - start_time
        return docs
