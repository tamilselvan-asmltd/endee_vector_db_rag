import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever

def test_retriever_chunks(query: str):
    print(f"[*] Testing Retriever for query: '{query}'")
    
    # 1. Initialize Services
    embeddings = EmbeddingService()
    db = DatabaseService()
    index = db.get_index()
    
    # 2. Setup Retriever
    retriever = HybridEndeeRetriever(
        index=index,
        embedding_service=embeddings
    )
    
    # 3. Retrieve Documents
    print("[*] Retrieving chunks...")
    docs = retriever.invoke(query)
    
    # 4. Print Results
    print("\n" + "="*80)
    print(f"RETRIEVED CHUNKS: {len(docs)}")
    print(f"LATENCY: {retriever.last_retrieval_time:.4f} seconds")
    print("="*80 + "\n")
    
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Score/Meta: {doc.metadata}")
        print(f"Content Preview: {doc.page_content[:200]}...")
        print("-" * 20 + "\n")

    print("="*80)

if __name__ == "__main__":
    test_query = "What is the role of AI in EDI?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    
    test_retriever_chunks(test_query)
