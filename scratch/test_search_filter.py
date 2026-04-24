import sys
import os

# Add current dir to sys.path
sys.path.append(os.getcwd())

from core.database import DatabaseService
from core.embeddings import EmbeddingService
from core.retriever import HybridEndeeRetriever

def test_search_filter():
    db = DatabaseService()
    index = db.get_index()
    embeddings = EmbeddingService()
    
    # We'll use a tag that we know might exist or we just created in the previous test
    # Actually, let's use the one from test_filters.py if it's still there, 
    # but since we deleted it, let's create a new one for this test.
    
    test_tag = "search_test_id"
    test_val = "12345"
    
    point = {
        "id": "search_test_point",
        "vector": [0.2] * 768,
        "sparse_indices": [0],
        "sparse_values": [1.0],
        "filter": {test_tag: test_val},
        "meta": {"text": "Specific content for search filtering test.", test_tag: test_val}
    }
    db.upsert_batch(index, [point])
    
    # 1. Search with correct filter
    retriever_with_filt = HybridEndeeRetriever(
        index=index, 
        embedding_service=embeddings,
        base_filter=[{test_tag: {"$eq": test_val}}]
    )
    
    print("[*] Searching with matching filter...")
    docs = retriever_with_filt.invoke("Specific content")
    if any(test_val in doc.metadata.get(test_tag, "") for doc in docs):
        print("[+] Found document with matching filter.")
    else:
        print("[!] Document NOT found with matching filter.")

    # 2. Search with non-matching filter
    retriever_with_wrong_filt = HybridEndeeRetriever(
        index=index, 
        embedding_service=embeddings,
        base_filter=[{test_tag: {"$eq": "wrong_val"}}]
    )
    
    print("[*] Searching with non-matching filter...")
    docs_wrong = retriever_with_wrong_filt.invoke("Specific content")
    if not docs_wrong:
        print("[+] Correctly found NO documents with non-matching filter.")
    else:
        print(f"[!] Found {len(docs_wrong)} documents even with non-matching filter!")

    # Cleanup
    print("[*] Cleaning up test point...")
    index.delete_with_filter(filter=[{test_tag: {"$eq": test_val}}])

if __name__ == "__main__":
    try:
        test_search_filter()
    except Exception as e:
        print(f"[!] Test failed: {e}")
