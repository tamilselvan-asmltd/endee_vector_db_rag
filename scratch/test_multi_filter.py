import sys
import os

# Add current dir to sys.path
sys.path.append(os.getcwd())

from core.database import DatabaseService
from core.embeddings import EmbeddingService
from core.retriever import HybridEndeeRetriever

def test_multi_filter():
    db = DatabaseService()
    index = db.get_index()
    embeddings = EmbeddingService()
    
    test_tag1 = "book"
    test_val1 = "aiml"
    test_tag2 = "page_num"
    test_val2 = 500
    
    point = {
        "id": "multi_filter_test",
        "vector": [0.3] * 768,
        "sparse_indices": [0],
        "sparse_values": [1.0],
        "filter": {test_tag1: test_val1, test_tag2: test_val2},
        "meta": {"text": "Multi-filter test content.", test_tag1: test_val1, test_tag2: test_val2}
    }
    db.upsert_batch(index, [point])
    
    # Search with only one filter
    retriever_one = HybridEndeeRetriever(
        index=index, 
        embedding_service=embeddings,
        base_filter=[{test_tag2: {"$eq": test_val2}}]
    )
    print(f"[*] Searching with single filter: {test_tag2}={test_val2}")
    docs_one = retriever_one.invoke("Multi-filter test")
    print(f"[*] Found {len(docs_one)} documents with single filter.")

    # Search with AND filter (multiple dicts in list)
    retriever = HybridEndeeRetriever(
        index=index, 
        embedding_service=embeddings,
        base_filter=[
            {test_tag1: {"$eq": test_val1}},
            {test_tag2: {"$eq": test_val2}}
        ]
    )
    
    print(f"[*] Searching with AND filter: {test_tag1}={test_val1} AND {test_tag2}={test_val2}")
    docs = retriever.invoke("Multi-filter test")
    
    if any(doc.metadata.get(test_tag1) == test_val1 and doc.metadata.get(test_tag2) == test_val2 for doc in docs):
        print("[+] Found document matching BOTH filters.")
    else:
        print("[!] Document NOT found with AND filter.")

    # Cleanup
    print("[*] Cleaning up...")
    index.delete_with_filter(filter=[{test_tag1: {"$eq": test_val1}}])

if __name__ == "__main__":
    try:
        test_multi_filter()
    except Exception as e:
        print(f"[!] Test failed: {e}")
