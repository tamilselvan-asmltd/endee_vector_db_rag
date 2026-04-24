import sys
import os
import hashlib
from typing import List, Dict, Any

# Add current dir to sys.path
sys.path.append(os.getcwd())

from core.database import DatabaseService
from main import ingest, delete_by_filter

def test_filters():
    db = DatabaseService()
    db.ensure_index()
    index = db.get_index()
    
    test_tag = "test_run_id"
    test_val = hashlib.sha256(b"test_run").hexdigest()[:8]
    extra_meta = {test_tag: test_val}
    
    print(f"[*] Starting verification with tag: {test_tag}={test_val}")
    
    # 1. Mock ingestion of a small snippet
    # We can't easily mock the PDF loader here without a file, 
    # but we can manually upsert a point to verify delete_with_filter works.
    
    point_id = "test_point_123"
    vector = [0.1] * 768 # Dummy vector
    point = {
        "id": point_id,
        "vector": vector,
        "sparse_indices": [0],
        "sparse_values": [1.0],
        "filter": {
            test_tag: test_val
        },
        "meta": {
            "text": "This is a test chunk.",
            test_tag: test_val
        }
    }
    
    print("[*] Upserting test point...")
    db.upsert_batch(index, [point])
    
    # 2. Verify it exists via query
    print("[*] Querying without filter...")
    res_no_filt = index.query(
        vector=vector,
        sparse_indices=[0],
        sparse_values=[1.0],
        top_k=5
    )
    print(f"[*] Found {len(res_no_filt)} points without filter.")

    print("[*] Querying with filter ($eq operator)...")
    res = index.query(
        vector=vector,
        sparse_indices=[0],
        sparse_values=[1.0],
        top_k=TOP_K if 'TOP_K' in locals() else 5,
        filter=[{test_tag: {"$eq": test_val}}]
    )
    
    if any(h['id'] == point_id for h in res):
        print("[+] Test point found in index.")
    else:
        print("[!] Test point NOT found in index after upsert.")
        return

    # 3. Delete by filter
    print("[*] Deleting by filter...")
    delete_by_filter([{test_tag: {"$eq": test_val}}])
    
    # 4. Verify it's gone
    print("[*] Querying again after deletion...")
    res_after = index.query(
        vector=vector,
        sparse_indices=[0],
        sparse_values=[1.0],
        top_k=5,
        filter=[{test_tag: {"$eq": test_val}}]
    )
    
    if any(h['id'] == point_id for h in res_after):
        print("[!] Test point still exists after deletion!")
    else:
        print("[+] Test point successfully deleted.")

if __name__ == "__main__":
    try:
        test_filters()
    except Exception as e:
        print(f"[!] Test failed with error: {e}")
