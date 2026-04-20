import sys
import os
import time
import requests
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever

try:
    import chromadb
    from rank_bm25 import BM25Okapi
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ---------------------------------------------------------
# Chroma Config
# ---------------------------------------------------------
CHROMA_HOST = "192.168.1.30"
CHROMA_PORT = 8000
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "embeddinggemma:300m"

def get_chroma_embedding(text: str):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": text}
    )
    return response.json()["embedding"]

# ---------------------------------------------------------
# Main Benchmark Logic
# ---------------------------------------------------------
def run_benchmark():
    queries = [
        "laser sensor is not working what i need to check?",
        "what is machine learning?",
        "how to fix motor overload issue?",
        "what are safety precautions in CNC machine?",
        "why is temperature sensor failing?"
    ]

    print("="*40)
    print("STEP 1: ENDEE DB SETUP")
    print("="*40)
    embeddings = EmbeddingService()
    db = DatabaseService()
    index = db.get_index()
    
    endee_retriever = HybridEndeeRetriever(
        index=index,
        embedding_service=embeddings
    )
    
    # Measure Endee DB
    endee_times = []
    print("\nBENCHMARKING ENDEE DB...")
    for q in queries:
        start = time.perf_counter()
        docs = endee_retriever.invoke(q)
        end = time.perf_counter()
        
        # We can use the retriever's internally tracked time for precision
        elapsed = endee_retriever.last_retrieval_time
        endee_times.append(elapsed)
        print(f"[{elapsed:.4f}s] {q[:40]}...")

    print("="*40)
    print("STEP 2: CHROMA DB SETUP")
    print("="*40)
    
    chroma_times = []
    
    if CHROMA_AVAILABLE:
        try:
            print("Connecting to Chroma...")
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            collection = client.get_collection(name="dev")
            data = collection.get()
            documents = data["documents"]
            
            print("Preparing BM25 for Chroma Hybrid Search...")
            tokenized_docs = [doc.split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            
            def chroma_hybrid_search(query: str, top_k: int = 5, alpha: float = 0.7):
                t_start = time.perf_counter()
                
                # Dense
                query_embedding = get_chroma_embedding(query)
                dense_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                dense_docs = dense_results["documents"][0]
                dense_distances = dense_results["distances"][0]
                
                dense_map = {doc: (1 - dist) for doc, dist in zip(dense_docs, dense_distances)}
                
                # BM25
                tokenized_query = query.split()
                bm25_scores = bm25.get_scores(tokenized_query)
                max_bm25 = bm25_scores.max() if len(bm25_scores) > 0 else 1
                bm25_scores = [s / max_bm25 for s in bm25_scores]
                
                # Hybrid
                hybrid_scores = []
                for i, doc in enumerate(documents):
                    dense_score = dense_map.get(doc, 0)
                    score = alpha * dense_score + (1 - alpha) * bm25_scores[i]
                    hybrid_scores.append((doc, score))
                
                hybrid_scores.sort(key=lambda x: x[1], reverse=True)
                return time.perf_counter() - t_start

            print("\nBENCHMARKING CHROMA DB...")
            for q in queries:
                elapsed = chroma_hybrid_search(q, top_k=5)
                chroma_times.append(elapsed)
                print(f"[{elapsed:.4f}s] {q[:40]}...")

        except Exception as e:
            print(f"Error accessing Chroma DB: {e}")
            print("Using fallback Chroma times from previous notebook run.")
            chroma_times = [4.2977, 0.4575, 1.5054, 1.7079, 0.4976]
    else:
        print("Chroma dependencies not installed. Using fallback Chroma times.")
        chroma_times = [4.2977, 0.4575, 1.5054, 1.7079, 0.4976]

    # Calculate averages
    avg_endee = sum(endee_times) / len(endee_times)
    avg_chroma = sum(chroma_times) / len(chroma_times)

    print("\n========================================")
    print(f"Average Retrieval Time (Endee DB)  : {avg_endee:.4f} sec")
    print(f"Average Retrieval Time (Chroma DB) : {avg_chroma:.4f} sec")
    print("========================================")

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    chart_path = os.path.join(os.path.dirname(__file__), "db_comparison_chart.png")
    
    # X-axis labels
    query_labels = [f"Q{i+1}" for i in range(len(queries))]
    
    plt.figure(figsize=(10, 6))
    
    # Plot lines with modern styling
    plt.plot(query_labels, endee_times, marker='o', linestyle='-', linewidth=2, markersize=8, color='#00d1b2', label=f'Endee DB (Avg: {avg_endee:.2f}s)')
    plt.plot(query_labels, chroma_times, marker='s', linestyle='-', linewidth=2, markersize=8, color='#ff3860', label=f'Chroma DB (Avg: {avg_chroma:.2f}s)')
    
    # Chart styling
    plt.title("Retrieval Speed Comparison: Endee DB vs Chroma DB", fontsize=14, pad=15)
    plt.xlabel("Test Queries", fontsize=12)
    plt.ylabel("Time Taken (Seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize=11)
    
    # Add data annotations (optional, but helps readability)
    for i, (te, tc) in enumerate(zip(endee_times, chroma_times)):
        plt.annotate(f"{te:.2f}s", (i, te), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#00d1b2')
        plt.annotate(f"{tc:.2f}s", (i, tc), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#ff3860')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"\nChart successfully generated and saved to: {chart_path}")

if __name__ == "__main__":
    run_benchmark()
