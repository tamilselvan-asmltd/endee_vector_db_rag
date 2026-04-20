import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever

def calculate_retriever_speed():
    print("[*] Initializing services...")
    # 1. Initialize Services
    embeddings = EmbeddingService()
    db = DatabaseService()
    index = db.get_index()
    
    # 2. Setup Retriever
    retriever = HybridEndeeRetriever(
        index=index,
        embedding_service=embeddings
    )
    
    # 3. Define 5 test questions
    questions = [
        "What is the role of AI in EDI?",
        "How do we handle document ingestion?",
        "What are the benefits of using a vector database?",
        "Explain the hybrid search mechanism.",
        "How is security managed in the RAG pipeline?"
    ]
    
    total_time = 0.0
    print("\n" + "="*80)
    print("RETRIEVER SPEED TEST")
    print("="*80)
    
    # 4. Measure speed for each question
    for i, q in enumerate(questions):
        print(f"\n[Question {i+1}] {q}")
        
        start_time = time.perf_counter()
        # Retrieve chunks (retriever.invoke also calculates internally, but we can measure entire call)
        docs = retriever.invoke(q)
        end_time = time.perf_counter()
        
        # We can also use retriever.last_retrieval_time
        time_taken = retriever.last_retrieval_time
        total_time += time_taken
        
        print(f"Retrieved {len(docs)} chunks in {time_taken:.4f} seconds")
        
    # 5. Calculate average time
    avg_time = total_time / len(questions)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total Time for 5 questions : {total_time:.4f} seconds")
    print(f"Average Time per question  : {avg_time:.4f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    calculate_retriever_speed()
