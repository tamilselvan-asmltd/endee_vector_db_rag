import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever

def run_advanced_tests():
    print("[*] Initializing services for advanced testing...")
    
    # 1. Initialize Services
    embeddings = EmbeddingService()
    db = DatabaseService()
    index = db.get_index()
    
    # 2. Setup Retriever
    retriever = HybridEndeeRetriever(
        index=index,
        embedding_service=embeddings
    )
    
    # 3. Define Test Questions
    questions = [
        # Ambiguity / Multiple Answer Traps
        "What is the correct next number after 2, 4, 8? Explain all possibilities.",
        "Can intelligence be defined objectively, or is it context-dependent?",
        "Is AlphaZero truly intelligent or just optimized? Why?",
        
        # Contradiction-Based Questions
        "If weak AI ignores human-like structure, why is strong AI still pursued?",
        "Can a system pass the Turing Test and still not be intelligent?",
        "Are heuristics better than algorithms for all AI problems?",
        
        # Multi-Section Reasoning
        "How do heuristics, genetic algorithms, and expert systems relate to each other?",
        "Compare machine learning and expert systems."
    ]
    
    total_time = 0.0
    print("\n" + "="*80)
    print("ADVANCED MODEL REASONING - RETRIEVER SPEED TEST")
    print("="*80)
    
    # 4. Measure speed for each question
    for i, q in enumerate(questions):
        print(f"\n[Question {i+1}] {q}")
        
        # Retrieve chunks
        # Because we only want to test the retriever, we use invoke and check last_retrieval_time
        docs = retriever.invoke(q)
        
        time_taken = retriever.last_retrieval_time
        total_time += time_taken
        
        print(f"Retrieved {len(docs)} chunks in {time_taken:.4f} seconds")
        
    # 5. Calculate average time
    avg_time = total_time / len(questions)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total Time for {len(questions)} questions : {total_time:.4f} seconds")
    print(f"Average Time per question  : {avg_time:.4f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_advanced_tests()
