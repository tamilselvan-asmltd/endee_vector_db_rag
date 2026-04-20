import argparse
import sys
import os
import hashlib
from tqdm import tqdm
from config.settings import settings

# --- Configure NLTK for Offline Use (Must happen before core imports) ---
os.environ["NLTK_DATA"] = settings.nltk_data_path
try:
    import nltk
    # Mock download to be a no-op to prevent network checks
    def noop_download(*args, **kwargs):
        pass
    nltk.download = noop_download
    
    if settings.nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, settings.nltk_data_path)
except ImportError:
    pass
# ------------------------------------------------------------------------

from core.loader import PDFLoader
from core.splitter import TextSplitter
from core.embeddings import EmbeddingService
from core.database import DatabaseService
from core.retriever import HybridEndeeRetriever
from core.generator import GenerationService

def ingest(pdf_path: str, recreate: bool = False):
    """Ingests a PDF into the vector database."""
    print(f"[*] Ingesting PDF: {pdf_path}")
    
    loader = PDFLoader()
    splitter = TextSplitter(settings.chunk_size, settings.chunk_overlap)
    embeddings = EmbeddingService()
    db = DatabaseService()
    
    # 1. Load PDF
    pages = loader.load(pdf_path)
    
    # 2. Process and Split
    all_points = []
    print("[*] Chunking and embedding...")
    
    for page_text, metadata in pages:
        chunks = splitter.split(page_text)
        filename = metadata.get("filename", "unknown")
        
        # Batch dense embeddings for current page chunks
        dense_vecs = embeddings.get_dense_embeddings_batch(chunks)
        
        for i, (chunk, dvec) in enumerate(zip(chunks, dense_vecs)):
            # Generate unique ID based on content hash
            point_id = hashlib.sha256(f"{filename}_{chunk}".encode()).hexdigest()
            
            # Sparse embedding
            sv = embeddings.get_sparse_embedding(chunk)
            
            if sv is None or not sv.indices.tolist():
                continue
                
            all_points.append({
                "id": point_id,
                "vector": dvec,
                "sparse_indices": sv.indices.tolist(),
                "sparse_values": sv.values.tolist(),
                "meta": {
                    "text": chunk,
                    **metadata,
                    "chunk_id": i
                },
            })

    # 3. Deduplicate points by ID (Endee forbids duplicates in a single batch)
    unique_points = {p['id']: p for p in all_points}
    all_points = list(unique_points.values())
    
    if not all_points:
        print("[!] No new unique chunks to ingest. Index is up to date.")
        return

    # 4. Upsert to DB
    print(f"[*] Upserting {len(all_points)} unique chunks to Endee...")
    db.ensure_index(recreate=recreate)
    index = db.get_index()
    
    # Batch upsert (64 at a time as per notebook pattern)
    batch_size = 64
    for i in range(0, len(all_points), batch_size):
        db.upsert_batch(index, all_points[i : i + batch_size])
    
    print("[+] Ingestion complete.")

def ask(query: str):
    """Runs the RAG query pipeline."""
    print(f"[*] Querying system for: '{query}'")
    
    print("[*] Initializing RAG services...")
    embeddings = EmbeddingService()
    db = DatabaseService()
    index = db.get_index()
    
    retriever = HybridEndeeRetriever(
        index=index,
        embedding_service=embeddings
    )
    
    generator = GenerationService(retriever)
    print("[*] Executing RAG pipeline...")
    result, metrics = generator.run_with_metrics(query)
    
    print(f"[*] Performance Metrics:")
    print(f"    - Retrieval Speed: {metrics['retrieval_time']:.3f} sec")
    print(f"    - LLM Completion: {metrics['llm_time']:.3f} sec")
    print(f"    - LLM TPS:        {metrics['tps']:.2f} tokens/sec")
    
    # 4. Display Unique Source Links
    docs = result.get("source_documents", [])
    unique_links = sorted(list(set(doc.metadata.get("link") for doc in docs if doc.metadata.get("link"))))
    
    if unique_links:
        print("\n[*] Unique Source Links:")
        for link in unique_links:
            print(f"    - {link}")
            
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="Endee RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document")
    ingest_parser.add_argument("--path", required=True, help="Path to the PDF file")
    ingest_parser.add_argument("--recreate", action="store_true", help="Recreate index before ingestion")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Query the RAG system")
    ask_parser.add_argument("--query", required=True, help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        # 1. Identify PDF files to process
        pdf_files = []
        if os.path.isdir(args.path):
            print(f"[*] Scanning directory: {args.path}")
            for f in os.listdir(args.path):
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(args.path, f))
            if not pdf_files:
                print(f"[!] No PDF files found in {args.path}")
                sys.exit(0)
            print(f"[*] Found {len(pdf_files)} PDF(s) to process.")
        elif os.path.isfile(args.path):
            pdf_files = [args.path]
        else:
            print(f"[!] Invalid path: {args.path}")
            sys.exit(1)

        # 2. Confirmation for Destructive Action
        if args.recreate:
            confirm = input(f"\n[!] WARNING: This will delete ALL existing data in the index '{settings.endee_index_name}'.\n[?] Are you sure you want to proceed with {len(pdf_files)} file(s)? (y/N): ").lower()
            if confirm not in ['y', 'yes']:
                print("[*] Ingestion cancelled.")
                sys.exit(0)
        
        # 3. Process Batch
        print("-" * 30)
        for i, pdf_path in enumerate(pdf_files):
            print(f"[*] [{i+1}/{len(pdf_files)}] Processing file: {pdf_path}")
            # We only recreate on the very first file if requested
            run_recreate = args.recreate if i == 0 else False
            try:
                ingest(pdf_path, run_recreate)
            except Exception as e:
                print(f"[!] Error processing {pdf_path}: {e}")
            print("-" * 30)
            
        print(f"\n[+] Batch Ingestion Complete. Processed {len(pdf_files)} file(s).")
    
    elif args.command == "ask":
        ask(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
