# Ingestion Pipeline

## Overview

The ingestion pipeline transforms raw PDF engineering manuals into searchable vector embeddings stored in the Endee database. It follows a **Load → Split → Embed → Store** architecture with parallel processing for high throughput.

## End-to-End Flow

```mermaid
flowchart TD
    A["📄 PDF Upload<br/>(Streamlit UI)"] --> B["PDFLoader.load()"]
    B --> C["Raw Page Text<br/>+ Metadata"]
    C --> D["TextSplitter.split()"]
    D --> E["Text Chunks<br/>(700 chars, 120 overlap)"]
    E --> F["EmbeddingService"]
    F --> G["Dense Embedding<br/>(Ollama nomic-embed-text)<br/>768 dimensions"]
    F --> H["Sparse Embedding<br/>(endee BM25 model)"]
    G --> I["Batch Upsert"]
    H --> I
    I --> J["Endee Vector DB<br/>(HNSW + BM25 Index)"]

    style A fill:#f59e0b,color:#000
    style J fill:#4f46e5,color:#fff
    style F fill:#059669,color:#fff
```

## Pipeline Stages

### Stage 1: Document Loading (`core/loader.py`)

```mermaid
flowchart LR
    A["PDF File Path"] --> B["PdfReader<br/>(pypdf)"]
    B --> C["Page Iterator"]
    C --> D["extract_text()"]
    D --> E["clean_text()<br/>Remove nulls<br/>Normalize whitespace"]
    E --> F["(text, metadata) tuples"]
```

**Key Behaviors:**
- Skips pages with no extractable text
- Removes null characters (`\x00`) and collapses whitespace
- Attaches metadata: `filename`, `page`, `source`

### Stage 2: Text Splitting (`core/splitter.py`)

```mermaid
flowchart LR
    A["Full Page Text"] --> B["Sliding Window<br/>chunk_size=700"]
    B --> C["Overlap Region<br/>chunk_overlap=120"]
    C --> D["Chunk 1"]
    C --> E["Chunk 2"]
    C --> F["Chunk N"]
```

**Configuration:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chunk_size` | 700 | Characters per chunk |
| `chunk_overlap` | 120 | Overlap between adjacent chunks |

**Overlap Logic:** Ensures that sentences split at a chunk boundary are preserved in the next chunk, preventing information loss during retrieval.

### Stage 3: Embedding Generation (`core/embeddings.py`)

```mermaid
flowchart TD
    A["Text Chunk"] --> B{Redis Cache<br/>embed_cache:sha256}
    B -->|HIT| C["Return Cached Vector"]
    B -->|MISS| D["Ollama API<br/>/api/embeddings"]
    D --> E["768-dim Dense Vector"]
    E --> F["Store in Redis<br/>TTL: 7 days"]
    F --> C

    A --> G["SparseModel<br/>(endee/bm25)"]
    G --> H["Sparse Indices + Values"]

    style B fill:#f59e0b,color:#000
    style D fill:#4f46e5,color:#fff
```

**Triple-Layer Caching:**
1. **LRU Cache** (`@lru_cache(128)`): In-memory, per-process
2. **Redis Cache** (`embed_cache:{sha256}`): Cross-process, 7-day TTL
3. **Parallel Batch** (`ThreadPoolExecutor`): Up to 5 concurrent embedding calls

### Stage 4: Batch Upsert

```mermaid
sequenceDiagram
    participant App as Streamlit App
    participant DB as DatabaseService
    participant Endee as Endee Server

    App->>DB: ensure_index(recreate=False)
    loop For each PDF
        App->>App: load() → split() → embed()
        App->>DB: upsert_batch(index, points)
        DB->>Endee: index.upsert(points)
    end
    Note over Endee: HNSW + BM25 index updated
```

## Metadata Schema

Each ingested chunk carries the following metadata:

```json
{
  "text": "The hydraulic pump operates at...",
  "filename": "/uploads/cnc_manual.pdf",
  "page": 42,
  "source": "pdf",
  "dept": "maintenance",
  "link": "http://localhost:8003/cnc_manual.pdf"
}
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Embedding Model | `nomic-embed-text:latest` (768-dim) |
| Sparse Model | `endee/bm25` (local, offline) |
| Embedding Cache TTL | 7 days |
| Batch Threading | 5 workers |
| Chunk Size | 700 characters |
| Overlap | 120 characters |
