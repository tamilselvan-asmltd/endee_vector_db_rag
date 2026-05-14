# Retrieval Pipeline

## Overview

The retrieval pipeline implements a **Hybrid Search + Cross-Encoder Reranking** strategy. It fetches candidate documents using both semantic (dense vector) and keyword (sparse BM25) matching, then reranks them with a CrossEncoder model to deliver the most precisely relevant results.

## End-to-End Flow

```mermaid
flowchart TD
    A["User Query"] --> B["EmbeddingService"]
    B --> C["Dense Embedding<br/>(768-dim via Ollama)"]
    B --> D["Sparse Embedding<br/>(BM25 via endee_model)"]
    C --> E["Endee Hybrid Search"]
    D --> E
    E --> F["Top-K Candidates<br/>(K=15 if reranking)"]
    F --> G{"Reranker Enabled?"}
    G -->|Yes| H["CrossEncoder Reranking<br/>(ms-marco-MiniLM-L-6-v2)"]
    H --> I["Scored & Sorted Results"]
    I --> J["Final Top-K Documents<br/>(K=5)"]
    G -->|No| J

    style E fill:#4f46e5,color:#fff
    style H fill:#dc2626,color:#fff
    style J fill:#059669,color:#fff
```

## Detailed Stage Breakdown

### Stage 1: Dual Embedding Generation

```mermaid
flowchart LR
    Q["Query Text"] --> D["get_dense_embedding()"]
    Q --> S["get_sparse_embedding(is_query=True)"]
    D --> DV["768-dim Float Vector"]
    S --> SV["Sparse Indices + Values"]
```

The query is simultaneously transformed into:
- **Dense Vector**: Captures semantic meaning via `nomic-embed-text`
- **Sparse Vector**: Captures exact keyword importance via BM25

### Stage 2: Hybrid Search on Endee

```mermaid
sequenceDiagram
    participant R as Retriever
    participant E as Endee Index

    R->>E: query(vector, sparse_indices, sparse_values, top_k=15, filter)
    Note over E: HNSW Approximate NN Search<br/>+ BM25 Keyword Match<br/>+ Internal Score Fusion
    E-->>R: 15 candidate documents with scores
```

**Why Hybrid?**
- Dense search excels at understanding "What does this mean?"
- Sparse search excels at finding "Which document mentions this exact term?"
- Fusion combines both for maximum recall

### Stage 3: Cross-Encoder Reranking

```mermaid
flowchart TD
    A["15 Candidate Documents"] --> B["CrossEncoder<br/>ms-marco-MiniLM-L-6-v2"]
    B --> C["Score Each<br/>(query, document) Pair"]
    C --> D["Sort by Rerank Score<br/>Descending"]
    D --> E["Return Top 5"]

    style B fill:#dc2626,color:#fff
```

**How Reranking Works:**
1. Forms `(query, document_text)` pairs for all 15 candidates
2. The CrossEncoder scores each pair independently (more accurate than bi-encoder similarity)
3. Results are sorted by this refined score
4. Only the top 5 are returned to the LLM

### Stage 4: Document Formatting

```mermaid
flowchart LR
    A["Ranked Documents"] --> B["Inject Doc Links<br/>(doc_server_url)"]
    B --> C["Convert to LangChain<br/>Document objects"]
    C --> D["Return to Generator"]
```

## Reranker Model Management

```mermaid
flowchart TD
    A["HybridEndeeRetriever.__init__()"] --> B{Global Cache<br/>_RERANKER_INSTANCE?}
    B -->|Loaded| C["Reuse Cached Model"]
    B -->|None| D{Local Path Exists?<br/>models/reranker/}
    D -->|Yes| E["Load from Disk<br/>(Offline Mode)"]
    D -->|No| F["Download from HuggingFace"]
    F --> G["Save to Local Disk<br/>for Future Offline Use"]
    E --> H["Set Global Cache"]
    G --> H
    H --> C

    style B fill:#f59e0b,color:#000
    style E fill:#059669,color:#fff
```

**Key Design Decisions:**
- **Global Singleton**: The reranker is loaded once and cached globally (`_RERANKER_INSTANCE`). This prevents expensive reloading when Streamlit reruns the script.
- **Offline-First**: The model is first checked at `models/reranker/`. If found, it loads locally. If not, it downloads from HuggingFace and saves locally for future offline deployments.

## Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `top_k` | 5 | Final documents returned |
| `rerank_top_k` | 15 | Candidates fetched before reranking |
| `use_reranker` | `True` | Toggle reranking on/off |
| `reranker_model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model |
| `reranker_model_path` | `models/reranker/` | Local cache path |

## Performance Metrics

The retriever tracks granular timing for observability:

| Metric | Description |
|--------|-------------|
| `last_retrieval_time` | Total time (hybrid + rerank) |
| `last_hybrid_time` | Time for Endee hybrid query only |
| `last_rerank_time` | Time for CrossEncoder scoring only |

These are surfaced in the Streamlit UI as:
```
🚀 Speed: 12.5 tokens/s | Latency: 2.3s | Retrieval: 0.8s (Hybrid: 0.3s, Rerank: 0.5s)
```

## Implementation: `core/retriever.py`

| Method | Description |
|--------|-------------|
| `__init__()` | Loads or downloads the CrossEncoder reranker |
| `_get_relevant_documents(query)` | Full hybrid search + rerank pipeline |

The retriever extends LangChain's `BaseRetriever`, making it compatible with the `create_history_aware_retriever` chain.
