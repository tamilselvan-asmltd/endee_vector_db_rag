# Semantic Retriever Cache & Parallelized Pre-Retrieval

## Overview

The **Retriever Semantic Cache** is a Redis Stack-powered caching layer that stores the final **reranked document chunks** (not LLM answers) for a given query embedding. When a semantically similar query arrives, the system skips both the **Endee Hybrid Search** and the **Cross-Encoder Reranking** phases entirely, returning cached `Document` objects in under 2ms.

The **Parallelized Pre-Retrieval** system runs follow-up suggestion generation concurrently with the main LLM response stream, and internally parallelizes the 3 individual chunk retrievals needed for each suggestion.

Together, these features reduce retrieval latency by **~400x** and eliminate perceived wait time for follow-up suggestions.

## Architecture — Retriever Cache

```mermaid
flowchart TD
    A["Incoming Query"] --> B["Generate Dense Embedding<br/>(768-dim, Ollama)"]
    B --> C["Redis HNSW Vector Search<br/>(KNN=1 on ret_cache index)"]
    C --> D{Similarity ≥ 0.80?}
    D -->|Yes| E["🎯 Cache HIT<br/>Return Cached Documents<br/>Skip DB + Reranker"]
    D -->|No| F["Cache MISS<br/>Continue Pipeline"]
    F --> G["Sparse Embedding<br/>(BM25)"]
    G --> H["Endee Hybrid Search<br/>(Dense + Sparse RRF)"]
    H --> I["Cross-Encoder Reranking<br/>(ms-marco-MiniLM-L6-v2)"]
    I --> J["Top-K Documents"]
    J --> K["Store in Retriever Cache<br/>(SHA-256 key + TTL)"]
    K --> L["Return Documents"]
    E --> L

    style C fill:#4f46e5,color:#fff
    style E fill:#059669,color:#fff
    style F fill:#dc2626,color:#fff
    style I fill:#f59e0b,color:#000
```

## Cache vs No-Cache — Decision Flow

```mermaid
flowchart LR
    A["_get_relevant_documents(query)"] --> B["get_dense_embedding(query)"]
    B --> C{"cache.search(embedding)"}
    C -->|HIT| D["Return cached List of Document<br/>⏱️ ~1ms"]
    C -->|MISS| E["get_sparse_embedding(query)"]
    E --> F["index.query(dense, sparse)"]
    F --> G["CrossEncoder.predict(pairs)"]
    G --> H["Sort + Top-K"]
    H --> I["cache.store(query, embedding, docs)"]
    I --> J["Return docs<br/>⏱️ ~400ms"]

    style D fill:#059669,color:#fff
    style J fill:#dc2626,color:#fff
```

## Redis Index Schema — Retriever Cache

```mermaid
graph LR
    A["Redis Hash Key<br/>ret_cache:tenant_id:sha256_hash"] --> B["query (TextField)"]
    A --> C["chunks_json (TextField)<br/>Serialized List of Document"]
    A --> D["embedding (VectorField<br/>768-dim, HNSW, COSINE)"]
    A --> E["tenant_id (TagField)"]
    A --> F["doc_version (TagField)"]
    A --> G["created_at (NumericField)"]
```

### Serialization Format

Documents are serialized to JSON for storage and reconstructed on retrieval:

```python
# Storing
chunks_data = [
    {"page_content": doc.page_content, "metadata": doc.metadata}
    for doc in documents
]
json.dumps(chunks_data)

# Retrieving
data = json.loads(chunks_json)
docs = [
    Document(page_content=d["page_content"], metadata=d["metadata"])
    for d in data
]
```

This preserves all metadata fields including `filename`, `page`, `link`, `rerank_score`, and `chunk_id`.

## Cache Lifecycle

```mermaid
sequenceDiagram
    participant Ret as HybridEndeeRetriever
    participant Cache as RetrieverSemanticCache
    participant Redis as Redis Stack
    participant DB as Endee Vector DB
    participant RR as CrossEncoder Reranker

    Ret->>Ret: get_dense_embedding(query)
    Ret->>Cache: search(embedding, tenant_id)
    Cache->>Redis: FT.SEARCH KNN=1 with filters
    Redis-->>Cache: Best match + similarity score

    alt Cache HIT (similarity ≥ 0.80)
        Cache-->>Ret: List of Document objects
        Note over Ret: last_cache_hit = True
        Note over Ret: last_hybrid_time = 0.0
        Note over Ret: last_rerank_time = 0.0
    else Cache MISS
        Cache-->>Ret: None
        Note over Ret: last_cache_hit = False
        Ret->>Ret: get_sparse_embedding(query)
        Ret->>DB: Hybrid Query (Dense + Sparse)
        DB-->>Ret: Raw hits
        Ret->>RR: Rerank candidates
        RR-->>Ret: Scored documents
        Ret->>Cache: store(query, embedding, docs)
        Cache->>Redis: HSET + EXPIRE (TTL: 3600s)
    end
```

## Parallelized Pre-Retrieval Architecture

```mermaid
flowchart TD
    A["User Asks Question"] --> B["Retrieval Pipeline"]
    B --> C["Context Chunks Retrieved"]
    
    C --> D["🧵 Thread 1: LLM Stream<br/>(Main Thread)"]
    C --> E["🧵 Thread 2: Suggestion Generator<br/>(Background Thread)"]
    
    D --> F["Streaming Response<br/>to UI"]
    
    E --> G["LLM Generates<br/>3 Follow-up Questions"]
    G --> H["ThreadPoolExecutor<br/>(max_workers=3)"]
    
    H --> I["🧵 Worker 1:<br/>retriever.invoke(Q1)"]
    H --> J["🧵 Worker 2:<br/>retriever.invoke(Q2)"]
    H --> K["🧵 Worker 3:<br/>retriever.invoke(Q3)"]
    
    I --> L["Collected Results"]
    J --> L
    K --> L
    
    F --> M["LLM Stream Ends"]
    M --> N{"Suggestions Ready?"}
    N -->|Yes| O["Display 3 Suggestion<br/>Buttons Instantly"]
    N -->|No| P["Brief Wait<br/>then Display"]

    style D fill:#4f46e5,color:#fff
    style E fill:#059669,color:#fff
    style H fill:#f59e0b,color:#000
```

## Two-Tier Concurrency Model

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant T1 as Main Thread
    participant T2 as Suggestion Thread
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant W3 as Worker 3
    participant LLM as Ollama LLM
    participant Ret as Retriever

    UI->>T1: User submits query
    T1->>T1: Retrieve context docs

    par LLM Streaming + Suggestion Generation
        T1->>LLM: stream(final_prompt)
        LLM-->>T1: chunk by chunk...
        Note over T1: Rendering to UI in real-time
    and
        T2->>LLM: invoke(suggestion_prompt)
        LLM-->>T2: 3 raw questions
        par Parallel Pre-Retrieval (3 workers)
            T2->>W1: retriever.invoke(Q1)
            W1->>Ret: Hybrid Search + Rerank
            Ret-->>W1: docs
        and
            T2->>W2: retriever.invoke(Q2)
            W2->>Ret: Hybrid Search + Rerank
            Ret-->>W2: docs
        and
            T2->>W3: retriever.invoke(Q3)
            W3->>Ret: Hybrid Search + Rerank
            Ret-->>W3: docs
        end
        T2-->>T2: Collect all 3 results
    end

    T1->>T1: LLM stream ends
    T1->>T2: future_suggestions.result()
    T2-->>T1: 3 suggestions with pre-docs
    T1->>UI: Display suggestion buttons
```

## Key Hashing Strategy

```mermaid
flowchart LR
    A["Raw Query<br/>'What is CNC?'"] --> B["SHA-256 Hash<br/>hashlib.sha256()"]
    B --> C["Hex Digest<br/>e3b0c44298fc..."]
    C --> D["Redis Key<br/>ret_cache:default:e3b0c44..."]

    style B fill:#4f46e5,color:#fff
```

**Why SHA-256 instead of Python's `hash()`?**
- Python's `hash()` returns different values across process restarts (randomized by default since Python 3.3)
- SHA-256 is deterministic, ensuring cached entries remain valid across app restarts and container recreation

## Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `retriever_cache_enabled` | `True` | Feature toggle for the retriever cache |
| `retriever_cache_threshold` | `0.80` | Minimum cosine similarity for a cache hit |
| `retriever_cache_ttl` | `3600` | Cache entry TTL in seconds (1 hour) |
| `retriever_cache_index_name` | `retriever_cache_idx` | Redis search index name |
| `retriever_cache_prefix` | `ret_cache:` | Redis key prefix for all cache entries |

## Persistent UI Metrics

Every assistant response in the chat history includes a persistent metrics bar:

| Metric | Description |
|---|---|
| 🚀 tokens/s | LLM generation throughput |
| Latency | Total end-to-end response time |
| Retriever Cache | ✅ HIT or ❌ MISS — whether cached chunks were used |
| Answer Cache | ✅ HIT or ❌ MISS — whether the semantic answer cache was used |
| ⏱️ Retrieval | Breakdown: Hybrid Search time + Rerank time |

These metrics are stored in the message dictionary and survive across page reruns and session reloads.

## Administrative Operations

| Method | Location | Description |
|---|---|---|
| `search(embedding, tenant_id)` | `retriever_cache.py` | KNN-1 vector search with tenant/version filters |
| `store(query, embedding, docs)` | `retriever_cache.py` | Serializes and stores Document list with TTL |
| `clear_all()` | `retriever_cache.py` | Purges all retriever cache entries |
| `generate_suggestions()` | `generator.py` | Parallel suggestion generation + pre-retrieval |

## Performance Comparison

| Scenario | Retrieval Latency | Rerank Latency | Total Retrieval | Improvement |
|---|---|---|---|---|
| **No Cache (Cold)** | ~200ms | ~150ms | ~400ms | Baseline |
| **Retriever Cache HIT** | ~1ms | 0ms (skipped) | ~1ms | **~400x faster** |
| **Suggestion Pre-Retrieval (Sequential)** | — | — | ~1200ms (3 × 400ms) | Baseline |
| **Suggestion Pre-Retrieval (Parallel)** | — | — | ~400ms (max of 3) | **~3x faster** |
| **Suggestion Pre-Retrieval (All Cached)** | — | — | ~3ms (max of 3 × 1ms) | **~400x faster** |

## Difference from Semantic Answer Cache

```mermaid
flowchart TD
    A["Incoming Query"] --> B["Semantic Answer Cache<br/>(sem_cache)"]
    B -->|HIT| C["Return Full LLM Response Text<br/>Skip Everything"]
    B -->|MISS| D["Retriever Chunk Cache<br/>(ret_cache)"]
    D -->|HIT| E["Return Cached Documents<br/>Skip DB + Reranker<br/>Still Run LLM"]
    D -->|MISS| F["Full Pipeline<br/>DB + Reranker + LLM"]

    style B fill:#059669,color:#fff
    style D fill:#4f46e5,color:#fff
    style F fill:#dc2626,color:#fff
```

| Feature | Semantic Answer Cache | Retriever Chunk Cache |
|---|---|---|
| **What is cached** | Final LLM response text | Reranked List of Document objects |
| **What is skipped on HIT** | Retrieval + Reranking + LLM | Retrieval + Reranking only |
| **LLM still runs?** | No | Yes (fresh answer from cached context) |
| **Threshold** | 0.82 | 0.80 |
| **TTL** | 3600s (1 hour) | 3600s (1 hour) |
| **Key prefix** | `sem_cache:` | `ret_cache:` |
| **Index name** | `semantic_cache_idx` | `retriever_cache_idx` |

## Error Handling & Resilience

```mermaid
flowchart TD
    A["Cache Operation"] --> B{Redis Available?}
    B -->|Yes| C["Normal Operation"]
    B -->|No| D["Graceful Fallback"]
    
    D --> E["search() returns None"]
    D --> F["store() silently skips"]
    D --> G["Pipeline continues<br/>without cache"]
    
    H["Index Missing?"] --> I["Auto-create on init"]
    J["Pre-retrieval fails?"] --> K["Return empty pre_docs<br/>Suggestion still shown"]

    style D fill:#f59e0b,color:#000
    style G fill:#059669,color:#fff
```

All cache operations are wrapped in `try/except` blocks. If Redis is unavailable or the index is corrupted, the system gracefully falls back to the standard retrieval pipeline with zero user-facing errors.

## File References

| File | Component |
|---|---|
| `core/retriever_cache.py` | `RetrieverSemanticCache` — full cache implementation |
| `core/retriever.py` | `HybridEndeeRetriever` — cache integration in `_get_relevant_documents()` |
| `core/generator.py` | `generate_suggestions()` — parallelized pre-retrieval with `ThreadPoolExecutor` |
| `config/settings.py` | Retriever cache configuration parameters |
| `streamlit_app.py` | Parallel suggestion thread launch + persistent metrics rendering |
