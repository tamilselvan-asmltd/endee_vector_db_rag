# Semantic Cache

## Overview

The Semantic Cache is a Redis Stack-powered vector similarity cache that prevents redundant LLM calls. Instead of matching queries by exact text, it uses **cosine similarity** on embedding vectors to detect semantically equivalent questions and return cached answers instantly.

## Architecture

```mermaid
flowchart TD
    A["Incoming Query"] --> B["Normalize Query<br/>(lowercase, remove stopwords)"]
    B --> C["Generate Dense Embedding<br/>(768-dim)"]
    C --> D["Redis Vector Search<br/>(HNSW KNN=1)"]
    D --> E{Similarity ≥ Threshold?<br/>Default: 0.82}
    E -->|Yes| F["🎯 Cache HIT<br/>Return Stored Response"]
    E -->|No| G["Cache MISS<br/>Proceed to LLM"]
    G --> H["LLM Generates Answer"]
    H --> I["Store in Cache<br/>(query + embedding + response)"]

    style D fill:#4f46e5,color:#fff
    style F fill:#059669,color:#fff
    style G fill:#dc2626,color:#fff
```

## Multi-Tenant Isolation

```mermaid
flowchart TD
    A["Search Request"] --> B["Build Filter String"]
    B --> C["@tenant_id:{user_id}"]
    B --> D["@doc_version:{1.0}"]
    C --> E["Redis FT.SEARCH<br/>Filtered KNN"]
    D --> E
    E --> F["Results Scoped to<br/>User + Version Only"]

    style E fill:#4f46e5,color:#fff
```

**Key Principle:** User A's cached answers are **never** returned to User B. Every cache lookup is filtered by `tenant_id` (which maps to `user_id`), ensuring strict data isolation.

## Redis Index Schema

```mermaid
graph LR
    A["Redis Hash Key<br/>sem_cache:user_id:timestamp"] --> B["query (TextField)"]
    A --> C["response (TextField)"]
    A --> D["embedding (VectorField, 768-dim HNSW)"]
    A --> E["tenant_id (TagField)"]
    A --> F["doc_version (TagField)"]
    A --> G["prompt_hash (TagField)"]
    A --> H["created_at (NumericField)"]
```

## Query Normalization

Before searching the cache, the query is normalized for better matching:

```mermaid
flowchart LR
    A["How do I fix the hydraulic pump?"] --> B["Lowercase"]
    B --> C["Remove Punctuation"]
    C --> D["Remove Stopwords<br/>(NLTK English)"]
    D --> E["fix hydraulic pump"]
```

This ensures that "How do I fix the hydraulic pump?" and "Fix the hydraulic pump" hit the same cache entry.

## Cache Lifecycle

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant Cache as SemanticCache
    participant Redis as Redis Stack
    participant LLM as Ollama LLM

    UI->>Cache: search(query_embedding, tenant_id)
    Cache->>Redis: FT.SEARCH with KNN + Filters
    Redis-->>Cache: Best match + similarity score

    alt Cache HIT (similarity ≥ 0.82)
        Cache-->>UI: Cached response + search_time
        Note over UI: Display with "🎯 Cache Hit!" badge
    else Cache MISS
        Cache-->>UI: miss signal
        UI->>LLM: Generate response (streaming)
        LLM-->>UI: Full response
        UI->>Cache: store(query, embedding, response, tenant_id)
        Cache->>Redis: HSET + EXPIRE (TTL: 3600s)
    end
```

## Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `semantic_cache_threshold` | `0.82` | Minimum cosine similarity for a hit |
| `semantic_cache_ttl` | `3600` | Cache entry TTL in seconds (1 hour) |
| `semantic_cache_index_name` | `semantic_cache_idx` | Redis search index name |
| `semantic_cache_prefix` | `sem_cache:` | Redis key prefix |

## Administrative Operations

| Method | Description |
|--------|-------------|
| `search()` | KNN-1 vector search with tenant/version filters |
| `store()` | Stores a new cache entry with TTL |
| `invalidate_version(version)` | Wipes all entries for a specific doc version |
| `clear_all()` | Purges the entire cache across all tenants |
| `clear_tenant(tenant_id)` | Wipes cache for a specific user |

## Performance Impact

| Scenario | Latency | LLM Cost |
|----------|---------|----------|
| Cache MISS | ~3-8 seconds (retrieval + LLM) | Full inference |
| Cache HIT | ~5-15 milliseconds | Zero |

A well-warmed cache can reduce average query latency by **95%+** for repeated engineering questions.
