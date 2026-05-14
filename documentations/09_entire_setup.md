# Entire System Setup - Complete Architecture

## Overview

This is a **100% offline, production-grade, multi-tenant RAG system** built with Ollama, Endee Vector Database, Redis Stack, and Streamlit. It transforms engineering PDF manuals into an intelligent Q&A assistant with semantic caching, conversation memory, and administrative controls.

## System Architecture

```mermaid
flowchart TD
    subgraph UI["Streamlit Frontend"]
        A["AI Chat Page"]
        B["System Management Page"]
        C["Sidebar Controls"]
    end
    
    subgraph Core["Core Engine - Python"]
        D["GenerationService<br/>Orchestrator"]
        E["HybridEndeeRetriever<br/>+ CrossEncoder Reranker"]
        F["EmbeddingService<br/>Dense + Sparse"]
        G["RedisSemanticCache<br/>Vector Similarity Cache"]
        H["ChatHistoryManager<br/>Multi-User Memory"]
    end
    
    subgraph Infra["Infrastructure - Docker"]
        I["Ollama Server<br/>LLM + Embeddings"]
        J["Endee Vector DB<br/>HNSW + BM25"]
        K["Redis Stack<br/>Cache + Search + History"]
    end
    
    A --> D
    B --> H
    C --> H
    D --> E
    D --> G
    D --> H
    E --> F
    F --> I
    E --> J
    G --> K
    H --> K
    
    style UI fill:#f59e0b,color:#000
    style Core fill:#4f46e5,color:#fff
    style Infra fill:#059669,color:#fff
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit
    participant Cache as SemanticCache
    participant History as ChatHistory
    participant Retriever as HybridRetriever
    participant Reranker as CrossEncoder
    participant LLM as Ollama LLM
    participant Suggest as SuggestionGen

    User->>UI: Ask question
    UI->>History: Fetch last 5 turns
    UI->>Cache: Check semantic cache
    
    alt Cache HIT
        Cache-->>UI: Cached answer
        UI->>History: Save Q+A
        UI->>Suggest: Generate 3 suggestions
        Suggest-->>UI: Questions + pre-docs
        UI-->>User: Answer + 3 bubble buttons
    else Cache MISS
        UI->>Retriever: Hybrid search (dense+sparse)
        Retriever->>Reranker: Rerank 15 candidates
        Reranker-->>Retriever: Top 5 documents
        Retriever-->>UI: Context documents
        UI->>LLM: Stream answer with context
        LLM-->>UI: Token-by-token response
        UI->>Cache: Store answer
        UI->>History: Save Q+A
        UI->>Suggest: Generate 3 suggestions
        Suggest-->>UI: Questions + pre-docs
        UI-->>User: Answer + sources + 3 bubbles
    end
```

## Component Map

```mermaid
flowchart LR
    subgraph Config
        A["config/settings.py<br/>Pydantic Settings"]
        B[".env<br/>Environment Variables"]
    end
    
    subgraph Core
        C["core/loader.py<br/>PDF Text Extraction"]
        D["core/splitter.py<br/>Chunk with Overlap"]
        E["core/embeddings.py<br/>Dense + Sparse + Cache"]
        F["core/database.py<br/>Endee CRUD Operations"]
        G["core/retriever.py<br/>Hybrid Search + Rerank"]
        H["core/generator.py<br/>LLM Orchestration"]
        I["core/semantic_cache.py<br/>Redis Vector Cache"]
        J["core/history_manager.py<br/>Redis Chat Memory"]
    end
    
    subgraph App
        K["streamlit_app.py<br/>Full UI Application"]
        L["main.py<br/>CLI Pipeline"]
    end
    
    B --> A
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
    A --> H
    A --> I
    A --> J
    C --> K
    D --> K
    H --> K
    
    style Config fill:#f59e0b,color:#000
    style Core fill:#4f46e5,color:#fff
    style App fill:#059669,color:#fff
```

## Infrastructure Services

### Service Dependency Map

```mermaid
flowchart TD
    A["Streamlit App<br/>Port 8501"] --> B["Ollama Server<br/>Port 11434"]
    A --> C["Endee Vector DB<br/>Port 8080"]
    A --> D["Redis Stack<br/>Port 6379"]
    A --> E["Doc Server<br/>Port 8003"]
    
    B --> F["nomic-embed-text<br/>Embedding Model"]
    B --> G["gpt-oss:120b-cloud<br/>LLM Model"]
    
    D --> H["RediSearch Module<br/>Vector Index"]
    D --> I["RedisJSON Module<br/>Data Storage"]
    
    style A fill:#f59e0b,color:#000
    style B fill:#4f46e5,color:#fff
    style C fill:#059669,color:#fff
    style D fill:#dc2626,color:#fff
```

### Port Configuration

| Service | Port | Purpose |
|---|---|---|
| Streamlit | 8501 | Web UI |
| Ollama | 11434 | LLM + Embeddings API |
| Endee | 8080 | Vector Database |
| Redis Stack | 6379 | Cache + History + Search |
| Doc Server | 8003 | PDF file serving for source links |

## Redis Namespace Map

```mermaid
graph TD
    A["Redis Keyspace"] --> B["users:registry<br/>Type: SET<br/>Global user list"]
    A --> C["chat:user:session<br/>Type: LIST<br/>Conversation messages"]
    A --> D["sem_cache:tenant:timestamp<br/>Type: HASH<br/>Cached Q+A + embeddings"]
    A --> E["embed_cache:sha256<br/>Type: STRING<br/>Cached embedding vectors"]
    A --> F["semantic_cache_idx<br/>Type: FT.INDEX<br/>HNSW vector search index"]
    
    style B fill:#f59e0b,color:#000
    style C fill:#4f46e5,color:#fff
    style D fill:#059669,color:#fff
    style E fill:#dc2626,color:#fff
```

## Caching Strategy - Triple Layer

```mermaid
flowchart TD
    A["Incoming Query"] --> B{"Layer 1:<br/>Semantic Cache<br/>(Redis Vector Search)"}
    B -->|HIT| C["Return Cached Answer<br/>~5ms latency"]
    B -->|MISS| D{"Layer 2:<br/>Embedding Cache<br/>(Redis Key-Value)"}
    D -->|HIT| E["Reuse Cached Embedding"]
    D -->|MISS| F["Generate via Ollama"]
    F --> G["Store in Embedding Cache<br/>TTL: 7 days"]
    E --> H["Layer 3:<br/>LRU In-Memory Cache<br/>(Python @lru_cache)"]
    H --> I["Proceed to Retrieval + LLM"]
    I --> J["Store Answer in Semantic Cache<br/>TTL: 1 hour"]
    
    style B fill:#059669,color:#fff
    style D fill:#4f46e5,color:#fff
    style H fill:#f59e0b,color:#000
```

## Environment Configuration

### .env File Structure

| Section | Variables |
|---|---|
| **Ollama** | OLLAMA_URL, OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL |
| **Endee** | ENDEE_URL, ENDEE_INDEX_NAME |
| **RAG** | TOP_K, DOC_SERVER_URL, NLTK_DATA_PATH |
| **Redis Cache** | REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, SEMANTIC_CACHE_THRESHOLD, SEMANTIC_CACHE_TTL |
| **Redis History** | REDIS_CHAT_HISTORY_PREFIX, MAX_HISTORY_MESSAGES |
| **Redis Embeddings** | REDIS_EMBED_CACHE_PREFIX, EMBED_CACHE_TTL |

### Pydantic Settings (config/settings.py)

All environment variables are loaded via Pydantic BaseSettings with:
- Type validation (str, int, float, bool)
- Default values for every parameter
- Automatic .env file loading
- extra="ignore" to prevent crashes from unknown variables

## Setup Steps

### Prerequisites

1. Python 3.11+ with venv
2. Docker and Docker Compose
3. Ollama installed locally

### Installation

```mermaid
flowchart TD
    A["1. Clone Repository"] --> B["2. Create venv<br/>python -m venv venv"]
    B --> C["3. Install Dependencies<br/>pip install -r requirements.txt"]
    C --> D["4. Start Docker Services"]
    D --> E["5. Pull Ollama Models"]
    E --> F["6. Configure .env"]
    F --> G["7. Launch Streamlit<br/>streamlit run streamlit_app.py"]
    
    D --> D1["docker run redis/redis-stack"]
    D --> D2["docker run endee-server"]
    
    E --> E1["ollama pull nomic-embed-text"]
    E --> E2["ollama pull gpt-oss:120b-cloud"]
```

### Docker Services

| Service | Image | Ports |
|---|---|---|
| Redis Stack | redis/redis-stack:latest | 6379, 8001 (RedisInsight) |
| Endee Server | endee-server:latest | 8080 |

## Offline Deployment

This system is designed to run 100% offline:

- **LLM:** Ollama runs locally with downloaded model weights
- **Embeddings:** nomic-embed-text runs locally via Ollama
- **Sparse Model:** endee/bm25 is a local model file
- **Reranker:** CrossEncoder is downloaded once and saved to models/reranker/ for offline reuse
- **Vector DB:** Endee runs as a local Docker container
- **Cache/History:** Redis runs as a local Docker container
- **No external API calls** are made during operation

## Documentation Index

| # | Document | Description |
|---|---|---|
| 01 | endee_vector_database.md | Endee DB architecture, HNSW config, hybrid queries |
| 02 | ingestion_pipeline.md | PDF loading, chunking, embedding, batch upsert |
| 03 | retrieval_pipeline.md | Hybrid search, CrossEncoder reranking, metrics |
| 04 | semantic_cache.md | Redis vector cache, multi-tenant isolation, TTL |
| 05 | history_aware_retriever.md | Query rewriting, dual-LLM strategy, LangChain chain |
| 06 | chat_history_management.md | Redis persistence, user registry, session lifecycle |
| 07 | system_management.md | Admin dashboard, user onboarding, cache flushing |
| 08 | question_suggestions.md | Follow-up generation, pre-retrieval, bubble UI |
| 09 | entire_setup.md | This document - complete system overview |
