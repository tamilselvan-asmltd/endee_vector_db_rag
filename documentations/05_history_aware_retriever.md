# History-Aware Retriever

## Overview

The History-Aware Retriever solves a fundamental problem in multi-turn RAG systems: **follow-up questions lack context**. When a user asks "What about the oil filter?", a standard retriever doesn't know what "that" refers to. This component uses the LLM to **rewrite ambiguous queries** into standalone questions before retrieval.

## Problem Statement

```mermaid
flowchart LR
    A["User: How does the hydraulic system work?"] --> B["AI: The hydraulic system uses..."]
    B --> C["User: What about the oil filter?"]
    C --> D{"Standard Retriever"}
    D --> E["Searches for 'oil filter' only<br/>Misses hydraulic context"]
    C --> F{"History-Aware Retriever"}
    F --> G["Rewrites to:<br/>'Role of the oil filter<br/>in the hydraulic system'"]
```

## Architecture

```mermaid
flowchart TD
    A["User Query"] --> B["ChatOllama<br/>Query Rewriter"]
    C["Chat History<br/>Last 5 Turns"] --> B
    B --> D["Reformulated Standalone Query"]
    D --> E["HybridEndeeRetriever"]
    E --> F["Relevant Documents"]

    style B fill:#4f46e5,color:#fff
    style E fill:#059669,color:#fff
```

## Implementation Flow

```mermaid
sequenceDiagram
    participant User
    participant Gen as GenerationService
    participant Rewriter as ChatOllama
    participant Retriever as HybridEndeeRetriever
    participant History as ChatHistoryManager

    User->>Gen: Follow-up question
    Gen->>History: get_history(session_id)
    History-->>Gen: Last 5 turns

    alt Has Chat History
        Gen->>Rewriter: Rewrite with context
        Rewriter-->>Gen: Standalone question
        Gen->>Retriever: invoke(reformulated)
    else No History
        Gen->>Retriever: invoke(original)
    end

    Retriever-->>Gen: Relevant Documents
```

## Query Rewriting Prompt

The rewriter uses a dedicated ChatOllama instance with temperature=0 for deterministic reformulation:

> **System Prompt:** Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

## Dual-LLM Strategy

| LLM Instance | Model | Temperature | Purpose |
|---|---|---|---|
| `self.llm` (OllamaLLM) | gpt-oss:120b-cloud | 0.0 | Main answer generation with streaming |
| `self.chat_llm` (ChatOllama) | gpt-oss:120b-cloud | 0 | Query rewriting and suggestion generation |

**Why Two Instances?**
- `OllamaLLM` supports streaming for real-time token display in the UI
- `ChatOllama` handles structured message history (system/human/ai) required by LangChain's `create_history_aware_retriever` chain

## LangChain Integration

The retriever is built using LangChain's `create_history_aware_retriever`:

- Takes `chat_llm`, `base_retriever`, and a `ChatPromptTemplate` with `MessagesPlaceholder`
- Accepts `{"input": query, "chat_history": messages}` as input
- Internally rewrites the query using the LLM, then passes the standalone question to the retriever
- Returns a list of LangChain `Document` objects

## Decision Flow in Generator

```mermaid
flowchart TD
    A["User Query Arrives"] --> B{Chat History Exists?}
    B -->|Yes| C["history_aware_retriever.invoke()<br/>Rewrites + Retrieves"]
    B -->|No| D["base_retriever.invoke()<br/>Direct Hybrid Search"]
    C --> E["Context Documents"]
    D --> E
    E --> F["Format Context + History"]
    F --> G["LLM Generates Answer"]

    style C fill:#4f46e5,color:#fff
    style D fill:#059669,color:#fff
```

## History Window Configuration

| Parameter | Value | Purpose |
|---|---|---|
| `history_window_size` | 5 | Number of conversation turns passed to the LLM prompt |
| `max_history_messages` | 10 | Maximum messages stored in Redis per session |

The history window ensures the rewriter has enough context to understand references, while the storage limit keeps Redis memory usage bounded.

## File Reference

**Implementation:** `core/generator.py` (lines 35-54)

| Component | Description |
|---|---|
| `contextualize_q_system_prompt` | System instruction for query rewriting |
| `contextualize_q_prompt` | ChatPromptTemplate with MessagesPlaceholder |
| `history_aware_retriever` | LangChain chain combining rewriter + retriever |
