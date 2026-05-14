# Question Suggestions After Response

## Overview

The Question Suggestion system provides an intelligent, context-aware follow-up experience. After every AI response, the system generates 3 strategic follow-up questions based on the retrieved document chunks and conversation history. Each suggestion comes with **pre-retrieved context chunks**, enabling near-instant responses when clicked.

## End-to-End Flow

```mermaid
flowchart TD
    A["User Asks Question"] --> B["Retrieval Pipeline"]
    B --> C["Context Chunks Retrieved"]
    C --> D["LLM Generates Answer"]
    D --> E["Answer Displayed"]
    E --> F["Suggestion Generator"]
    
    C --> F
    G["Chat History<br/>Last 5 Turns"] --> F
    D --> F
    
    F --> H["LLM Generates 3 Questions"]
    H --> I["Pre-Retrieve Chunks<br/>for Each Question"]
    I --> J["Display 3 Bubble Buttons"]
    
    J --> K{User Clicks Bubble?}
    K -->|Yes| L["Skip Retrieval Phase<br/>Use Pre-Retrieved Chunks"]
    L --> M["LLM Generates Answer<br/>Near-Instant"]
    K -->|No| N["User Types Own Question"]
    N --> B
    
    style F fill:#4f46e5,color:#fff
    style L fill:#059669,color:#fff
    style J fill:#f59e0b,color:#000
```

## Suggestion Generation Pipeline

```mermaid
sequenceDiagram
    participant Gen as GenerationService
    participant LLM as ChatOllama
    participant Ret as HybridEndeeRetriever

    Note over Gen: After main answer is generated
    Gen->>LLM: suggestion_prompt(context, history, last_answer)
    LLM-->>Gen: Raw text with 3 bullet points
    Gen->>Gen: Parse bullet points into list
    
    loop For each of 3 questions
        Gen->>Ret: invoke(suggestion_question)
        Ret-->>Gen: Pre-retrieved documents
        Gen->>Gen: Store {question, pre_docs}
    end
    
    Gen-->>Gen: Return list of 3 suggestion dicts
```

## Suggestion Prompt Design

The prompt is carefully engineered to generate questions that are:
1. **Answerable** from the existing technical documentation
2. **Relevant** to the current conversation thread
3. **Strategic** — helping users explore deeper into the topic

The prompt receives:
- **Context (Retrieved Chunks):** The same document snippets used to answer the current question
- **Conversation History:** Last 5 turns for topic continuity
- **AI's Last Response:** To avoid suggesting the same question that was just answered

## Pre-Retrieval Strategy

```mermaid
flowchart TD
    A["3 Suggested Questions"] --> B["Question 1"]
    A --> C["Question 2"]
    A --> D["Question 3"]
    
    B --> E["base_retriever.invoke(Q1)"]
    C --> F["base_retriever.invoke(Q2)"]
    D --> G["base_retriever.invoke(Q3)"]
    
    E --> H["Store in session_state<br/>suggestions list with pre_docs"]
    F --> H
    G --> H
    
    style H fill:#059669,color:#fff
```

**Why Pre-Retrieve?**
- Eliminates the retrieval phase latency (typically 0.3-0.8s) when the user clicks a suggestion
- The UI shows a lightning bolt indicator when pre-retrieved context is used
- If the user types their own question instead, normal retrieval proceeds as usual

## UI Interaction Flow

```mermaid
flowchart TD
    A["Page Loads / Reruns"] --> B{Suggestions Exist?}
    B -->|Yes| C["Render 3 Bubble Buttons<br/>with lightbulb icons"]
    B -->|No| D["Show Only Chat Input"]
    
    C --> E{User Clicks Bubble?}
    E -->|Yes| F["Set suggested_query<br/>Set suggested_docs"]
    F --> G["Rerun Page"]
    G --> H["Skip st.chat_input<br/>Use suggested_query"]
    H --> I["Skip Retrieval<br/>Use suggested_docs"]
    I --> J["LLM Generates Answer"]
    
    E -->|No| K["User Types in Chat Input"]
    K --> L["Normal RAG Pipeline"]
    
    style C fill:#f59e0b,color:#000
    style I fill:#059669,color:#fff
```

## Session Isolation

```mermaid
flowchart TD
    A["Session Change Detected"] --> B["Clear suggestions list"]
    A --> C["Clear suggested_docs"]
    A --> D["Clear suggested_query"]
    B --> E["No stale suggestions<br/>from other users or chats"]
    
    style A fill:#dc2626,color:#fff
```

Suggestions are automatically cleared when:
- The user switches to a different User ID
- The user switches to a different Chat Session
- A new chat is created via the plus icon

## Data Structure

Each suggestion is stored as a dictionary:

| Field | Type | Description |
|---|---|---|
| question | str | The suggested follow-up question text |
| pre_docs | List of Document | Pre-retrieved LangChain Document objects |

## Performance Comparison

| Scenario | Retrieval Time | Total Latency |
|---|---|---|
| Normal Question (typed) | 0.3 - 0.8s | 3 - 8s |
| Suggestion Click (pre-retrieved) | ~0.01s | 2 - 6s |
| Cache Hit | 0s | 0.005 - 0.015s |

## File References

| File | Component |
|---|---|
| core/generator.py | suggestion_prompt, generate_suggestions() |
| streamlit_app.py | Bubble button rendering, suggested_query handling |
