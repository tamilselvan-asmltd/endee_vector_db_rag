# Chat History Management

## Overview

The Chat History Management system provides **persistent, multi-user, multi-session** conversation memory using Redis. It ensures that every engineer's conversation threads are stored, retrievable, and isolated from other users, enabling context-aware responses across multiple chat sessions.

## Architecture

```mermaid
flowchart TD
    A["Streamlit UI"] --> B["ChatHistoryManager"]
    B --> C["Redis Server"]
    C --> D["users:registry<br/>Global User Set"]
    C --> E["chat:user_id:conv_id<br/>Session History Lists"]
    
    B --> F["RedisChatMessageHistory<br/>LangChain Abstraction"]
    F --> C
    
    style B fill:#4f46e5,color:#fff
    style C fill:#dc2626,color:#fff
```

## Redis Key Structure

```mermaid
graph TD
    A["Redis Keyspace"] --> B["users:registry<br/>Type: SET<br/>Members: tamil, selvan, admin"]
    A --> C["chat:tamil:general<br/>Type: LIST<br/>Messages: HumanMsg, AIMsg..."]
    A --> D["chat:tamil:Chat 2026-05-14 14:22:30<br/>Type: LIST<br/>Messages: HumanMsg, AIMsg..."]
    A --> E["chat:selvan:general<br/>Type: LIST"]
    
    style B fill:#f59e0b,color:#000
    style C fill:#059669,color:#fff
    style D fill:#059669,color:#fff
```

## Core Operations

### User Lifecycle

```mermaid
flowchart TD
    A["System Management Page"] --> B["register_user(user_id)"]
    B --> C["SADD users:registry user_id"]
    
    D["list_all_users()"] --> E["SMEMBERS users:registry"]
    E --> F["SCAN chat:* keys<br/>Backup Discovery"]
    F --> G["Union + Sort"]
    
    H["delete_user(user_id)"] --> I["SREM users:registry user_id"]
    I --> J["DEL chat:user_id:*"]
    
    style B fill:#059669,color:#fff
    style H fill:#dc2626,color:#fff
```

### Session Lifecycle

```mermaid
sequenceDiagram
    participant UI as Streamlit
    participant HM as ChatHistoryManager
    participant Redis as Redis

    Note over UI: User clicks New Chat
    UI->>UI: Generate timestamp title<br/>"Chat 2026-05-14 14:22:30"
    UI->>UI: Set session_state.conv_id
    
    Note over UI: User sends first message
    UI->>HM: add_user_message(session_id, msg)
    HM->>Redis: RPUSH chat:tamil:Chat_2026... msg
    
    UI->>HM: add_ai_message(session_id, response)
    HM->>Redis: RPUSH chat:tamil:Chat_2026... response
    
    Note over UI: User clicks Rename
    UI->>HM: rename_session(user, old, new)
    HM->>Redis: RENAME old_key new_key
    
    Note over UI: User clicks Delete
    UI->>HM: clear_history(session_id)
    HM->>Redis: DEL chat:tamil:session
```

### History Trimming

```mermaid
flowchart TD
    A["New Message Added"] --> B["_trim_history()"]
    B --> C{Messages > max_messages?}
    C -->|No| D["No Action"]
    C -->|Yes| E["Get All Messages"]
    E --> F["Slice Last N Messages"]
    F --> G["Clear Session"]
    G --> H["Re-add Last N Messages"]
    
    style C fill:#f59e0b,color:#000
```

**Max Messages:** 10 (configurable via `MAX_HISTORY_MESSAGES` in `.env`)

## Session ID Format

The session ID is constructed as: `{user_id}:{conversation_id}`

Examples:
- `tamil:general` — Default session for user "tamil"
- `tamil:Chat 2026-05-14 14:22:30` — Timestamped auto-generated session
- `selvan:hydraulic_pump_fix` — User-renamed session

## Colon Handling

Since Redis uses colons as separators and chat titles contain colons (timestamps), the system uses `split(":", 2)` (maxsplit=2) to correctly parse keys:

| Key | Prefix | User | Conv ID |
|---|---|---|---|
| `chat:tamil:general` | chat | tamil | general |
| `chat:tamil:Chat 2026-05-14 14:22:30` | chat | tamil | Chat 2026-05-14 14:22:30 |

## API Reference

### File: core/history_manager.py

| Method | Description |
|---|---|
| `get_history(session_id)` | Returns all messages for a session |
| `add_user_message(session_id, msg)` | Appends a human message |
| `add_ai_message(session_id, msg)` | Appends an AI message |
| `get_recent_messages(session_id, limit)` | Returns the last N messages |
| `clear_history(session_id)` | Deletes all history for a session |
| `clear_all_histories()` | Wipes ALL chat keys globally |
| `register_user(user_id)` | Adds user to persistent registry |
| `list_all_users()` | Returns sorted list of all known users |
| `list_user_sessions(user_id)` | Returns sorted list of session IDs for a user |
| `rename_session(user_id, old, new)` | Renames a Redis key |
| `delete_user(user_id)` | Purges user from registry + all data |

## Configuration

| Parameter | Value | Source |
|---|---|---|
| `REDIS_HOST` | localhost | .env |
| `REDIS_PORT` | 6379 | .env |
| `REDIS_PASSWORD` | (empty) | .env |
| `REDIS_CHAT_HISTORY_PREFIX` | chat: | .env |
| `MAX_HISTORY_MESSAGES` | 10 | .env |

## Data Safety

- **Defensive Deletion:** Before calling `redis.delete(*keys)`, the system always checks `if keys:` to prevent the "wrong number of arguments for DEL" error
- **Persistent Registry:** Users survive even if all their chat keys are deleted, because the registry is a separate SET
- **Cross-Session Isolation:** Each session is a separate Redis list, preventing any data mixing
