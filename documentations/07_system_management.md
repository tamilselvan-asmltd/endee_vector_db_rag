# System Management Dashboard

## Overview

The System Management page provides a centralized administrative interface for managing users, chat histories, and performance caches. It follows a card-based UI design with color-coded user profiles and granular controls for data lifecycle management.

## Dashboard Layout

```mermaid
flowchart TD
    A["System Management Page"] --> B["Onboard New User Card"]
    A --> C["User Directory Cards"]
    A --> D["Engine Optimization Cards"]
    
    B --> B1["Text Input + Validation"]
    B --> B2["Create and Login Button"]
    
    C --> C1["Per-User Profile Card"]
    C1 --> C2["Recent Conversations List"]
    C1 --> C3["Delete Individual Sessions"]
    C1 --> C4["Nuclear Purge - Delete User"]
    
    D --> D1["Flush Semantic Cache"]
    D --> D2["Flush Embedding Cache"]
    
    style A fill:#4f46e5,color:#fff
    style B fill:#059669,color:#fff
    style D fill:#dc2626,color:#fff
```

## User Onboarding Flow

```mermaid
sequenceDiagram
    participant Admin as Admin User
    participant UI as System Management Page
    participant HM as ChatHistoryManager
    participant Redis as Redis

    Admin->>UI: Enter new User ID
    UI->>UI: Validate (strip, non-empty)
    
    alt Valid ID
        UI->>HM: register_user(clean_id)
        HM->>Redis: SADD users:registry clean_id
        UI->>UI: Set session_state.user_id
        UI->>UI: Set conv_id = "general"
        UI-->>Admin: Success toast + redirect
    else Empty or Whitespace
        UI-->>Admin: Error - Please enter a valid User ID
    end
```

## User Directory

Each registered user gets a profile card with:

```mermaid
flowchart LR
    A["User Card"] --> B["Color-Coded Header<br/>HSL Theme via get_user_theme()"]
    A --> C["Recent Conversations<br/>List with delete icons"]
    A --> D["Security Section<br/>Nuclear Purge Button"]
```

### HSL Theme Engine

Each user gets a unique color based on their username hash:

| Username | Hue | Primary Color |
|---|---|---|
| tamil | 127 | hsl(127, 70%, 45%) |
| selvan | 243 | hsl(243, 70%, 45%) |
| admin | 89 | hsl(89, 70%, 45%) |

This ensures visual distinction between user cards without manual color assignment.

## Administrative Actions

### Per-Session Actions

```mermaid
flowchart TD
    A["Session Entry in Card"] --> B["Delete Button"]
    B --> C["clear_history(user:session)"]
    C --> D["Redis DEL chat:user:session"]
    D --> E["Toast Notification"]
    E --> F["Rerun UI"]
```

### Per-User Nuclear Purge

```mermaid
flowchart TD
    A["Delete User Button"] --> B["delete_user(user_id)"]
    B --> C["SREM users:registry user_id"]
    B --> D["DEL chat:user_id:* keys"]
    A --> E["clear_tenant(user_id)"]
    E --> F["DEL sem_cache:user_id:* keys"]
    F --> G["Success Message"]
    G --> H["Rerun UI"]
    
    style A fill:#dc2626,color:#fff
```

### Global Cache Optimization

```mermaid
flowchart LR
    A["Flush Semantic Cache"] --> B["semantic_cache.clear_all()"]
    B --> C["SCAN + DEL sem_cache:*"]
    
    D["Flush Embedding Cache"] --> E["embedding_service.clear_all()"]
    E --> F["SCAN + DEL embed_cache:*"]
```

## Sidebar vs Management Page

| Feature | Sidebar | System Management |
|---|---|---|
| Create New User | No | Yes |
| Switch User | Yes (dropdown) | No |
| Create New Chat | Yes (plus icon) | No |
| Rename Chat | Yes | No |
| Delete Chat | Yes | Yes (per-user cards) |
| Delete User | No | Yes (Nuclear Purge) |
| Flush Caches | No | Yes |

## Access Control Design

- Only existing registered users appear in the sidebar dropdown
- New users must be created through the System Management page
- This prevents accidental user creation and ensures administrative oversight
- Empty or whitespace-only User IDs are rejected with validation

## Configuration

All management operations are powered by the ChatHistoryManager class in `core/history_manager.py` and the RedisSemanticCache class in `core/semantic_cache.py`.
