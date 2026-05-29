# LLM Metrics Collection & Dashboard — Implementation Plan

## 1. Overview

Build a **Streamlit-based metrics dashboard** that tracks, persists, and visualizes every RAG pipeline operation. Currently the system collects per-query metrics (latency, tokens/s, retrieval breakdowns, cache hits) but only displays them ephemerally in the chat UI. This plan adds a **persistent metrics layer** with a dedicated dashboard page.

---

## 2. Metrics Surface

### 2.1 Traceable Events

Every `GenerationService.run_with_metrics()` call already produces:

| Metric | Source | Example |
|--------|--------|---------|
| `total_time` | generator.py:219 | 4.23s |
| `retrieval_time` | generator.py:220 | 0.87s |
| `hybrid_time` | retriever.py:146 | 0.31s |
| `rerank_time` | retriever.py:144 | 0.56s |
| `llm_time` | generator.py:221 | 3.15s |
| `tps` (tokens/s) | generator.py:222 | 38.4 |
| `token_count` | generator.py:223 | 121 |
| `cache_hit` (semantic) | generator.py:224 | False |
| `retriever_cache_hit` | retriever.py:78 | False |

### 2.2 New Metrics to Inject

| Metric | Where to Capture | Why |
|--------|-----------------|-----|
| `embedding_latency` | `EmbeddingService.get_dense_embedding()` | Track embedding speed |
| `sparse_latency` | `EmbeddingService.get_sparse_embedding()` | Track BM25 overhead |
| `cache_search_time` | `RedisSemanticCache.search()` | Track cache overhead |
| `cache_store_time` | `RedisSemanticCache.store()` | Track write latency |
| `retriever_cache_search_time` | `RetrieverSemanticCache.search()` | Track retriever cache overhead |
| `query_length` | Generator entry | Track prompt size trends |
| `context_chunks` | Generator retrieval phase | Track context window size |
| `context_char_count` | Generator `_format_context()` | Track context token usage |
| `user_id` / `session_id` | Generator entry | Per-user/ per-session breakdown |
| `doc_version` | Generator entry | Track version drift |
| `timestamp` | Generator entry | Time-series axis |

---

## 3. Storage Layer

### 3.1 Option — Redis Timeseries (Recommended)

Use **RedisTimeSeries** module (available in `redis/redis-stack`) to store metrics natively.

**Data model per metric:**
```
TS.CREATE metrics:llm_latency LABELS type llm user tamil
TS.CREATE metrics:retrieval_time LABELS type retrieval user tamil
TS.CREATE metrics:tokens_per_sec LABELS type throughput user tamil
```

**Key schema:**
```
metrics:{metric_name}:{tenant_id}
```

**Redis commands:**
```python
# Write (one per query)
redis_client.ts().add(
    f"metrics:llm_time:{tenant_id}",
    timestamp=int(time.time() * 1000),  # milliseconds
    value=llm_duration,
    labels={"type": "llm", "user": tenant_id}
)

# Read (for dashboard)
redis_client.ts().range(
    f"metrics:llm_time:{tenant_id}",
    from_time=start_ts,
    to_time=end_ts,
    aggregation_type="avg",
    bucket_size_msec=60000  # 1-minute buckets
)
```

### 3.2 Fallback — SQLite (via sqlite3)

If RedisTimeSeries is unavailable, fall back to SQLite:

**Schema (`metrics/metrics.db`):**
```sql
CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT,
    doc_version TEXT DEFAULT '1.0',
    query_length INTEGER DEFAULT 0,
    context_chunks INTEGER DEFAULT 0,
    context_chars INTEGER DEFAULT 0,
    embedding_latency REAL DEFAULT 0.0,
    sparse_latency REAL DEFAULT 0.0,
    retrieval_time REAL DEFAULT 0.0,
    hybrid_time REAL DEFAULT 0.0,
    rerank_time REAL DEFAULT 0.0,
    llm_time REAL DEFAULT 0.0,
    total_time REAL DEFAULT 0.0,
    token_count INTEGER DEFAULT 0,
    tps REAL DEFAULT 0.0,
    cache_hit INTEGER DEFAULT 0,
    retriever_cache_hit INTEGER DEFAULT 0,
    cache_search_time REAL DEFAULT 0.0,
    cache_store_time REAL DEFAULT 0.0
);
```

---

## 4. Collection Layer — `metrics/collector.py`

New module responsible for gathering and storing metrics.

```
metrics/
  ├── __init__.py
  ├── collector.py      # Core collector class
  ├── storage.py        # Redis TS + SQLite abstraction
  └── dashboard.py      # Streamlit dashboard pages
```

### 4.1 `MetricsCollector` API

```python
class MetricsCollector:
    def record_query(self, metrics: dict) -> None
    def record_embedding(self, user_id: str, latency: float) -> None
    def record_cache_operation(self, cache_type: str, hit: bool, latency: float) -> None
    def get_user_stats(self, user_id: str, since: int) -> dict
    def get_global_stats(self, since: int) -> dict
    def get_time_series(self, metric: str, user_id: str, since: int) -> list
```

### 4.2 Integration Points

**`GenerationService.run_with_metrics()`** — Add at end (before return):

```python
self.metrics_collector.record_query({
    "timestamp": int(time.time() * 1000),
    "user_id": tenant_id,
    "session_id": session_id,
    "doc_version": doc_version,
    "query_length": len(query),
    "context_chunks": len(docs) if docs else 0,
    "context_chars": len(context_text) if context_text else 0,
    "embedding_latency": getattr(self.base_retriever, "last_embedding_time", 0.0),
    "retrieval_time": retrieval_time,
    "hybrid_time": getattr(self.base_retriever, "last_hybrid_time", 0.0),
    "rerank_time": getattr(self.base_retriever, "last_rerank_time", 0.0),
    "llm_time": llm_duration,
    "total_time": total_duration + retrieval_time,
    "token_count": token_count,
    "tps": tps,
    "cache_hit": cached_hit.get("hit", False) if cached_hit else False,
    "retriever_cache_hit": getattr(self.base_retriever, "last_cache_hit", False),
})
```

**`HybridEndeeRetriever._get_relevant_documents()`** — Track embedding times:

```python
t0 = time.perf_counter()
q_dense = self.embedding_service.get_dense_embedding(query)
self.last_embedding_time = time.perf_counter() - t0
```

**`RedisSemanticCache.search()`** — Already tracks `search_time`; expose it:

```python
# Already implemented (semantic_cache.py:154)
start_time = time.perf_counter()
results = self.client.ft(self.index_name).search(q, query_params=params)
search_time = time.perf_counter() - start_time
```

---

## 5. Dashboard — Streamlit Pages

### 5.1 Page Structure

```
Dashboard
├── 📊 Overview          — Global KPIs, live throughput gauge
├── 👤 Per-User Stats    — Select user → latency/usage/cache charts
├── ⏱️ Latency Explorer  — Drill-down: total → retrieval → llm → rerank
├── 💾 Cache Analytics   — Hit rates, store times, eviction trends
└── 📋 Query Log         — Raw table with search/filter/sort
```

### 5.2 Page: 📊 Overview

```
┌─────────────────────────────────────┐
│  📊 LLM Metrics Dashboard — Overview │
├─────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│ │ Avg  │ │Total │ │ P95  │ │Cache │ │
│ │Latency│ │Queries│ │Latency│ │Rate  │ │
│ │ 2.3s │ │ 1,247│ │ 6.1s │ │ 72%  │ │
│ └──────┘ └──────┘ └──────┘ └──────┘ │
│                                       │
│ ┌─────────────────────────────────┐   │
│ │ Latency Time-Series (24h)       │   │
│ │   ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁              │   │
│ │   └─────────────┬──────────────┘   │
│ │             per-minute avg         │
│ └─────────────────────────────────┘   │
│                                       │
│ ┌─────────┐ ┌─────────────────────┐   │
│ │Avg Breakdown      │ Token Throughput  │
│ │ Retrieval: 0.8s   │ ████████▁▁▁▁ 42/s │
│ │ LLM:      1.5s   │                   │
│ │ Total:    2.3s   │                   │
│ └─────────┘ └─────────────────────┘   │
└─────────────────────────────────────┘
```

**Code sketch:**
```python
def overview_page(collector):
    since = time_selector()  # 1h, 6h, 24h, 7d
    stats = collector.get_global_stats(since)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Latency", f"{stats['avg_latency']:.1f}s")
    col2.metric("Total Queries", stats["total_queries"])
    col3.metric("P95 Latency", f"{stats['p95_latency']:.1f}s")
    col4.metric("Cache Rate", f"{stats['cache_rate']:.0f}%")

    # Time-series chart
    ts = collector.get_time_series("total_time", since=since)
    st.line_chart(pd.DataFrame(ts, columns=["time", "latency"]).set_index("time"))
```

### 5.3 Page: 👤 Per-User Stats

```
┌─────────────────────────────────────┐
│  👤 User: tamil                     │
├─────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│ │ Avg  │ │Peak  │ │Total │ │Query │ │
│ │Latency│ │Latency│ │Tokens│ │Count │ │
│ │ 1.8s │ │ 4.2s │ │15.2K│ │ 342  │ │
│ └──────┘ └──────┘ └──────┘ └──────┘ │
│                                       │
│ User Selector: [tamil ▼]              │
│ Time Range: [24h ▼]                   │
│                                       │
│ ┌─────────────────────────────────┐   │
│ │ Per-Session Breakdown            │   │
│ │ ┌─────────┬──────┬──────┬──────┐│   │
│ │ │Session  │Qrys  │Avg L │Cache%││   │
│ │ │general  │ 124  │1.2s  │ 85%  ││   │
│ │ │Chat 5.. │  87  │2.4s  │ 60%  ││   │
│ │ │hydrauli │  42  │3.1s  │ 45%  ││   │
│ │ └─────────┴──────┴──────┴──────┘│   │
│ └─────────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 5.4 Page: ⏱️ Latency Explorer

```
┌─────────────────────────────────────┐
│  ⏱️ Latency Explorer                 │
├─────────────────────────────────────┤
│ Show: [Breakdown ▾]  Filter: [______]│
│                                       │
│ Stacked Bar Chart (per-query)         │
│ ┌─────────────────────────────────┐   │
│ │ ████ Embedding                  │   │
│ │ ██████████ Hybrid Search        │   │
│ │ ██████ Rerank                   │   │
│ │ ████████████████████ LLM Gen    │   │
│ └─────────────────────────────────┘   │
│                                       │
│ Detailed Table:                       │
│ ┌────┬───────┬──────┬──────┬──────┐   │
│ │ #  │Total │Retr.│LLM  │TPS   │   │
│ │ 1  │ 3.2s │ 0.9 │ 2.1 │ 45.2 │   │
│ │ 2  │ 2.1s │ 0.7 │ 1.3 │ 38.4 │   │
│ │ ...│       │      │      │      │   │
│ └────┴───────┴──────┴──────┴──────┘   │
└─────────────────────────────────────┘
```

### 5.5 Page: 💾 Cache Analytics

```
┌─────────────────────────────────────┐
│  💾 Cache Analytics                  │
├─────────────────────────────────────┤
│ ┌────────────────┐ ┌────────────────┐│
│ │ Semantic Cache     │ Retriever Cache    │
│ │ Hit Rate: 68%     │ Hit Rate: 82%      │
│ │ Avg search: 2ms   │ Avg search: 1.5ms  │
│ │ Entries: 342      │ Entries: 521       │
│ │ TTL: 3600s        │ TTL: 3600s         │
│ └────────────────┘ └────────────────┘│
│                                       │
│ Hit Rate Over Time:                   │
│   ████████░░░░ 68% (24h avg)          │
│                                       │
│ ┌─────────────────────────────────┐   │
│ │ Cache Miss Top Queries           │   │
│ │ "hydraulic pump pressure" × 12   │   │
│ │ "spindle calibration"      ×  8   │   │
│ │ "coolant temperature"      ×  5   │   │
│ └─────────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 5.6 Page: 📋 Query Log

```
┌─────────────────────────────────────┐
│  📋 Query Log                        │
├─────────────────────────────────────┤
│ Search: [_____________________]      │
│ User: [All ▼]  Status: [All ▼]       │
│                                       │
│ ┌────┬──────┬────────┬────┬────┬────┐│
│ │Time│User  │Query   │Lat.│TPS │Cache││
│ ├────┼──────┼────────┼────┼────┼────┤│
│ │14: │tamil │how do..│2.1 │42  │✅  ││
│ │14: │admin │what is.│4.3 │18  │❌  ││
│ │... │      │        │    │    │    ││
│ └────┴──────┴────────┴────┴────┴────┘│
│                                       │
│ 📥 Export CSV     📊 Download Report  │
└─────────────────────────────────────┘
```

---

## 6. Integration into Streamlit App

### 6.1 Navigation

Add a third page option alongside the existing two:

**`streamlit_app.py:255`** — change:
```python
page = st.sidebar.radio("Navigate", [
    "🤖 AI Chat",
    "⚙️ System Management",
    "📊 LLM Metrics"
], index=0)
```

### 6.2 Page Router

**`streamlit_app.py:491`** — add a new route:

```python
elif page == "📊 LLM Metrics":
    from metrics.dashboard import render_dashboard
    render_dashboard(collector)
```

### 6.3 Initialize Collector in `initialize_services()`

**`streamlit_app.py:206`**:
```python
from metrics.collector import MetricsCollector
st.session_state.metrics_collector = MetricsCollector()
```

### 6.4 Wire into Generator

- Add `metrics_collector` parameter to `GenerationService.__init__()`
- Call `self.metrics_collector.record_query(...)` inside `run_with_metrics()`
- Pass collector to generator at init in `streamlit_app.py:214`

---

## 7. Files to Create

| File | Purpose |
|------|---------|
| `metrics/__init__.py` | Package init |
| `metrics/collector.py` | `MetricsCollector` class with `record_query()`, `get_user_stats()`, `get_global_stats()`, `get_time_series()` |
| `metrics/storage.py` | `MetricsStorage` abstract base + `RedisTimeseriesStorage` + `SqliteStorage` implementations |
| `metrics/dashboard.py` | `render_dashboard()` — all 5 Streamlit pages |
| `requirements.txt` add | `redistimeseries` (if using Redis TS) or nothing (sqlite3 stdlib) |

## 8. Files to Modify

| File | Changes |
|------|---------|
| `core/generator.py` | Inject `MetricsCollector`, call `record_query()` in `run_with_metrics()` |
| `core/retriever.py` | Track `last_embedding_time`, `last_sparse_time` |
| `core/semantic_cache.py` | Expose `search_time` and `store_time` on the instance |
| `streamlit_app.py` | Add "📊 LLM Metrics" page route, init collector, pass to generator |

---

## 9. Implementation Order

```
Week 1 — Storage & Collection
├── metrics/__init__.py + metrics/storage.py (Redis TS + SQLite)
├── metrics/collector.py (record_query, get_*, singleton)
├── Wire collector into GenerationService.run_with_metrics()
└── Wire collector into HybridEndeeRetriever (embedding timing)

Week 2 — Dashboard Pages
├── metrics/dashboard.py — 5 pages (Overview, Per-User, Latency, Cache, Query Log)
├── Streamlit page router integration
├── Cache analytics page (top cache-miss queries)
└── Query log page with search/filter/export

Week 3 — Polish & Alerts
├── P95/P99 latency gauges
├── Per-user trend comparison charts
├── CSV/JSON export on Query Log page
└── Optional: Slack/email alert when latency > threshold
```

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage primary | RedisTimeSeries | Already runs Redis Stack; no new infra; time-series native |
| Storage fallback | SQLite | Zero deps; file-based; survives restarts |
| Collection scope | Every `run_with_metrics()` call | No sampling — full observability |
| Tenant isolation | `:tenant_id` key suffix | Matches existing cache key pattern |
| Time-series granularity | Per-query | Aggregation happens at query time via dashboard |
| Extensibility | `MetricsStorage` ABC | Swap backends without changing collector |

---

## 11. Dashboard Dependencies

```python
# metrics/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px      # or altair (bundled with Streamlit)
import plotly.graph_objects as go
from datetime import datetime, timedelta
from metrics.collector import MetricsCollector
```

**Plotly** is recommended over raw `st.line_chart` for:
- Stacked bar charts (latency breakdown)
- Hover tooltips with exact values
- Range sliders for time-series zoom
