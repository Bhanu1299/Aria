# Phase 5D — Memory Upgrade Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 5A (core engine)  
**Unlocks:** Semantic long-term memory, context-aware responses across sessions

---

## Goal

Aria stops forgetting. Replace key-value SQLite fact storage with vector embeddings so Aria can search semantically across everything she's ever learned. Inject relevant memories into every LLM call automatically.

---

## What changes and what stays

| Component | Change |
|---|---|
| `memory.py` | `store_fact()` also embeds + upserts into ChromaDB |
| `memory_extractor.py` | No change — still extracts facts, memory.py handles storage |
| `auto_dream.py` | New step: score + promote top facts to `identity.json` |
| `compact.py` | Smarter trigger: fire when context hits 80% of model window |
| `db.py` | No change — SQLite stays as write-ahead log |

---

## Architecture

### Plugin structure

```
plugins/
  memory/
    __init__.py          ← MemoryPlugin(PluginBase)
    vector_store.py      ← ChromaStore
    embedder.py          ← Embedder (sentence-transformers)
    context_injector.py  ← builds system prompt addition
```

---

## Embedding model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Size:** 22MB  
**Dimensions:** 384  
**Load time:** ~1s on Apple Silicon  
**Loaded:** once at startup alongside Whisper, shared across all components

```python
# embedder.py
class Embedder:
    _instance = None

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()
```

---

## Vector store

**Library:** `chromadb` — pure Python, persistent, no Rust compile step  
**Location:** `~/.aria/memory/vectors/`  
**Collection:** `aria_facts`

### Fact schema

Each document in ChromaDB:
```
id:           str          # UUID, same as SQLite fact ID
document:     str          # the fact text
embedding:    list[float]  # 384-dim vector
metadata:
  source:     str          # "conversation" | "cron" | "email" | "dream"
  session_ids: str         # JSON-encoded list of session IDs where fact appeared
  timestamp:  float        # unix timestamp of first extraction
  recall_count: int        # incremented on each retrieval
```

`session_ids` is stored as a JSON string (ChromaDB metadata values must be scalar). On each new extraction of the same fact, the current session ID is appended if not already present. `unique_session_count = len(json.loads(session_ids))`.

### ChromaStore interface

```python
class ChromaStore:
    def upsert(self, fact_id: str, text: str, metadata: dict)
    def search(self, query: str, k: int = 5, min_score: float = 0.6) -> list[dict]
    def get(self, fact_id: str) -> dict | None
    def delete(self, fact_id: str)
    def increment_recall(self, fact_id: str)
    def count(self) -> int
```

`search()` returns facts with cosine similarity ≥ `min_score`, sorted by score descending. `min_score=0.6` filters noise — facts below this threshold are not injected.

---

## Memory storage flow

`memory.py` `store_fact()` updated:

```
1. Save to SQLite (existing behavior, unchanged)
2. Embed fact text via Embedder.get().embed(text)
3. Upsert into ChromaStore with metadata
```

Embedding happens synchronously inside the existing async lock — adds ~5ms per fact, imperceptible in practice.

---

## Context injection

`context_injector.py` runs before every `agent.run()` call:

```
1. Take last user message as query
2. Search ChromaStore: top 5 facts with score ≥ 0.6
3. Increment recall_count for each retrieved fact
4. Format as system prompt addition:

   "Relevant things I know about you:
   - [fact 1]
   - [fact 2]
   - [fact 3]"
```

If no facts score above threshold, nothing is injected — no hallucinated context.

Token budget: injected memories capped at 300 tokens. If top 5 facts exceed 300 tokens, truncate to top 3.

---

## Short-term promotion (auto_dream upgrade)

`auto_dream.py` gets a new consolidation step after its existing Groq-based summary:

```
1. Query ChromaStore for all facts with recall_count >= 3
   AND unique_session_count >= 2 (appeared in multiple sessions)
2. Score each: recall_count * 0.4 + recency_score * 0.3 + session_spread * 0.3
3. Top 5 scoring facts → upsert into identity.json under "learned_facts"
4. These become permanent — survive memory wipes
```

`identity.json` gains a new field:
```json
{
  "learned_facts": [
    {"fact": "User is targeting ML engineering roles", "promoted_at": "2026-05-06"},
    {"fact": "User prefers morning focus sessions", "promoted_at": "2026-05-06"}
  ]
}
```

Learned facts are always injected into the system prompt regardless of semantic search results.

---

## Context compaction upgrade

`compact.py` currently fires when session notes exceed 3000 chars. New trigger:

```
1. After every agent turn, estimate token count of full message history
2. If > 80% of model context window (Claude Sonnet = 200k, Groq Llama = 128k):
   a. Keep last 10 turns verbatim
   b. Summarize older turns into one "Prior context" message via Groq
   c. Replace old turns with summary
3. Log: "Compacted {n} turns into summary ({tokens_before} → {tokens_after} tokens)"
```

Token estimation: `len(text) / 4` (rough but fast — no tokenizer needed).

---

## Migration

On first run after upgrade, `MemoryPlugin` runs a one-time migration:

```
1. Read all existing facts from SQLite
2. Embed each in batches of 50
3. Upsert into ChromaDB
4. Write migration marker: ~/.aria/memory/.migrated
```

Migration runs in background thread — Aria is usable immediately, migration completes silently. ~1-2 minutes for typical fact count.

---

## Error handling

| Error | Behavior |
|---|---|
| ChromaDB write fails | Log warning, continue — SQLite is source of truth |
| Embedding model not loaded | Fall back to no memory injection, log error |
| Context injection timeout (>200ms) | Skip injection for this turn, log warning |
| Migration fails | Log error, mark migration incomplete, retry next startup |

---

## Testing

- `tests/test_vector_store.py` — upsert, search with scoring, recall increment, delete
- `tests/test_embedder.py` — model loads, embed returns correct shape
- `tests/test_context_injector.py` — injection with facts, empty result when none qualify, token cap
- `tests/test_memory_migration.py` — SQLite → ChromaDB migration, idempotent re-run
- `tests/test_auto_dream_promote.py` — scoring logic, identity.json update

---

## Dependencies

```
chromadb>=0.4.22
sentence-transformers>=2.6.1
```

---

## Definition of done

- [ ] `Embedder` singleton loaded at startup
- [ ] `ChromaStore` with upsert, search, recall increment
- [ ] `memory.py` `store_fact()` dual-writes to SQLite + Chroma
- [ ] `context_injector.py` injects top-5 relevant facts per turn
- [ ] `auto_dream.py` short-term promotion to `identity.json`
- [ ] `compact.py` token-budget-based compaction trigger
- [ ] One-time migration from SQLite to ChromaDB
- [ ] All existing memory tests still pass
- [ ] New tests for vector store, injector, promotion, migration
