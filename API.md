# API Reference

Complete API documentation for Persistent AI Memory System.

## Quick Links
- **New to API?** → See [Quick Examples](#quick-examples) first
- **Setting up?** → See [INSTALL.md](INSTALL.md)
- **Need config?** → See [CONFIGURATION.md](CONFIGURATION.md)

---

## Core Classes

### AIMemorySystem

Main class for all memory operations. Manages databases, embeddings, and API.

#### Initialization

```python
from ai_memory_core import AIMemorySystem

# Create system (async)
system = await AIMemorySystem.create()

# Use system...

# Clean up when done
await system.close()
```

---

## Memory Operations

### store_memory

Store a persistent memory with optional metadata.

```python
memory_id = await system.store_memory(
    content: str,
    metadata: Dict[str, Any] = None,
    memory_bank: str = "memories",
    user_id: str = "default"
)
```

**Parameters:**
- `content` (str) - Memory text to store
- `metadata` (dict, optional) - Additional metadata (category, tags, source, etc.)
- `memory_bank` (str, optional) - Which memory bank to store in (default: "memories")
- `user_id` (str, optional) - User identifier for multi-user systems (default: "default")

**Returns:**
- `memory_id` (str) - Unique ID of stored memory

**Example:**
```python
memory_id = await system.store_memory(
    "Python async/await allows non-blocking I/O operations",
    metadata={
        "category": "learning",
        "topic": "python",
        "confidence": 0.95
    }
)
print(f"Stored memory: {memory_id}")
```

---

### search_memories

Find memories using semantic similarity search.

```python
results = await system.search_memories(
    query: str,
    limit: int = 10,
    user_id: str = "default",
    similarity_threshold: float = 0.0
)
```

**Parameters:**
- `query` (str) - Search query text
- `limit` (int, optional) - Maximum results to return (default: 10)
- `user_id` (str, optional) - User ID to search (default: "default")
- `similarity_threshold` (float, optional) - Minimum similarity score 0-1 (default: 0.0)

**Returns:**
- `results` (list) - List of matching memories with scores:
  ```python
  [
    {
      "id": "mem_123",
      "content": "Memory text...",
      "similarity": 0.95,
      "metadata": {...},
      "created_at": "2026-02-23T10:30:00"
    }
  ]
  ```

**Example:**
```python
results = await system.search_memories(
    "async programming in Python",
    limit=5,
    similarity_threshold=0.7
)

for result in results:
    print(f"Score: {result['similarity']:.2f} - {result['content'][:50]}")
```

---

### list_recent_memories

Get most recent memories without searching.

```python
memories = await system.list_recent_memories(
    limit: int = 10,
    user_id: str = "default"
)
```

**Parameters:**
- `limit` (int, optional) - Number of memories to retrieve (default: 10)
- `user_id` (str, optional) - User ID (default: "default")

**Returns:**
- `memories` (list) - List of recent memory objects

**Example:**
```python
recent = await system.list_recent_memories(limit=20)
for memory in recent:
    print(f"{memory['created_at']}: {memory['content'][:50]}...")
```

---

## Conversation Tracking

### store_conversation

Store a single conversation turn (message from user or assistant).

```python
turn_id = await system.store_conversation(
    role: str,
    content: str,
    metadata: Dict[str, Any] = None,
    user_id: str = "default"
)
```

**Parameters:**
- `role` (str) - "user", "assistant", "system", etc.
- `content` (str) - Message text
- `metadata` (dict, optional) - Additional data (source platform, model name, etc.)
- `user_id` (str, optional) - User identifier (default: "default")

**Returns:**
- `turn_id` (str) - Unique ID of stored conversation turn

**Example:**
```python
user_turn = await system.store_conversation(
    role="user",
    content="What is machine learning?",
    metadata={"platform": "discord"}
)

assistant_turn = await system.store_conversation(
    role="assistant",
    content="Machine learning is a subset of AI that...",
    metadata={"model": "llama2"}
)
```

---

### search_conversations

Find conversations using semantic search.

```python
conversations = await system.search_conversations(
    query: str,
    limit: int = 10,
    user_id: str = "default"
)
```

**Parameters:**
- `query` (str) - Search query
- `limit` (int, optional) - Maximum results (default: 10)
- `user_id` (str, optional) - User ID (default: "default")

**Returns:**
- `conversations` (list) - Matching conversation turns

**Example:**
```python
results = await system.search_conversations(
    "How to use async in Python",
    limit=5
)

for conv in results:
    print(f"{conv['role']}: {conv['content'][:50]}...")
```

---

### get_conversation_history

Retrieve conversation history in chronological order.

```python
history = await system.get_conversation_history(
    limit: int = 100,
    user_id: str = "default",
    offset: int = 0
)
```

**Parameters:**
- `limit` (int, optional) - Number of turns to retrieve (default: 100)
- `user_id` (str, optional) - User ID (default: "default")
- `offset` (int, optional) - Skip first N turns for pagination (default: 0)

**Returns:**
- `history` (list) - Conversation turns in order

**Example:**
```python
# Get last 50 conversation turns
history = await system.get_conversation_history(limit=50)

# Print conversation
for turn in history:
    print(f"{turn['role']}: {turn['content']}")
    print()
```

---

## Tool Call Logging

### log_tool_call

Log an MCP tool invocation.

```python
call_id = await system.log_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    result: Any,
    metadata: Dict[str, Any] = None,
    user_id: str = "default"
)
```

**Parameters:**
- `tool_name` (str) - Name of the tool (e.g., "search_memories")
- `arguments` (dict) - Arguments passed to tool
- `result` (any) - Return value from tool
- `metadata` (dict, optional) - Additional metadata
- `user_id` (str, optional) - User ID (default: "default")

**Returns:**
- `call_id` (str) - Unique ID of logged call

**Example:**
```python
call_id = await system.log_tool_call(
    tool_name="search_memories",
    arguments={"query": "async programming", "limit": 10},
    result=["memory_1", "memory_2", "memory_3"],
    metadata={"execution_time_ms": 125}
)
```

---

### get_tool_call_history

Retrieve tool call history.

```python
history = await system.get_tool_call_history(
    tool_name: str = None,
    limit: int = 100,
    user_id: str = "default"
)
```

**Parameters:**
- `tool_name` (str, optional) - Filter by tool name (default: None = all tools)
- `limit` (int, optional) - Number of calls to retrieve (default: 100)
- `user_id` (str, optional) - User ID (default: "default")

**Returns:**
- `history` (list) - List of tool calls with timestamps

**Example:**
```python
# All tool calls
all_calls = await system.get_tool_call_history(limit=50)

# Only search_memories calls
search_calls = await system.get_tool_call_history(
    tool_name="search_memories",
    limit=20
)

for call in search_calls:
    print(f"{call['tool_name']}: {call['result']}")
```

---

### reflect_on_tool_usage

Get AI-generated insights about tool usage patterns.

```python
reflection = await system.reflect_on_tool_usage(
    user_id: str = "default"
)
```

**Parameters:**
- `user_id` (str, optional) - User ID (default: "default")

**Returns:**
- `reflection` (str) - Text analysis of tool patterns

**Example:**
```python
insights = await system.reflect_on_tool_usage()
print(insights)
# Output: "You frequently use search_memories followed by store_memory,
#          suggesting a learn-and-remember workflow..."
```

---

## System Health

### get_system_health

Check overall system status and database health.

```python
health = await system.get_system_health()
```

**Parameters:** None

**Returns:**
- `health` (dict) - System status information:
  ```python
  {
    "status": "healthy",  # or "degraded", "critical"
    "databases": {
      "ai_memories": {"status": "ok", "row_count": 1503},
      "conversations": {"status": "ok", "row_count": 5240},
      "schedule": {"status": "ok", "row_count": 12},
      "mcp_tool_calls": {"status": "ok", "row_count": 892}
    },
    "embeddings": {
      "cache_size_mb": 45.2,
      "cached_embeddings": 1503
    },
    "providers": {
      "primary": "lm_studio",
      "status": "available"
    },
    "last_check": "2026-02-23T10:30:00"
  }
  ```

**Example:**
```python
health = await system.get_system_health()

if health["status"] == "healthy":
    print("✓ System is healthy")
    print(f"Memories: {health['databases']['ai_memories']['row_count']}")
else:
    print(f"⚠ System status: {health['status']}")
    for db, info in health['databases'].items():
        print(f"  {db}: {info['status']}")
```

---

## Quick Examples

### Example 1: Store and Retrieve a Memory

```python
import asyncio
from ai_memory_core import AIMemorySystem

async def demo():
    # Initialize system
    system = await AIMemorySystem.create()
    
    # Store a memory
    memory_id = await system.store_memory(
        "Persistent AI Memory System provides semantic search over memories",
        metadata={"topic": "AI", "importance": "high"}
    )
    print(f"Stored memory: {memory_id}")
    
    # Search for it
    results = await system.search_memories("AI memory and semantic search")
    print(f"Found {len(results)} results")
    for result in results:
        print(f"  - {result['content'][:50]}... (score: {result['similarity']:.2f})")
    
    # Clean up
    await system.close()

asyncio.run(demo())
```

### Example 2: Track a Conversation

```python
async def track_conversation():
    system = await AIMemorySystem.create()
    
    # Store conversation
    turns = [
        ("user", "What is async/await?"),
        ("assistant", "Async/await is syntax for working with coroutines..."),
        ("user", "Can you give an example?"),
        ("assistant", "async def hello(): await asyncio.sleep(1)"),
    ]
    
    for role, content in turns:
        turn_id = await system.store_conversation(role, content)
        print(f"Stored {role} turn: {turn_id}")
    
    # Retrieve conversation
    history = await system.get_conversation_history(limit=10)
    print("\nConversation:")
    for turn in history:
        print(f"{turn['role']}: {turn['content']}")
    
    await system.close()

asyncio.run(track_conversation())
```

### Example 3: Monitor Tool Usage

```python
async def monitor_tools():
    system = await AIMemorySystem.create()
    
    # Log some tool calls
    await system.log_tool_call(
        tool_name="search_memories",
        arguments={"query": "async"},
        result={"matches": 5}
    )
    
    # Get history
    history = await system.get_tool_call_history(limit=10)
    print(f"Tool calls: {len(history)}")
    
    # Get insights
    insights = await system.reflect_on_tool_usage()
    print(f"Insights: {insights}")
    
    await system.close()

asyncio.run(monitor_tools())
```

---

## Error Handling

### Common Exceptions

```python
from ai_memory_core import AIMemorySystem, DatabaseError

try:
    system = await AIMemorySystem.create()
    results = await system.search_memories("query")
except DatabaseError as e:
    print(f"Database error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    await system.close()
```

---

## Performance Notes

- **Search times**: ~100-500ms depending on database size
- **Storage**: ~500 bytes per memory (varies with metadata)
- **Concurrent operations**: System supports multiple concurrent calls
- **Embedding caching**: Repeated searches use cached embeddings

---

## Thread Safety

The system is **NOT thread-safe**. For multi-threaded use:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Use in separate threads
with ThreadPoolExecutor() as executor:
    executor.submit(run_async, system.store_memory("..."))
```

---

## See Also
- [INSTALL.md](INSTALL.md) - Installation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration
- [TESTING.md](TESTING.md) - Testing
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
