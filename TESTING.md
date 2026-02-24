# Testing & Validation Guide

This guide covers how to verify your installation, run tests, and validate system health.

## Quick Links
- **Just installed?** → Start here with Health Check
- **Want to run full tests?** → See Test Suite section
- **Having problems?** → See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## ✅ Health Check (Start Here!)

The **health check** is the first validation after installation. It verifies all required components are installed and working.

```bash
python tests/test_health_check.py
```

### What It Validates

✅ **Python Module Imports**
- `ai_memory_core` - Main memory system
- `database_maintenance` - Database tools
- `tag_manager` - Tag extraction system
- `settings` - Configuration system

✅ **Configuration Files**
- `embedding_config.json` exists and is valid JSON
- `memory_config.json` exists and is valid (optional)
- Paths are accessible and writable

✅ **Database Initialization**
- Can create memory databases
- Can connect to configured providers
- System health check returns valid status

✅ **Provider Connectivity**
- Embedding service is reachable
- Fallback providers work
- Timeout handling is correct

### Expected Output

```
============================================================
    Persistent AI Memory System Health Check
============================================================

[✓] Imported ai_memory_core
[✓] Imported database_maintenance
[✓] Imported tag_manager
[✓] Imported settings
[✓] Found embedding_config.json at ~/.ai_memory/embedding_config.json
[✓] Valid JSON in embedding_config.json
[✓] Created system successfully
[✓] System health check passed
[✓] All health checks passed! System is ready to use.
```

### Troubleshooting Health Check

If health check fails, it provides specific error messages:

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'ai_memory_core'` | Run `pip install -e .` in repo root, then `pip install -r requirements.txt` |
| `FileNotFoundError: embedding_config.json` | Run installation script or create template (see [CONFIGURATION.md](CONFIGURATION.md)) |
| `ConnectionError` to embedding service | Check Ollama/LM Studio is running and URL in embedding_config.json is correct |
| `Permission denied` creating databases | See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#permission-denied-on-linuxmacos) |

---

## 📊 Running Full Test Suite

### Individual Test Modules

**Test memory operations:**
```bash
python tests/test_memory_operations.py
```
Validates: store_memory, search_memories, list_recent_memories

**Test conversation tracking:**
```bash
python tests/test_conversation_tracking.py
```
Validates: store_conversation, search_conversations, get_conversation_history

**Test MCP integration:**
```bash
python tests/test_mcp_integration.py
```
Validates: MCP server works, tool registration, tool execution

**Test tag management:**
```bash
python tests/test_tag_manager.py
```
Validates: Tag extraction, normalization, registry building

### Run All Tests with pytest

```bash
# Install pytest if needed
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_health_check.py -v

# Run specific test function
pytest tests/test_memory_operations.py::test_store_memory -v
```

### Test Coverage Report

```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 🏗️ Test Structure

```
tests/
├── __pycache__/
├── test_health_check.py           # System validation
├── test_memory_operations.py       # Core memory API
├── test_conversation_tracking.py   # Conversation storage
├── test_mcp_integration.py        # MCP server tests
├── test_tag_manager.py            # Tag extraction tests
├── fixtures.py                    # Shared test data
└── README.md                      # Detailed test documentation
```

### Test Conventions

- **Async tests:** Use `@pytest.mark.asyncio` decorator
- **Database isolation:** Each test uses in-memory database
- **Fixtures:** Use pytest fixtures for shared setup
- **Naming:** `test_<function>_<scenario>` (e.g., `test_store_memory_with_metadata`)

---

## 🧪 Common Test Scenarios

### Test Memory Storage

```python
import asyncio
from ai_memory_core import AIMemorySystem

async def test_memory():
    system = await AIMemorySystem.create()
    
    # Store a memory
    memory_id = await system.store_memory(
        "Python async/await tutorial",
        metadata={"category": "learning", "topic": "python"}
    )
    
    # Search it back
    results = await system.search_memories("async tutorial")
    print(f"Found {len(results)} results")
    
    await system.close()

asyncio.run(test_memory())
```

### Test Conversation Tracking

```python
async def test_conversation():
    system = await AIMemorySystem.create()
    
    # Store conversation turns
    await system.store_conversation("user", "What is machine learning?")
    await system.store_conversation("assistant", "Machine learning is...")
    await system.store_conversation("user", "Can you give an example?")
    
    # Retrieve conversation
    history = await system.get_conversation_history(limit=3)
    for turn in history:
        print(f"{turn['role']}: {turn['content'][:50]}...")
    
    await system.close()

asyncio.run(test_conversation())
```

### Test Tool Logging

```python
async def test_tool_logging():
    system = await AIMemorySystem.create()
    
    # Log a tool call
    await system.log_tool_call(
        tool_name="search_memories",
        arguments={"query": "test"},
        result=["memory1", "memory2"]
    )
    
    # Retrieve tool history
    history = await system.get_tool_call_history(limit=10)
    print(f"Logged {len(history)} tool calls")
    
    await system.close()

asyncio.run(test_tool_logging())
```

---

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r requirements-dev.txt
      
      - name: Run health check
        run: python tests/test_health_check.py
      
      - name: Run tests
        run: pytest tests/ -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 📈 Performance Testing

### Benchmark Memory Operations

```bash
python tests/test_performance.py
```

Expected on modern hardware:
- Store memory: < 100ms
- Search 1000 memories: < 500ms
- Get conversation history: < 50ms

### Load Testing

```python
import time
import asyncio
from ai_memory_core import AIMemorySystem

async def load_test():
    system = await AIMemorySystem.create()
    
    # Store 1000 memories
    start = time.time()
    for i in range(1000):
        await system.store_memory(f"Memory {i}")
    duration = time.time() - start
    
    print(f"Stored 1000 memories in {duration:.2f}s")
    print(f"Average: {duration/1000*1000:.2f}ms per memory")
    
    await system.close()

asyncio.run(load_test())
```

---

## ✨ Tips for Effective Testing

1. **Start with health check** - Validates basic setup
2. **Run tests early** - Catch issues before they propagate
3. **Check logs** - Look in `~/.ai_memory/logs/` for detailed info
4. **Isolate failure** - Run specific test to narrow down problem
5. **Clean database** - Delete `~/.ai_memory/*.db` for fresh start
6. **Check dependencies** - Ensure all providers (Ollama, LM Studio) are running

---

## 📖 See Also
- [INSTALL.md](INSTALL.md) - Installation instructions
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration details
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solutions
