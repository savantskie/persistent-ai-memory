# Persistent AI Memory System v1.5.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v1.5.0-green.svg)](https://github.com/savantskie/persistent-ai-memory)

> 🌟 **Community Call to Action**: Have you made improvements or additions to this system? Submit a pull request! Every contributor will be properly credited in the final product.

**GITHUB LINK** - https://github.com/savantskie/persistent-ai-memory.git

---

## 🆕 What's New in v1.5.0 (March 28, 2026)

**Major Architectural Rewrite: OpenWebUI-Native Integration**
- ✅ **OpenWebUI-first design** - AI Memory System now deeply integrated into OpenWebUI via plugin (primary deployment method)
- ✅ **Advanced short-term memory** - sophisticated memory extraction, filtering, and injection for chat conversations
- ✅ **User ID & Model ID isolation** - strict multi-tenant support with configurable enforcement for security and tracking
- ✅ **Complete system portability** - all hardcoded paths replaced with environment variables (works anywhere)
- ✅ **Generic class names** - removed all Friday-specific branding (FridayMemorySystem → AIMemorySystem)
- ✅ **Production-ready** - enhanced error handling, validation, and logging throughout

**Upgrade from v1.1.0:** See [CHANGELOG.md](CHANGELOG.md) for migration guide.

---

## 📚 Documentation Guide

**Choose your starting point:**

| **I want to...** | **Read this** | **Time** |
|---|---|---|
| Get started quickly | [REDDIT_QUICKSTART.md](REDDIT_QUICKSTART.md) | 5 min |
| Install the system | [INSTALL.md](INSTALL.md) | 10 min |
| Understand configuration | [CONFIGURATION.md](CONFIGURATION.md) | 15 min |
| Check system health | [TESTING.md](TESTING.md) | 10 min |
| Use the API | [API.md](API.md) | 20 min |
| Deploy to production | [DEPLOYMENT.md](DEPLOYMENT.md) | 15 min |
| Fix a problem | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | varies |
| See examples | [examples/README.md](examples/README.md) | 15 min |

---

## 🚀 Quick Start (30 seconds)

### Installation
```bash
# Linux/macOS
pip install git+https://github.com/savantskie/persistent-ai-memory.git

# Windows (same command, just use Command Prompt or PowerShell)
pip install git+https://github.com/savantskie/persistent-ai-memory.git
```

### First Validation
```bash
python tests/test_health_check.py
```

Expected output:
```
[✓] Imported ai_memory_core
[✓] Found embedding_config.json
[✓] System health check passed
[✓] All health checks passed! System is ready to use.
```

---

## 💡 What This System Does

**Persistent AI Memory** provides sophisticated memory management for AI assistants:

- 📝 **OpenWebUI Short-Term Memory Plugin** - Intelligent memory extraction and injection directly in chat conversations
- 🧠 **Persistent Memory Storage** - SQLite databases for structured, searchable long-term memories
- 🔍 **Semantic Search** - Vector embeddings for intelligent memory retrieval and relevance scoring
- 💬 **Conversation Tracking** - Multi-platform conversation history capture with context linking
- 🎯 **Smart Memory Filtering** - Advanced blacklist/whitelist and relevance scoring to inject only what matters
- 🧮 **Tool Call Logging** - Track and analyze AI tool usage patterns and performance
- 🔄 **Self-Reflection** - AI insights into its own behavior and memory patterns
- 📱 **Multi-Platform Support** - Works with OpenWebUI (primary), LM Studio, VS Code, and any MCP-compatible assistant
- 🎨 **MCP Server** - Standard Model Context Protocol for cross-platform integration

---

## ⚙️ System Architecture

### Five Specialized Databases
```
~/.ai_memory/
├── conversations.db      # Chat messages and conversation history
├── ai_memories.db       # Curated long-term memories
├── schedule.db          # Appointments and reminders
├── mcp_tool_calls.db    # Tool usage logs and reflections
└── vscode_project.db    # Development session context
```

### Configuration Files
```
~/.ai_memory/
├── embedding_config.json   # Embedding provider setup
└── memory_config.json      # Memory system defaults
```

---

## 🎯 Core Features

### Memory Operations
- `store_memory()` - Save important information persistently
- `search_memories()` - Find memories using semantic search
- `list_recent_memories()` - Get recent memories without searching

### Conversation Tracking
- `store_conversation()` - Store user/assistant messages
- `search_conversations()` - Search through conversation history
- `get_conversation_history()` - Retrieve chronological conversations

### Tool Integration
- `log_tool_call()` - Record MCP tool invocations
- `get_tool_call_history()` - Analyze tool usage patterns
- `reflect_on_tool_usage()` - Get AI insights on tool patterns

### System Health
- `get_system_health()` - Check databases, embeddings, providers
- `built-in health check` - `python tests/test_health_check.py`

---

## 🔌 Embedding Providers

Choose your embedding service:

| Provider | Speed | Quality | Cost |
|----------|-------|---------|------|
| **Ollama** (local) | ⚡⚡ | ⭐⭐⭐ | FREE |
| **LM Studio** (local) | ⚡ | ⭐⭐⭐⭐ | FREE |
| **OpenAI** (cloud) | ⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ |

See [CONFIGURATION.md](CONFIGURATION.md) for setup instructions for each provider.

---

## � Important: User ID & Model ID Requirements

**All memory operations require `user_id` and `model_id` parameters for data isolation and tracking.**

This ensures:
- ✅ **Multi-user safety** - Each user's memories are completely isolated
- ✅ **Model tracking** - Different AI models can maintain separate memories
- ✅ **Audit trail** - All operations are traceable to the user and model

### Configuration Options

By default, `user_id` and `model_id` are **required**. You can change this in `memory_config.json`:

```json
{
  "tool_requirements": {
    "require_user_id": true,
    "require_model_id": true,
    "default_user_id": "default_user",
    "default_model_id": "default_model"
  }
}
```

- `require_user_id/require_model_id: true` → Strict mode (recommended for production, security-focused, or multi-user systems)
- `require_user_id/require_model_id: false` → Use defaults instead (simpler for single-user/single-model setups)

### For AI Assistants: Auto-Fill in System Prompt

To make your AI automatically provide these values, add this to its **system prompt**:

```
When using memory system tools (store_memory, search_memories, etc.), 
ALWAYS include these parameters:
- user_id='your_user_identifier' (e.g., 'nate_user_1')
- model_id='your_model_name' (e.g., 'llama-2:7b' or 'gpt-4')

If the actual values are unknown, use safe defaults:
- user_id='default_user'
- model_id='default_model'

This isolates memories per user and tracks which AI model generated each memory.
```

### Examples

**With user_id and model_id:**
```python
# Memories are stored with full isolation
await system.store_memory(
    "User likes Python", 
    user_id="alice", 
    model_id="gpt-4"
)

# Search returns only this user's memories for this model
results = await system.search_memories(
    "programming", 
    user_id="alice", 
    model_id="gpt-4"
)
```

**Without strict requirements (if disabled):**
```python
# Uses defaults from memory_config.json
await system.store_memory("User likes Python")  # user_id="default_user", model_id="default_model"
```

See [API.md](API.md) for complete parameter documentation.

---

## �🔄 Integration Methods (Choose One)

### 1. OpenWebUI Plugin (Recommended)
**Primary deployment method** - Deep integration for sophisticated memory management:
- Deploy `ai_memory_short_term.py` as an OpenWebUI Function
- Automatically extracts memories from conversations
- Intelligently injects relevant memories before AI response
- Configurable memory scoring, filtering, and injection preferences
- No additional setup required beyond copying file into OpenWebUI Functions editor

**Installation:**
1. In OpenWebUI: **Settings → Functions → +New Function**
2. Paste entire `ai_memory_short_term.py` file
3. Set trigger to `Inlet` (runs before model response)
4. Configure memory preferences via function settings

### 2. MCP Server (Alternative Platforms)
Use with any MCP-compatible AI assistant (Claude, custom integrations, etc.):
```bash
# Via mcpo
python -m ai_memory_mcp_server

# Or make streamable for OpenWebUI's alternative integration
# (OpenWebUI supports both plugin and streamable MCP methods)
```

### 3. Standalone Library (Custom Implementations)
Use memory capabilities directly in your Python code:
```python
from ai_memory_core import AIMemorySystem
system = AIMemorySystem()
await system.store_memory("Important information", user_id="user1", model_id="model1")
results = await system.search_memories("query", user_id="user1", model_id="model1")
```

---

## 🛠️ Development & Examples

Ready-to-use examples:
```bash
python examples/basic_usage.py          # Store and search memories
python examples/advanced_usage.py       # Conversation tracking and tool logging
python examples/performance_tests.py    # Benchmark operations
```

Full API reference: [API.md](API.md)

---

## 📖 Learning Resources

- **New to AI memory systems?** → [REDDIT_QUICKSTART.md](REDDIT_QUICKSTART.md)
- **Troubleshooting issues?** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)  
- **Need configuration help?** → [CONFIGURATION.md](CONFIGURATION.md)
- **Want to deploy to production?** → [DEPLOYMENT.md](DEPLOYMENT.md)
- **Need the full API?** → [API.md](API.md)

---

## � System Sophistication

This is a significantly enhanced version of traditional memory systems:

| Feature | Traditional | AI Memory System |
|---------|-------------|------------------|
| Memory Extraction | Manual/Static | LLM-powered intelligent extraction |
| Filtering | Simple keyword matching | Multi-layer semantic + relevance scoring |
| Memory Injection | All available memories | Smart filtering - only inject relevant |
| Duplicate Prevention | Text matching | Embedding-based semantic deduplication |
| Importance Scoring | Not tracked | Dynamic importance analysis |
| Memory Normalization | N/A | Automatic format standardization |
| Context Awareness | Limited | Full conversation context integration |
| Tool Integration | Basic logging | Deep reflection and pattern analysis |
| Error Handling | Minimal | Comprehensive validation and recovery |
| Performance | N/A | Optimized with async operations |

**Result:** An AI assistant that truly learns from and adapts to your preferences over time.

---

## �🤝 Contributing

We welcome contributions! See [CONTRIBUTORS.md](CONTRIBUTORS.md) for:
- Development setup instructions
- How to run tests
- Code style guidelines
- Contribution process

---

## 📄 License

MIT License - Feel free to use this in your own AI projects!

See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project represents a unique collaboration:
- **@savantskie** - Project vision, architecture, testing
- **GitHub Copilot** - Core implementation and system design
- **ChatGPT** - Architectural guidance and insights

**Special thanks** to the AI and open-source communities for inspiration and support.

---

## 📞 Need Help?

1. **Start with**: [TESTING.md](TESTING.md) → Run health check
2. **Then check**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → Find your issue
3. **Or visit**: [COMMUNITY.md](COMMUNITY.md) → Get help from community
4. **Or open**: [GitHub Issues](https://github.com/savantskie/persistent-ai-memory/issues)

---

**⭐ If this project helps you build better AI assistants, please give it a star!**

Built with determination, debugged with patience, designed for the future of AI.
