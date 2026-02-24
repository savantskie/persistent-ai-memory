# Persistent AI Memory System v1.1.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v1.1.0-green.svg)](https://github.com/savantskie/persistent-ai-memory)

> 🌟 **Community Call to Action**: Have you made improvements or additions to this system? Submit a pull request! Every contributor will be properly credited in the final product.

**GITHUB LINK** - https://github.com/savantskie/persistent-ai-memory.git

---

## 🆕 What's New in v1.1.0 (February 23, 2026)

**Path Independence & Configuration System Update**
- ✅ **Complete path independence** - works on any system in any directory
- ✅ **Configuration system** - `memory_config.json` and `embedding_config.json` for flexible deployments
- ✅ **Tag management** - automatic tag extraction and normalization
- ✅ **Improved health checks** - better diagnostics with helpful error messages
- ✅ **Docker enhancements** - full container support with synced registries
- ✅ **Documentation** - 5 comprehensive guides (config, testing, API, deployment, troubleshooting)

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

**Persistent AI Memory** provides:

- 🧠 **Persistent Memory Storage** - SQLite databases for structured, searchable storage
- 🔍 **Semantic Search** - Vector embeddings for intelligent memory retrieval
- 💬 **Conversation Tracking** - Multi-platform conversation history capture
- 🧮 **Tool Call Logging** - Track and analyze AI tool usage patterns
- 🔄 **Self-Reflection** - AI insights into its own behavior and performance
- 📱 **Multi-Platform** - Works with LM Studio, VS Code, OpenWebUI, custom applications
- 🎯 **Easy Integration** - MCP server for any AI assistant compatible with Model Context Protocol

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

## 🔄 Three Ways to Use

### 1. Standalone Functions
Use memory capabilities directly in your Python code:
```python
from ai_memory_core import AIMemorySystem
system = await AIMemorySystem.create()
await system.store_memory("Important information")
results = await system.search_memories("query")
```

### 2. OpenWebUI Plugin
Deploy as an OpenWebUI filter to auto-inject memories into chat:
- Automatically extracts memories from conversations
- Injects relevant memories before responding
- Configurable via OpenWebUI filter settings

### 3. MCP Server
Use with any MCP-compatible AI assistant (Claude, custom tools, etc.):
```bash
python -m ai_memory_mcp_server
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

## 🤝 Contributing

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
