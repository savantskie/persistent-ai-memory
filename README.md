# Persistent AI Memory System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive AI memory system that provides persistent, searchable storage for AI assistants with conversation tracking, MCP tool call logging, and intelligent scheduling.

**🎯 Multiple Installation Options Available:** We've created **4 different ways** to install this system - from one-command installation to manual setup - so you can get started immediately regardless of your platform or preference!

> 👋 **New to this from Reddit?** Check out the [Reddit Quick Start Guide](REDDIT_QUICKSTART.md) for a super simple setup!

## 🚀 Quick Installation - Choose Your Method!

### ⚡ Option 1: One-Command Installation (Linux/macOS) - **FASTEST**
```bash
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.sh | bash
```

### 🪟 Option 2: Windows One-Click Installation - **EASIEST**
```cmd
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.bat -o install.bat && install.bat
```

### 🔧 Option 3: Manual Installation - **MOST CONTROL**
```bash
git clone https://github.com/savantskie/persistent-ai-memory.git
cd persistent-ai-memory
pip install -r requirements.txt
pip install -e .
```

### 📦 Option 4: Direct pip Installation - **SIMPLEST**
```bash
pip install git+https://github.com/savantskie/persistent-ai-memory.git
```

## 🏥 Health Check
After installation, verify everything is working:
```bash
python tests/test_health_check.py
```

## 🎯 Features

- **Persistent Memory**: SQLite-based storage for conversations, AI memories, schedules, VS Code projects, and MCP tool calls
- **Vector Search**: Semantic search using embeddings via LM Studio
- **Real-time Monitoring**: File-based conversation capture with watchdog
- **MCP Integration**: Model Context Protocol server with tool call logging and AI self-reflection
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Zero Configuration**: Works out-of-the-box with sensible defaults

## 📚 Quick Start

### Basic Usage
```python
import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def main():
    # Initialize the memory system
    memory = PersistentAIMemorySystem()
    
    # Store a memory
    await memory.store_memory("I learned about Python async programming today")
    
    # Search memories
    results = await memory.search_memories("Python programming")
    print(f"Found {len(results)} related memories")
    
    # Store conversation
    await memory.store_conversation("user", "What is async programming?")
    await memory.store_conversation("assistant", "Async programming allows...")

if __name__ == "__main__":
    asyncio.run(main())
```

### MCP Server (for Claude Desktop, etc.)
```python
# Run as MCP server
python ai_memory_core.py
```

### File Monitoring
```python
# Monitor conversation files (like ChatGPT exports)
from ai_memory_core import PersistentAIMemorySystem

memory = PersistentAIMemorySystem()
memory.start_conversation_monitoring("/path/to/conversation/files")
```

## 🏗️ Architecture

The system includes 5 specialized databases:

1. **Conversations**: Chat history with embeddings for semantic search
2. **AI Memories**: Long-term persistent AI memories
3. **Schedule**: Time-based events and reminders
4. **VS Code Projects**: Project context and file tracking
5. **MCP Tool Calls**: Model Context Protocol interaction logging

## 🔧 Configuration

The system works with zero configuration but can be customized:

```python
memory = PersistentAIMemorySystem(
    db_path="custom_memory.db",
    embedding_service_url="http://localhost:1234/v1/embeddings"
)
```

## 🧪 Examples

Check the `examples/` directory for:
- Basic memory operations
- Conversation tracking
- MCP server setup
- Vector search demonstrations

## 🧪 Testing

Run the complete test suite:
```bash
python tests/test_health_check.py
python tests/test_memory_operations.py
python tests/test_conversation_tracking.py
python tests/test_mcp_integration.py
```

## � API Reference

### Core Methods

#### Memory Operations
- `store_memory(content, metadata=None)` - Store a persistent memory
- `search_memories(query, limit=10)` - Semantic search of memories
- `list_recent_memories(limit=10)` - Get recent memories

#### Conversation Tracking
- `store_conversation(role, content, metadata=None)` - Store conversation turn
- `search_conversations(query, limit=10)` - Search conversation history
- `get_conversation_history(limit=100)` - Get recent conversations

#### MCP Tool Calls
- `log_tool_call(tool_name, arguments, result, metadata=None)` - Log MCP tool usage
- `get_tool_call_history(tool_name=None, limit=100)` - Get tool usage history
- `reflect_on_tool_usage()` - AI self-reflection on tool patterns

#### System Health
- `get_system_health()` - Check system status and database health

## 🛠️ Development

### Setting up for Development
```bash
git clone https://github.com/savantskie/persistent-ai-memory.git
cd persistent-ai-memory
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest tests/
```

## 🤝 Contributing

We welcome contributions! This system is designed to be:

- **Modular**: Easy to extend with new memory types
- **Platform-agnostic**: Works with any AI assistant that supports MCP
- **Scalable**: Handles large conversation histories efficiently

## 📋 Roadmap

- [ ] **Semantic Tagging Assistant** - AI-powered memory categorization
- [ ] **Memory Summarization** - Automatic TL;DR for long conversations  
- [ ] **Deferred Retry Queue** - Resilient file import with retry logic
- [ ] **Memory Reflection Engine** - Meta-insights from memory patterns
- [ ] **Export/Import Tools** - Backup and migration utilities

## 📄 License

MIT License - feel free to use this in your own AI projects!

## � Contributors

This project is the result of a collaborative effort between humans and AI assistants:

- **@yourusername** - Project vision, architecture design, and testing
- **GitHub Copilot** - Core implementation, database design, MCP server development, and tool call logging system
- **ChatGPT** - Initial concept development, feature recommendations, and architectural guidance over 3 months of development

## �🙏 Acknowledgments

This project represents a unique collaboration between human creativity and AI assistance. After 3 months of conceptual development with ChatGPT and intensive implementation with GitHub Copilot, we've created something that could genuinely change how AI assistants maintain memory and context.

**Special thanks to:**
- **ChatGPT** for the original insight that "*If this ever becomes open source? It'll become the standard.*"
- **GitHub Copilot** for the breakthrough implementation that solved foreign key constraints and made real-time conversation capture work flawlessly
- **The open source community** for inspiring us to share this foundational technology

Built with determination, debugged with patience, and designed for the future of AI assistance.

---

**⭐ If this project helps you build better AI assistants, please give it a star!**
