# Persistent AI Memory System (PAMS)

🧠 **A comprehensive, real-time memory system for AI assistants with cross-platform conversation capture, semantic search, and MCP server integration.**

## 🌟 Features

- **📚 Multi-Database Architecture**: 5 specialized SQLite databases for different memory types
- **🔄 Real-Time Conversation Capture**: Automatic import from VS Code, LM Studio, and other platforms
- **🔍 Semantic Search**: Vector embeddings with similarity search across all memory types
- **🛠️ MCP Server Integration**: Standardized tool interface for AI assistants
- **📊 Health Monitoring**: Comprehensive system diagnostics and statistics
- **🔧 Tool Call Logging**: Track and analyze AI assistant tool usage patterns
- **⚡ Cross-Platform Support**: Windows, Linux, and macOS compatibility

## 🏗️ Architecture

### Core Components

1. **`ai_memory_core.py`** - Main memory system with all database managers
2. **`mcp_server.py`** - Model Context Protocol server providing AI tool interface
3. **`conversation_monitor.py`** - Real-time file monitoring for conversation capture
4. **`embedding_service.py`** - LM Studio integration for semantic embeddings

### Database Structure

- **`conversations.db`** - Chat messages with auto-threading and embeddings
- **`ai_memories.db`** - Curated memories and insights with importance scoring
- **`schedule.db`** - Appointments and reminders with due date tracking
- **`vscode_project.db`** - Development sessions, insights, and code context
- **`mcp_tool_calls.db`** - Tool call logging for reflection and debugging

## 🚀 Quick Start

### Prerequisites

```bash
pip install aiohttp watchdog numpy sqlite3
```

### Basic Usage

```python
from ai_memory_core import PersistentAIMemorySystem

# Initialize the memory system
memory = PersistentAIMemorySystem()

# Start real-time conversation monitoring
await memory.start_file_monitoring()

# Store a memory
result = await memory.create_memory(
    content="User prefers concise technical explanations",
    memory_type="preference",
    importance_level=7,
    tags=["communication", "technical"]
)

# Search memories semantically
results = await memory.search_memories("How should I explain technical concepts?")
```

### MCP Server

```bash
# Start the MCP server for AI assistant integration
python mcp_server.py
```

## 🔧 Configuration

The system automatically detects conversation storage locations for:

- **VS Code**: `~/.config/Code/User/workspaceStorage/*/chatSessions`
- **LM Studio**: `~/.lmstudio/conversations`
- **Custom paths**: Add via `add_watch_directory()`

## 📊 Monitoring & Health

```python
# Get comprehensive system health
health = await memory.get_system_health()
print(f"Status: {health['status']}")
print(f"Total messages: {health['databases']['conversations']['message_count']}")
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
