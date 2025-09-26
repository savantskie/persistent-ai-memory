# Persistent AI Memory System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🌟 **Community Call to Action**: Have you made improvements or additions to this system? We want to include your work! Every contributor will be properly credited in the final product. Whether it's bug fixes, new features, or documentation improvements - your contributions matter and will help shape the future of AI memory systems. Submit a pull request today!

**GITHUB LINK** - https://github.com/savantskie/persistent-ai-memory.git

---
🆕 **Recent Changes (2025-09-04)**
- **🧠 Enhanced Embedding System**: Implemented intelligent embedding service with primary/fallback providers
  - **Preservation Strategy**: All existing embeddings (15,000+) are automatically preserved
  - **LM Studio Primary**: High-quality embeddings for new content via LM Studio
  - **Ollama Fallback**: Fast local embeddings when LM Studio unavailable
  - **Real-time Provider Monitoring**: Automatic availability detection and graceful fallback
- **⚙️ Standardized Configuration**: Added `embedding_config.json` for easy provider management
- **📁 Improved Organization**: Moved all test files to proper `tests/` folder structure
- **🔧 Enhanced Core Systems**: Updated `ai_memory_core.py` with intelligent provider selection
- **🛡️ Backward Compatibility**: All existing functionality preserved while adding new capabilities
- **📊 Performance Optimization**: Better semantic search quality with preserved data integrity

Previous Changes (2025-09-01):
- Updated `ai-memory-mcp_server.py` to include enhanced tool registration logic for `update_memory` and other tools.
- Improved MCP server functionality to dynamically detect and register tools based on client context.
- Added robust error handling and logging for tool execution.
- Enhanced automatic maintenance tasks with centralized scheduling and improved database optimization routines.
---

A comprehensive AI memory system that provides persistent, searchable storage for AI assistants with conversation tracking, MCP tool call logging, and intelligent scheduling.

## 🎉 Exciting News: Desktop App Coming Soon!

We're thrilled to announce the development of a new desktop application that will make the Persistent AI Memory System even more powerful and user-friendly! 

### 🚀 Upcoming Desktop Features:

- **Universal LLM Integration**:
  - LM Studio - Direct API integration and conversation tracking
  - Ollama - Real-time chat capture and model switching
  - llama.cpp - Native support for local models
  - Text Generation WebUI - Full conversation history
  - KoboldCpp - Seamless integration
  - More platforms coming soon!

- **Enhanced GUI Features**:
  - Real-time conversation visualization
  - Advanced memory search interface
  - Interactive context management
  - Visual relationship mapping
  - Customizable dashboard
  - Dark/Light theme support

- **Extended Capabilities**:
  - Multiple MCP protocol support
  - Cross-platform conversation sync
  - Enhanced embedding options
  - Visual memory navigation
  - Bulk import/export tools
  - Custom plugin support

Stay tuned for the beta release! Follow this repository for updates.

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

## 🛠️ Available Tools

### Core Memory Tools (Available in All Environments)
These tools are available in all environments (LM Studio, VS Code, etc.):

- **Memory Management**:
  - `search_memories` - Search through stored memories using semantic similarity
  - `store_conversation` - Store conversation messages
  - `create_memory` - Create a new curated memory entry
  - `update_memory` - Update an existing memory entry
  - `get_recent_context` - Get recent conversation context

- **Schedule Management**:
  - `create_appointment` - Create calendar appointments
  - `create_reminder` - Set reminders with priorities

- **System Tools**:
  - `get_system_health` - Check system status and database health
  - `get_tool_usage_summary` - Get AI tool usage statistics
  - `reflect_on_tool_usage` - AI self-reflection on tool patterns
  - `get_ai_insights` - Get AI's insights and patterns

### IDE-Specific Tools
These tools are only available in specific development environments:

#### VS Code Tools
- `save_development_session` - Save VS Code development context
- `store_project_insight` - Store development insights
- `search_project_history` - Search project development history
- `link_code_context` - Link conversations to specific code
- `get_project_continuity` - Get context for continuing development work

## 🎯 Features

- **Enhanced Memory System**:
  - SQLite-based persistent storage across all databases
  - Registry-based extensible import system
  - **Advanced duplicate detection and migration logic in all major database classes**
  - **Centralized, generic maintenance and deduplication routines**
  - **Robust, explicit startup (no auto background tasks)**
  - Database-backed deduplication across all sources
  - Incremental imports (only new messages)
  - Enhanced error handling with detailed logging
  - Automatic system maintenance and optimization
  - AI-driven self-reflection and pattern analysis
  - Cross-database relationship tracking
  - Smart memory pruning and archival

- **Dedicated Chat Format Support**:
  - Independent parsers for each chat GUI
  - No merged/refactored import logic
  - Easy addition of new chat formats
  - Format-specific metadata preservation
  - Source-aware deduplication

- **Core Features**:
  - Vector Search using LM Studio embeddings
  - Real-time conversation monitoring
  - MCP server with tool call logging
  - Advanced AI self-reflection system:
    - Usage pattern detection and analysis
    - Automated performance optimization
    - Tool effectiveness tracking
    - Learning from past interactions
    - Continuous system improvement
  - Multi-platform compatibility
  - Zero configuration needed
  - **Full feature parity with Friday’s main repo**

- **Platform Support**:
  - LM Studio integration
  - VS Code & GitHub Copilot
  - Koboldcpp compatibility
  - Ollama chat tracking
  - Cross-platform (Windows/Linux/macOS)

## 📚 Quick Start

    - Learning from past interactions

## ⚙️ Embedding Configuration

The system supports multiple embedding providers with automatic fallback for optimal performance:

### Supported Providers

- **Ollama** (Default): Uses `qwen2.5:1.5b` for fast, lightweight embeddings
- **LM Studio**: Uses `text-embedding-nomic-embed-text-v1.5` for quality embeddings
- **OpenAI**: Uses `text-embedding-3-small` for high-quality cloud embeddings
- **Custom**: Support for custom embedding servers

### Configuration

Edit `embedding_config.json` to customize your embedding setup:

```json
{
  "embedding_configuration": {
    "primary": {
      "provider": "ollama",
      "model": "qwen2.5:1.5b",
      "base_url": "http://localhost:11434"
    },
    "fallback": {
      "provider": "lm_studio", 
      "model": "text-embedding-nomic-embed-text-v1.5",
      "base_url": "http://localhost:1234"
    }
  }
}
```

### Setup Instructions

1. **For Ollama** (Recommended): 
   ```bash
   ollama pull qwen2.5:1.5b
   ```

2. **For LM Studio**: Load an embedding model in LM Studio

3. **For OpenAI**: Add your API key to the config file

4. **For Custom**: Configure your server URL and model name

The system will automatically try the primary provider first, then fallback to the secondary if needed.

## 🚀 Quick Start

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

The system includes 5 specialized databases with enhanced cross-source integration:

1. **Conversations**: 
   - Multi-source chat history with embeddings
   - Registry-based extensible import system
   - Independent parsers per chat format
   - Database-backed deduplication
   - Source tracking and sync status
   - Cross-conversation relationships
   - Incremental import tracking
   - Comprehensive metadata per source

2. **AI Memories**: 
   - Long-term persistent AI memories
   - Cross-source knowledge synthesis
   - Relationship tracking between memories

3. **Schedule**: 
   - Time-based events and reminders
   - Cross-platform calendar integration
   - Smart scheduling with context

4. **VS Code Projects**: 
   - Project context and file tracking
   - Development conversation tracking
   - Code change history integration
   - Context-aware project insights

5. **MCP Tool Calls**: 
   - Model Context Protocol interaction logging
   - Tool usage analytics
   - Self-reflection capabilities
   - Performance monitoring

## 🔧 Configuration

The system works with zero configuration but can be customized:

```python
memory = PersistentAIMemorySystem(
    db_path="custom_memory.db",
    embedding_service_url="http://localhost:1234/v1/embeddings"
)
```

## 🛡️ System Maintenance

The system now includes automatic and **centralized maintenance features**:

- **Centralized Maintenance & Deduplication**:
  - Generic, registry-based maintenance routines
  - Advanced duplicate detection and migration logic for all database classes
  - No format-specific or auto-startup tasks; all maintenance is explicit and robust

- **Database Optimization**:
  - Automatic vacuum and reindex
  - Smart memory pruning
  - Performance monitoring
  - Index optimization

- **Error Management**:
  - Comprehensive error logging
  - Automatic recovery procedures
  - Failed operation retry
  - Data consistency checks

- **AI Self-Reflection**:
  - Tool usage pattern analysis
  - Performance optimization suggestions
  - Automated system improvements
  - Usage statistics and insights

## 🧪 Examples

Check the `examples/` directory for:
- Basic memory operations
- Conversation tracking
- MCP server setup
- Vector search demonstrations
- Custom chat format integration
- Deduplication system usage
- Registry-based importing
- Source tracking setup

## 🔌 Platform Integration Guides

### AI Platforms
- **[Koboldcpp Integration](KOBOLDCPP_INTEGRATION.md)** - Complete setup guide for Koboldcpp compatibility
- **LM Studio** - Built-in support for embeddings and conversation capture
- **VS Code** - MCP server integration for development workflows
- **SillyTavern** - MCP server support with character-specific memory tools
- **Ollama** - Compatible through file monitoring and HTTP API approaches

### Integration Methods
- **File Monitoring** - Automatic conversation capture from chat logs
- **HTTP API** - Real-time memory access via REST endpoints  
- **MCP Protocol** - Standardized tool interface for compatible platforms

### Cross-Source Memory Integration

The system now provides comprehensive cross-source memory management:

- **Source Tracking**: 
  - Automatic source detection and monitoring
  - Per-source metadata and sync status
  - Error tracking and recovery
  - Active source health monitoring

- **Relationship Management**:
  - Cross-conversation linking
  - Context preservation across platforms
  - Conversation continuation tracking
  - Reference and fork management

- **Supported Sources**:
  - VS Code/GitHub Copilot
  - ChatGPT desktop app
  - Claude/Anthropic
  - Character.ai
  - SillyTavern (file monitoring + MCP server)
  - text-generation-webui
  - Ollama
  - Generic text/markdown formats
  - Custom source support via plugins

- **Sync Features**:
  - Real-time sync status tracking
  - Source-specific metadata preservation
  - Robust deduplication across sources
  - Failure recovery and retry logic

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

## 🌟 Community Requests & Platform Support

**Recently Added Platforms** (Based on Reddit Community Feedback & Local Storage Verification):
- ✅ **SillyTavern** - AI character chat interface with conversation logging
- ✅ **Gemini CLI** - Google's Gemini command line interface support  
- ✅ **Open WebUI** - Local web-based LLM interface (multiple install locations)
- ❌ **ChatGPT & Claude Desktop** - Removed after verification (cloud-only, no local storage)

**Total Platform Support**: **9 Chat Platforms** (Based on Verified Local Storage)
- ✅ **LM Studio** - Local conversations in JSON format  
- ✅ **Ollama** - SQLite database with chat history
- ✅ **VS Code Copilot** - Development conversation tracking
- ✅ **Open WebUI** - SQLite database with conversation storage  
- ✅ **Text Generation WebUI** - Local chat logs and history
- ✅ **SillyTavern** - AI character chat interface with conversation logging
- ✅ **Gemini CLI** - Google's Gemini command line interface support
- ✅ **Jan AI** - Local AI assistant with conversation storage
- ✅ **Perplexity** - Local conversation tracking (where applicable)

**❌ Cloud-Only Applications** (No Local Storage - Removed After Verification):
- ❌ **ChatGPT Desktop** - Cloud-only, no local conversation storage
- ❌ **Claude Desktop** - Cloud-only, no local conversation files
- ❌ **Perplexity Web** - Cloud-based, no local storage

**Upcoming Community Requests**:
- [ ] **GraphDB Integration** - Graph database support for relationship mapping (community requested)
- [ ] **Discord Bot Integration** - Chat logging for Discord AI bots
- [ ] **Telegram Bot Support** - Conversation tracking for Telegram bots
- [ ] **API Standardization** - Universal chat format for easier platform integration

*Have a platform request? Open an issue or submit a PR - all contributors get credited!*

## 📄 License

MIT License - feel free to use this in your own AI projects!

## � Contributors

This project is the result of a collaborative effort between humans and AI assistants:

- **@savantskie** - Project vision, architecture design, and testing
- **GitHub Copilot** - Core implementation, database design, MCP server development, and tool call logging system
- **ChatGPT** - Initial concept development, feature recommendations, and architectural guidance over 3 months of development

## �🙏 Acknowledgments

This project represents a unique collaboration between human creativity and AI assistance. After 3 months of conceptual development with ChatGPT and intensive implementation with GitHub Copilot, we've created something that could genuinely change how AI assistants maintain memory and context.

**Special thanks to:**
- **ChatGPT** for the original insight that "*If this ever becomes open source? It'll become the standard.*"
- **GitHub Copilot** for the breakthrough implementation that solved foreign key constraints and made real-time conversation capture work flawlessly
- **The open source community** for inspiring us to share this foundational technology

Built with determination, debugged with patience, and designed for the future of AI assistance.

**GITHUB LINK** - https://github.com/savantskie/persistent-ai-memory.git
---

**⭐ If this project helps you build better AI assistants, please give it a star!**
