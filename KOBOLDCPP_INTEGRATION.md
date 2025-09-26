# Friday Memory System + Koboldcpp Integration Guide

## 🎯 **Yes, Friday's Memory System Works with Koboldcpp!**

Friday's memory system is designed to be platform-agnostic and can integrate with Koboldcpp through multiple approaches:

## 🔧 **Integration Options**

### **Option 1: File-Based Monitoring (Recommended - Available Now)**
- **How it works**: Friday monitors conversation files from Koboldcpp
- **Setup**: Point Friday's file monitor to Koboldcpp's chat logs
- **Benefits**: Automatic, zero-configuration, works with any Koboldcpp UI

```python
# Example configuration
friday_memory = FridayMemorySystem()
friday_memory.add_watch_directory("/path/to/koboldcpp/conversations")
```

### **Option 2: HTTP API Integration (Custom Implementation)**
- **How it works**: Expose Friday's memory as REST API endpoints
- **Setup**: Run Friday's memory server alongside Koboldcpp
- **Benefits**: Real-time memory access, programmatic control

```python
# Friday could expose endpoints like:
# POST /api/memory/store - Store new memories
# GET /api/memory/search - Search existing memories
# GET /api/memory/context - Get conversation context
```

### **Option 3: MCP Integration (Future)**
- **How it works**: If Koboldcpp adds MCP support, use the same server
- **Setup**: Same MCP server that works with VS Code/LM Studio
- **Benefits**: Standardized tool interface, rich capabilities

## 🚀 **Quick Setup for Koboldcpp**

### **Step 1: File Monitoring Setup**
1. Find where Koboldcpp saves conversations
2. Configure Friday to monitor that directory
3. Start Friday's memory system
4. Chat in Koboldcpp - memories build automatically!

### **Step 2: Manual Memory Integration**
```python
from friday_memory_system import FridayMemorySystem
import asyncio

async def koboldcpp_integration():
    friday = FridayMemorySystem()
    
    # Store important information
    await friday.create_memory(
        content="User prefers technical explanations",
        memory_type="preference",
        importance_level=8
    )
    
    # Search memories
    results = await friday.search_memories("technical preferences")
    return results

# Use in your Koboldcpp integration
memories = asyncio.run(koboldcpp_integration())
```

## 📋 **Available Memory Tools for Koboldcpp**

### **Core Memory Operations**
- `search_memories()` - Semantic search with importance filtering
- `create_memory()` - Store information with importance levels
- `store_conversation()` - Save chat history automatically
- `get_recent_context()` - Retrieve conversation context

### **Advanced Features**
- **Importance Levels**: 1-10 scale for memory prioritization
- **Memory Types**: preferences, facts, skills, safety guidelines
- **Semantic Search**: Vector-based similarity search
- **Cross-session Context**: Remember across different chat sessions

## 🛠️ **Implementation Examples**

### **Basic Integration**
```python
# In your Koboldcpp integration script
from friday_memory_system import FridayMemorySystem

async def enhance_koboldcpp_with_memory():
    memory = FridayMemorySystem()
    
    # Before generating response, search for relevant context
    context = await memory.search_memories(user_input, limit=3)
    
    # Include context in prompt to Koboldcpp
    enhanced_prompt = f"Context: {context}\n\nUser: {user_input}"
    
    # After getting response, store the conversation
    await memory.store_conversation(
        content=user_input,
        role="user",
        session_id="koboldcpp_session"
    )
    await memory.store_conversation(
        content=ai_response,
        role="assistant", 
        session_id="koboldcpp_session"
    )
```

### **Advanced Context Management**
```python
async def smart_context_for_koboldcpp(user_input, session_id):
    memory = FridayMemorySystem()
    
    # Get relevant memories
    memories = await memory.search_memories(user_input, limit=5)
    
    # Get recent conversation context
    recent_context = await memory.get_recent_context(
        session_id=session_id,
        limit=10
    )
    
    # Combine for rich context
    full_context = {
        "relevant_memories": memories,
        "recent_conversation": recent_context,
        "session_id": session_id
    }
    
    return full_context
```

## 🎯 **Benefits for Koboldcpp Users**

- **Persistent Memory**: Remember important information across sessions
- **Smart Context**: Automatically surface relevant past conversations
- **User Preferences**: Learn and remember how you like to interact
- **Project Continuity**: Maintain context for long-term projects
- **Cross-Platform**: Same memories work across all your AI tools

## 📦 **Installation for Koboldcpp**

1. **Install Friday's Memory System**:
   ```bash
   git clone https://github.com/savantskie/persistent-ai-memory.git
   cd persistent-ai-memory
   pip install -r requirements.txt
   ```

2. **Basic Setup**:
   ```python
   from friday_memory_system import FridayMemorySystem
   memory = FridayMemorySystem()
   # Ready to use with Koboldcpp!
   ```

3. **File Monitoring** (for automatic conversation capture):
   ```python
   # Point to Koboldcpp's conversation storage
   memory.add_watch_directory("/path/to/koboldcpp/conversations")
   await memory.start_file_monitoring()
   ```

## 💬 **Community & Support**

- **GitHub**: [persistent-ai-memory](https://github.com/savantskie/persistent-ai-memory)
- **Compatible with**: Koboldcpp, LM Studio, VS Code, Ollama, and more
- **Open Source**: MIT licensed, fully extensible

**Friday's memory system is designed to work with ANY AI interface - Koboldcpp included!** 🧠✨
