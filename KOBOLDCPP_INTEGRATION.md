# AI Memory System + Koboldcpp Integration Guide

## 🎯 **AI Memory System Works with Koboldcpp!**

The AI Memory System is designed with platform flexibility and can integrate with Koboldcpp through multiple approaches:

## 🔧 **Integration Options**

### **Option 1: File-Based Monitoring (Recommended)**
- **How it works**: System monitors conversation files from Koboldcpp
- **Setup**: Configure file monitor to watch Koboldcpp's chat logs
- **Benefits**: Automatic memory extraction, zero configuration

```python
# Configuration (using MCP server tools)
from ai_memory_core import AIMemorySystem
import asyncio

async def monitor_koboldcpp():
    system = AIMemorySystem()
    # System will automatically track conversations
    # from configured directories
```

### **Option 2: MCP Server Integration**
- **How it works**: Run AI Memory MCP server alongside Koboldcpp
- **Setup**: Start MCP server and connect via standard protocol
- **Benefits**: Access to 36+ memory and tool management functions

```bash
# run the MCP server
python -m ai_memory_mcp_server
```

### **Option 3: Direct Memory API**
- **How it works**: Call memory functions directly from your scripts
- **Setup**: Use Python library directly
- **Benefits**: Programmatic control and custom workflows

## 🚀 **Quick Setup for Koboldcpp**

### **Step 1: Start AI Memory System**
1. Install the system: `pip install git+https://github.com/savantskie/persistent-ai-memory.git`
2. Start MCP server: `python -m ai_memory_mcp_server`
3. Or use directly in Python for custom integration
4. Conversations are automatically tracked and memories build automatically!

### **Step 2: Using Memory Functions in Code**
```python
from ai_memory_core import AIMemorySystem
import asyncio

async def koboldcpp_integration():
    system = AIMemorySystem()
    
    # Store important information
    await system.create_memory(
        content="User prefers technical explanations",
        user_id="user1",
        model_id="koboldcpp",
        memory_type="preference",
        importance_level=8
    )
    
    # Search memories
    results = await system.search_memories(
        query="technical preferences",
        user_id="user1",
        model_id="koboldcpp"
    )
    return results

# Use in your Koboldcpp integration
memories = asyncio.run(koboldcpp_integration())
```

## 📋 **Available Memory Tools**

### **Core Memory Operations**
- `search_memories()` - Semantic search with filtering
- `create_memory()` - Store information with tags and importance
- `store_conversation()` - Automatic conversation tracking
- `get_relevant_memories()` - Smart context retrieval

### **Advanced Features**
- **Importance Scoring**: Multi-factor relevance analysis
- **Memory Types**: preferences, facts, observations, insights
- **Semantic Search**: Vector-based similarity matching
- **Multi-user Support**: Isolated memory per user+model combination

## 🛠️ **Implementation Examples**

### **Basic Integration**
```python
# In your Koboldcpp integration script
from ai_memory_core import AIMemorySystem
import asyncio

async def enhance_koboldcpp():
    system = AIMemorySystem()
    
    # Before generating response, search for relevant context
    context = await system.search_memories(
        query=user_message,
        user_id="user1",
        model_id="koboldcpp",
        limit=3
    )
    
    # Include context in prompt to Koboldcpp
    enhanced_prompt = f"Context: {[m['content'] for m in context]}\n\nUser: {user_message}"
    
    # After getting response, store the conversation
    await system.store_conversation(
        content=user_message,
        role="user",
        source_type="koboldcpp",
        user_id="user1",
        model_id="koboldcpp"
    )
    await system.store_conversation(
        content=ai_response,
        role="assistant", 
        source_type="koboldcpp",
        user_id="user1",
        model_id="koboldcpp"
    )
```

### **Advanced Context Management**
```python
async def smart_context_for_koboldcpp(user_message, session_id):
    system = AIMemorySystem()
    
    # Get relevant memories
    memories = await system.search_memories(
        query=user_message,
        user_id="user1",
        model_id="koboldcpp",
        limit=5
    )
    
    # Get recent conversation context
    recent_context = await system.search_conversations(
        query=user_message,
        limit=10
    )
    
    # Combine for rich context
    full_context = {
        "relevant_memories": [m['content'] for m in memories],
        "recent_conversation": [c['content'] for c in recent_context],
        "session_id": session_id
    }
    
    return full_context
```

## 🎯 **Benefits for Koboldcpp Users**

- **Persistent Memory**: Remember important information across sessions
- **Smart Context**: Automatically surface relevant past interactions
- **User Preferences**: Learn and remember user interaction patterns
- **Continuity**: Maintain context for long-term conversations
- **Cross-Platform**: Same memories work with other AI platforms

## 📦 **Installation for Koboldcpp**

1. **Install AI Memory System**:
   ```bash
   git clone https://github.com/savantskie/persistent-ai-memory.git
   cd persistent-ai-memory
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Quick Start**:
   ```python
   from ai_memory_core import AIMemorySystem
   
   async def setup():
       system = AIMemorySystem()
       print("✅ System ready for Koboldcpp integration!")
   
   import asyncio
   asyncio.run(setup())
   ```

3. **In Your Koboldcpp Integration**:
   - Import AIMemorySystem
   - Call search_memories() before generating responses
   - Call store_conversation() after each exchange
   - Watch your memory context grow!

## 💬 **Community & Support**

- **GitHub**: [persistent-ai-memory](https://github.com/savantskie/persistent-ai-memory)
- **Compatible with**: Koboldcpp, LM Studio, VS Code, Ollama, OpenWebUI, and more
- **Open Source**: MIT licensed, fully extensible

**The AI Memory System works with ANY AI interface - including Koboldcpp!** 🧠✨
