# Examples for Persistent AI Memory System

This directory contains practical examples demonstrating how to use the Persistent AI Memory System in real-world scenarios. These examples show the full potential of AI assistants with persistent memory and self-reflection capabilities.

## Example Files Overview

### Getting Started

#### `basic_usage.py`
- **Purpose**: Introduction to core functionality
- **Demonstrates**: Memory storage, search, conversation handling, file monitoring
- **Best for**: New users, understanding basic concepts
- **Key features**: Step-by-step examples with explanations

### Advanced Features

#### `advanced_usage.py`
- **Purpose**: Complex scenarios and advanced features
- **Demonstrates**: MCP tool integration, AI self-reflection, strategic memory management
- **Best for**: Power users, production implementations
- **Key features**: Real-world workflows, optimization strategies

### Performance and Testing

#### `performance_tests.py`
- **Purpose**: Benchmarking and performance validation
- **Demonstrates**: Load testing, concurrent operations, performance metrics
- **Best for**: System optimization, capacity planning
- **Key features**: Detailed performance analysis, bottleneck identification

## Use Case Examples

### 🤖 AI Assistant Memory
```python
# Store user preferences
await store_memory(
    content="User prefers detailed explanations with examples",
    memory_type="user_preference",
    importance_level=8,
    tags=["communication", "style", "preference"]
)

# Later, AI retrieves context
memories = await search_memories("user communication style")
# AI now knows to provide detailed explanations!
```

### 💬 Conversation Context
```python
# Automatic conversation storage
await store_conversation(
    user_message="Help me debug this Python error",
    assistant_response="I see the issue. Let me check similar problems you've had...",
    session_id="debugging_session_001"
)

# AI learns from conversation history
history = await get_conversation_history("debugging_session_001")
# AI provides context-aware help based on past interactions
```

### 🧠 AI Self-Reflection
```python
# AI analyzes its own tool usage
reflection = await reflect_on_tool_usage(days=7)
print(reflection['insights'])
# "I notice I'm frequently searching for database-related memories. 
#  The user seems to be working on database optimization projects."

# AI adapts behavior based on self-analysis
recommendations = reflection['recommendations']
# ["Consider storing more detailed database performance memories",
#  "Increase importance level for SQL optimization insights"]
```

### 🔄 Learning from Patterns
```python
# AI discovers patterns in user behavior
patterns = await get_tool_usage_summary(days=30)
insights = patterns['insights']

if insights['most_used_tool'] == 'search_memories':
    # AI realizes user relies heavily on memory search
    # Could optimize search algorithms or suggest better tagging
    pass
```

## Real-World Scenarios

### Scenario 1: Software Development Assistant

An AI assistant helping with coding projects:

1. **Learns coding style**: Stores preferences for code formatting, naming conventions
2. **Remembers project context**: Tracks architecture decisions, design patterns used
3. **Builds debugging knowledge**: Stores solutions to specific errors encountered
4. **Reflects on effectiveness**: Analyzes which types of help are most useful

Example workflow:
```python
# Day 1: Store project context
await store_memory(
    content="Project uses microservices architecture with FastAPI and PostgreSQL",
    memory_type="project_context",
    importance_level=9,
    tags=["architecture", "fastapi", "postgresql", "microservices"]
)

# Day 5: AI uses context to provide relevant help
context_memories = await search_memories("project architecture database")
# AI suggests PostgreSQL-specific optimizations for FastAPI project
```

### Scenario 2: Research Assistant

An AI helping with academic research:

1. **Tracks research topics**: Stores paper summaries, key findings
2. **Connects related concepts**: Uses semantic search to find connections
3. **Monitors progress**: Reflects on research direction and productivity
4. **Suggests next steps**: Based on gaps in current knowledge

### Scenario 3: Personal Productivity Assistant

An AI managing daily tasks and learning user habits:

1. **Learns work patterns**: Stores peak productivity times, preferred task types
2. **Adapts scheduling**: Uses memory to suggest optimal task timing
3. **Tracks goal progress**: Monitors long-term objectives and milestones
4. **Provides contextual reminders**: Based on current projects and deadlines

## Implementation Patterns

### Pattern 1: Importance-Based Memory Management
```python
# Critical information gets high importance
await store_memory(
    content="Emergency contact: John Doe - 555-0123",
    importance_level=10,  # Maximum importance
    memory_type="emergency_contact"
)

# Regular preferences get medium importance
await store_memory(
    content="Prefers coffee over tea in meetings",
    importance_level=5,  # Medium importance
    memory_type="personal_preference"
)
```

### Pattern 2: Strategic Tagging
```python
# Use hierarchical tags for better organization
tags = [
    "project:friday",           # Project identifier
    "component:memory_system",  # System component
    "issue:performance",        # Issue type
    "status:resolved"          # Current status
]

await store_memory(content="...", tags=tags)
```

### Pattern 3: Context-Aware Search
```python
# Combine multiple search terms for better context
query = f"user preferences {current_project} {current_task_type}"
relevant_memories = await search_memories(query, limit=5)

# Use memory importance for prioritization
high_priority = [m for m in relevant_memories if m['importance_level'] >= 8]
```

## Performance Optimization Examples

### Batch Operations
```python
# Store multiple memories efficiently
memories_to_store = [
    {"content": "Memory 1", "memory_type": "batch_test"},
    {"content": "Memory 2", "memory_type": "batch_test"},
    {"content": "Memory 3", "memory_type": "batch_test"},
]

# Process in batches for better performance
for memory_data in memories_to_store:
    await store_memory(**memory_data)
```

### Optimized Search
```python
# Use specific queries for faster results
specific_query = "Python database optimization SQLite"  # Good
vague_query = "help with code"  # Less effective

# Limit results to what you actually need
results = await search_memories(specific_query, limit=3)  # Faster
results = await search_memories(specific_query, limit=100)  # Slower
```

## Error Handling Examples

### Graceful Degradation
```python
try:
    memories = await search_memories(user_query)
    if not memories:
        # Fallback to broader search
        memories = await search_memories(f"general {user_query}")
except Exception as e:
    # Log error but continue operation
    print(f"Memory search failed: {e}")
    memories = []  # Empty list allows system to continue

# Always provide helpful response, even without memory context
if memories:
    response = f"Based on what I remember about {user_query}..."
else:
    response = f"I'll help you with {user_query}..."
```

### Retry Logic
```python
import asyncio

async def robust_memory_operation(operation_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await operation_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Integration Examples

### With Web Applications
```python
# FastAPI integration
from fastapi import FastAPI
from mcp_server import PersistentAIMemoryMCPServer

app = FastAPI()
memory_server = PersistentAIMemoryMCPServer()

@app.post("/chat")
async def chat_endpoint(message: str, session_id: str):
    # Search for relevant context
    context = await memory_server.handle_mcp_request({
        "tool": "search_memories",
        "parameters": {"query": message, "limit": 5}
    }, client_id=session_id)
    
    # Generate AI response using context
    ai_response = generate_response(message, context)
    
    # Store conversation for future context
    await memory_server.handle_mcp_request({
        "tool": "store_conversation",
        "parameters": {
            "user_message": message,
            "assistant_response": ai_response,
            "session_id": session_id
        }
    }, client_id=session_id)
    
    return {"response": ai_response}
```

### With CLI Applications
```python
# Command-line AI assistant
import sys
from mcp_server import PersistentAIMemoryMCPServer

async def cli_assistant():
    memory_server = PersistentAIMemoryMCPServer()
    session_id = "cli_session"
    
    print("AI Assistant with Memory - Type 'quit' to exit")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Search for relevant memories
        context_request = {
            "tool": "search_memories",
            "parameters": {"query": user_input, "limit": 3}
        }
        context = await memory_server.handle_mcp_request(context_request, session_id)
        
        # Generate response (implement your AI logic here)
        ai_response = f"Based on my memory of similar topics: {user_input}"
        
        # Store conversation
        conv_request = {
            "tool": "store_conversation",
            "parameters": {
                "user_message": user_input,
                "assistant_response": ai_response,
                "session_id": session_id
            }
        }
        await memory_server.handle_mcp_request(conv_request, session_id)
        
        print(f"AI: {ai_response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(cli_assistant())
```

## Running Examples

### Prerequisites
1. Ensure the system is properly installed
2. LM Studio running (for embedding examples)
3. Database initialized (run a health check first)

### Execute Examples
```bash
# Start with basic usage
python basic_usage.py

# Explore advanced features
python advanced_usage.py

# Test performance
python performance_tests.py
```

### Customization
All examples can be modified for your specific use case:
- Change embedding service URLs
- Adjust memory importance levels
- Modify tagging strategies
- Add custom search logic

## Best Practices from Examples

1. **Strategic Memory Storage**: Use importance levels and tags thoughtfully
2. **Context-Aware Operations**: Combine multiple data sources for better AI responses
3. **Error Resilience**: Always handle failures gracefully
4. **Performance Monitoring**: Use the reflection tools to optimize system usage
5. **Incremental Learning**: Let the AI build up knowledge over time

## Next Steps

After exploring these examples:

1. **Adapt for your use case**: Modify examples to fit your specific needs
2. **Monitor performance**: Use the performance tests to optimize your setup
3. **Extend functionality**: Add custom tools and memory types
4. **Share learnings**: Contribute back improvements and new examples

---

🎯 **Goal**: These examples demonstrate how AI assistants can become truly intelligent by learning from interactions, building contextual understanding, and continuously improving their effectiveness through persistent memory and self-reflection.

🚀 **Result**: AI assistants that get better over time, understand user preferences, and provide increasingly relevant and helpful responses based on accumulated knowledge and experience!
