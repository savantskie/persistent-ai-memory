#!/usr/bin/env python3
"""
Basic usage examples for the Persistent AI Memory System

This script demonstrates the core functionality and how to get started.
"""

import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def basic_usage_example():
    """Demonstrate basic memory operations"""
    
    print("🧠 Persistent AI Memory System - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the memory system
    print("🔧 Initializing memory system...")
    memory = PersistentAIMemorySystem()
    
    # Store some memories
    print("\n📝 Storing memories...")
    
    # Store a user preference
    result1 = await memory.create_memory(
        content="User prefers concise technical explanations with code examples",
        memory_type="preference",
        importance_level=8,
        tags=["communication", "technical", "preferences"]
    )
    print(f"   ✅ Stored preference memory: {result1['memory_id'][:8]}...")
    
    # Store a fact
    result2 = await memory.create_memory(
        content="The user is working on a Python project called 'Friday' - an AI assistant",
        memory_type="fact",
        importance_level=7,
        tags=["project", "python", "ai"]
    )
    print(f"   ✅ Stored fact memory: {result2['memory_id'][:8]}...")
    
    # Store an insight
    result3 = await memory.create_memory(
        content="User tends to work late at night and prefers detailed documentation",
        memory_type="insight",
        importance_level=6,
        tags=["behavior", "documentation"]
    )
    print(f"   ✅ Stored insight memory: {result3['memory_id'][:8]}...")
    
    # Search memories
    print("\n🔍 Searching memories...")
    search_results = await memory.search_memories("technical explanations")
    print(f"   📊 Found {len(search_results.get('results', []))} results for 'technical explanations'")
    
    # Check system health
    print("\n💚 System health check...")
    health = await memory.get_system_health()
    print(f"   🌟 Status: {health['status']}")
    if 'databases' in health and 'ai_memories' in health['databases']:
        db_info = health['databases']['ai_memories']
        if 'memory_count' in db_info:
            print(f"   💾 Total memories: {db_info['memory_count']}")
        else:
            print(f"   💾 Database status: {db_info['status']}")
    
    print("\n✅ Basic usage example completed!")
    print("💡 The memories are now stored and searchable.")


async def conversation_storage_example():
    """Demonstrate conversation storage and retrieval"""
    
    print("\n💬 Conversation Storage Example")
    print("=" * 40)
    
    memory = PersistentAIMemorySystem()
    
    # Simulate storing a conversation
    print("📝 Storing a sample conversation...")
    
    # Store user message
    user_msg = await memory.conversations_db.store_message(
        content="Can you help me understand how embeddings work?",
        role="user",
        metadata={"source": "example_conversation"}
    )
    session_id = user_msg["session_id"]
    conversation_id = user_msg["conversation_id"]
    
    # Store assistant response
    await memory.conversations_db.store_message(
        content="Embeddings are vector representations of text that capture semantic meaning. They allow us to measure similarity between pieces of text mathematically.",
        role="assistant",
        session_id=session_id,
        conversation_id=conversation_id,
        metadata={"source": "example_conversation"}
    )
    
    print(f"   ✅ Conversation stored in session: {session_id[:8]}...")
    
    # Retrieve recent messages
    print("\n📜 Retrieving recent conversation context...")
    recent = await memory.conversations_db.get_recent_messages(limit=5)
    
    for i, msg in enumerate(recent):
        role_emoji = "👤" if msg["role"] == "user" else "🤖"
        content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"   {i+1}. {role_emoji} [{msg['role']}]: {content_preview}")
    
    print("\n✅ Conversation storage example completed!")


async def file_monitoring_example():
    """Demonstrate file monitoring capabilities"""
    
    print("\n📁 File Monitoring Example")
    print("=" * 30)
    
    print("🔍 File monitoring capabilities...")
    print("   This system can be configured to monitor:")
    print("   • VS Code chat sessions")
    print("   • LM Studio conversations")
    print("   • Custom conversation files")
    print("   • Real-time conversation imports")
    
    print("\n💡 Note: File monitoring requires additional setup")
    print("   See the documentation for configuration details")
    
    print("\n✅ File monitoring example completed!")
    print("🎯 Ready for production conversation tracking!")


async def mcp_server_example():
    """Show how to use the MCP server"""
    
    print("\n🛠️ MCP Server Usage Example")
    print("=" * 35)
    
    print("To use the MCP server for AI assistant integration:")
    print()
    print("1. Start the MCP server:")
    print("   python mcp_server.py")
    print()
    print("2. Your AI assistant can then call tools like:")
    print("   • store_memory(content, memory_type, importance_level)")
    print("   • search_memories(query, limit)")  
    print("   • get_system_health()")
    print("   • get_tool_usage_summary(days)")
    print("   • reflect_on_tool_usage(days)")
    print()
    print("3. All tool calls are logged for self-reflection!")
    print("   AI assistants can analyze their own behavior patterns.")
    print()
    print("✅ MCP server provides standardized AI tool interface!")


async def main():
    """Run all examples"""
    
    try:
        await basic_usage_example()
        await conversation_storage_example()
        await file_monitoring_example()
        await mcp_server_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Run 'python mcp_server.py' to start the MCP server")
        print("   2. Connect your AI assistant to use the memory tools")
        print("   3. Watch as your AI builds persistent memory!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
