#!/usr/bin/env python3
"""
Advanced usage examples showing the full power of the Persistent AI Memory System
"""

import asyncio
import json
from datetime import datetime, timezone
from mcp_server import PersistentAIMemoryMCPServer

async def mcp_tool_integration_example():
    """Demonstrate MCP server tool integration with logging"""
    
    print("🔧 MCP Tool Integration with Logging Example")
    print("=" * 50)
    
    # Initialize MCP server
    server = PersistentAIMemoryMCPServer()
    
    print("📝 Simulating AI assistant tool calls...")
    
    # Simulate storing a memory via MCP
    request1 = {
        "tool": "store_memory",
        "parameters": {
            "content": "User enjoys working with SQLite databases and appreciates clean schema design",
            "memory_type": "preference",
            "importance_level": 7,
            "tags": ["database", "sqlite", "design"]
        }
    }
    
    result1 = await server.handle_mcp_request(request1, client_id="example_ai")
    print(f"   ✅ Store memory result: {result1['status']}")
    print(f"   📊 Execution time: {result1.get('execution_time_ms', 0):.2f}ms")
    print(f"   🔗 Call ID: {result1.get('call_id', 'N/A')[:8]}...")
    
    # Simulate searching memories
    request2 = {
        "tool": "search_memories",
        "parameters": {
            "query": "database preferences",
            "limit": 5
        }
    }
    
    result2 = await server.handle_mcp_request(request2, client_id="example_ai")
    print(f"   ✅ Search memories result: {result2['status']}")
    print(f"   📊 Execution time: {result2.get('execution_time_ms', 0):.2f}ms")
    
    # Get tool usage summary (AI self-reflection!)
    request3 = {
        "tool": "get_tool_usage_summary",
        "parameters": {"days": 1}
    }
    
    result3 = await server.handle_mcp_request(request3, client_id="example_ai")
    if result3['status'] == 'success':
        insights = result3['result']['insights']
        print(f"\n🧠 AI Self-Reflection Results:")
        print(f"   📊 Total tool calls: {insights['total_tool_calls']}")
        print(f"   ✅ Success rate: {insights['success_rate_percent']}%")
        print(f"   💭 AI's reflection: {insights['reflection']}")
    
    # Demonstrate reflection on tool usage patterns
    request4 = {
        "tool": "reflect_on_tool_usage",
        "parameters": {"days": 1}
    }
    
    result4 = await server.handle_mcp_request(request4, client_id="example_ai")
    if result4['status'] == 'success':
        reflection = result4['result']['reflection']
        print(f"\n🔍 Usage Pattern Analysis:")
        if reflection['patterns']['peak_usage_tools']:
            for tool_pattern in reflection['patterns']['peak_usage_tools'][:2]:
                print(f"   📈 {tool_pattern['insight']}")
        
        if reflection['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in reflection['recommendations']:
                print(f"     • {rec}")
    
    print("\n✅ MCP tool integration example completed!")


async def advanced_memory_management():
    """Show advanced memory management techniques"""
    
    print("\n🎯 Advanced Memory Management Example")
    print("=" * 45)
    
    server = PersistentAIMemoryMCPServer()
    
    # Store different types of memories with strategic importance levels
    memories_to_store = [
        {
            "content": "User's primary programming language is Python, secondary is JavaScript",
            "memory_type": "skill_profile",
            "importance_level": 9,
            "tags": ["skills", "programming", "languages"]
        },
        {
            "content": "Project 'Friday' uses SQLite for persistence, aiohttp for async HTTP, watchdog for file monitoring",
            "memory_type": "project_context",
            "importance_level": 8,
            "tags": ["friday", "project", "dependencies", "architecture"]
        },
        {
            "content": "User prefers detailed commit messages and comprehensive documentation",
            "memory_type": "work_style",
            "importance_level": 7,
            "tags": ["git", "documentation", "style"]
        },
        {
            "content": "Breakthrough: Fixed foreign key constraints by letting store_message auto-create sessions",
            "memory_type": "technical_insight", 
            "importance_level": 10,
            "tags": ["breakthrough", "database", "foreign_keys", "solution"]
        }
    ]
    
    print("📝 Storing strategically important memories...")
    stored_memories = []
    
    for memory_data in memories_to_store:
        request = {
            "tool": "store_memory",
            "parameters": memory_data
        }
        
        result = await server.handle_mcp_request(request, client_id="advanced_example")
        if result['status'] == 'success':
            stored_memories.append(result['result']['memory_id'])
            print(f"   ✅ Stored {memory_data['memory_type']}: importance {memory_data['importance_level']}")
    
    # Demonstrate semantic search across different memory types
    search_queries = [
        "programming languages and skills",
        "database and persistence solutions", 
        "breakthrough solutions and insights"
    ]
    
    print(f"\n🔍 Testing semantic search across memory types...")
    for query in search_queries:
        request = {
            "tool": "search_memories",
            "parameters": {"query": query, "limit": 3}
        }
        
        result = await server.handle_mcp_request(request, client_id="advanced_example")
        print(f"   🎯 '{query}': {len(result.get('result', {}).get('results', []))} matches")
    
    print("\n✅ Advanced memory management example completed!")


async def real_world_workflow_simulation():
    """Simulate a real-world AI assistant workflow"""
    
    print("\n🌟 Real-World AI Assistant Workflow Simulation")
    print("=" * 55)
    
    server = PersistentAIMemoryMCPServer()
    
    # Simulate a development session where AI helps with coding
    workflow_steps = [
        {
            "action": "store_memory",
            "params": {
                "content": "User is working on implementing MCP tool call logging for AI self-reflection",
                "memory_type": "current_task",
                "importance_level": 8,
                "tags": ["mcp", "logging", "current"]
            },
            "description": "AI remembers current task"
        },
        {
            "action": "search_memories", 
            "params": {"query": "database logging patterns", "limit": 5},
            "description": "AI searches for relevant past knowledge"
        },
        {
            "action": "store_memory",
            "params": {
                "content": "Tool call logging should include execution time, parameters, results, and error handling",
                "memory_type": "design_decision",
                "importance_level": 7,
                "tags": ["logging", "design", "requirements"]
            },
            "description": "AI stores design insights"
        },
        {
            "action": "get_system_health",
            "params": {},
            "description": "AI checks system status"
        },
        {
            "action": "reflect_on_tool_usage",
            "params": {"days": 1},
            "description": "AI reflects on its own behavior"
        }
    ]
    
    print("🤖 Simulating AI assistant workflow...")
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"\n{i}. {step['description']}...")
        
        request = {
            "tool": step["action"],
            "parameters": step["params"]
        }
        
        result = await server.handle_mcp_request(request, client_id="workflow_simulation")
        status_emoji = "✅" if result['status'] == 'success' else "❌"
        print(f"   {status_emoji} {step['action']}: {result['status']}")
        
        if step["action"] == "reflect_on_tool_usage" and result['status'] == 'success':
            reflection = result['result']['reflection']
            print(f"   💭 AI Self-Assessment: Analyzed {reflection['period_days']} days of tool usage")
    
    # Final reflection on the entire workflow
    print(f"\n🎯 Final Tool Usage Analysis...")
    final_request = {
        "tool": "get_tool_usage_summary", 
        "parameters": {"days": 1}
    }
    
    final_result = await server.handle_mcp_request(final_request, client_id="workflow_simulation")
    if final_result['status'] == 'success':
        insights = final_result['result']['insights']
        print(f"   📊 Session summary: {insights['total_tool_calls']} tool calls, {insights['success_rate_percent']}% success rate")
        print(f"   🧠 AI's self-reflection: {insights['reflection']}")
    
    print("\n✅ Real-world workflow simulation completed!")
    print("💡 This demonstrates how AI assistants can:")
    print("   • Store and retrieve contextual memories")
    print("   • Learn from past interactions")
    print("   • Reflect on their own behavior patterns")
    print("   • Continuously improve their effectiveness")


async def main():
    """Run all advanced examples"""
    
    print("🚀 Advanced Examples for Persistent AI Memory System")
    print("=" * 60)
    print("These examples show the full potential of AI self-reflection and learning!")
    print()
    
    try:
        await mcp_tool_integration_example()
        await advanced_memory_management()
        await real_world_workflow_simulation()
        
        print("\n🎉 All advanced examples completed successfully!")
        print("\n🌟 Key Takeaways:")
        print("   • AI assistants can now maintain persistent memories")
        print("   • Tool call logging enables AI self-reflection")
        print("   • Semantic search provides contextual memory retrieval")
        print("   • Memory importance levels prioritize critical information")
        print("   • Cross-platform conversation capture works automatically")
        print("\n🚀 This is the foundation for truly intelligent AI assistants!")
        
    except Exception as e:
        print(f"\n❌ Error running advanced examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
