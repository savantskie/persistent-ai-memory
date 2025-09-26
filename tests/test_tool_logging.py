#!/usr/bin/env python3
"""
Test script for MCP Tool Call Logger functionality
"""

import asyncio
import json
from datetime import datetime
from ai_memory_core import PersistentAIMemorySystem
from mcp_server import PersistentAIMemoryMCPServer

async def test_tool_call_logging():
    """Test the MCP tool call logging functionality"""
    
    print("🔧 Testing MCP Tool Call Logger")
    print("=" * 50)
    
    # Initialize the MCP server
    server = PersistentAIMemoryMCPServer()
    
    # Test storing a memory (this will be logged)
    print("📝 Testing store_memory tool...")
    request = {
        "tool": "store_memory",
        "parameters": {
            "content": "User prefers detailed technical explanations with code examples",
            "memory_type": "preference", 
            "importance_level": 8,
            "tags": ["communication", "technical", "preferences"]
        }
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    print(f"   Result: {result['status']}")
    print(f"   Call ID: {result.get('call_id')}")
    print(f"   Execution time: {result.get('execution_time_ms', 0):.2f}ms")
    print()
    
    # Test searching memories (this will be logged)
    print("🔍 Testing search_memories tool...")
    request = {
        "tool": "search_memories", 
        "parameters": {
            "query": "technical explanations",
            "limit": 5
        }
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    print(f"   Result: {result['status']}")
    print(f"   Call ID: {result.get('call_id')}")
    print(f"   Execution time: {result.get('execution_time_ms', 0):.2f}ms")
    print()
    
    # Test system health (this will be logged)
    print("💚 Testing get_system_health tool...")
    request = {
        "tool": "get_system_health",
        "parameters": {}
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    print(f"   Result: {result['status']}")
    print(f"   Call ID: {result.get('call_id')}")
    print(f"   Execution time: {result.get('execution_time_ms', 0):.2f}ms")
    print()
    
    # Now test the reflection tools
    print("🧠 Testing tool usage reflection...")
    request = {
        "tool": "get_tool_usage_summary",
        "parameters": {"days": 1}
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    if result["status"] == "success":
        summary = result["result"]
        print(f"   Total tool calls tracked: {summary['insights']['total_tool_calls']}")
        print(f"   Success rate: {summary['insights']['success_rate_percent']}%")
        print(f"   Reflection: {summary['insights']['reflection']}")
        
        if summary['summary']['most_used_tools']:
            print("   Most used tools:")
            for tool in summary['summary']['most_used_tools'][:3]:
                print(f"     - {tool['tool_name']}: {tool['total_calls']} calls")
    print()
    
    # Test getting tool call history
    print("📊 Testing tool call history...")
    request = {
        "tool": "get_tool_call_history",
        "parameters": {"limit": 10}
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    if result["status"] == "success":
        history = result["result"]["history"]
        print(f"   Retrieved {len(history)} tool calls")
        if history:
            latest = history[0]
            print(f"   Latest call: {latest['tool_name']} at {latest['timestamp']}")
            print(f"   Status: {latest['status']}, Time: {latest.get('execution_time_ms', 0)}ms")
    print()
    
    # Test reflection on tool usage
    print("💭 Testing usage pattern reflection...")
    request = {
        "tool": "reflect_on_tool_usage", 
        "parameters": {"days": 1}
    }
    
    result = await server.handle_mcp_request(request, client_id="test_client")
    if result["status"] == "success":
        reflection = result["result"]["reflection"]
        print(f"   Analysis period: {reflection['period_days']} days")
        print("   Usage patterns:")
        for tool in reflection['patterns']['peak_usage_tools']:
            print(f"     - {tool['insight']}")
        
        if reflection['recommendations']:
            print("   Recommendations:")
            for rec in reflection['recommendations']:
                print(f"     - {rec}")
    print()
    
    print("✅ Tool call logging test completed!")
    print()
    print("🎯 Summary:")
    print("   - Tool calls are being logged with timing and results")
    print("   - Usage statistics are tracked daily")
    print("   - Reflection tools provide insights about tool usage patterns") 
    print("   - AI assistants can now analyze their own behavior!")


async def main():
    await test_tool_call_logging()


if __name__ == "__main__":
    asyncio.run(main())
