#!/usr/bin/env python3
"""
Persistent AI Memory System - MCP Server

Model Context Protocol server providing standardized tool interface for AI assistants
with comprehensive tool call logging and reflection capabilities.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

# Import the core memory system
from ai_memory_core import (
    PersistentAIMemorySystem, 
    MCPToolCallDatabase
)

class MCPToolLogger:
    """Handles logging and reflection for all MCP tool calls"""
    
    def __init__(self, tool_call_db: MCPToolCallDatabase):
        self.tool_call_db = tool_call_db
    
    async def log_and_execute_tool(self, tool_name: str, parameters: Dict, 
                                 tool_function, client_id: str = None) -> Dict:
        """Execute a tool and log the call with timing and results"""
        
        start_time = time.time()
        call_id = None
        
        try:
            # Execute the tool function
            result = await tool_function(**parameters)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Log successful tool call
            call_id = await self.tool_call_db.log_tool_call(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                status="success",
                execution_time_ms=execution_time_ms,
                client_id=client_id
            )
            
            return {
                "status": "success",
                "result": result,
                "call_id": call_id,
                "execution_time_ms": execution_time_ms
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Log failed tool call
            call_id = await self.tool_call_db.log_tool_call(
                tool_name=tool_name,
                parameters=parameters,
                status="error",
                execution_time_ms=execution_time_ms,
                error_message=str(e),
                client_id=client_id
            )
            
            return {
                "status": "error",
                "error": str(e),
                "call_id": call_id,
                "execution_time_ms": execution_time_ms
            }


class PersistentAIMemoryMCPServer:
    """MCP Server for Persistent AI Memory System with tool call logging"""
    
    def __init__(self):
        self.memory_system = PersistentAIMemorySystem()
        self.tool_logger = MCPToolLogger(self.memory_system.tool_call_db)
        self.client_sessions = {}  # Track client sessions
    
    async def handle_mcp_request(self, request: Dict, client_id: str = None) -> Dict:
        """Handle incoming MCP requests with tool call logging"""
        
        tool_name = request.get("tool")
        parameters = request.get("parameters", {})
        
        if not tool_name:
            return {"status": "error", "error": "No tool specified"}
        
        # Map tool names to functions
        tool_functions = {
            # Memory operations
            "store_memory": self._store_memory,
            "search_memories": self._search_memories,
            "update_memory": self._update_memory,
            "get_recent_conversations": self._get_recent_conversations,
            
            # Schedule operations  
            "create_appointment": self._create_appointment,
            "create_reminder": self._create_reminder,
            "get_schedule": self._get_schedule,
            
            # VS Code project operations
            "store_project_insight": self._store_project_insight,
            "link_code_context": self._link_code_context,
            "search_project_history": self._search_project_history,
            
            # System operations
            "get_system_health": self._get_system_health,
            
            # 🔧 NEW: Tool reflection and logging operations
            "get_tool_usage_summary": self._get_tool_usage_summary,
            "get_tool_call_history": self._get_tool_call_history,
            "reflect_on_tool_usage": self._reflect_on_tool_usage,
        }
        
        tool_function = tool_functions.get(tool_name)
        if not tool_function:
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}
        
        # Execute tool with logging
        return await self.tool_logger.log_and_execute_tool(
            tool_name, parameters, tool_function, client_id
        )
    
    # Core memory operations
    async def _store_memory(self, content: str, memory_type: str = None, 
                           importance_level: int = 5, tags: List[str] = None,
                           source_conversation_id: str = None) -> Dict:
        """Store a curated memory"""
        return await self.memory_system.create_memory(
            content, memory_type, importance_level, tags, source_conversation_id
        )
    
    async def _search_memories(self, query: str, limit: int = 10, 
                              memory_type: str = None) -> Dict:
        """Search memories using semantic similarity"""
        return await self.memory_system.search_memories(query, limit, memory_type)
    
    async def _update_memory(self, memory_id: str, content: str = None,
                            importance_level: int = None, tags: List[str] = None) -> Dict:
        """Update an existing memory"""
        return await self.memory_system.update_memory(memory_id, content, importance_level, tags)
    
    async def _get_recent_conversations(self, limit: int = 10, session_id: str = None) -> Dict:
        """Get recent conversation context"""
        return await self.memory_system.get_recent_context(limit, session_id)
    
    # Schedule operations
    async def _create_appointment(self, title: str, scheduled_datetime: str,
                                 description: str = None, location: str = None,
                                 source_conversation_id: str = None) -> Dict:
        """Create an appointment"""
        return await self.memory_system.create_appointment(
            title, scheduled_datetime, description, location, source_conversation_id
        )
    
    async def _create_reminder(self, content: str, due_datetime: str,
                              priority_level: int = 5, source_conversation_id: str = None) -> Dict:
        """Create a reminder"""
        return await self.memory_system.create_reminder(
            content, due_datetime, priority_level, source_conversation_id
        )
    
    async def _get_schedule(self, days_ahead: int = 7) -> Dict:
        """Get upcoming schedule items"""
        return await self.memory_system.get_upcoming_schedule(days_ahead)
    
    # VS Code project operations
    async def _store_project_insight(self, content: str, insight_type: str = None,
                                    related_files: List[str] = None, importance_level: int = 5,
                                    source_conversation_id: str = None) -> Dict:
        """Store a project development insight"""
        return await self.memory_system.store_project_insight(
            content, insight_type, related_files, importance_level, source_conversation_id
        )
    
    async def _link_code_context(self, file_path: str, function_name: str = None,
                                description: str = "", purpose: str = None) -> Dict:
        """Link code context for future reference"""
        return await self.memory_system.link_code_context(
            file_path, function_name, description, purpose
        )
    
    async def _search_project_history(self, query: str, limit: int = 10) -> Dict:
        """Search VS Code project history"""
        return await self.memory_system.search_project_history(query, limit)
    
    # System operations
    async def _get_system_health(self) -> Dict:
        """Get comprehensive system health and statistics"""
        return await self.memory_system.get_system_health()
    
    # 🔧 NEW: Tool reflection and logging operations
    async def _get_tool_usage_summary(self, days: int = 7) -> Dict:
        """Get AI assistant tool usage summary for reflection"""
        
        summary = await self.memory_system.tool_call_db.get_tool_usage_summary(days)
        
        # Add insights for the AI assistant
        total_calls = sum(stat["call_count"] for stat in summary["daily_stats"])
        success_rate = 0
        if total_calls > 0:
            total_successes = sum(stat["success_count"] for stat in summary["daily_stats"])
            success_rate = (total_successes / total_calls) * 100
        
        return {
            "status": "success",
            "summary": summary,
            "insights": {
                "total_tool_calls": total_calls,
                "success_rate_percent": round(success_rate, 2),
                "period_days": days,
                "reflection": self._generate_usage_reflection(summary, total_calls, success_rate)
            }
        }
    
    async def _get_tool_call_history(self, tool_name: str = None, limit: int = 50) -> Dict:
        """Get recent tool call history for analysis"""
        
        history = await self.memory_system.tool_call_db.get_tool_call_history(tool_name, limit)
        
        return {
            "status": "success",
            "history": history,
            "filter": tool_name,
            "count": len(history)
        }
    
    async def _reflect_on_tool_usage(self, days: int = 7) -> Dict:
        """Generate reflection insights about tool usage patterns"""
        
        summary = await self.memory_system.tool_call_db.get_tool_usage_summary(days)
        
        # Analyze patterns
        patterns = self._analyze_usage_patterns(summary)
        
        return {
            "status": "success",
            "reflection": {
                "period_days": days,
                "patterns": patterns,
                "recommendations": self._generate_usage_recommendations(patterns),
                "summary": summary
            }
        }
    
    def _generate_usage_reflection(self, summary: Dict, total_calls: int, success_rate: float) -> str:
        """Generate natural language reflection on tool usage"""
        
        if total_calls == 0:
            return "I haven't used any memory tools in this period."
        
        most_used = summary["most_used_tools"]
        top_tool = most_used[0]["tool_name"] if most_used else "unknown"
        
        reflection_parts = [
            f"I made {total_calls} tool calls with a {success_rate:.1f}% success rate.",
            f"My most frequently used tool was '{top_tool}'."
        ]
        
        if success_rate < 90:
            reflection_parts.append("I should investigate the failures to improve my tool usage.")
        elif success_rate > 95:
            reflection_parts.append("My tool usage has been very reliable.")
        
        return " ".join(reflection_parts)
    
    def _analyze_usage_patterns(self, summary: Dict) -> Dict:
        """Analyze tool usage patterns for insights"""
        
        patterns = {
            "peak_usage_tools": [],
            "error_prone_tools": [],
            "efficiency_insights": [],
            "usage_trends": []
        }
        
        # Find tools with high usage
        for tool in summary["most_used_tools"][:3]:
            patterns["peak_usage_tools"].append({
                "tool": tool["tool_name"],
                "calls": tool["total_calls"],
                "insight": f"Heavy usage of {tool['tool_name']} suggests it's a core operation"
            })
        
        # Find error-prone tools
        for stat in summary["daily_stats"]:
            if stat["error_count"] > 0 and stat["call_count"] > 0:
                error_rate = (stat["error_count"] / stat["call_count"]) * 100
                if error_rate > 10:  # More than 10% error rate
                    patterns["error_prone_tools"].append({
                        "tool": stat["tool_name"],
                        "error_rate": round(error_rate, 1),
                        "date": stat["date"]
                    })
        
        return patterns
    
    def _generate_usage_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on usage patterns"""
        
        recommendations = []
        
        if patterns["error_prone_tools"]:
            recommendations.append("Consider investigating error-prone tools to improve reliability")
        
        if len(patterns["peak_usage_tools"]) > 0:
            top_tool = patterns["peak_usage_tools"][0]["tool"]
            recommendations.append(f"Optimize '{top_tool}' performance as it's heavily used")
        
        if not patterns["peak_usage_tools"]:
            recommendations.append("Explore using more memory tools to enhance capabilities")
        
        return recommendations


# Tool definitions for MCP clients
MCP_TOOLS = {
    "store_memory": {
        "description": "Store a curated memory for future reference",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "memory_type": {"type": "string", "description": "Type of memory (preference, fact, insight, etc.)"},
                "importance_level": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Importance level (1-10)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                "source_conversation_id": {"type": "string", "description": "ID of source conversation"}
            },
            "required": ["content"]
        }
    },
    
    "search_memories": {
        "description": "Search stored memories using semantic similarity",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Maximum results to return"},
                "memory_type": {"type": "string", "description": "Filter by memory type"}
            },
            "required": ["query"]
        }
    },
    
    "get_tool_usage_summary": {
        "description": "🔧 NEW: Get summary of AI assistant tool usage for self-reflection",
        "parameters": {
            "type": "object", 
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Number of days to analyze"}
            }
        }
    },
    
    "reflect_on_tool_usage": {
        "description": "🔧 NEW: Generate insights about tool usage patterns and efficiency",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Analysis period in days"}
            }
        }
    },
    
    "get_system_health": {
        "description": "Get comprehensive system health and statistics",
        "parameters": {"type": "object", "properties": {}}
    }
}


async def main():
    """Run the MCP server"""
    server = PersistentAIMemoryMCPServer()
    
    # Start file monitoring
    await server.memory_system.start_file_monitoring()
    
    print("🧠 Persistent AI Memory System - MCP Server Started")
    print("🔧 Tool call logging and reflection enabled")
    print("📁 File monitoring active")
    print("🌐 Ready for MCP client connections")
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MCP server...")
        await server.memory_system.stop_file_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
