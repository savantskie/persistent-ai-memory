#!/usr/bin/env python3
"""
Persistent AI Memory System - MCP Server

Model Context Protocol server providing standardized tool interface for AI assistants
with comprehensive tool call logging and reflection capabilities.
"""

import asyncio
import json
import logging
import time
import jsonschema
from typing import Any, Dict, List, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# MCP imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    TextContent,
    Tool,
)

# Import the core memory system
from ai_memory_core import (
    PersistentAIMemorySystem, 
    MCPToolCallDatabase,
    FileMonitor
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
            
            # Format response for MCP clients
            if isinstance(result, dict) and "status" in result:
                # Result already formatted
                return result
            else:
                # Standard response format
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


class AutoMaintenanceMixin:
    """Mixin to add automatic maintenance functionality to MCP server"""
    
    def _start_automatic_maintenance(self):
        """Start automatic database maintenance background task"""
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("🔧 Automatic database maintenance started (3-hour intervals)")
    
    async def _maintenance_loop(self):
        """Background loop for automatic database maintenance"""
        # Wait a bit after startup before first maintenance
        await asyncio.sleep(300)  # 5 minutes initial delay
        
        while True:
            try:
                logger.info("🧹 Running automatic database maintenance...")
                await self._run_database_maintenance()
                logger.info("✅ Automatic maintenance completed")
                    
            except Exception as e:
                logger.error(f"❌ Automatic maintenance failed: {e}")
            
            # Wait 3 hours before next maintenance
            await asyncio.sleep(3 * 60 * 60)
    
    async def _run_database_maintenance(self):
        """Run database maintenance tasks"""
        try:
            # Run maintenance through memory system which uses database_maintenance.py
            result = await self.memory_system.run_database_maintenance()
            
            # Log maintenance results
            if result.get("success"):
                logger.info(f"✅ Maintenance completed: {len(result.get('optimization_results', {}))} databases optimized")
            else:
                logger.warning(f"⚠️ Maintenance issues: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"Database maintenance error: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources when server stops"""
        if hasattr(self, '_maintenance_task') and self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            logger.info("🔧 Automatic maintenance stopped")
        
        if self.file_monitor:
            await self.file_monitor.stop()
            logger.info("📁 File monitoring stopped")


class PersistentAIMemoryMCPServer(AutoMaintenanceMixin):
    """MCP Server for Persistent AI Memory System with tool call logging and automatic maintenance"""
    
    def __init__(self):
        self.memory_system = PersistentAIMemorySystem()
        self.server = Server("friday-memory")
        self.tool_logger = MCPToolLogger(self.memory_system.mcp_db)
        self.client_context = {}  # Track client-specific context
        self._maintenance_task = None  # Background maintenance task
        self.file_monitor = None  # File monitoring task
        
        # Register MCP handlers
        self._register_handlers()
        
        # Start automatic maintenance
        self._start_automatic_maintenance()
        
        # Start file monitoring
        self.file_monitor = FileMonitor(self.memory_system)
        
    def _validate_parameters(self, parameters: Dict, schema: Dict) -> None:
        """Validate parameters against the tool's JSON schema"""
        try:
            jsonschema.validate(parameters, schema)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Parameter validation failed: {e.message}")
    
    def _create_error_response(self, message: str, code: str = None) -> Dict:
        """Create a standardized error response"""
        content = {
            "type": "text",
            "text": f"Error: {message}",
            "highlights": None,
            "meta": {"error_code": code} if code else None
        }
        
        return {
            "content": [content],
            "success": False,
            "structuredContent": None,
            "isError": True,
            "meta": {"error_code": code} if code else None
        }
    
    def _create_success_response(self, result: Any) -> Dict:
        """Create a standardized success response"""
        # Format the result as text content
        if isinstance(result, (dict, list)):
            result_text = json.dumps(result, indent=2, default=str)
        else:
            result_text = str(result)
            
        content = {
            "type": "text",
            "text": result_text,
            "highlights": None,
            "meta": None
        }
        
        return {
            "content": [content],
            "success": True,
            "structuredContent": None,
            "isError": False,
            "meta": None
        }
    
    def _register_handlers(self):
        """Register MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            # Convert MCP_TOOLS to list of Tool objects
            tools = []
            for name, info in MCP_TOOLS.items():
                tools.append(Tool(
                    name=name,
                    description=info["description"],
                    inputSchema=info["parameters"]
                ))
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Execute tool with proper response formatting"""
            response = await self.handle_mcp_request({"tool": name, "parameters": arguments})
            
            # Convert response to CallToolResult format
            if response.get("isError"):
                return CallToolResult(
                    content=[TextContent(**content) for content in response["content"]],
                    success=False,
                    structuredContent=None,
                    isError=True,
                    meta=response.get("meta")
                )
            else:
                return CallToolResult(
                    content=[TextContent(**content) for content in response["content"]],
                    success=True,
                    structuredContent=None,
                    isError=False,
                    meta=response.get("meta")
                )
    
    async def handle_mcp_request(self, request: Dict, client_id: str = None) -> Dict:
        """Handle incoming MCP requests with tool call logging"""
        
        try:
            tool_name = request.get("tool")
            parameters = request.get("parameters", {})
            
            if not tool_name:
                return self._create_error_response("No tool specified", "TOOL_MISSING")
                
            # Validate tool parameters against schema if defined
            tool_schema = MCP_TOOLS.get(tool_name, {}).get("parameters")
            if tool_schema:
                try:
                    self._validate_parameters(parameters, tool_schema)
                except ValueError as e:
                    return self._create_error_response(f"Invalid parameters: {str(e)}", "INVALID_PARAMS")
                    
            # Execute the tool with logging
            # Map tool names to functions  
            tool_functions = {
                # Memory operations
                "store_memory": self.memory_system.create_memory,
                "search_memories": self.memory_system.search_memories,
                "store_conversation": self.memory_system.store_conversation,
                "update_memory": self.memory_system.update_memory,
                "get_recent_context": self.memory_system.get_recent_context,
                "get_current_time": self._get_current_time,
                
                # Schedule operations  
                "create_appointment": self.memory_system.create_appointment,
                "create_reminder": self.memory_system.create_reminder,
                "get_schedule": self.memory_system.get_upcoming_schedule,
                
                # Project development operations
                "save_development_session": self.memory_system.save_development_session,
                "store_project_insight": self.memory_system.store_project_insight,
                "search_project_history": self.memory_system.search_project_history,
                "link_code_context": self.memory_system.link_code_context,
                "get_project_continuity": self.memory_system.get_project_continuity,
                
                # System and reflection operations
                "get_system_health": self.memory_system.get_system_health,
                "get_tool_usage_summary": self.memory_system.get_tool_usage_summary,
                "reflect_on_tool_usage": self.memory_system.reflect_on_tool_usage,
                "get_ai_insights": self.memory_system.get_ai_insights
            }
            
            tool_function = tool_functions.get(tool_name)
            if not tool_function:
                return self._create_error_response(f"Unknown tool: {tool_name}", "UNKNOWN_TOOL")
            
            # Execute tool with logging
            try:
                result = await self.tool_logger.log_and_execute_tool(
                    tool_name, parameters, tool_function, client_id
                )
                return self._create_success_response(result)
            except Exception as e:
                logger.error(f"Tool execution error: {str(e)}")
                return self._create_error_response(str(e), "TOOL_EXEC_ERROR")
                
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return self._create_error_response(f"Internal server error: {str(e)}", "SERVER_ERROR")
        
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
    
    async def _get_current_time(self, format: str = "ISO") -> Dict:
        """Get the current time for appointment awareness"""
        current_time = datetime.now()
        
        if format.upper() == "ISO":
            formatted_time = current_time.isoformat()
        else:
            try:
                formatted_time = current_time.strftime(format)
            except ValueError:
                formatted_time = current_time.isoformat()
        
        return {
            "current_time": formatted_time,
            "timestamp": current_time.timestamp(),
            "timezone": current_time.astimezone().tzname()
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
    # Memory operations
    "search_memories": {
        "description": "Search memories using semantic similarity with importance and type filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
                "database_filter": {"type": "string", "description": "Filter by database type", "enum": ["conversations", "ai_memories", "schedule", "all"], "default": "all"},
                "min_importance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Minimum importance level to include (1-10)"},
                "max_importance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum importance level to include (1-10)"},
                "memory_type": {"type": "string", "description": "Filter by memory type"}
            },
            "required": ["query"]
        }
    },

    "store_memory": {
        "description": "Create a curated memory entry",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "memory_type": {"type": "string", "description": "Type of memory"},
                "importance_level": {"type": "integer", "description": "Importance (1-10)", "default": 5},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Memory tags"},
                "source_conversation_id": {"type": "string", "description": "Source conversation ID"}
            },
            "required": ["content"]
        }
    },

    "store_conversation": {
        "description": "Store conversation automatically",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Conversation content"},
                "role": {"type": "string", "description": "Role (user/assistant)"},
                "session_id": {"type": "string", "description": "Session identifier"},
                "metadata": {"type": "object", "description": "Additional metadata"}
            },
            "required": ["content", "role"]
        }
    },

    "update_memory": {
        "description": "Update an existing curated memory",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to update"},
                "content": {"type": "string", "description": "Updated content"},
                "importance_level": {"type": "integer", "description": "Updated importance"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Updated tags"}
            },
            "required": ["memory_id"]
        }
    },

    # Schedule operations
    "create_appointment": {
        "description": "Create an appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Appointment title"},
                "description": {"type": "string", "description": "Appointment description"},
                "scheduled_datetime": {"type": "string", "description": "ISO format datetime"},
                "location": {"type": "string", "description": "Location"}
            },
            "required": ["title", "scheduled_datetime"]
        }
    },

    "create_reminder": {
        "description": "Create a reminder",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Reminder content"},
                "due_datetime": {"type": "string", "description": "ISO format datetime"},
                "priority_level": {"type": "integer", "description": "Priority (1-10)", "default": 5}
            },
            "required": ["content", "due_datetime"]
        }
    },

    "get_recent_context": {
        "description": "Get recent conversation context",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of recent items", "default": 5},
                "session_id": {"type": "string", "description": "Specific session ID"}
            }
        }
    },

    "get_system_health": {
        "description": "Get comprehensive system health, statistics, and database status",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    },

    # Tool reflection and analysis
    "get_tool_usage_summary": {
        "description": "Get AI tool usage summary and insights for self-reflection",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days to analyze", "default": 7},
                "client_id": {"type": "string", "description": "Specific client ID to analyze"}
            }
        }
    },

    "reflect_on_tool_usage": {
        "description": "AI self-reflection on tool usage patterns and effectiveness",
        "parameters": {
            "type": "object", 
            "properties": {
                "days": {"type": "integer", "description": "Days to analyze", "default": 7},
                "client_id": {"type": "string", "description": "Specific client ID to analyze"}
            }
        }
    },

    "get_ai_insights": {
        "description": "Get recent AI self-reflection insights and patterns",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of insights", "default": 5},
                "insight_type": {"type": "string", "description": "Type of insight to filter"}
            }
        }
    },

    # Project development tools
    "save_development_session": {
        "description": "Save development session context",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_path": {"type": "string", "description": "Workspace path"},
                "active_files": {"type": "array", "items": {"type": "string"}, "description": "Active files"},
                "git_branch": {"type": "string", "description": "Current git branch"},
                "session_summary": {"type": "string", "description": "Session summary"}
            },
            "required": ["workspace_path"]
        }
    },

    "store_project_insight": {
        "description": "Store development insight or decision",
        "parameters": {
            "type": "object",
            "properties": {
                "insight_type": {"type": "string", "description": "Type of insight"},
                "content": {"type": "string", "description": "Insight content"},
                "related_files": {"type": "array", "items": {"type": "string"}, "description": "Related files"},
                "importance_level": {"type": "integer", "description": "Importance (1-10)", "default": 5}
            },
            "required": ["content"]
        }
    },

    "search_project_history": {
        "description": "Search development history",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10}
            },
            "required": ["query"]
        }
    },

    "link_code_context": {
        "description": "Link conversation to specific code context",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"},
                "function_name": {"type": "string", "description": "Function name"},
                "description": {"type": "string", "description": "Context description"},
                "conversation_id": {"type": "string", "description": "Related conversation ID"}
            },
            "required": ["file_path", "description"]
        }
    },

    "get_project_continuity": {
        "description": "Get context to continue development work",
        "parameters": {
            "type": "object",
            "properties": {
                "workspace_path": {"type": "string", "description": "Workspace path"},
                "limit": {"type": "integer", "description": "Context items", "default": 5}
            }
        }
    },

    "get_current_time": {
        "description": "Get the current time and date for appointment awareness",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "description": "Optional format string for the time/date", "default": "ISO"}
            }
        }
    }
}


async def main():
    """Run the MCP server"""
    server = PersistentAIMemoryMCPServer()
    
    # Start file monitoring
    await server.memory_system.start_file_monitoring()
    
    print("🧠 Persistent AI Memory System - MCP Server Started")
    print("🔧 Tool call logging and reflection enabled") 
    print("� Automatic database maintenance enabled (3-hour intervals)")
    print("�📁 File monitoring active")
    print("🌐 Ready for MCP client connections")
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MCP server...")
        await server.memory_system.stop_file_monitoring()
        await server.cleanup()  # Clean up maintenance task


if __name__ == "__main__":
    asyncio.run(main())
