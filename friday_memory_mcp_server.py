#!/usr/bin/env python3
"""
Friday Memory MCP Server

Acts as an interface layer between MCP clients (VS Code, LM Studio, Ollama UIs)
and the Friday Memory System. Provides standardized tools for memory operations
while maintaining client-specific access controls.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

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

# Local imports (will be implemented)
from friday_memory_system import FridayMemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FridayMemoryMCPServer:
    """MCP Server for Friday's Memory System"""
    
    def __init__(self):
        self.memory_system = FridayMemorySystem()
        self.server = Server("friday-memory")
        self.client_context = {}  # Track client-specific context
        self._maintenance_task = None  # Background maintenance task
        
        # Register MCP handlers
        self._register_handlers()
        
        # Start automatic maintenance
        self._start_automatic_maintenance()
    
    def _register_handlers(self):
        """Register MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools based on client context"""
            return await self._get_client_tools()
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Execute tool based on client and parameters"""
            return await self._execute_tool(name, arguments or {})
    
    async def _get_client_tools(self) -> List[Tool]:
        """Return tools available to the current client"""
        
        # Common tools available to all clients
        common_tools = [
            Tool(
                name="search_memories",
                description="Search memories using semantic similarity with importance and type filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10},
                        "database_filter": {"type": "string", "description": "Filter by database type", "enum": ["conversations", "ai_memories", "schedule", "all"], "default": "all"},
                        "min_importance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Minimum importance level to include (1-10)"},
                        "max_importance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum importance level to include (1-10)"},
                        "memory_type": {"type": "string", "description": "Filter by memory type (e.g., 'safety', 'preference', 'skill', 'general')"}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="store_conversation",
                description="Store conversation automatically",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Conversation content"},
                        "role": {"type": "string", "description": "Role (user/assistant)"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "metadata": {"type": "object", "description": "Additional metadata"}
                    },
                    "required": ["content", "role"]
                }
            ),
            Tool(
                name="create_memory",
                description="Create a curated memory entry",
                inputSchema={
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
            ),
            Tool(
                name="update_memory",
                description="Update an existing curated memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID to update"},
                        "content": {"type": "string", "description": "Updated content"},
                        "importance_level": {"type": "integer", "description": "Updated importance"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Updated tags"}
                    },
                    "required": ["memory_id"]
                }
            ),
            Tool(
                name="create_appointment",
                description="Create an appointment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Appointment title"},
                        "description": {"type": "string", "description": "Appointment description"},
                        "scheduled_datetime": {"type": "string", "description": "ISO format datetime"},
                        "location": {"type": "string", "description": "Location"}
                    },
                    "required": ["title", "scheduled_datetime"]
                }
            ),
            Tool(
                name="create_reminder",
                description="Create a reminder",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Reminder content"},
                        "due_datetime": {"type": "string", "description": "ISO format datetime"},
                        "priority_level": {"type": "integer", "description": "Priority (1-10)", "default": 5}
                    },
                    "required": ["content", "due_datetime"]
                }
            ),
            Tool(
                name="get_recent_context",
                description="Get recent conversation context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of recent items", "default": 5},
                        "session_id": {"type": "string", "description": "Specific session ID"}
                    }
                }
            ),
            Tool(
                name="get_system_health",
                description="Get comprehensive system health, statistics, and database status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="get_tool_usage_summary",
                description="Get AI tool usage summary and insights for self-reflection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "description": "Days to analyze", "default": 7},
                        "client_id": {"type": "string", "description": "Specific client ID to analyze"}
                    }
                }
            ),
            Tool(
                name="reflect_on_tool_usage",
                description="AI self-reflection on tool usage patterns and effectiveness",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "days": {"type": "integer", "description": "Days to analyze", "default": 7},
                        "client_id": {"type": "string", "description": "Specific client ID to analyze"}
                    }
                }
            ),
            Tool(
                name="get_ai_insights",
                description="Get recent AI self-reflection insights and patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of insights", "default": 5},
                        "insight_type": {"type": "string", "description": "Type of insight to filter"}
                    }
                }
            )
        ]
        
        # VS Code specific tools
        vscode_tools = [
            Tool(
                name="save_development_session",
                description="Save VS Code development session context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_path": {"type": "string", "description": "Workspace path"},
                        "active_files": {"type": "array", "items": {"type": "string"}, "description": "Active files"},
                        "git_branch": {"type": "string", "description": "Current git branch"},
                        "session_summary": {"type": "string", "description": "Session summary"}
                    },
                    "required": ["workspace_path"]
                }
            ),
            Tool(
                name="store_project_insight",
                description="Store development insight or decision",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "insight_type": {"type": "string", "description": "Type of insight"},
                        "content": {"type": "string", "description": "Insight content"},
                        "related_files": {"type": "array", "items": {"type": "string"}, "description": "Related files"},
                        "importance_level": {"type": "integer", "description": "Importance (1-10)", "default": 5}
                    },
                    "required": ["content"]
                }
            ),
            Tool(
                name="search_project_history",
                description="Search VS Code project development history",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="link_code_context",
                description="Link conversation to specific code context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File path"},
                        "function_name": {"type": "string", "description": "Function name"},
                        "description": {"type": "string", "description": "Context description"},
                        "conversation_id": {"type": "string", "description": "Related conversation ID"}
                    },
                    "required": ["file_path", "description"]
                }
            ),
            Tool(
                name="get_project_continuity",
                description="Get context to continue development work",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace_path": {"type": "string", "description": "Workspace path"},
                        "limit": {"type": "integer", "description": "Context items", "default": 5}
                    }
                }
            )
        ]
        
        # Determine client type and return appropriate tools
        # For now, return all tools - we'll add client detection later
        return common_tools + vscode_tools
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Execute the requested tool with logging for AI self-reflection"""
        
        import time
        
        # Start timing and get client info
        start_time = time.perf_counter()
        client_id = self.client_context.get("current_client", "unknown")
        
        try:
            # Route to appropriate handler
            if tool_name == "search_memories":
                result = await self.memory_system.search_memories(**arguments)
            elif tool_name == "store_conversation":
                result = await self.memory_system.store_conversation(**arguments)
            elif tool_name == "create_memory":
                result = await self.memory_system.create_memory(**arguments)
            elif tool_name == "update_memory":
                result = await self.memory_system.update_memory(**arguments)
            elif tool_name == "create_appointment":
                result = await self.memory_system.create_appointment(**arguments)
            elif tool_name == "create_reminder":
                result = await self.memory_system.create_reminder(**arguments)
            elif tool_name == "get_recent_context":
                result = await self.memory_system.get_recent_context(**arguments)
            elif tool_name == "get_system_health":
                result = await self.memory_system.get_system_health()
            elif tool_name == "save_development_session":
                result = await self.memory_system.save_development_session(**arguments)
            elif tool_name == "store_project_insight":
                result = await self.memory_system.store_project_insight(**arguments)
            elif tool_name == "search_project_history":
                result = await self.memory_system.search_project_history(**arguments)
            elif tool_name == "link_code_context":
                result = await self.memory_system.link_code_context(**arguments)
            elif tool_name == "get_project_continuity":
                result = await self.memory_system.get_project_continuity(**arguments)
            elif tool_name == "get_tool_usage_summary":
                result = await self.memory_system.get_tool_usage_summary(**arguments)
            elif tool_name == "reflect_on_tool_usage":
                result = await self.memory_system.reflect_on_tool_usage(**arguments)
            elif tool_name == "get_ai_insights":
                result = await self.memory_system.get_ai_insights(**arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Calculate execution time and log successful call
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Log tool call for AI self-reflection (async, don't wait)
            asyncio.create_task(self.memory_system.log_tool_call(
                client_id=client_id,
                tool_name=tool_name,
                parameters=arguments,
                execution_time_ms=execution_time_ms,
                status="success",
                result=result
            ))
            
            # Create the text content
            result_text = json.dumps(result, indent=2, default=str)
            text_content = TextContent(type="text", text=result_text)
            
            return CallToolResult(content=[text_content])
            
        except Exception as e:
            # Calculate execution time and log failed call
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Log tool call failure for AI self-reflection (async, don't wait)
            asyncio.create_task(self.memory_system.log_tool_call(
                client_id=client_id,
                tool_name=tool_name,
                parameters=arguments,
                execution_time_ms=execution_time_ms,
                status="error",
                error_message=str(e)
            ))
            
            logger.error(f"Error executing tool {tool_name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    def _start_automatic_maintenance(self):
        """Start automatic database maintenance background task"""
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("🔧 Automatic database maintenance started")
    
    async def _maintenance_loop(self):
        """Background loop for automatic database maintenance"""
        # Wait a bit after startup before first maintenance
        await asyncio.sleep(300)  # 5 minutes initial delay
        
        while True:
            try:
                logger.info("🧹 Running automatic database maintenance...")
                result = await self.memory_system.run_database_maintenance()
                
                # Log maintenance results
                if result.get("success"):
                    logger.info(f"✅ Automatic maintenance completed - optimized {len(result.get('optimization_results', {}))} databases")
                else:
                    logger.warning(f"⚠️ Automatic maintenance had issues: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ Automatic maintenance failed: {e}")
            
            # Wait 3 hours before next maintenance
            await asyncio.sleep(3 * 60 * 60)
    
    async def cleanup(self):
        """Cleanup resources when server stops"""
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            logger.info("🔧 Automatic maintenance stopped")


async def main():
    """Main entry point for the MCP server"""
    mcp_server = FridayMemoryMCPServer()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="friday-memory",
                    server_version="1.0.0",
                    capabilities=mcp_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    finally:
        # Cleanup when server stops
        await mcp_server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
