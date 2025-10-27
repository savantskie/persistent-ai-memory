#!/usr/bin/env python3
"""
Persistent AI Memory System - MCP Server

Acts as an interface layer between MCP clients (VS Code, LM Studio, Ollama UIs)
and the AI Memory System. Provides standardized tools for memory operations
while maintaining client-specific access controls.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import time
import warnings
import os
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
from ai_memory_core import PersistentAIMemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIMemoryMCPServer:
    """MCP Server for Friday's Memory System"""

    async def update_memory(self, memory_id: str, content: str = None, importance_level: int = None, tags: List[str] = None) -> Dict:
        """Update an existing memory"""
        
        success = await self.ai_memory_db.update_memory(memory_id, content, importance_level, tags)
        
        # If content was updated, regenerate embedding
        if content is not None:
            asyncio.create_task(self._add_embedding_to_memory(memory_id, content))
        
        return {
            "status": "success" if success else "error",
            "memory_id": memory_id
        }   
        
    def __init__(self):
        self.memory_system = PersistentAIMemorySystem()
        self.server = Server("ai-memory")
        self.client_context = {}  # Track client-specific context
        self._maintenance_task = None  # Background maintenance task
        
        # Enable debug logging for MCP server
        logging.getLogger("mcp.server").setLevel(logging.DEBUG)
        
        # Register MCP handlers
        self._register_handlers()
        
        # Start automatic maintenance
        self._start_automatic_maintenance()
        
        logger.info("AIMemoryMCPServer initialized successfully")
    
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
        logger.debug("Getting client tools")
        
        # Detect client type based on user agent or connection context
        client_type = self._detect_client_type()
        logger.info(f"Detected client type: {client_type}")
        
        try:
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
                description="Create an appointment, optionally recurring (e.g., weekly mental health appointments)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Appointment title"},
                        "description": {"type": "string", "description": "Appointment description"},
                        "scheduled_datetime": {"type": "string", "description": "ISO format datetime for first appointment"},
                        "location": {"type": "string", "description": "Location"},
                        "recurrence_pattern": {"type": "string", "description": "Recurrence pattern: 'daily', 'weekly', 'monthly', 'yearly'", "enum": ["daily", "weekly", "monthly", "yearly"]},
                        "recurrence_count": {"type": "integer", "description": "Number of appointments to create (including first), e.g., 12 for 12 weeks", "minimum": 1},
                        "recurrence_end_date": {"type": "string", "description": "End date for recurrences (ISO format), alternative to recurrence_count"}
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
                name="get_reminders",
                description="Get up to 5 active (uncompleted) reminders, sorted by due date.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of reminders to return", "default": 5}
                    }
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
            ),
            Tool(
                name="get_active_reminders",
                description="Get active (not completed) reminders",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of reminders to return", "default": 10},
                        "days_ahead": {"type": "integer", "description": "Only show reminders due within X days", "default": 30}
                    }
                }
            ),
            Tool(
                name="get_completed_reminders",
                description="Get recently completed reminders",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "description": "Look back X days", "default": 7}
                    }
                }
            ),
            Tool(
                name="complete_reminder",
                description="Mark a reminder as completed",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reminder_id": {"type": "string", "description": "ID of the reminder to complete"}
                    },
                    "required": ["reminder_id"]
                }
            ),
            Tool(
                name="reschedule_reminder",
                description="Update the due date of a reminder",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reminder_id": {"type": "string", "description": "ID of the reminder"},
                        "new_due_datetime": {"type": "string", "description": "New ISO datetime (e.g., 2025-08-03T14:00:00Z)"}
                    },
                    "required": ["reminder_id", "new_due_datetime"]
                }
            ),
            Tool(
                name="delete_reminder",
                description="Permanently delete a reminder",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reminder_id": {"type": "string", "description": "ID of the reminder to delete"}
                    },
                    "required": ["reminder_id"]
                }
            ),
            Tool(
                name="cancel_appointment",
                description="Cancel a scheduled appointment",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "string", "description": "ID of the appointment to cancel"}
                    },
                    "required": ["appointment_id"]
                }
            ),
            Tool(
                name="complete_appointment",
                description="Mark an appointment as completed",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "string", "description": "ID of the appointment to complete"}
                    },
                    "required": ["appointment_id"]
                }
            ),
            Tool(
                name="get_upcoming_appointments",
                description="Get upcoming appointments (not cancelled)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number to return", "default": 5},
                        "days_ahead": {"type": "integer", "description": "Only show within X days", "default": 30}
                    }
                }
            ),
            Tool(
                name="get_appointments",
                description="Get recent appointments, optionally filtered by date range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of appointments to return", "default": 5},
                        "days_ahead": {"type": "integer", "description": "Only show appointments scheduled within X days", "default": 30}
                    }
                }
            ),
            Tool(
                name="store_ai_reflection",
                description="Store an AI self-reflection/insight record (manual write)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Freeform write-up of the reflection"},
                        "reflection_type": {"type": "string", "description": "Category (e.g., tool_usage_analysis, memory, general)", "default": "general"},
                        "insights": {"type": "array", "items": {"type": "string"}, "description": "Bullet insights derived from the analysis"},
                        "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Recommended next actions"},
                        "confidence_level": {"type": "number", "description": "Confidence 0.0–1.0", "default": 0.7},
                        "source_period_days": {"type": "integer", "description": "Days of data this reflection summarizes"}
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="write_ai_insights",
                description="Alias of store_ai_reflection – write an AI self-reflection/insight record",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Freeform write-up of the reflection"},
                        "reflection_type": {"type": "string", "description": "Category (e.g., tool_usage_analysis, memory, general)", "default": "general"},
                        "insights": {"type": "array", "items": {"type": "string"}, "description": "Bullet insights derived from the analysis"},
                        "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Recommended next actions"},
                        "confidence_level": {"type": "number", "description": "Confidence 0.0–1.0", "default": 0.7},
                        "source_period_days": {"type": "integer", "description": "Days of data this reflection summarizes"}
                    },
                    "required": ["content"],
                    "additionalProperties": False
                }
            ),
            Tool(
                name="get_current_time",
                description="Get the current server time in ISO format (UTC and local)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            ),
            Tool(
                name="get_weather_open_meteo",
                description="Open-Meteo forecast (no API key). Defaults to Motley, MN and caches once per local day.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "latitude": {"type": ["number", "null"], "description": "Ignored unless override=True"},
                        "longitude": {"type": ["number", "null"], "description": "Ignored unless override=True"},
                        "timezone_str": {"type": ["string", "null"], "description": "Ignored unless override=True"},
                        "force_refresh": {"type": "boolean", "description": "Ignore same-day cache", "default": False},
                        "return_changes_only": {"type": "boolean", "description": "If true, return only a summary of changed fields for today.", "default": False},
                        "update_today": {"type": "boolean", "description": "If true (default), fetch and merge changes into today's file before returning.", "default": True},
                        "severe_update": {"type": "boolean", "description": "If true, shrink the update window to 30 minutes for severe weather.", "default": False}
                    }
                }
            )
        ]
        except Exception as e:
            logger.error(f"Error creating common tools: {e}")
            common_tools = []
        
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
        
        try:
            # Return appropriate tools based on client type
            if client_type == "sillytavern":
                # SillyTavern gets memory tools + character/roleplay specific tools
                sillytavern_tools = [
                    Tool(
                        name="get_character_context",
                        description="Get relevant context about characters from memory",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "character_name": {"type": "string", "description": "Character name to search for"},
                                "context_type": {"type": "string", "description": "Type of context (personality, relationships, history)"},
                                "limit": {"type": "integer", "description": "Max results", "default": 5}
                            },
                            "required": ["character_name"]
                        }
                    ),
                    Tool(
                        name="store_roleplay_memory",
                        description="Store important roleplay moments or character developments",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "character_name": {"type": "string", "description": "Character involved"},
                                "event_description": {"type": "string", "description": "What happened"},
                                "importance_level": {"type": "integer", "description": "Importance (1-10)", "default": 5},
                                "tags": {"type": "array", "items": {"type": "string"}, "description": "Relevant tags"}
                            },
                            "required": ["character_name", "event_description"]
                        }
                    ),
                    Tool(
                        name="search_roleplay_history",
                        description="Search past roleplay interactions and character development",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "character_name": {"type": "string", "description": "Focus on specific character"},
                                "limit": {"type": "integer", "description": "Max results", "default": 10}
                            },
                            "required": ["query"]
                        }
                    )
                ]
                return common_tools + sillytavern_tools
            
            elif client_type == "vscode":
                # VS Code gets development-specific tools
                return common_tools + vscode_tools
            
            else:
                # Default: LM Studio, Ollama UIs, etc. get core memory tools only
                return common_tools
                
        except Exception as e:
            logger.error(f"Error combining tool lists: {e}")
            return []

    def _detect_client_type(self) -> str:
        """Detect the type of MCP client connecting"""
        # Detect the type of MCP client connecting
        client_type = "unknown"
        if "VS Code" in os.getenv("USER_AGENT", ""):
            client_type = "vscode"
        elif "LM Studio" in os.getenv("USER_AGENT", ""):
            client_type = "lm_studio"
        elif "Silly Tavern" in os.getenv("USER_AGENT", ""):
            client_type = "sillytavern"
        elif "Ollama" in os.getenv("USER_AGENT", ""):
            client_type = "ollama"
        logger.info(f"Detected client type: {client_type}")
        return client_type
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Execute the requested tool with logging for AI self-reflection"""
        
        import time
        
        # Start timing and get client info
        start_time = time.perf_counter()
        client_id = self.client_context.get("current_client", "unknown")
        
        try:
            logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
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
            elif tool_name == "get_reminders":
                limit = arguments.get("limit", 5)
                reminders = await self.memory_system.get_active_reminders()
                result = reminders[:limit] if reminders else []
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
            elif tool_name == "get_active_reminders":
                result = await self.memory_system.get_active_reminders(**arguments)
            elif tool_name == "get_completed_reminders":
                result = await self.memory_system.get_completed_reminders(**arguments)
            elif tool_name == "complete_reminder":
                result = await self.memory_system.complete_reminder(**arguments)
            elif tool_name == "reschedule_reminder":
                result = await self.memory_system.reschedule_reminder(**arguments)
            elif tool_name == "delete_reminder":
                result = await self.memory_system.delete_reminder(**arguments)
            elif tool_name == "cancel_appointment":
                result = await self.memory_system.cancel_appointment(**arguments)
            elif tool_name == "complete_appointment":
                result = await self.memory_system.complete_appointment(**arguments)
            elif tool_name == "get_upcoming_appointments":
                result = await self.memory_system.get_upcoming_appointments(**arguments)
            elif tool_name == "get_appointments":
                result = await self.memory_system.get_appointments(**arguments)
            elif tool_name == "store_ai_reflection" or tool_name == "write_ai_insights":
                result = await self.memory_system.store_ai_reflection(**arguments)
            elif tool_name == "get_current_time":
                result = await self.memory_system.get_current_time()
            elif tool_name == "get_weather_open_meteo":
                result = await self.memory_system.get_weather_open_meteo(**arguments)
            # SillyTavern-specific tools
            elif tool_name == "get_character_context":
                result = await self.memory_system.get_character_context(**arguments)
            elif tool_name == "store_roleplay_memory":
                result = await self.memory_system.store_roleplay_memory(**arguments)
            elif tool_name == "search_roleplay_history":
                result = await self.memory_system.search_roleplay_history(**arguments)
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
            
            # Format the result as a proper TextContent object
            if isinstance(result, (dict, list)):
                result_text = json.dumps(result, indent=2, default=str)
            else:
                result_text = str(result)
            
            text_content = {
                "type": "text",
                "text": result_text,
                "highlights": None,
                "meta": None
            }
            
            return {
                "content": [text_content],
                "success": True,
                "structuredContent": None,
                "isError": False,
                "meta": None
            }
            
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
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: {str(e)}",
                    "highlights": None,
                    "meta": None
                }],
                "success": False,
                "structuredContent": None,
                "isError": True,
                "meta": None
            }
    
    def _start_automatic_maintenance(self):
        """Start automatic database maintenance background task"""
        try:
            loop = asyncio.get_running_loop()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            self._maintenance_task = loop.create_task(self._maintenance_loop())

        except RuntimeError:
            logger.warning("Event loop not running. Call `_start_automatic_maintenance()` after loop starts.")
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
    



async def start_http_server(mcp_server: AIMemoryMCPServer, host: str = "127.0.0.1", port: int = 11434):
    """Start the HTTP API server if needed"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        
        app = FastAPI(title="Friday Memory API")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "server": "ai-memory"}
            
        # Start server without blocking
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        return await server.serve()
    except ImportError:
        logger.info("FastAPI not installed - HTTP API disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to start HTTP server: {e}")
        return None

async def main():
    """Main entry point for the MCP server"""
    logger.info("AI Memory MCP Server starting...")
    
    # Set debug logging for MCP components
    logging.getLogger("mcp").setLevel(logging.DEBUG)
    logging.getLogger("mcp.server").setLevel(logging.DEBUG)
    
    mcp_server = AIMemoryMCPServer()
    
    logger.debug("Server initialized, starting stdio interface for LM Studio...")
    
    try:
        # Only use stdio for LM Studio - no HTTP server needed
        logger.info("Waiting for stdio connection from LM Studio...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("LM Studio connected via stdio")
            await mcp_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="ai-memory",
                    server_version="1.0.0",
                    capabilities=mcp_server.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await mcp_server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
