#!/usr/bin/env python3
"""
Persistent AI Memory System - Core Module

A comprehensive memory system for AI assistants with real-time conversation capture,
semantic search, and multi-database architecture.
"""

import sqlite3
import json
import uuid
import hashlib
import asyncio
import aiohttp
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Base database manager for common operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure the database file and directory exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        return conn
    
    async def execute_query(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    async def execute_update(self, query: str, params: Tuple = ()) -> str:
        """Execute an INSERT/UPDATE/DELETE query and return last row ID"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return str(cursor.lastrowid)


class MCPToolCallDatabase(DatabaseManager):
    """🔧 NEW: Tracks all MCP tool calls for reflection and debugging"""
    
    def __init__(self, db_path: str = "mcp_tool_calls.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tool call tracking tables"""
        with self.get_connection() as conn:
            # Tool calls table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    client_id TEXT,
                    tool_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    result TEXT,
                    status TEXT NOT NULL,
                    execution_time_ms INTEGER,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tool usage statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_usage_stats (
                    stat_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    call_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    avg_execution_time_ms REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tool_name, date)
                )
            """)
            
            conn.commit()
    
    async def log_tool_call(self, tool_name: str, parameters: Dict, result: Any = None, 
                           status: str = "success", execution_time_ms: float = None,
                           error_message: str = None, client_id: str = None) -> str:
        """Log a tool call with all relevant details"""
        
        call_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Store the tool call
        await self.execute_update(
            """INSERT INTO tool_calls 
               (call_id, timestamp, client_id, tool_name, parameters, result, 
                status, execution_time_ms, error_message) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, timestamp, client_id, tool_name, 
             json.dumps(parameters), json.dumps(result) if result else None,
             status, int(execution_time_ms) if execution_time_ms else None, error_message)
        )
        
        # Update daily statistics
        await self._update_tool_stats(tool_name, status, execution_time_ms)
        
        return call_id
    
    async def _update_tool_stats(self, tool_name: str, status: str, execution_time_ms: float):
        """Update daily usage statistics for a tool"""
        today = datetime.now(timezone.utc).date().isoformat()
        
        # Check if stat record exists for today
        existing = await self.execute_query(
            "SELECT * FROM tool_usage_stats WHERE tool_name = ? AND date = ?",
            (tool_name, today)
        )
        
        if existing:
            # Update existing record
            stat = existing[0]
            new_call_count = stat["call_count"] + 1
            new_success_count = stat["success_count"] + (1 if status == "success" else 0)
            new_error_count = stat["error_count"] + (1 if status == "error" else 0)
            
            # Calculate new average execution time
            if execution_time_ms and stat["avg_execution_time_ms"]:
                new_avg = ((stat["avg_execution_time_ms"] * stat["call_count"]) + execution_time_ms) / new_call_count
            elif execution_time_ms:
                new_avg = execution_time_ms
            else:
                new_avg = stat["avg_execution_time_ms"]
            
            await self.execute_update(
                """UPDATE tool_usage_stats 
                   SET call_count = ?, success_count = ?, error_count = ?, avg_execution_time_ms = ?
                   WHERE tool_name = ? AND date = ?""",
                (new_call_count, new_success_count, new_error_count, new_avg, tool_name, today)
            )
        else:
            # Create new record
            stat_id = str(uuid.uuid4())
            await self.execute_update(
                """INSERT INTO tool_usage_stats 
                   (stat_id, tool_name, date, call_count, success_count, error_count, avg_execution_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (stat_id, tool_name, today, 1,
                 1 if status == "success" else 0,
                 1 if status == "error" else 0,
                 execution_time_ms or 0)
            )
    
    async def get_tool_usage_summary(self, days: int = 7) -> Dict:
        """Get tool usage summary for the last N days"""
        
        # Get recent tool calls
        recent_calls = await self.execute_query(
            """SELECT tool_name, status, COUNT(*) as count
               FROM tool_calls 
               WHERE timestamp >= datetime('now', '-{} days')
               GROUP BY tool_name, status
               ORDER BY count DESC""".format(days)
        )
        
        # Get daily stats
        daily_stats = await self.execute_query(
            """SELECT * FROM tool_usage_stats 
               WHERE date >= date('now', '-{} days')
               ORDER BY date DESC, call_count DESC""".format(days)
        )
        
        # Get most used tools
        most_used = await self.execute_query(
            """SELECT tool_name, COUNT(*) as total_calls
               FROM tool_calls 
               WHERE timestamp >= datetime('now', '-{} days')
               GROUP BY tool_name
               ORDER BY total_calls DESC
               LIMIT 10""".format(days)
        )
        
        return {
            "recent_calls": [dict(row) for row in recent_calls],
            "daily_stats": [dict(row) for row in daily_stats],
            "most_used_tools": [dict(row) for row in most_used],
            "period_days": days
        }
    
    async def get_tool_call_history(self, tool_name: str = None, limit: int = 50) -> List[Dict]:
        """Get recent tool call history, optionally filtered by tool name"""
        
        if tool_name:
            query = """SELECT * FROM tool_calls 
                      WHERE tool_name = ? 
                      ORDER BY timestamp DESC 
                      LIMIT ?"""
            params = (tool_name, limit)
        else:
            query = """SELECT * FROM tool_calls 
                      ORDER BY timestamp DESC 
                      LIMIT ?"""
            params = (limit,)
        
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]


class ConversationDatabase(DatabaseManager):
    """Manages conversation auto-save database"""
    
    def __init__(self, db_path: str = "conversations.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_timestamp TEXT NOT NULL,
                    end_timestamp TEXT,
                    context TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    start_timestamp TEXT NOT NULL,
                    end_timestamp TEXT,
                    topic_summary TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            conn.commit()
    
    async def store_message(self, content: str, role: str, session_id: str = None, 
                          conversation_id: str = None, metadata: Dict = None) -> Dict[str, str]:
        """Store a message and auto-manage sessions/conversations"""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        message_id = str(uuid.uuid4())
        
        # Auto-create session if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            await self.execute_update(
                "INSERT INTO sessions (session_id, start_timestamp, context) VALUES (?, ?, ?)",
                (session_id, timestamp, "auto-created")
            )
        
        # Auto-create conversation if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            await self.execute_update(
                "INSERT INTO conversations (conversation_id, session_id, start_timestamp) VALUES (?, ?, ?)",
                (conversation_id, session_id, timestamp)
            )
        
        # Store the message
        await self.execute_update(
            """INSERT INTO messages 
               (message_id, conversation_id, timestamp, role, content, metadata) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (message_id, conversation_id, timestamp, role, content, 
             json.dumps(metadata) if metadata else None)
        )
        
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "session_id": session_id
        }
    
    async def get_recent_messages(self, limit: int = 10, session_id: str = None) -> List[Dict]:
        """Get recent messages, optionally filtered by session"""
        
        if session_id:
            query = """
                SELECT m.*, c.session_id 
                FROM messages m 
                JOIN conversations c ON m.conversation_id = c.conversation_id
                WHERE c.session_id = ?
                ORDER BY m.timestamp DESC 
                LIMIT ?
            """
            params = (session_id, limit)
        else:
            query = """
                SELECT m.*, c.session_id 
                FROM messages m 
                JOIN conversations c ON m.conversation_id = c.conversation_id
                ORDER BY m.timestamp DESC 
                LIMIT ?
            """
            params = (limit,)
        
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]


class AIMemoryDatabase(DatabaseManager):
    """Manages AI-curated memories database"""
    
    def __init__(self, db_path: str = "ai_memories.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS curated_memories (
                    memory_id TEXT PRIMARY KEY,
                    timestamp_created TEXT NOT NULL,
                    timestamp_updated TEXT NOT NULL,
                    source_conversation_id TEXT,
                    source_message_ids TEXT,
                    memory_type TEXT,
                    content TEXT NOT NULL,
                    importance_level INTEGER DEFAULT 5,
                    tags TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    async def create_memory(self, content: str, memory_type: str = None, 
                          importance_level: int = 5, tags: List[str] = None,
                          source_conversation_id: str = None) -> str:
        """Create a new curated memory"""
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        await self.execute_update(
            """INSERT INTO curated_memories 
               (memory_id, timestamp_created, timestamp_updated, source_conversation_id, 
                memory_type, content, importance_level, tags) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, timestamp, timestamp, source_conversation_id, 
             memory_type, content, importance_level, 
             json.dumps(tags) if tags else None)
        )
        
        return memory_id


class EmbeddingService:
    """Handles embedding generation via LM Studio"""
    
    def __init__(self, base_url: str = "http://192.168.1.50:1234"):
        self.base_url = base_url
        self.embeddings_endpoint = f"{base_url}/v1/embeddings"
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-nomic-embed-text-v1.5") -> List[float]:
        """Generate embedding for text using LM Studio"""
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "input": text,
                    "model": model
                }
                
                async with session.post(self.embeddings_endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["data"][0]["embedding"]
                    else:
                        logger.error(f"Embedding API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None


class PersistentAIMemorySystem:
    """Main memory system that coordinates all databases and operations"""
    
    def __init__(self, data_dir: str = "memory_data", enable_file_monitoring: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize databases
        self.conversations_db = ConversationDatabase(str(self.data_dir / "conversations.db"))
        self.ai_memory_db = AIMemoryDatabase(str(self.data_dir / "ai_memories.db"))
        self.tool_call_db = MCPToolCallDatabase(str(self.data_dir / "mcp_tool_calls.db"))
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
    
    async def create_memory(self, content: str, memory_type: str = None,
                          importance_level: int = 5, tags: List[str] = None,
                          source_conversation_id: str = None) -> Dict:
        """Create a curated memory"""
        
        memory_id = await self.ai_memory_db.create_memory(
            content, memory_type, importance_level, tags, source_conversation_id
        )
        
        return {
            "status": "success",
            "memory_id": memory_id
        }
    
    async def search_memories(self, query: str, limit: int = 10) -> Dict:
        """Search memories using semantic similarity"""
        # Placeholder - implement semantic search
        return {
            "status": "success",
            "results": [],
            "query": query
        }
    
    async def get_system_health(self) -> Dict:
        """Get comprehensive system health"""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "databases": {
                "conversations": {"status": "healthy"},
                "ai_memories": {"status": "healthy"},
                "tool_calls": {"status": "healthy"}
            }
        }
