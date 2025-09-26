#!/usr/bin/env python3
"""
Persistent AI Memory System - Core Module

A comprehensive memory system designed for long-term persistence, semantic search,
and AI assistant augmentation. This standalone version includes all core functionality
with enhanced features for broader use.

Key Features:
- Specialized Database Architecture:
  * Conversations with automatic session management
  * AI-curated memories with importance levels and tags
  * Appointment and reminder scheduling
  * VS Code project context and development tracking
  * MCP tool call logging with AI self-reflection

- Advanced Search and Retrieval:
  * Vector-based semantic search across all databases
  * Project-specific search capabilities
  * Code context linking and retrieval
  * Importance-weighted memory search
  * Fallback text-based search when embeddings unavailable

- Enhanced AI Capabilities:
  * Automatic embedding generation
  * Usage pattern detection and analysis
  * AI self-reflection on tool usage
  * Pattern-based recommendations
  * Confidence scoring for insights

- Real-time Monitoring:
  * Conversation file monitoring
  * Multiple chat source support (VS Code, LM Studio, ChatGPT, etc.)
  * Deduplication across sources
  * MCP server integration
  
- System Management:
  * Comprehensive health monitoring
  * Automated database maintenance
  * Error tracking and logging
  * Performance optimization

- Development Tools:
  * Project continuity tracking
  * Code context management
  * Development session history
  * Insight storage and retrieval

All timestamps are stored in the local timezone using ISO format. This ensures
that timestamps are correctly displayed and interpreted in the local time context.

For usage examples and integration guides, see the documentation in /docs.
"""

import asyncio
import sqlite3
import json
import uuid
import logging
import aiohttp
import numpy as np
import hashlib
import os
import re
import time
import socket
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

# Get local timezone
def get_local_timezone() -> ZoneInfo:
    """Get local timezone based on system settings"""
    try:
        import time
        return ZoneInfo(time.tzname[0])
    except:
        # Fallback to a common timezone if detection fails
        return ZoneInfo("America/Chicago")  # Central Time fallback
    
def get_current_timestamp() -> str:
    """Get current timestamp in local timezone ISO format"""
    return datetime.now(get_local_timezone()).isoformat()
    
def datetime_to_local_isoformat(dt: datetime) -> str:
    """Convert any datetime to local timezone ISO format"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_local_timezone())
    return dt.astimezone(get_local_timezone()).isoformat()

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import sqlite3
import json
import uuid
import hashlib
import asyncio
import aiohttp
import logging
import os
import re
import time
import socket
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from database_maintenance import DatabaseMaintenance
from settings import get_settings

# Configure logging with minimal output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# Only show important messages and errors
logger.setLevel(logging.WARNING)


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
    
    async def execute_update(self, query: str, params: Tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return number of affected rows"""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                logger.error(f"Database error: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                raise
                
    def parse_timestamp(self, timestamp: Union[str, int, float, None], fallback: Optional[datetime] = None) -> str:
        """Parse various timestamp formats into ISO format string.
        
        Args:
            timestamp: Input timestamp (string, unix timestamp, or None)
            fallback: Optional fallback datetime if parsing fails
            
        Returns:
            ISO format datetime string
        """
        if not timestamp:
            return (fallback or datetime.now(get_local_timezone())).isoformat()
            
        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                dt = datetime.fromtimestamp(timestamp, timezone.utc)
            elif isinstance(timestamp, str):
                # Try various string formats
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    # Try parsing with dateutil as fallback
                    from dateutil import parser
                    dt = parser.parse(timestamp)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
            else:
                raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")
                
            return dt.isoformat()
            
        except Exception as e:
            logger.warning(f"Error parsing timestamp {timestamp}: {e}")
            return (fallback or datetime.now(get_local_timezone())).isoformat()


class MCPToolCallDatabase(DatabaseManager):
    """Tracks all MCP tool calls for reflection and debugging"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_settings().mcp_db_path)
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
            
            # AI reflections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_reflections (
                    reflection_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    reflection_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    insights TEXT,
                    recommendations TEXT,
                    confidence_level REAL DEFAULT 0.5,
                    source_period_days INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Usage patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    analysis_period_days INTEGER NOT NULL,
                    confidence_score REAL DEFAULT 0.5,
                    supporting_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def log_tool_call(self, tool_name: str, parameters: Dict, result: Any = None, 
                           status: str = "success", execution_time_ms: float = None,
                           error_message: str = None, client_id: str = None) -> str:
        """Log a tool call with all relevant details"""
        
        call_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
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
        today = datetime.now(get_local_timezone()).date().isoformat()
        
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
        
    async def store_ai_reflection(self, reflection_type: str, content: str,
                                insights: List[str] = None, recommendations: List[str] = None,
                                confidence_level: float = 0.5, source_period_days: int = None) -> str:
        """Store AI self-reflection on tool usage and patterns.
        
        Args:
            reflection_type: Type of reflection (e.g., usage_patterns, performance, suggestions)
            content: Main reflection content
            insights: List of specific insights gained
            recommendations: List of action recommendations
            confidence_level: Confidence in the reflection (0-1)
            source_period_days: Period of data analyzed
            
        Returns:
            str: Reflection ID
        """
        reflection_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO ai_reflections 
               (reflection_id, timestamp, reflection_type, content, insights, 
                recommendations, confidence_level, source_period_days)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (reflection_id, timestamp, reflection_type, content,
             json.dumps(insights) if insights else None,
             json.dumps(recommendations) if recommendations else None,
             confidence_level, source_period_days)
        )
        
        return reflection_id
        
    async def store_usage_pattern(self, pattern_type: str, insight: str, 
                                analysis_period_days: int, confidence_score: float = 0.5,
                                supporting_data: Dict = None) -> str:
        """Store identified usage pattern from AI analysis.
        
        Args:
            pattern_type: Type of usage pattern
            insight: Description of the pattern
            analysis_period_days: Period analyzed to identify pattern
            confidence_score: Confidence in pattern (0-1)
            supporting_data: Additional data supporting the pattern
            
        Returns:
            str: Pattern ID
        """
        pattern_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO usage_patterns
               (pattern_id, timestamp, pattern_type, insight, analysis_period_days,
                confidence_score, supporting_data)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pattern_id, timestamp, pattern_type, insight, analysis_period_days,
             confidence_score, json.dumps(supporting_data) if supporting_data else None)
        )
        
        return pattern_id
        
    async def get_recent_reflections(self, limit: int = 5, reflection_type: str = None) -> List[Dict]:
        """Get recent AI reflections, optionally filtered by type.
        
        Args:
            limit: Maximum number of reflections to return
            reflection_type: Optional filter by reflection type
            
        Returns:
            List of reflection entries
        """
        if reflection_type:
            query = """SELECT * FROM ai_reflections
                      WHERE reflection_type = ?
                      ORDER BY timestamp DESC
                      LIMIT ?"""
            params = (reflection_type, limit)
        else:
            query = """SELECT * FROM ai_reflections
                      ORDER BY timestamp DESC
                      LIMIT ?"""
            params = (limit,)
            
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]


class ConversationDatabase(DatabaseManager):
    """Manages conversation auto-save database"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_settings().conversations_db_path)
        super().__init__(db_path)
        self.initialize_tables()

    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # --- Migration logic for messages table ---
            expected_columns = [
                'message_id', 'conversation_id', 'timestamp', 'role', 'content', 'source_type',
                'source_id', 'source_url', 'source_metadata', 'sync_status', 'last_sync',
                'metadata', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(messages)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in expected_columns:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating messages table to new schema!")
                old_rows = conn.execute("SELECT * FROM messages").fetchall()
                conn.execute("DROP TABLE IF EXISTS messages")
                conn.execute("""
                    CREATE TABLE messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_type TEXT,
                        source_id TEXT,
                        source_url TEXT,
                        source_metadata TEXT,
                        sync_status TEXT,
                        last_sync TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in expected_columns:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO messages ({', '.join(expected_columns)}) VALUES ({', '.join(['?' for _ in expected_columns])})",
                        tuple(row_dict[col] for col in expected_columns)
                    )
                print(f"Restored {len(old_rows)} messages after migration.")
            else:
                # Create table if not exists (normal path)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_type TEXT,
                        source_id TEXT,
                        source_url TEXT,
                        source_metadata TEXT,
                        sync_status TEXT,
                        last_sync TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)

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

            # Source metadata table for tracking chat sources
            conn.execute("""
                CREATE TABLE IF NOT EXISTS source_tracking (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_path TEXT,
                    last_check TEXT NOT NULL,
                    last_sync TEXT,
                    status TEXT NOT NULL,
                    error_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cross-source relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_conversation_id TEXT NOT NULL,
                    related_conversation_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_conversation_id) REFERENCES conversations (conversation_id),
                    FOREIGN KEY (related_conversation_id) REFERENCES conversations (conversation_id)
                )
            """)

            conn.commit()
    
    async def store_message(self, content: str, role: str, session_id: str = None, 
                          conversation_id: str = None, metadata: Dict = None) -> Dict[str, str]:
        """Store a message and auto-manage sessions/conversations with duplicate detection"""
        timestamp = get_current_timestamp()
        message_id = str(uuid.uuid4())

        # Advanced duplicate detection: check for existing message with same content, role, and session in last hour
        if session_id:
            existing = await self.execute_query(
                """SELECT message_id FROM messages 
                   WHERE conversation_id IN (
                       SELECT conversation_id FROM conversations WHERE session_id = ?
                   ) AND role = ? AND content = ? 
                   AND datetime(timestamp) > datetime('now', '-1 hour')""",
                (session_id, role, content)
            )
            if existing:
                print(f"Skipping duplicate message in session {session_id}")
                return {
                    "message_id": existing[0]["message_id"],
                    "conversation_id": None,
                    "session_id": session_id,
                    "duplicate": True
                }

        # Auto-create session if not provided or doesn't exist
        if not session_id:
            session_id = str(uuid.uuid4())
            await self.execute_update(
                "INSERT INTO sessions (session_id, start_timestamp, context) VALUES (?, ?, ?)",
                (session_id, timestamp, "auto-created")
            )
        else:
            existing_session = await self.execute_query(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            if not existing_session:
                await self.execute_update(
                    "INSERT INTO sessions (session_id, start_timestamp, context) VALUES (?, ?, ?)",
                    (session_id, timestamp, "imported-session")
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
            "session_id": session_id,
            "duplicate": False
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
    """Manages AI-curated memories database with enhanced operations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_settings().ai_memories_db_path)
        super().__init__(db_path)
        self.initialize_tables()

    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            expected_columns = [
                'memory_id', 'timestamp_created', 'timestamp_updated', 'source_conversation_id',
                'source_message_ids', 'memory_type', 'content', 'importance_level', 'tags',
                'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(curated_memories)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in expected_columns:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating curated_memories table to new schema!")
                old_rows = conn.execute("SELECT * FROM curated_memories").fetchall()
                conn.execute("DROP TABLE IF EXISTS curated_memories")
                conn.execute("""
                    CREATE TABLE curated_memories (
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
                for row in old_rows:
                    row_dict = dict(row)
                    for col in expected_columns:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO curated_memories ({', '.join(expected_columns)}) VALUES ({', '.join(['?' for _ in expected_columns])})",
                        tuple(row_dict[col] for col in expected_columns)
                    )
                print(f"Restored {len(old_rows)} curated memories after migration.")
            else:
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
        """Create a new curated memory with duplicate detection"""
        memory_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()

        # Advanced duplicate detection: check for existing memory with same content, type, and source
        existing = await self.execute_query(
            """SELECT memory_id FROM curated_memories 
                   WHERE content = ? AND memory_type = ? AND source_conversation_id IS ?""",
            (content, memory_type, source_conversation_id)
        )
        if existing:
            print("Skipping duplicate curated memory entry.")
            return existing[0]["memory_id"]

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
        """Run database maintenance tasks.
        
        Args:
            force: Whether to force maintenance even if recent
            
        Returns:
            Dict containing maintenance results
        """
        try:
            # Check last maintenance
            last_maintenance = await self.execute_query(
                "SELECT value FROM metadata WHERE key = 'last_maintenance'"
            )
            
            if not force and last_maintenance:
                last_time = datetime.fromisoformat(last_maintenance[0]["value"])
                if datetime.now(get_local_timezone()) - last_time < timedelta(days=7):
                    return {
                        "status": "skipped",
                        "message": "Maintenance ran recently",
                        "last_run": last_time.isoformat()
                    }
            
            with self.get_connection() as conn:
                # Optimize indexes
                conn.execute("ANALYZE")
                
                # Clean up any orphaned records
                conn.execute("""
                    DELETE FROM curated_memories 
                    WHERE source_conversation_id NOT IN (
                        SELECT conversation_id FROM conversations
                    ) AND source_conversation_id IS NOT NULL
                """)
                
                # Update metadata
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    ("last_maintenance", get_current_timestamp())
                )
                
                conn.commit()
                
            return {
                "status": "success",
                "message": "Maintenance completed successfully",
                "timestamp": get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Maintenance error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": get_current_timestamp()
            }
    
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
        timestamp = get_current_timestamp()
        
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


class ScheduleDatabase(DatabaseManager):
    """Manages appointments and reminders database"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_settings().schedule_db_path)
        super().__init__(db_path)
        self.initialize_tables()

    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # Appointments table migration
            appointments_expected = [
                'appointment_id', 'timestamp_created', 'scheduled_datetime', 'title', 'description',
                'location', 'cancelled_at', 'completed_at', 'status', 'source_conversation_id', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(appointments)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in appointments_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating appointments table to new schema!")
                old_rows = conn.execute("SELECT * FROM appointments").fetchall()
                conn.execute("DROP TABLE IF EXISTS appointments")
                conn.execute("""
                    CREATE TABLE appointments (
                        appointment_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        scheduled_datetime TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        location TEXT,
                        status TEXT DEFAULT 'scheduled' CHECK(status IN ('scheduled', 'cancelled', 'completed')),
                        cancelled_at TEXT,
                        completed_at TEXT,
                        source_conversation_id TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in appointments_expected:
                        if col not in row_dict:
                            if col == 'timestamp_created' or col == 'scheduled_datetime' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'status':
                                row_dict[col] = 'scheduled'  # Default status for migrated appointments
                            elif col == 'cancelled_at' or col == 'completed_at':
                                row_dict[col] = None  # Default to None for new timestamp columns
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO appointments ({', '.join(appointments_expected)}) VALUES ({', '.join(['?' for _ in appointments_expected])})",
                        tuple(row_dict[col] for col in appointments_expected)
                    )
                print(f"Restored {len(old_rows)} appointments after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS appointments (
                        appointment_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        scheduled_datetime TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        location TEXT,
                        status TEXT DEFAULT 'scheduled' CHECK(status IN ('scheduled', 'cancelled', 'completed')),
                        cancelled_at TEXT,
                        completed_at TEXT,
                        source_conversation_id TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Reminders table migration
            reminders_expected = [
                'reminder_id', 'timestamp_created', 'due_datetime', 'content', 'priority_level',
                'completed', 'is_completed', 'completed_at', 'source_conversation_id', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(reminders)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in reminders_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating reminders table to new schema!")
                old_rows = conn.execute("SELECT * FROM reminders").fetchall()
                conn.execute("DROP TABLE IF EXISTS reminders")
                conn.execute("""
                    CREATE TABLE reminders (
                        reminder_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        due_datetime TEXT NOT NULL,
                        content TEXT NOT NULL,
                        priority_level INTEGER DEFAULT 5,
                        completed INTEGER DEFAULT 0,
                        is_completed INTEGER DEFAULT 0,
                        completed_at TEXT,
                        source_conversation_id TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in reminders_expected:
                        if col not in row_dict:
                            if col == 'is_completed':
                                row_dict[col] = row_dict.get('completed', 0)
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO reminders ({', '.join(reminders_expected)}) VALUES ({', '.join(['?' for _ in reminders_expected])})",
                        tuple(row_dict[col] for col in reminders_expected)
                    )
                print(f"Restored {len(old_rows)} reminders after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reminders (
                        reminder_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        due_datetime TEXT NOT NULL,
                        content TEXT NOT NULL,
                        priority_level INTEGER DEFAULT 5,
                        completed INTEGER DEFAULT 0,
                        is_completed INTEGER DEFAULT 0,
                        completed_at TEXT,
                        source_conversation_id TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            conn.commit()
    
    async def create_appointment(self, title: str, scheduled_datetime: str, 
                               description: str = None, location: str = None,
                               source_conversation_id: str = None,
                               recurrence_pattern: str = None,
                               recurrence_count: int = None,
                               recurrence_end_date: str = None) -> Union[str, List[str]]:
        """Create a new appointment, optionally recurring
        
        Args:
            title: Appointment title
            scheduled_datetime: ISO format datetime for first appointment
            description: Optional description
            location: Optional location
            source_conversation_id: Optional source conversation ID
            recurrence_pattern: Optional recurrence pattern ('weekly', 'monthly', 'daily')
            recurrence_count: Optional number of recurrences (including first appointment)
            recurrence_end_date: Optional end date for recurrences (ISO format)
            
        Returns:
            Single appointment_id if no recurrence, list of appointment_ids if recurring
        """
        from dateutil.relativedelta import relativedelta
        from dateutil.parser import parse as parse_date
        
        appointment_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()

        # Duplicate detection: check for existing appointment with same title, datetime, location, and source
        existing = await self.execute_query(
            """SELECT appointment_id FROM appointments 
                   WHERE title = ? AND scheduled_datetime = ? AND location IS ? AND source_conversation_id IS ?""",
            (title, scheduled_datetime, location, source_conversation_id)
        )
        if existing:
            print("Skipping duplicate appointment entry.")
            return existing[0]["appointment_id"]

        await self.execute_update(
            """INSERT INTO appointments 
               (appointment_id, timestamp_created, scheduled_datetime, title, description, location, source_conversation_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (appointment_id, timestamp, scheduled_datetime, title, description, location, source_conversation_id)
        )
        
        appointment_ids = [appointment_id]
        
        # Handle recurring appointments
        if recurrence_pattern and (recurrence_count or recurrence_end_date):
            try:
                base_datetime = parse_date(scheduled_datetime)
                if recurrence_end_date:
                    end_date = parse_date(recurrence_end_date)
                else:
                    end_date = None
                
                # Determine the increment pattern
                if recurrence_pattern.lower() == 'daily':
                    delta = relativedelta(days=1)
                elif recurrence_pattern.lower() == 'weekly':
                    delta = relativedelta(weeks=1)
                elif recurrence_pattern.lower() == 'monthly':
                    delta = relativedelta(months=1)
                elif recurrence_pattern.lower() == 'yearly':
                    delta = relativedelta(years=1)
                else:
                    raise ValueError(f"Unsupported recurrence pattern: {recurrence_pattern}")
                
                current_datetime = base_datetime
                created_count = 1  # Already created the first one
                
                # Create recurring appointments
                while True:
                    # Check if we should stop
                    if recurrence_count and created_count >= recurrence_count:
                        break
                    if end_date and current_datetime >= end_date:
                        break
                    
                    # Calculate next occurrence
                    current_datetime += delta
                    
                    # Check end date again after increment
                    if end_date and current_datetime > end_date:
                        break
                    
                    # Skip duplicate detection for recurring appointments
                    # Create the recurring appointment
                    recurring_id = str(uuid.uuid4())
                    recurring_datetime = current_datetime.isoformat()
                    
                    await self.execute_update(
                        """INSERT INTO appointments 
                           (appointment_id, timestamp_created, scheduled_datetime, title, description, location, source_conversation_id) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (recurring_id, timestamp, recurring_datetime, title, description, location, source_conversation_id)
                    )
                    
                    appointment_ids.append(recurring_id)
                    created_count += 1
                    
            except Exception as e:
                logger.error(f"Error creating recurring appointments: {e}")
                # Return the first appointment ID even if recurring failed
                return appointment_id
        
        # Return single ID if no recurrence, list if recurring
        return appointment_id if len(appointment_ids) == 1 else appointment_ids
    
    async def create_reminder(self, content: str, due_datetime: str, 
                            priority_level: int = 5, source_conversation_id: str = None) -> str:
        """Create a new reminder with duplicate detection"""
        reminder_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()

        # Duplicate detection: check for existing reminder with same content, due_datetime, and source
        existing = await self.execute_query(
            """SELECT reminder_id FROM reminders 
                   WHERE content = ? AND due_datetime = ? AND source_conversation_id IS ?""",
            (content, due_datetime, source_conversation_id)
        )
        if existing:
            print("Skipping duplicate reminder entry.")
            return existing[0]["reminder_id"]

        await self.execute_update(
            """INSERT INTO reminders 
               (reminder_id, timestamp_created, due_datetime, content, priority_level, source_conversation_id) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (reminder_id, timestamp, due_datetime, content, priority_level, source_conversation_id)
        )
        return reminder_id
    
    async def get_upcoming_appointments(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming appointments within specified days"""
        
        future_date = datetime.now(get_local_timezone()) + timedelta(days=days_ahead)
        
        rows = await self.execute_query(
            """SELECT * FROM appointments 
               WHERE scheduled_datetime >= ? AND scheduled_datetime <= ?
               ORDER BY scheduled_datetime ASC""",
            (get_current_timestamp(), future_date.isoformat())
        )
        
        return [dict(row) for row in rows]
    
    async def get_active_reminders(self) -> List[Dict]:
        """Get all uncompleted reminders"""
        
        rows = await self.execute_query(
            "SELECT * FROM reminders WHERE completed = 0 ORDER BY due_datetime ASC"
        )
        
        return [dict(row) for row in rows]

    async def auto_complete_overdue_reminders(self) -> Dict[str, int]:
        """
        Auto-complete overdue reminders at midnight.
        Returns count of reminders that were auto-completed.
        """
        current_time = datetime.now()
        
        # Find overdue reminders that aren't already completed
        overdue_reminders = await self.execute_query(
            """SELECT reminder_id, content, due_datetime 
               FROM reminders 
               WHERE due_datetime < ? 
               AND completed = 0 
               AND is_completed = 0""",
            (current_time.isoformat(),)
        )
        
        completed_count = 0
        for reminder in overdue_reminders:
            # Mark as completed
            await self.execute_update(
                """UPDATE reminders 
                   SET completed = 1, is_completed = 1, completed_at = ?
                   WHERE reminder_id = ?""",
                (current_time.isoformat(), reminder['reminder_id'])
            )
            completed_count += 1
            
            logger.info(f"Auto-completed overdue reminder: {reminder['content']}")
        
        return {
            "completed_count": completed_count,
            "processed_at": current_time.isoformat()
        }


class VSCodeProjectDatabase(DatabaseManager):
    """Manages VS Code project context and development sessions"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_settings().vscode_db_path)
        super().__init__(db_path)
        self.initialize_tables()

    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # Project sessions table migration
            sessions_expected = [
                'session_id', 'start_timestamp', 'end_timestamp', 'workspace_path', 'active_files',
                'git_branch', 'session_summary', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(project_sessions)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in sessions_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating project_sessions table to new schema!")
                old_rows = conn.execute("SELECT * FROM project_sessions").fetchall()
                conn.execute("DROP TABLE IF EXISTS project_sessions")
                conn.execute("""
                    CREATE TABLE project_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_timestamp TEXT NOT NULL,
                        end_timestamp TEXT,
                        workspace_path TEXT NOT NULL,
                        active_files TEXT,
                        git_branch TEXT,
                        session_summary TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in sessions_expected:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO project_sessions ({', '.join(sessions_expected)}) VALUES ({', '.join(['?' for _ in sessions_expected])})",
                        tuple(row_dict[col] for col in sessions_expected)
                    )
                print(f"Restored {len(old_rows)} project sessions after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_timestamp TEXT NOT NULL,
                        end_timestamp TEXT,
                        workspace_path TEXT NOT NULL,
                        active_files TEXT,
                        git_branch TEXT,
                        session_summary TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Project insights table migration
            insights_expected = [
                'insight_id', 'timestamp_created', 'timestamp_updated', 'insight_type', 'content',
                'related_files', 'source_conversation_id', 'importance_level', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(project_insights)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in insights_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating project_insights table to new schema!")
                old_rows = conn.execute("SELECT * FROM project_insights").fetchall()
                conn.execute("DROP TABLE IF EXISTS project_insights")
                conn.execute("""
                    CREATE TABLE project_insights (
                        insight_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        timestamp_updated TEXT NOT NULL,
                        insight_type TEXT,
                        content TEXT NOT NULL,
                        related_files TEXT,
                        source_conversation_id TEXT,
                        importance_level INTEGER DEFAULT 5,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in insights_expected:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO project_insights ({', '.join(insights_expected)}) VALUES ({', '.join(['?' for _ in insights_expected])})",
                        tuple(row_dict[col] for col in insights_expected)
                    )
                print(f"Restored {len(old_rows)} project insights after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_insights (
                        insight_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        timestamp_updated TEXT NOT NULL,
                        insight_type TEXT,
                        content TEXT NOT NULL,
                        related_files TEXT,
                        source_conversation_id TEXT,
                        importance_level INTEGER DEFAULT 5,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Code context table migration
            codectx_expected = [
                'context_id', 'timestamp', 'file_path', 'function_name', 'description', 'purpose',
                'related_insights', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(code_context)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in codectx_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating code_context table to new schema!")
                old_rows = conn.execute("SELECT * FROM code_context").fetchall()
                conn.execute("DROP TABLE IF EXISTS code_context")
                conn.execute("""
                    CREATE TABLE code_context (
                        context_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        function_name TEXT,
                        description TEXT NOT NULL,
                        purpose TEXT,
                        related_insights TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in codectx_expected:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO code_context ({', '.join(codectx_expected)}) VALUES ({', '.join(['?' for _ in codectx_expected])})",
                        tuple(row_dict[col] for col in codectx_expected)
                    )
                print(f"Restored {len(old_rows)} code contexts after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS code_context (
                        context_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        function_name TEXT,
                        description TEXT NOT NULL,
                        purpose TEXT,
                        related_insights TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Development conversations table migration
            devcon_expected = [
                'conversation_id', 'session_id', 'timestamp', 'chat_context_id', 'conversation_content',
                'decisions_made', 'code_changes', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(development_conversations)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in devcon_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                print("Migrating development_conversations table to new schema!")
                old_rows = conn.execute("SELECT * FROM development_conversations").fetchall()
                conn.execute("DROP TABLE IF EXISTS development_conversations")
                conn.execute("""
                    CREATE TABLE development_conversations (
                        conversation_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp TEXT NOT NULL,
                        chat_context_id TEXT,
                        conversation_content TEXT NOT NULL,
                        decisions_made TEXT,
                        code_changes TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES project_sessions (session_id)
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in devcon_expected:
                        if col not in row_dict:
                            row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO development_conversations ({', '.join(devcon_expected)}) VALUES ({', '.join(['?' for _ in devcon_expected])})",
                        tuple(row_dict[col] for col in devcon_expected)
                    )
                print(f"Restored {len(old_rows)} development conversations after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS development_conversations (
                        conversation_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp TEXT NOT NULL,
                        chat_context_id TEXT,
                        conversation_content TEXT NOT NULL,
                        decisions_made TEXT,
                        code_changes TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES project_sessions (session_id)
                    )
                """)

            conn.commit()
    
    async def save_development_session(self, workspace_path: str, active_files: List[str] = None,
                                     git_branch: str = None, session_summary: str = None) -> str:
        """Save a development session"""
        
        session_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO project_sessions 
               (session_id, start_timestamp, workspace_path, active_files, git_branch, session_summary) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, timestamp, workspace_path, 
             json.dumps(active_files) if active_files else None,
             git_branch, session_summary)
        )
        
        return session_id
    
    async def store_development_conversation(self, content: str, session_id: str = None,
                                          chat_context_id: str = None, decisions_made: str = None,
                                          code_changes: Dict = None) -> str:
        """Store a development conversation from VS Code
        
        Args:
            content: The conversation content
            session_id: Optional project session ID (will create new if none)
            chat_context_id: Optional VS Code chat context ID
            decisions_made: Summary of decisions made in conversation
            code_changes: Dictionary of files changed and their changes
        """
        conversation_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        # Create session if none provided
        if not session_id:
            session_id = await self.save_development_session(
                workspace_path=os.getcwd(),  # Current workspace
                session_summary="Auto-created session for development conversation"
            )
        
        # Store conversation
        await self.execute_update(
            """INSERT INTO development_conversations 
               (conversation_id, session_id, timestamp, chat_context_id,
                conversation_content, decisions_made, code_changes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, session_id, timestamp, chat_context_id,
             content, decisions_made, json.dumps(code_changes) if code_changes else None)
        )
        
        return conversation_id

    async def store_project_insight(self, content: str, insight_type: str = None,
                                  related_files: List[str] = None, importance_level: int = 5,
                                  source_conversation_id: str = None) -> str:
        """Store a project insight with duplicate detection"""
        insight_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()

        # Duplicate detection: check for existing insight with same content, type, and source
        existing = await self.execute_query(
            """SELECT insight_id FROM project_insights 
                   WHERE content = ? AND insight_type IS ? AND source_conversation_id IS ?""",
            (content, insight_type, source_conversation_id)
        )
        if existing:
            print("Skipping duplicate project insight entry.")
            return existing[0]["insight_id"]

        await self.execute_update(
            """INSERT INTO project_insights 
               (insight_id, timestamp_created, timestamp_updated, insight_type, content, 
                related_files, source_conversation_id, importance_level) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, timestamp, timestamp, insight_type, content,
             json.dumps(related_files) if related_files else None,
             source_conversation_id, importance_level)
        )
        return insight_id


class ConversationFileMonitor:
    def __init__(self, memory_system, watch_directories):
        self.memory_system = memory_system
        self.watch_directories = watch_directories
        self.vscode_db = memory_system.vscode_db
        self.conversations_db = memory_system.conversations_db  # Add this to maintain compatibility
        self.curated_db = memory_system.curated_db  # Add this to maintain compatibility
        # Do NOT start file monitoring or background tasks automatically here
        
    def _parse_character_ai_format(self, data: Dict) -> List[Dict]:
        """Parse Character.ai conversation format (list of messages under 'conversation')"""
        conversations = []
        try:
            messages = data.get('conversation', [])
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    conversations.append({
                        'role': msg.get('character', msg.get('role', 'unknown')),
                        'content': msg['content'],
                        'timestamp': msg.get('timestamp')
                    })
        except Exception as e:
            logger.error(f"Error parsing Character.ai format: {e}")
        return conversations

    def _parse_text_gen_format(self, data: Dict) -> List[Dict]:
        """Parse text-generation-webui format (list of messages under 'history')"""
        conversations = []
        try:
            history = data.get('history', [])
            for msg in history:
                if isinstance(msg, dict) and 'content' in msg:
                    conversations.append({
                        'role': msg.get('role', 'unknown'),
                        'content': msg['content'],
                        'timestamp': msg.get('timestamp')
                    })
        except Exception as e:
            logger.error(f"Error parsing text-generation-webui format: {e}")
        return conversations
    """Monitors files for conversation changes and auto-imports them.
    
    Features:
    - Automatic MCP server detection to avoid duplicate message processing
    - Real-time file monitoring with hash-based change detection
    - Support for VS Code, LM Studio, and Ollama chat files
    - Message deduplication across sources
    """
    
    def __init__(self, memory_system, watch_directories: List[str] = None, mcp_port: int = 1234):
        self.memory_system = memory_system
        self.watch_directories = watch_directories or []
        self.observer = None
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.file_hashes = {}  # Track file content hashes to detect changes
        self.processed_messages = {}  # Track processed messages per file: {file_path: set(message_hashes)}
        self.mcp_port = mcp_port  # Port to check for MCP server
        self.mcp_server_running = False  # Will be updated periodically
        self.last_mcp_check = 0  # Timestamp of last MCP server check
        
    def _get_default_chat_directories(self) -> List[str]:
        """Get default chat storage directories for different platforms"""
        home = Path.home()
        documents = home / "Documents"
        downloads = home / "Downloads"
        directories = []
        
        # NOTE: ChatGPT and Claude desktop apps DO NOT store conversations locally
        # They are cloud-only applications. Removed these paths after verification.
        
        # LM Studio conversation directories
        lm_studio_paths = [
            home / ".lmstudio" / "conversations",  # Windows/Linux/macOS (new location)
            home / "AppData" / "Roaming" / "LM Studio" / "conversations",  # Windows (old location)
            home / ".config" / "lm-studio" / "conversations",  # Linux (old location)
            home / "Library" / "Application Support" / "LM Studio" / "conversations"  # macOS (old location)
        ]
        
        # Ollama database paths (SQLite database instead of files)
        ollama_db_paths = [
            home / "AppData" / "Local" / "Ollama" / "db.sqlite",  # Windows
            home / ".local" / "share" / "ollama" / "db.sqlite",  # Linux
            home / "Library" / "Application Support" / "Ollama" / "db.sqlite"  # macOS
        ]
        
        # Additional popular chat platforms
        perplexity_paths = [
            home / "AppData" / "Roaming" / "Perplexity" / "conversations",  # Windows
            home / ".config" / "perplexity" / "conversations",  # Linux
            home / "Library" / "Application Support" / "Perplexity" / "conversations"  # macOS
        ]
        
        jan_ai_paths = [
            home / "AppData" / "Roaming" / "Jan" / "conversations",  # Windows
            home / ".config" / "jan" / "conversations",  # Linux
            home / "Library" / "Application Support" / "Jan" / "conversations"  # macOS
        ]
        
        open_webui_paths = [
            home / ".open-webui" / "data" / "chats",  # All platforms
            home / "open-webui" / "data" / "chats",  # Alternative location
        ]
        
        # OpenWebUI database paths (SQLite database)
        open_webui_db_paths = [
            home / ".open-webui" / "data" / "webui.db",  # Default location
            home / "open-webui" / "data" / "webui.db",  # Alternative location
            home / "OpenWebUI" / "data" / "webui.db",  # Capitalized variant
            documents / "OpenWebUI" / "data" / "webui.db",  # Documents folder
            downloads / "OpenWebUI" / "data" / "webui.db",  # Downloads folder
        ]
        
        # Text generation WebUI paths  
        text_gen_webui_paths = [
            home / "text-generation-webui" / "logs",
            home / "text-generation-webui" / "characters",
            home / "Documents" / "text-generation-webui" / "logs"
        ]
        
        # SillyTavern paths (requested by Reddit community)
        sillytavern_paths = [
            home / "SillyTavern" / "data" / "chats",  # Default installation
            home / "AppData" / "Roaming" / "SillyTavern" / "chats",  # Windows
            home / ".config" / "sillytavern" / "chats",  # Linux
            home / "Library" / "Application Support" / "SillyTavern" / "chats",  # macOS
            documents / "SillyTavern" / "chats",  # User documents
            downloads / "SillyTavern" / "data" / "chats"  # Downloaded version
        ]
        
        # Gemini CLI paths (requested by Reddit community)
        gemini_cli_paths = [
            home / ".gemini" / "conversations",  # Linux/macOS
            home / "AppData" / "Roaming" / "gemini-cli" / "conversations",  # Windows
            home / ".config" / "gemini" / "conversations",  # Linux alternative
            home / "Library" / "Application Support" / "Gemini" / "conversations"  # macOS
        ]
        
        # VS Code workspace storage directories
        vscode_base_paths = [
            home / "AppData" / "Roaming" / "Code" / "User" / "workspaceStorage",  # Windows
            home / ".config" / "Code" / "User" / "workspaceStorage",  # Linux
            home / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage"  # macOS
        ]
        
        # Helper function to add paths with logging
        def add_paths_if_exist(paths: List[Path], app_name: str):
            for path in paths:
                if path.exists():
                    directories.append(str(path))
                    logger.info(f"Found {app_name} conversations: {path}")
        
        # Add paths for each application (ChatGPT and Claude removed - cloud-only)
        add_paths_if_exist(lm_studio_paths, "LM Studio")
        add_paths_if_exist(perplexity_paths, "Perplexity")
        add_paths_if_exist(jan_ai_paths, "Jan AI")
        add_paths_if_exist(open_webui_paths, "Open WebUI")
        add_paths_if_exist(text_gen_webui_paths, "Text Generation WebUI")
        add_paths_if_exist(sillytavern_paths, "SillyTavern")
        add_paths_if_exist(gemini_cli_paths, "Gemini CLI")
        
        # Special handling for Ollama database
        for db_path in ollama_db_paths:
            if db_path.exists():
                directories.append(str(db_path))
                logger.info(f"Found Ollama database: {db_path}")
        
        # Special handling for OpenWebUI database
        for db_path in open_webui_db_paths:
            if db_path.exists():
                directories.append(str(db_path))
                logger.info(f"Found OpenWebUI database: {db_path}")
        
        # Add VS Code workspace storage paths - find specific workspace hashes
        for vscode_base in vscode_base_paths:
            if vscode_base.exists():
                try:
                    # Look for workspace hashes (directories with chatSessions folders)
                    for workspace_hash in vscode_base.iterdir():
                        if workspace_hash.is_dir():
                            chat_sessions_dir = workspace_hash / "chatSessions"
                            if chat_sessions_dir.exists():
                                directories.append(str(chat_sessions_dir))
                                logger.info(f"Found VS Code chat sessions: {chat_sessions_dir}")
                except Exception as e:
                    logger.error(f"Error scanning VS Code workspace storage: {e}")
        
        return directories

    def _check_mcp_server(self) -> bool:
        """Check if an MCP server is running by attempting a connection.
        
        Returns:
            bool: True if MCP server is running, False otherwise
        """
        # Only check every 60 seconds to avoid overhead
        current_time = time.time()
        if current_time - self.last_mcp_check < 60:
            return self.mcp_server_running
            
        try:
            # Try to connect to MCP server port
            with socket.create_connection(("localhost", self.mcp_port), timeout=1.0):
                self.mcp_server_running = True
        except (socket.timeout, ConnectionRefusedError):
            self.mcp_server_running = False
        
        self.last_mcp_check = current_time
        return self.mcp_server_running
        
    async def _is_message_in_mcp(self, msg_hash: str) -> bool:
        """Check if a message was manually stored through MCP server.
        
        Args:
            msg_hash: Hash of the message content to check
            
        Returns:
            bool: True if message exists in MCP storage, False otherwise
        """
        try:
            # Connect to MCP server
            reader, writer = await asyncio.open_connection('localhost', self.mcp_port)
            
            # Send check request
            request = json.dumps({
                'type': 'check_message',
                'hash': msg_hash
            }).encode() + b'\n'
            writer.write(request)
            await writer.drain()
            
            # Get response
            response = await reader.readline()
            writer.close()
            await writer.wait_closed()
            
            # Parse response
            result = json.loads(response.decode())
            return result.get('exists', False)
            
        except Exception as e:
            logger.debug(f"Failed to check message in MCP: {e}")
            return False  # If check fails, assume message doesn't exist
    
    def _get_mcp_start_time(self) -> Optional[datetime]:
        """Get the start time of the MCP server if running.
        
        Returns:
            Optional[datetime]: Server start time if available, None otherwise
        """
        if not self._check_mcp_server():
            return None
            
        try:
            with socket.create_connection(("localhost", self.mcp_port), timeout=1.0) as sock:
                sock.sendall(b"GET_START_TIME\n")
                response = sock.recv(1024).decode().strip()
                if response and response != "ERROR":
                    return datetime.fromisoformat(response)
        except Exception as e:
            logger.debug(f"Failed to get MCP start time: {e}")
        return None

    async def start_monitoring(self):
        """Start monitoring conversation files"""
        if not self.watch_directories:
            logger.info("No watch directories specified for file monitoring")
            return
            
        # Store reference to the current event loop
        self.loop = asyncio.get_running_loop()
        
        self.observer = Observer()
        
        for directory in self.watch_directories:
            if os.path.exists(directory):
                
                class ConversationFileHandler(FileSystemEventHandler):
                    def __init__(self, monitor):
                        self.monitor = monitor
                    
                    def on_modified(self, event):
                        if not event.is_directory:
                            try:
                                # Get the event loop from the main thread
                                loop = self.monitor.loop
                                if loop and loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self.monitor._process_file_change(event.src_path), 
                                        loop
                                    )
                            except Exception as e:
                                print(f"Error scheduling file change processing: {e}")
                    
                    def on_created(self, event):
                        if not event.is_directory:
                            try:
                                # Get the event loop from the main thread
                                loop = self.monitor.loop
                                if loop and loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        self.monitor._process_file_change(event.src_path), 
                                        loop
                                    )
                            except Exception as e:
                                print(f"Error scheduling file change processing: {e}")
                
                handler = ConversationFileHandler(self)
                self.observer.schedule(handler, directory, recursive=True)
                logger.info(f"Started monitoring directory: {directory}")
        
        self.observer.start()
        logger.info("File monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring conversation files"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File monitoring stopped")
    
    def add_watch_directory(self, directory: str):
        """Add a directory to monitor"""
        if directory not in self.watch_directories:
            self.watch_directories.append(directory)
            logger.info(f"Added watch directory: {directory}")
    
    async def _process_file_change(self, file_path: str):
        """Process a changed conversation file with MCP-aware deduplication"""
        try:
            # Check if file is a conversation file (JSON, txt, etc.)
            if not any(file_path.endswith(ext) for ext in ['.json', '.txt', '.md', '.log']):
                return
            
            # Calculate file hash to detect actual content changes
            with open(file_path, 'rb') as f:
                file_content = f.read()
                current_hash = hashlib.md5(file_content).hexdigest()
            
            # Skip if we've already processed this exact content
            if file_path in self.file_hashes and self.file_hashes[file_path] == current_hash:
                return
                
            self.file_hashes[file_path] = current_hash
            
            # Initialize message tracking for this file if needed
            if file_path not in self.processed_messages:
                self.processed_messages[file_path] = set()
            
            # Read and parse conversation content
            conversations = await self._extract_conversations(file_path)
            
            # Check with MCP server for manually stored messages
            if self._check_mcp_server():
                try:
                    filtered_conversations = []
                    for conv in conversations:
                        # Create a hash of the message content and metadata
                        msg_hash = hashlib.md5(
                            f"{conv['role']}:{conv['content']}".encode()
                        ).hexdigest()
                        
                        # Check if this exact message was manually stored
                        if not await self._is_message_in_mcp(msg_hash):
                            filtered_conversations.append(conv)
                    conversations = filtered_conversations
                except Exception as e:
                    logger.debug(f"Failed to check MCP messages: {e}")
                    # If we can't check MCP server, process all messages
            
            # For VS Code chat files, handle development conversations
            is_vscode_chat = 'vscode' in file_path.lower() or 'chatsessions' in file_path.lower()
            if is_vscode_chat:
                # Create development session
                dev_session_id = await self.memory_system.vscode_db.save_development_session(
                    workspace_path=os.path.dirname(file_path),
                    session_summary=f"Imported VS Code chat session from {os.path.basename(file_path)}"
                )
                full_conversation = []
            
            # Store conversations in database
            for conv in conversations:
                result = await self.memory_system.store_conversation(
                    content=conv['content'],
                    role=conv['role'],
                    metadata={'source_file': file_path, 'imported_at': get_current_timestamp()},
                    session_id=self._get_file_hash(file_path)  # Use file hash as session ID for grouping
                )
                
                if is_vscode_chat and not result.get("duplicate", False):
                    # Add to development conversation
                    full_conversation.append(f"{conv['role'].title()}: {conv['content']}")
            
            # Store development conversation if this is a VS Code chat
            if is_vscode_chat and full_conversation:
                await self.memory_system.vscode_db.store_development_conversation(
                    content="\n\n".join(full_conversation),
                    session_id=dev_session_id,
                    chat_context_id=self._get_file_hash(file_path)
                )
            
            logger.info(f"Imported {len(conversations)} conversations from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(hash(file_path))
    
    async def _extract_conversations(self, file_path: str) -> List[Dict]:
        """Extract conversations from various file formats with timestamps, using registry-based extensibility and robust deduplication"""
        conversations = []
        try:
            # Special handling for Ollama SQLite database
            if file_path.lower().endswith('db.sqlite') and 'ollama' in file_path.lower():
                conversations.extend(self._extract_ollama_database(file_path))
                return conversations
            
            # Special handling for OpenWebUI SQLite database
            if (file_path.lower().endswith('webui.db') or 
                (file_path.lower().endswith('.db') and 'openwebui' in file_path.lower()) or
                (file_path.lower().endswith('.db') and 'open-webui' in file_path.lower())):
                conversations.extend(self._extract_openwebui_database(file_path))
                return conversations
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            fallback_time = datetime.fromtimestamp(
                os.path.getmtime(file_path),
                timezone.utc
            ).isoformat()

            # Registry of format handlers: (predicate, handler)
            format_handlers = [
                (lambda fn, _: fn.endswith('.json'), self._handle_json_formats),
                (lambda fn, _: fn.endswith(('.txt', '.md', '.log')), self._parse_text_format),
            ]

            handled = False
            for predicate, handler in format_handlers:
                if predicate(file_path, content):
                    if handler == self._handle_json_formats:
                        conversations.extend(handler(content))
                    else:
                        conversations.extend(handler(content))
                    handled = True
                    break

            if not handled:
                logger.warning(f"No format handler found for {file_path}")

            # Ensure all conversations have timestamps
            for conv in conversations:
                if 'timestamp' not in conv or not conv['timestamp']:
                    conv['timestamp'] = fallback_time

            # Robust deduplication: by id (if present), timestamp (if present), and content hash
            seen = set()
            deduped = []
            for conv in conversations:
                # Use id if present, else None
                cid = conv.get('id') or conv.get('message_id') or None
                ts = conv.get('timestamp') or None
                content_hash = hashlib.md5(conv.get('content', '').encode('utf-8')).hexdigest()
                dedup_key = (cid, ts, content_hash)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    deduped.append(conv)
            return deduped
        except Exception as e:
            logger.error(f"Error extracting conversations from {file_path}: {e}")
            return []

    def _handle_json_formats(self, content: str) -> List[Dict]:
        """Handle all supported JSON conversation formats (add new ones here)"""
        conversations = []
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if self._is_lmstudio_format(data):
                    conversations.extend(self._parse_lmstudio_format(data))
                elif ('messages' in data or 'chat' in data) and self._is_sillytavern_format(data):
                    conversations.extend(self._parse_sillytavern_format(data))
                elif ('conversation' in data or ('messages' in data and self._is_gemini_cli_format(data))):
                    conversations.extend(self._parse_gemini_cli_format(data))
            elif isinstance(data, list):
                conversations.extend(self._parse_simple_array(data))
        except Exception as e:
            logger.error(f"Error handling JSON formats: {e}")
        return conversations
    
    def _parse_simple_array(self, data: List) -> List[Dict]:
        """Parse simple conversation array format with timestamps"""
        conversations = []
        
        for item in data:
            if isinstance(item, dict) and 'content' in item:
                # Look for timestamp in various formats
                timestamp = None
                for key in ['timestamp', 'time', 'created_at', 'date']:
                    if key in item:
                        try:
                            # Handle both ISO format strings and Unix timestamps
                            if isinstance(item[key], (int, float)):
                                timestamp = datetime.fromtimestamp(item[key], timezone.utc).isoformat()
                            else:
                                timestamp = datetime.fromisoformat(str(item[key])).isoformat()
                            break
                        except (ValueError, TypeError):
                            continue
                
                conversations.append({
                    'role': item.get('role', 'user'),
                    'content': str(item['content']),
                    'timestamp': timestamp
                })
        
        return conversations
    
    def _is_lmstudio_format(self, data: Dict) -> bool:
        """Check if data is in LM Studio format (has messages with versions structure)"""
        try:
            messages = data.get('messages', [])
            if not messages:
                return False
            # Check if first message has the LM Studio structure
            first_msg = messages[0] if isinstance(messages, list) else None
            return (isinstance(first_msg, dict) and 
                    'versions' in first_msg and 
                    'currentlySelected' in first_msg)
        except:
            return False

    def _is_sillytavern_format(self, data: Dict) -> bool:
        """Check if data is in SillyTavern format"""
        if not isinstance(data, dict):
            return False
        
        # SillyTavern specific indicators
        if 'messages' in data:
            messages = data.get('messages', [])
            if isinstance(messages, list) and messages:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    # SillyTavern specific fields
                    return 'is_user' in first_msg or 'mes' in first_msg or 'send_date' in first_msg
        
        # Alternative SillyTavern format
        if 'chat' in data:
            chat = data.get('chat', [])
            if isinstance(chat, list) and chat:
                first_msg = chat[0]
                if isinstance(first_msg, dict):
                    return 'is_user' in first_msg or 'mes' in first_msg
        
        return False

    def _is_gemini_cli_format(self, data: Dict) -> bool:
        """Check if data is in Gemini CLI format"""
        if not isinstance(data, dict):
            return False
        
        # Gemini CLI specific structure
        if 'conversation' in data:
            return True
        
        # Check for Gemini-specific message format
        if 'messages' in data:
            messages = data.get('messages', [])
            if isinstance(messages, list) and messages:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    # Gemini uses 'parts' array or 'model' role
                    return ('parts' in first_msg or 
                           first_msg.get('role') == 'model' or
                           'response' in first_msg)
        
        return False

    def _parse_lmstudio_format(self, data: Dict) -> List[Dict]:
        """Parse LM Studio conversation format with versions and complex content structure"""
        conversations = []
        try:
            messages = data.get('messages', [])
            conversation_timestamp = data.get('createdAt')
            base_timestamp = None
            
            if conversation_timestamp:
                try:
                    # LM Studio uses millisecond timestamps
                    base_timestamp = datetime.fromtimestamp(conversation_timestamp / 1000, timezone.utc)
                except (ValueError, TypeError):
                    pass
            
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict) or 'versions' not in msg:
                    continue
                
                versions = msg.get('versions', [])
                current_version = msg.get('currentlySelected', 0)
                
                if 0 <= current_version < len(versions):
                    version = versions[current_version]
                    
                    role = version.get('role', 'unknown')
                    content_parts = version.get('content', [])
                    
                    # Extract text content from LM Studio's complex content structure
                    text_content = []
                    for part in content_parts:
                        if isinstance(part, dict):
                            if part.get('type') == 'text':
                                text_content.append(part.get('text', ''))
                            elif part.get('type') == 'file':
                                # Handle file attachments
                                file_info = f"[File: {part.get('fileIdentifier', 'unknown')}]"
                                text_content.append(file_info)
                        elif isinstance(part, str):
                            text_content.append(part)
                    
                    # For assistant messages, handle multi-step responses
                    if version.get('type') == 'multiStep' and 'steps' in version:
                        for step in version.get('steps', []):
                            if step.get('type') == 'contentBlock':
                                step_content = step.get('content', [])
                                for step_part in step_content:
                                    if isinstance(step_part, dict) and step_part.get('type') == 'text':
                                        text_content.append(step_part.get('text', ''))
                    
                    final_content = ' '.join(text_content).strip()
                    if final_content:
                        # Calculate approximate timestamp for each message
                        timestamp = None
                        if base_timestamp:
                            # Spread messages over time based on their position
                            message_time = base_timestamp + timedelta(minutes=i * 2)
                            timestamp = message_time.isoformat()
                        
                        conversations.append({
                            'role': role,
                            'content': final_content,
                            'timestamp': timestamp,
                            'metadata': {
                                'source': 'LM_Studio',
                                'model': data.get('lastUsedModel', {}).get('name', 'unknown'),
                                'conversation_name': data.get('name', 'Unknown'),
                                'version_index': current_version,
                                'message_index': i
                            }
                        })
        except Exception as e:
            logger.error(f"Error parsing LM Studio format: {e}")
        
        return conversations
    
    def _extract_ollama_database(self, db_path: str) -> List[Dict]:
        """Extract conversations from Ollama SQLite database."""
        conversations = []
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all chats with their messages
            cursor.execute("""
                SELECT c.id, c.title, c.created_at,
                       m.role, m.content, m.model_name, m.created_at as message_created_at
                FROM chats c
                LEFT JOIN messages m ON c.id = m.chat_id
                ORDER BY c.created_at, m.created_at
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.debug(f"No conversations found in Ollama database: {db_path}")
                return conversations
            
            # Group messages by chat and convert to conversation format
            chats = {}
            for row in rows:
                chat_id, title, chat_created_at, role, content, model_name, msg_created_at = row
                
                if chat_id not in chats:
                    chats[chat_id] = {
                        'title': title,
                        'created_at': chat_created_at,
                        'messages': []
                    }
                
                if role and content:  # Only add if message exists
                    # Convert timestamp if needed
                    timestamp = None
                    if msg_created_at:
                        try:
                            # Parse ISO format timestamp
                            timestamp = datetime.fromisoformat(msg_created_at.replace('Z', '+00:00')).isoformat()
                        except (ValueError, TypeError):
                            pass
                    
                    conversations.append({
                        'role': role,
                        'content': content,
                        'timestamp': timestamp,
                        'metadata': {
                            'source': 'Ollama',
                            'model': model_name or 'unknown',
                            'chat_id': chat_id,
                            'chat_title': title
                        }
                    })
            
            logger.info(f"Extracted {len(conversations)} messages from {len(chats)} Ollama chats")
            
        except Exception as e:
            logger.error(f"Error extracting Ollama database {db_path}: {e}")
        
        return conversations
    
    def _extract_openwebui_database(self, db_path: str) -> List[Dict]:
        """Extract conversations from OpenWebUI SQLite database."""
        conversations = []
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all chats with their messages
            cursor.execute("""
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       m.id as message_id, m.role, m.content, m.created_at as message_created_at
                FROM chat c
                LEFT JOIN message m ON c.id = m.chat_id
                ORDER BY c.created_at, m.created_at
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.debug(f"No conversations found in OpenWebUI database: {db_path}")
                return conversations
            
            # Convert to conversation format
            for row in rows:
                chat_id, title, chat_created_at, chat_updated_at, message_id, role, content, msg_created_at = row
                
                if role and content:  # Only add if message exists
                    # Convert timestamp if needed
                    timestamp = None
                    if msg_created_at:
                        try:
                            # Parse timestamp (OpenWebUI typically uses ISO format)
                            if isinstance(msg_created_at, (int, float)):
                                timestamp = datetime.fromtimestamp(msg_created_at).isoformat()
                            else:
                                timestamp = datetime.fromisoformat(str(msg_created_at).replace('Z', '+00:00')).isoformat()
                        except (ValueError, TypeError):
                            pass
                    
                    conversations.append({
                        'role': role,
                        'content': content,
                        'timestamp': timestamp,
                        'metadata': {
                            'source': 'OpenWebUI',
                            'chat_id': chat_id,
                            'chat_title': title or f'Chat {chat_id}',
                            'message_id': message_id
                        }
                    })
            
            logger.info(f"Extracted {len(conversations)} messages from OpenWebUI database")
            
        except Exception as e:
            logger.error(f"Error extracting OpenWebUI database {db_path}: {e}")
        
        return conversations
    
    def _parse_text_format(self, content: str) -> List[Dict]:
        """Parse text-based conversation formats with timestamp detection"""
        conversations = []
        lines = content.split('\n')
        
        current_role = 'user'
        current_content = []
        current_timestamp = None
        
        # Common timestamp patterns
        timestamp_patterns = [
            r'\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]',  # ISO format
            r'\[(\d{2}:\d{2}(?::\d{2})?)\]',  # Time only
            r'\[(\d{4}-\d{2}-\d{2})\]',  # Date only
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract timestamp
            for pattern in timestamp_patterns:
                match = re.match(pattern, line)
                if match:
                    try:
                        ts = match.group(1)
                        # Handle time-only format by adding today's date
                        if re.match(r'\d{2}:\d{2}(?::\d{2})?$', ts):
                            ts = f"{datetime.now().date()}T{ts}"
                        current_timestamp = datetime.fromisoformat(ts).isoformat()
                        line = line[match.end():].strip()
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Detect role markers
            if line.lower().startswith(('user:', 'human:', 'me:')):
                if current_content:
                    conversations.append({
                        'role': current_role,
                        'content': '\n'.join(current_content),
                        'timestamp': current_timestamp
                    })
                    current_content = []
                current_role = 'user'
                content_part = line.split(':', 1)[1].strip() if ':' in line else line
                if content_part:
                    current_content.append(content_part)
                    
            elif line.lower().startswith(('assistant:', 'ai:', 'bot:', 'friday:')):
                if current_content:
                    conversations.append({
                        'role': current_role,
                        'content': '\n'.join(current_content)
                    })
                    current_content = []
                current_role = 'assistant'
                content_part = line.split(':', 1)[1].strip() if ':' in line else line
                if content_part:
                    current_content.append(content_part)
            else:
                current_content.append(line)
        
        # Add the last conversation
        if current_content:
            conversations.append({
                'role': current_role,
                'content': '\n'.join(current_content)
            })
        
        return conversations

    def _parse_sillytavern_format(self, data: Dict) -> List[Dict]:
        """Parse SillyTavern conversation format"""
        conversations = []
        
        try:
            # SillyTavern typically stores chats in nested format
            if 'messages' in data:
                for msg in data['messages']:
                    # SillyTavern message format
                    role = 'user' if msg.get('is_user', False) else 'assistant'
                    content = msg.get('mes', msg.get('message', ''))
                    timestamp = msg.get('send_date', msg.get('timestamp'))
                    
                    if content:
                        conversations.append({
                            'role': role,
                            'content': str(content),
                            'timestamp': self.parse_timestamp(timestamp),
                            'metadata': {'source_type': 'sillytavern'}
                        })
            
            # Alternative format for SillyTavern exports
            elif 'chat' in data and isinstance(data['chat'], list):
                for msg in data['chat']:
                    conversations.append({
                        'role': 'user' if msg.get('is_user') else 'assistant',
                        'content': str(msg.get('mes', '')),
                        'timestamp': self.parse_timestamp(msg.get('send_date')),
                        'metadata': {'source_type': 'sillytavern'}
                    })
                    
        except Exception as e:
            logger.warning(f"Error parsing SillyTavern format: {e}")
        
        return conversations

    def _parse_gemini_cli_format(self, data: Dict) -> List[Dict]:
        """Parse Gemini CLI conversation format"""
        conversations = []
        
        try:
            # Gemini CLI format (assuming similar to other CLI tools)
            if 'conversation' in data and isinstance(data['conversation'], list):
                for turn in data['conversation']:
                    # User input
                    if 'input' in turn:
                        conversations.append({
                            'role': 'user',
                            'content': str(turn['input']),
                            'timestamp': self.parse_timestamp(turn.get('timestamp')),
                            'metadata': {'source_type': 'gemini_cli'}
                        })
                    
                    # Assistant response
                    if 'response' in turn:
                        conversations.append({
                            'role': 'assistant',
                            'content': str(turn['response']),
                            'timestamp': self.parse_timestamp(turn.get('timestamp')),
                            'metadata': {'source_type': 'gemini_cli'}
                        })
            
            # Alternative format with messages array
            elif 'messages' in data:
                for msg in data['messages']:
                    role = msg.get('role', 'user')
                    if role == 'model':  # Gemini uses 'model' instead of 'assistant'
                        role = 'assistant'
                    
                    content = ''
                    if 'parts' in msg and isinstance(msg['parts'], list):
                        # Gemini format with parts array
                        content = ' '.join(str(part.get('text', part)) for part in msg['parts'])
                    else:
                        content = str(msg.get('content', msg.get('text', '')))
                    
                    if content:
                        conversations.append({
                            'role': role,
                            'content': content,
                            'timestamp': self.parse_timestamp(msg.get('timestamp')),
                            'metadata': {'source_type': 'gemini_cli'}
                        })
                        
        except Exception as e:
            logger.warning(f"Error parsing Gemini CLI format: {e}")
        
        return conversations


class EmbeddingService:
    """Intelligent embedding service that preserves existing embeddings while optimizing for quality"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize embedding service with intelligent configuration
        
        Args:
            config: Optional configuration dictionary. If None, loads from embedding_config.json
        """
        if config:
            self.primary_config = config
            self.fallback_config = config.get("fallback", {})
            self.full_config = {"primary": config, "fallback": self.fallback_config}
        else:
            self.full_config = self._load_full_config()
            self.primary_config = self.full_config.get("primary", {})
            self.fallback_config = self.full_config.get("fallback", {})
            
        self.provider_availability = {
            "lm_studio": None,  # Will be tested on first use
            "ollama": None,
            "openai": None
        }
        
    print("Intelligent Embedding Service Configuration")
    primary_provider = self.primary_config.get('provider', 'lm_studio')
    primary_model = self.primary_config.get('model', 'text-embedding-nomic-embed-text-v1.5')
    fallback_provider = self.fallback_config.get('provider', 'ollama')
    fallback_model = self.fallback_config.get('model', 'nomic-embed-text')
    
    print(f"Primary: {primary_provider} ({primary_model})")
    print(f"Fallback: {fallback_provider} ({fallback_model})")
    print(f"Preserving existing 768D embeddings, using best available for new ones")
    print("To customize, edit embedding_config.json in the project directory")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Backward compatibility property - returns primary config as expected format"""
        return {
            "provider": self.primary_config.get("provider"),
            "model": self.primary_config.get("model"),
            "base_url": self.primary_config.get("base_url"),
            "api_key": self.primary_config.get("api_key"),
            "fallback_provider": self.fallback_config.get("provider"),
            "fallback_model": self.fallback_config.get("model"),
            "fallback_base_url": self.fallback_config.get("base_url"),
            "fallback_api_key": self.fallback_config.get("api_key")
        }
    
    def _load_full_config(self) -> dict:
        """Load complete embedding configuration from JSON file"""
        try:
            config_path = Path(__file__).parent / "embedding_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return config_data.get("embedding_configuration", {})
        except Exception as e:
            logger.warning(f"Failed to load embedding config: {e}, using defaults")
        
        # Return default configuration
        return {
            "primary": {
                "provider": "lm_studio",
                "model": "text-embedding-nomic-embed-text-v1.5",
                "base_url": "http://localhost:1234",
                "description": "High-quality LM Studio embeddings for semantic search"
            },
            "fallback": {
                "provider": "ollama",
                "model": "nomic-embed-text", 
                "base_url": "http://localhost:11434",
                "description": "Fast local Ollama embeddings"
            }
        }
    
    @classmethod
    def create_with_user_config(cls) -> 'EmbeddingService':
        """Create embedding service with user configuration prompt"""
        try:
            print("🔧 Embedding Service Configuration")
            print("Loading configuration from embedding_config.json...")
            return cls()  # Use config file
            
        except Exception as e:
            logger.warning(f"Failed to get user config, using defaults: {e}")
            return cls()  # Fallback to defaults

    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embedding using intelligent provider selection with preservation strategy"""
        
        # Try primary provider first
        primary_provider = self.primary_config.get("provider", "lm_studio")
        
        try:
            if primary_provider == "lm_studio":
                result = await self._generate_lm_studio_embedding(text)
                if result:
                    self.provider_availability["lm_studio"] = True
                    return result
                else:
                    self.provider_availability["lm_studio"] = False
                    logger.warning("LM Studio unavailable, trying fallback")
                    
            elif primary_provider == "ollama":
                result = await self._generate_ollama_embedding(text)
                if result:
                    self.provider_availability["ollama"] = True
                    return result
                else:
                    self.provider_availability["ollama"] = False
                    logger.warning("Ollama unavailable, trying fallback")
                    
            elif primary_provider == "openai":
                result = await self._generate_openai_embedding(text)
                if result:
                    self.provider_availability["openai"] = True
                    return result
                else:
                    self.provider_availability["openai"] = False
                    logger.warning("OpenAI unavailable, trying fallback")
                    
        except Exception as e:
            logger.warning(f"Primary provider {primary_provider} failed: {e}")
        
        # Try fallback provider
        fallback_provider = self.fallback_config.get("provider")
        if fallback_provider and fallback_provider != primary_provider:
            try:
                if fallback_provider == "lm_studio":
                    result = await self._generate_lm_studio_embedding(text, fallback=True)
                    if result:
                        self.provider_availability["lm_studio"] = True
                        logger.info("Using LM Studio fallback for embedding")
                        return result
                        
                elif fallback_provider == "ollama":
                    result = await self._generate_ollama_embedding(text, fallback=True)
                    if result:
                        self.provider_availability["ollama"] = True
                        logger.info("Using Ollama fallback for embedding")
                        return result
                        
                elif fallback_provider == "openai":
                    result = await self._generate_openai_embedding(text, fallback=True)
                    if result:
                        self.provider_availability["openai"] = True
                        logger.info("Using OpenAI fallback for embedding")
                        return result
                        
            except Exception as e:
                logger.error(f"Fallback provider {fallback_provider} also failed: {e}")
        
        # If both primary and fallback fail, log the issue
        logger.error("All embedding providers failed - semantic search will be unavailable")
        return []
    
    async def _generate_ollama_embedding(self, text: str, fallback: bool = False) -> List[float]:
        """Generate embedding using Ollama"""
        if fallback:
            config = self.fallback_config
        else:
            config = self.primary_config if self.primary_config.get("provider") == "ollama" else self.fallback_config
            
        base_url = config.get("base_url", "http://localhost:11434")
        model = config.get("model", "nomic-embed-text")
        
        async with aiohttp.ClientSession() as session:
            payload = {"model": model, "prompt": text}
            async with session.post(f"{base_url}/api/embeddings", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding")
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    return None
    
    async def _generate_lm_studio_embedding(self, text: str, fallback: bool = False) -> List[float]:
        """Generate embedding using LM Studio"""
        if fallback:
            config = self.fallback_config
        else:
            config = self.primary_config if self.primary_config.get("provider") == "lm_studio" else self.fallback_config
            
        base_url = config.get("base_url", "http://localhost:1234")
        model = config.get("model", "text-embedding-nomic-embed-text-v1.5")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"model": model, "input": text}
                async with session.post(f"{base_url}/v1/embeddings", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and "data" in data and len(data["data"]) > 0:
                            embedding = data["data"][0].get("embedding")
                            if embedding:
                                return embedding
                        logger.error(f"Invalid LM Studio response format: {data}")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"LM Studio API error {response.status}: {error_text}")
                        return None
        except Exception as e:
            logger.error(f"LM Studio embedding error: {e}")
            return None
    
    async def _generate_openai_embedding(self, text: str, fallback: bool = False) -> List[float]:
        """Generate embedding using OpenAI"""
        if fallback:
            config = self.fallback_config
        else:
            config = self.primary_config if self.primary_config.get("provider") == "openai" else self.fallback_config
            
        api_key = config.get("api_key")
        if not api_key or api_key == "your-openai-api-key-here":
            logger.error("OpenAI API key not configured")
            return None
            
        model = config.get("model", "text-embedding-3-small")
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {"model": model, "input": text}
                async with session.post("https://api.openai.com/v1/embeddings", 
                                        json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and "data" in data and len(data["data"]) > 0:
                            return data["data"][0].get("embedding")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error {response.status}: {error_text}")
                        return None
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return None


class PersistentAIMemorySystem:
    """Main memory system that coordinates all databases and operations - FULL FEATURED VERSION"""
    
    def __init__(self, settings=None, enable_file_monitoring: bool = None, 
                 watch_directories: List[str] = None):
        # Use provided settings or get global settings
        if settings is None:
            settings = get_settings()
        self.settings = settings
        self.data_dir = settings.data_dir
        
        # Override file monitoring setting if explicitly provided
        if enable_file_monitoring is None:
            enable_file_monitoring = settings.enable_file_monitoring
        
        # Initialize all 5 databases using settings paths
        self.conversations_db = ConversationDatabase()
        self.ai_memory_db = AIMemoryDatabase()
        self.schedule_db = ScheduleDatabase()
        self.vscode_db = VSCodeProjectDatabase()
        self.mcp_db = MCPToolCallDatabase()
        
        # Initialize embedding service with user-configurable options
        self.embedding_service = EmbeddingService.create_with_user_config()
        
        # Initialize file monitoring
        self.file_monitor = None
        if enable_file_monitoring:
            # Use watch_directories parameter or fall back to settings
            watch_dirs = watch_directories or settings.watch_directories
            self.file_monitor = ConversationFileMonitor(self, watch_dirs)
    
    async def start_file_monitoring(self):
        """Start monitoring conversation files"""
        if self.file_monitor:
            await self.file_monitor.start_monitoring()
            logger.info("File monitoring started")
    
    async def stop_file_monitoring(self):
        """Stop monitoring conversation files"""
        if self.file_monitor:
            await self.file_monitor.stop_monitoring()
            logger.info("File monitoring stopped")
    
    def add_watch_directory(self, directory: str):
        """Add a directory to monitor for conversation files"""
        if self.file_monitor:
            self.file_monitor.add_watch_directory(directory)

    # =============================================================================
    # CONVERSATION OPERATIONS
    # =============================================================================
    
    async def store_conversation(self, content: str, role: str, session_id: str = None,
                               conversation_id: str = None, metadata: Dict = None) -> Dict:
        """Store a conversation message with automatic embedding generation"""
        
        result = await self.conversations_db.store_message(
            content, role, session_id, conversation_id, metadata
        )
        
        # Generate and store embedding asynchronously
        asyncio.create_task(self._add_embedding_to_message(result["message_id"], content))
        
        return {
            "status": "success",
            "message_id": result["message_id"],
            "conversation_id": result["conversation_id"],
            "session_id": result["session_id"]
        }
    
    async def get_conversation_history(self, limit: int = 20, session_id: str = None) -> List[Dict]:
        """Get recent conversation history"""
        
        messages = await self.conversations_db.get_recent_messages(limit, session_id)
        return [dict(msg) for msg in messages]

    async def get_recent_context(self, limit: int = 10, session_id: str = None) -> Dict:
        """Retrieve recent conversation context, optionally filtered by session."""
        try:
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

            rows = await self.conversations_db.execute_query(query, params)
            return {
                "status": "success",
                "recent_context": [dict(row) for row in rows]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # =============================================================================
    # AI MEMORY OPERATIONS
    # =============================================================================
    
    async def create_memory(self, content: str, memory_type: str = None,
                          importance_level: int = 5, tags: List[str] = None,
                          source_conversation_id: str = None) -> Dict:
        """Create a curated AI memory with automatic embedding generation"""
        
        memory_id = await self.ai_memory_db.create_memory(
            content, memory_type, importance_level, tags, source_conversation_id
        )
        
        # Generate and store embedding asynchronously
        asyncio.create_task(self._add_embedding_to_memory(memory_id, content))
        
        return {
            "status": "success",
            "memory_id": memory_id
        }

    # =============================================================================
    # SCHEDULE OPERATIONS
    # =============================================================================
    
    async def create_appointment(self, title: str, scheduled_datetime: str, 
                               description: str = None, location: str = None,
                               source_conversation_id: str = None) -> Dict:
        """Create an appointment with automatic embedding generation"""
        
        appointment_id = await self.schedule_db.create_appointment(
            title, scheduled_datetime, description, location, source_conversation_id
        )
        
        # Generate embedding for search (combine title and description)
        content_for_embedding = f"{title}"
        if description:
            content_for_embedding += f" {description}"
        
        asyncio.create_task(self._add_embedding_to_appointment(appointment_id, content_for_embedding))
        
        return {
            "status": "success",
            "appointment_id": appointment_id
        }
    
    async def create_reminder(self, content: str, due_datetime: str, 
                            priority_level: int = 5, source_conversation_id: str = None) -> Dict:
        """Create a reminder with automatic embedding generation"""
        
        reminder_id = await self.schedule_db.create_reminder(
            content, due_datetime, priority_level, source_conversation_id
        )
        
        # Generate and store embedding for the reminder content
        asyncio.create_task(self._add_embedding_to_reminder(reminder_id, content))
        
        return {
            "status": "success",
            "reminder_id": reminder_id
        }
    
    async def get_upcoming_schedule(self, days_ahead: int = 7) -> Dict:
        """Get upcoming appointments and reminders"""
        
        appointments = await self.schedule_db.get_upcoming_appointments(days_ahead)
        reminders = await self.schedule_db.get_active_reminders()
        
        return {
            "status": "success",
            "appointments": appointments,
            "active_reminders": reminders,
            "period_days": days_ahead
        }

    # =============================================================================
    # VSCODE PROJECT OPERATIONS
    # =============================================================================
    
    async def save_development_session(self, workspace_path: str, active_files: List[str] = None,
                                     git_branch: str = None, session_summary: str = None) -> Dict:
        """Save development session"""
        
        session_id = await self.vscode_db.save_development_session(
            workspace_path, active_files, git_branch, session_summary
        )
        
        return {
            "status": "success",
            "session_id": session_id
        }
    
    async def store_project_insight(self, content: str, insight_type: str = None,
                                  related_files: List[str] = None, importance_level: int = 5,
                                  source_conversation_id: str = None) -> Dict:
        """Store project insight with automatic embedding generation"""
        
        insight_id = await self.vscode_db.store_project_insight(
            content, insight_type, related_files, importance_level, source_conversation_id
        )
        
        # Generate and store embedding for the insight content
        asyncio.create_task(self._add_embedding_to_project_insight(insight_id, content))
        
        return {
            "status": "success",
            "insight_id": insight_id
        }

    # =============================================================================
    # MCP TOOL CALL OPERATIONS
    # =============================================================================
    
    async def log_tool_call(self, tool_name: str, parameters: Dict = None,
                          execution_time_ms: float = None, status: str = "success",
                          result: Any = None, error_message: str = None, client_id: str = None) -> str:
        """Log an MCP tool call for analysis and debugging"""
        
        return await self.mcp_db.log_tool_call(
            tool_name, parameters, result, status, execution_time_ms, error_message, client_id
        )
    
    async def get_tool_usage_summary(self, days: int = 7) -> Dict:
        """Get comprehensive tool usage summary"""
        
        return await self.mcp_db.get_tool_usage_summary(days)

    async def get_ai_insights(self, limit: int = 10, reflection_type: str = None, insight_type: str = None) -> Dict:
        """Unified method to retrieve AI insights and reflections from both MCP and VS Code project databases."""
        results = []
        
        # Get MCP reflections
        mcp_reflections = await self.mcp_db.get_recent_reflections(limit=limit, reflection_type=reflection_type)
        for reflection in mcp_reflections:
            results.append({
                "source": "mcp_reflection",
                "reflection_id": reflection.get("reflection_id"),
                "timestamp": reflection.get("timestamp"),
                "reflection_type": reflection.get("reflection_type"),
                "content": reflection.get("content"),
                "insights": json.loads(reflection["insights"]) if reflection.get("insights") else None,
                "recommendations": json.loads(reflection["recommendations"]) if reflection.get("recommendations") else None,
                "confidence_level": reflection.get("confidence_level"),
                "source_period_days": reflection.get("source_period_days")
            })
        
        # Get VS Code project insights
        query = "SELECT * FROM project_insights"
        params = []
        where_clauses = []
        
        if insight_type:
            where_clauses.append("insight_type = ?")
            params.append(insight_type)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY timestamp_created DESC LIMIT ?"
        params.append(limit)
        
        project_insights = await self.vscode_db.execute_query(query, tuple(params))
        for insight in project_insights:
            results.append({
                "source": "project_insight",
                "insight_id": insight.get("insight_id"),
                "timestamp_created": insight.get("timestamp_created"),
                "timestamp_updated": insight.get("timestamp_updated"),
                "insight_type": insight.get("insight_type"),
                "content": insight.get("content"),
                "related_files": json.loads(insight["related_files"]) if insight.get("related_files") else None,
                "importance_level": insight.get("importance_level"),
                "source_conversation_id": insight.get("source_conversation_id")
            })
        
        # Sort by timestamp (descending)
        results.sort(key=lambda x: x.get("timestamp", x.get("timestamp_created", "")), reverse=True)
        
        return {
            "status": "success",
            "count": len(results),
            "results": results[:limit]
        }

    # =============================================================================
    # ADVANCED SEARCH OPERATIONS
    # =============================================================================
    
    async def search_project_history(self, query: str, limit: int = 10) -> Dict:
        """Search project development history including conversations and insights.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            Dict containing search results from project context
        """
        query_embedding = await self.embedding_service.generate_embedding(query)
        if not query_embedding:
            return await self._text_based_project_search(query, limit)
            
        results = []
        
        # Search development conversations
        conv_results = await self._search_development_conversations(query_embedding, limit)
        results.extend(conv_results)
        
        # Search project insights
        insight_results = await self._search_project_insights(query_embedding, limit)
        results.extend(insight_results)
        
        # Search code context
        context_results = await self._search_code_context(query_embedding, limit)
        results.extend(context_results)
        
        # Sort by relevance and return
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return {
            "status": "success",
            "query": query,
            "results": results[:limit],
            "count": len(results[:limit])
        }
        
    async def link_code_context(self, file_path: str, description: str,
                              function_name: str = None, conversation_id: str = None) -> Dict:
        """Link conversation context to specific code location.
        
        Args:
            file_path: Path to the code file
            description: Description of the code context
            function_name: Optional function/method name
            conversation_id: Optional related conversation ID
            
        Returns:
            Dict containing the created context link
        """
        context_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.vscode_db.execute_update(
            """INSERT INTO code_context 
               (context_id, timestamp, file_path, function_name, description)
               VALUES (?, ?, ?, ?, ?)""",
            (context_id, timestamp, file_path, function_name, description)
        )
        
        if conversation_id:
            await self.vscode_db.execute_update(
                """UPDATE development_conversations
                   SET chat_context_id = ?
                   WHERE conversation_id = ?""",
                (context_id, conversation_id)
            )
            
        # Generate embedding for search
        asyncio.create_task(self._add_embedding_to_code_context(context_id, description))
        
        return {
            "status": "success",
            "context_id": context_id
        }
        
    async def get_project_continuity(self, workspace_path: str = None, limit: int = 5) -> Dict:
        """Get context for continuing development work.
        
        Args:
            workspace_path: Optional workspace path filter
            limit: Maximum number of context items
            
        Returns:
            Dict containing recent development context
        """
        # Get recent development sessions
        sessions_query = """
            SELECT * FROM project_sessions
            WHERE end_timestamp IS NULL
        """
        if workspace_path:
            sessions_query += " AND workspace_path = ?"
            sessions = await self.vscode_db.execute_query(
                sessions_query + " ORDER BY start_timestamp DESC LIMIT ?",
                (workspace_path, limit)
            )
        else:
            sessions = await self.vscode_db.execute_query(
                sessions_query + " ORDER BY start_timestamp DESC LIMIT ?",
                (limit,)
            )
            
        # Get associated conversations and insights
        context = {
            "active_sessions": [dict(session) for session in sessions],
            "recent_conversations": [],
            "relevant_insights": []
        }
        
        for session in sessions:
            # Get conversations for this session
            convs = await self.vscode_db.execute_query(
                """SELECT * FROM development_conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session["session_id"], limit)
            )
            context["recent_conversations"].extend([dict(conv) for conv in convs])
            
            # Get insights mentioning active files
            if session["active_files"]:
                active_files = json.loads(session["active_files"])
                for file in active_files:
                    insights = await self.vscode_db.execute_query(
                        """SELECT * FROM project_insights
                           WHERE related_files LIKE ?
                           ORDER BY timestamp_created DESC LIMIT ?""",
                        (f"%{file}%", limit)
                    )
                    context["relevant_insights"].extend([dict(insight) for insight in insights])
        
        return {
            "status": "success",
            "context": context
        }
            
    async def search_memories(self, query: str, limit: int = 10, 
                            min_importance: int = None, max_importance: int = None,
                            memory_type: str = None, database_filter: str = "all") -> Dict:
        """Advanced semantic search across all databases with filtering"""
        
        # Generate embedding for the search query
        query_embedding = await self.embedding_service.generate_embedding(query)
        if not query_embedding:
            # Fallback to text-based search if embedding fails
            return await self._text_based_search(query, limit, database_filter, min_importance, max_importance, memory_type)
        
        all_results = []
        
        # Search AI memories
        if database_filter in ["all", "ai_memories"]:
            memory_results = await self._search_ai_memories(query_embedding, limit, min_importance, max_importance, memory_type)
            all_results.extend(memory_results)
        
        # Search conversations
        if database_filter in ["all", "conversations"]:
            conversation_results = await self._search_conversations(query_embedding, limit)
            all_results.extend(conversation_results)
        
        # Search schedule items
        if database_filter in ["all", "schedule"]:
            schedule_results = await self._search_schedule(query_embedding, limit)
            all_results.extend(schedule_results)
        
        # Search project insights
        if database_filter in ["all", "projects"]:
            project_results = await self._search_project_insights(query_embedding, limit)
            all_results.extend(project_results)
        
        # Sort all results by similarity score and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "query": query,
            "results": all_results[:limit],
            "count": len(all_results[:limit]),
            "search_type": "semantic" if query_embedding else "text_based"
        }

    # =============================================================================
    # SYSTEM HEALTH AND MONITORING
    # =============================================================================
    
    async def get_system_health(self) -> Dict:
        """Get comprehensive system health and statistics"""
        health_data = {
            "status": "healthy",
            "timestamp": get_current_timestamp(),
            "databases": {},
            "file_monitoring": {},
            "embedding_service": {}
        }
        
        try:
            # Check conversations database
            conversations_count = await self.conversations_db.execute_query(
                "SELECT COUNT(*) as count FROM messages"
            )
            sessions_count = await self.conversations_db.execute_query(
                "SELECT COUNT(*) as count FROM sessions"
            )
            health_data["databases"]["conversations"] = {
                "status": "healthy",
                "message_count": conversations_count[0]["count"] if conversations_count else 0,
                "session_count": sessions_count[0]["count"] if sessions_count else 0,
                "database_path": self.conversations_db.db_path
            }
            
            # Check AI memories database
            memories_count = await self.ai_memory_db.execute_query(
                "SELECT COUNT(*) as count FROM curated_memories"
            )
            high_importance_count = await self.ai_memory_db.execute_query(
                "SELECT COUNT(*) as count FROM curated_memories WHERE importance_level >= 7"
            )
            health_data["databases"]["ai_memories"] = {
                "status": "healthy",
                "memory_count": memories_count[0]["count"] if memories_count else 0,
                "high_importance_count": high_importance_count[0]["count"] if high_importance_count else 0,
                "database_path": self.ai_memory_db.db_path
            }
            
            # Check schedule database
            appointments_count = await self.schedule_db.execute_query(
                "SELECT COUNT(*) as count FROM appointments"
            )
            reminders_count = await self.schedule_db.execute_query(
                "SELECT COUNT(*) as count FROM reminders"
            )
            health_data["databases"]["schedule"] = {
                "status": "healthy",
                "appointment_count": appointments_count[0]["count"] if appointments_count else 0,
                "reminder_count": reminders_count[0]["count"] if reminders_count else 0,
                "database_path": self.schedule_db.db_path
            }
            
            # Check VS Code project database
            project_sessions_count = await self.vscode_db.execute_query(
                "SELECT COUNT(*) as count FROM project_sessions"
            )
            insights_count = await self.vscode_db.execute_query(
                "SELECT COUNT(*) as count FROM project_insights"
            )
            health_data["databases"]["vscode_project"] = {
                "status": "healthy",
                "session_count": project_sessions_count[0]["count"] if project_sessions_count else 0,
                "insight_count": insights_count[0]["count"] if insights_count else 0,
                "database_path": self.vscode_db.db_path
            }
            
            # Check MCP tool calls database
            tool_calls_count = await self.mcp_db.execute_query(
                "SELECT COUNT(*) as count FROM tool_calls"
            )
            health_data["databases"]["mcp_tool_calls"] = {
                "status": "healthy",
                "total_tool_calls": tool_calls_count[0]["count"] if tool_calls_count else 0,
                "database_path": self.mcp_db.db_path
            }
            
            # Check file monitoring status
            if self.file_monitor:
                health_data["file_monitoring"] = {
                    "status": "enabled",
                    "watch_directories": len(self.file_monitor.watch_directories),
                    "directories": self.file_monitor.watch_directories
                }
            else:
                health_data["file_monitoring"] = {
                    "status": "disabled",
                    "message": "File monitoring is not enabled"
                }
            
            # Check embedding service
            try:
                # Try a simple ping to the embedding service
                test_embedding = await self.embedding_service.generate_embedding("test")
                if test_embedding:
                    health_data["embedding_service"] = {
                        "status": "healthy",
                        "endpoint": self.embedding_service.embeddings_endpoint,
                        "embedding_dimensions": len(test_embedding)
                    }
                else:
                    health_data["embedding_service"] = {
                        "status": "unhealthy",
                        "endpoint": self.embedding_service.embeddings_endpoint,
                        "error": "Failed to generate test embedding"
                    }
            except Exception as e:
                health_data["embedding_service"] = {
                    "status": "unhealthy",
                    "endpoint": self.embedding_service.embeddings_endpoint,
                    "error": str(e)
                }
            
            # Overall system status
            unhealthy_components = []
            if health_data["embedding_service"]["status"] == "unhealthy":
                unhealthy_components.append("embedding_service")
            
            if unhealthy_components:
                health_data["status"] = "degraded"
                health_data["issues"] = unhealthy_components
            
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
            logger.error(f"Error getting system health: {e}")
        
        return health_data

    # =============================================================================
    # INTERNAL HELPER METHODS
    # =============================================================================
    
    async def _search_ai_memories(self, query_embedding: List[float], limit: int,
                                min_importance: int = None, max_importance: int = None,
                                memory_type: str = None) -> List[Dict]:
        """Search AI curated memories using semantic similarity"""
        
        # Build SQL query with optional filters
        sql = "SELECT * FROM curated_memories WHERE embedding IS NOT NULL"
        params = []
        
        if min_importance is not None:
            sql += " AND importance_level >= ?"
            params.append(min_importance)
            
        if max_importance is not None:
            sql += " AND importance_level <= ?"
            params.append(max_importance)
            
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        
        rows = await self.ai_memory_db.execute_query(sql, params)
        results = []
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:  # Threshold for relevance
                    result = {
                        "type": "ai_memory",
                        "similarity_score": similarity,
                        "data": {
                            "memory_id": row["memory_id"],
                            "content": row["content"],
                            "importance_level": row["importance_level"],
                            "memory_type": row["memory_type"],
                            "timestamp_created": row["timestamp_created"],
                            "tags": json.loads(row["tags"]) if row["tags"] else []
                        }
                    }
                    results.append(result)
        
        # Boost results based on importance level
        for result in results:
            importance_boost = result["data"]["importance_level"] / 10.0 * 0.1
            result["similarity_score"] += importance_boost
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _search_conversations(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search conversation messages using semantic similarity"""
        
        query = """
            SELECT message_id, conversation_id, timestamp, role, content, metadata, embedding
            FROM messages 
            WHERE embedding IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        
        rows = await self.conversations_db.execute_query(query)
        results = []
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:
                    result = {
                        "type": "conversation",
                        "similarity_score": similarity,
                        "data": {
                            "message_id": row["message_id"],
                            "conversation_id": row["conversation_id"],
                            "timestamp": row["timestamp"],
                            "role": row["role"],
                            "content": row["content"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                        }
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _search_schedule(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search appointments and reminders using semantic similarity"""
        
        results = []
        
        # Search appointments
        appointment_query = """
            SELECT appointment_id, timestamp_created, scheduled_datetime, title, 
                   description, location, source_conversation_id, embedding
            FROM appointments 
            WHERE embedding IS NOT NULL
        """
        
        rows = await self.schedule_db.execute_query(appointment_query)
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:
                    result = {
                        "type": "appointment",
                        "similarity_score": similarity,
                        "data": {
                            "appointment_id": row["appointment_id"],
                            "scheduled_datetime": row["scheduled_datetime"],
                            "title": row["title"],
                            "description": row["description"],
                            "location": row["location"]
                        }
                    }
                    results.append(result)
        
        # Search reminders
        reminder_query = """
            SELECT reminder_id, timestamp_created, due_datetime, content, 
                   priority_level, completed, source_conversation_id, embedding
            FROM reminders 
            WHERE embedding IS NOT NULL
        """
        
        rows = await self.schedule_db.execute_query(reminder_query)
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:
                    result = {
                        "type": "reminder",
                        "similarity_score": similarity,
                        "data": {
                            "reminder_id": row["reminder_id"],
                            "due_datetime": row["due_datetime"],
                            "content": row["content"],
                            "priority_level": row["priority_level"],
                            "completed": bool(row["completed"])
                        }
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _search_project_insights(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search project insights using semantic similarity"""
        
        query = """
            SELECT insight_id, timestamp_created, insight_type, content, 
                   related_files, importance_level, embedding
            FROM project_insights 
            WHERE embedding IS NOT NULL
        """
        
        rows = await self.vscode_db.execute_query(query)
        results = []
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:
                    result = {
                        "type": "project_insight",
                        "similarity_score": similarity,
                        "data": {
                            "insight_id": row["insight_id"],
                            "timestamp_created": row["timestamp_created"],
                            "insight_type": row["insight_type"],
                            "content": row["content"],
                            "related_files": json.loads(row["related_files"]) if row["related_files"] else None,
                            "importance_level": row["importance_level"]
                        }
                    }
                    results.append(result)
        
        # Boost results based on importance level
        for result in results:
            importance_boost = result["data"]["importance_level"] / 10.0 * 0.15
            result["similarity_score"] += importance_boost
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _text_based_search(self, query: str, limit: int, database_filter: str,
                               min_importance: int = None, max_importance: int = None,
                               memory_type: str = None) -> Dict:
        """Fallback text-based search when embeddings are unavailable"""
        
        query_words = query.lower().split()
        results = []
        
        if database_filter in ["all", "ai_memories"]:
            # Search AI memories with text matching and filters
            sql = "SELECT * FROM curated_memories WHERE 1=1"
            params = []
            
            # Add content search
            content_conditions = []
            for word in query_words:
                content_conditions.append("LOWER(content) LIKE ?")
                params.append(f"%{word}%")
            
            if content_conditions:
                sql += f" AND ({' OR '.join(content_conditions)})"
            
            # Add importance filters
            if min_importance is not None:
                sql += " AND importance_level >= ?"
                params.append(min_importance)
                
            if max_importance is not None:
                sql += " AND importance_level <= ?"
                params.append(max_importance)
                
            if memory_type is not None:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            sql += " ORDER BY importance_level DESC LIMIT ?"
            params.append(limit)
            
            rows = await self.ai_memory_db.execute_query(sql, params)
            for row in rows:
                results.append({
                    "type": "ai_memory",
                    "similarity_score": 0.5,
                    "data": dict(row)
                })
        
        if database_filter in ["all", "conversations"]:
            # Search conversations with text matching
            for word in query_words:
                rows = await self.conversations_db.execute_query(
                    "SELECT * FROM messages WHERE LOWER(content) LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{word}%", limit)
                )
                for row in rows:
                    results.append({
                        "type": "conversation",
                        "similarity_score": 0.5,
                        "data": dict(row)
                    })
        
        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            key = f"{result['type']}_{result['data'].get('memory_id', result['data'].get('message_id', ''))}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "results": unique_results[:limit],
            "count": len(unique_results[:limit]),
            "search_type": "text_based",
            "note": "Used text-based search (embeddings unavailable)"
        }
    
    # SillyTavern-specific methods
    async def get_character_context(self, character_name: str, context_type: str = None, limit: int = 5) -> Dict:
        """Get relevant context about characters from memory"""
        try:
            # Search for character-specific memories
            query = f"character {character_name}"
            if context_type:
                query += f" {context_type}"
            
            # Search memories and conversations
            memories = await self.search_memories(query, limit=limit)
            
            # Filter and format results
            character_context = {
                "character_name": character_name,
                "context_type": context_type,
                "memories": memories.get("results", []),
                "total_found": len(memories.get("results", [])),
                "timestamp": get_current_timestamp()
            }
            
            logger.info(f"Retrieved {len(memories.get('results', []))} memories for character: {character_name}")
            return character_context
            
        except Exception as e:
            logger.error(f"Error getting character context: {e}")
            return {"error": str(e), "character_name": character_name}

    async def store_roleplay_memory(self, character_name: str, event_description: str, 
                                    importance_level: int = 5, tags: List[str] = None) -> Dict:
        """Store important roleplay moments or character developments"""
        try:
            # Format the memory content
            content = f"Character: {character_name}\nEvent: {event_description}"
            
            # Add roleplay-specific tags
            if tags is None:
                tags = []
            tags.extend(["roleplay", "character", character_name.lower()])
            
            # Create the memory
            result = await self.create_memory(
                content=content,
                memory_type="roleplay",
                importance_level=importance_level,
                tags=tags
            )
            
            logger.info(f"Stored roleplay memory for character: {character_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing roleplay memory: {e}")
            return {"error": str(e), "character_name": character_name}

    async def search_roleplay_history(self, query: str, character_name: str = None, limit: int = 10) -> Dict:
        """Search past roleplay interactions and character development"""
        try:
            # Build search query
            search_query = f"roleplay {query}"
            if character_name:
                search_query += f" character {character_name}"
            
            # Search with roleplay memory type filter
            results = await self.search_memories(
                query=search_query,
                limit=limit,
                memory_type="roleplay"
            )
            
            # Add roleplay-specific formatting
            results["search_type"] = "roleplay_history"
            results["character_filter"] = character_name
            results["original_query"] = query
            
            logger.info(f"Found {len(results.get('results', []))} roleplay history results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching roleplay history: {e}")
            return {"error": str(e), "query": query}

    # System maintenance
    async def run_database_maintenance(self, force: bool = False) -> Dict:
        """Run maintenance on all databases using the DatabaseMaintenance class.
        
        This includes:
        - Optimizing indexes
        - Cleaning up orphaned records
        - Updating statistics
        - Validating data consistency
        - Applying retention policies
        - Removing duplicates
        
        Args:
            force: Whether to force maintenance even if recent
            
        Returns:
            Dict containing maintenance results
        """
        try:
            # Create and use DatabaseMaintenance instance
            maintenance = DatabaseMaintenance(self)
            results = await maintenance.run_maintenance(force)
            return results
                
        except Exception as e:
            error_result = {
                "status": "error",
                "message": str(e),
                "timestamp": get_current_timestamp()
            }
            logger.error(f"System maintenance error: {e}")
            return error_result
    
    # Embedding helper methods (async background tasks)
    async def _add_embedding_to_message(self, message_id: str, content: str):
        """Add embedding to a message (background task)"""
        try:
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.conversations_db.execute_update(
                    "UPDATE messages SET embedding = ? WHERE message_id = ?",
                    (embedding_blob, message_id)
                )
        except Exception as e:
            logger.error(f"Error adding embedding to message {message_id}: {e}")
    
    async def _add_embedding_to_memory(self, memory_id: str, content: str):
        """Add embedding to a memory (background task)"""
        try:
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.ai_memory_db.execute_update(
                    "UPDATE curated_memories SET embedding = ? WHERE memory_id = ?",
                    (embedding_blob, memory_id)
                )
        except Exception as e:
            logger.error(f"Error adding embedding to memory {memory_id}: {e}")
    
    async def _add_embedding_to_appointment(self, appointment_id: str, content: str):
        """Add embedding to an appointment (background task)"""
        try:
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.schedule_db.execute_update(
                    "UPDATE appointments SET embedding = ? WHERE appointment_id = ?",
                    (embedding_blob, appointment_id)
                )
        except Exception as e:
            logger.error(f"Error adding embedding to appointment {appointment_id}: {e}")
    
    async def _add_embedding_to_reminder(self, reminder_id: str, content: str):
        """Add embedding to a reminder (background task)"""
        try:
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.schedule_db.execute_update(
                    "UPDATE reminders SET embedding = ? WHERE reminder_id = ?",
                    (embedding_blob, reminder_id)
                )
        except Exception as e:
            logger.error(f"Error adding embedding to reminder {reminder_id}: {e}")
    
    async def _add_embedding_to_project_insight(self, insight_id: str, content: str):
        """Add embedding to a project insight (background task)"""
        try:
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.vscode_db.execute_update(
                    "UPDATE project_insights SET embedding = ? WHERE insight_id = ?",
                    (embedding_blob, insight_id)
                )
        except Exception as e:
            logger.error(f"Error adding embedding to project insight {insight_id}: {e}")

    # =============================================================================
    # ADDITIONAL REMINDER AND APPOINTMENT METHODS
    # =============================================================================
    
    async def get_active_reminders(self, limit: int = 10, days_ahead: int = 30) -> List[Dict]:
        """Get active (not completed) reminders"""
        try:
            from datetime import datetime, timedelta
            end_date = (datetime.now() + timedelta(days=days_ahead)).isoformat()
            
            rows = await self.schedule_db.execute_query(
                """SELECT * FROM reminders 
                   WHERE is_completed = 0 AND due_datetime <= ? 
                   ORDER BY due_datetime ASC LIMIT ?""",
                (end_date, limit)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting active reminders: {e}")
            return []
    
    async def get_completed_reminders(self, days: int = 7) -> List[Dict]:
        """Get recently completed reminders"""
        try:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            rows = await self.schedule_db.execute_query(
                """SELECT * FROM reminders 
                   WHERE is_completed = 1 AND completed_at >= ? 
                   ORDER BY completed_at DESC""",
                (start_date,)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting completed reminders: {e}")
            return []
    
    async def complete_reminder(self, reminder_id: str) -> Dict:
        """Mark a reminder as completed"""
        try:
            from datetime import datetime
            completed_at = datetime.now().isoformat()
            
            await self.schedule_db.execute_update(
                "UPDATE reminders SET is_completed = 1, completed_at = ? WHERE reminder_id = ?",
                (completed_at, reminder_id)
            )
            return {"status": "success", "reminder_id": reminder_id, "completed_at": completed_at}
        except Exception as e:
            logger.error(f"Error completing reminder {reminder_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def reschedule_reminder(self, reminder_id: str, new_due_datetime: str) -> Dict:
        """Update the due date of a reminder"""
        try:
            await self.schedule_db.execute_update(
                "UPDATE reminders SET due_datetime = ? WHERE reminder_id = ?",
                (new_due_datetime, reminder_id)
            )
            return {"status": "success", "reminder_id": reminder_id, "new_due_datetime": new_due_datetime}
        except Exception as e:
            logger.error(f"Error rescheduling reminder {reminder_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def delete_reminder(self, reminder_id: str) -> Dict:
        """Permanently delete a reminder"""
        try:
            await self.schedule_db.execute_update(
                "DELETE FROM reminders WHERE reminder_id = ?",
                (reminder_id,)
            )
            return {"status": "success", "reminder_id": reminder_id}
        except Exception as e:
            logger.error(f"Error deleting reminder {reminder_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def cancel_appointment(self, appointment_id: str) -> Dict:
        """Cancel an appointment"""
        result = await self.schedule_db.execute_update(
            "UPDATE appointments SET status = 'cancelled', cancelled_at = ? WHERE appointment_id = ?",
            (get_current_timestamp(), appointment_id)
        )
        if result > 0:
            return {"status": "success", "message": f"Appointment {appointment_id} cancelled"}
        else:
            return {"status": "error", "message": "Appointment not found"}

    async def complete_appointment(self, appointment_id: str) -> Dict:
        """Mark an appointment as completed"""
        result = await self.schedule_db.execute_update(
            "UPDATE appointments SET status = 'completed', completed_at = ? WHERE appointment_id = ?",
            (get_current_timestamp(), appointment_id)
        )
        if result > 0:
            return {"status": "success", "message": f"Appointment {appointment_id} marked as completed"}
        else:
            return {"status": "error", "message": "Appointment not found"}
    
    async def get_upcoming_appointments(self, limit: int = 5, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming appointments (not cancelled)"""
        try:
            from datetime import datetime, timedelta
            end_date = (datetime.now() + timedelta(days=days_ahead)).isoformat()
            
            rows = await self.schedule_db.execute_query(
                """SELECT * FROM appointments 
                   WHERE is_cancelled = 0 AND scheduled_datetime <= ? 
                   ORDER BY scheduled_datetime ASC LIMIT ?""",
                (end_date, limit)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting upcoming appointments: {e}")
            return []
    
    async def get_appointments(self, limit: int = 5, days_ahead: int = 30) -> List[Dict]:
        """Get recent appointments, optionally filtered by date range"""
        try:
            from datetime import datetime, timedelta
            end_date = (datetime.now() + timedelta(days=days_ahead)).isoformat()
            
            rows = await self.schedule_db.execute_query(
                """SELECT * FROM appointments 
                   WHERE scheduled_datetime <= ? 
                   ORDER BY scheduled_datetime DESC LIMIT ?""",
                (end_date, limit)
            )
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting appointments: {e}")
            return []
    
    async def store_ai_reflection(self, content: str, reflection_type: str = "general", 
                                insights: List[str] = None, recommendations: List[str] = None,
                                confidence_level: float = 0.7, source_period_days: int = None) -> Dict:
        """Store an AI self-reflection/insight record"""
        try:
            reflection_id = await self.mcp_db.store_ai_reflection(
                content, reflection_type, insights, recommendations, confidence_level, source_period_days
            )
            return {"status": "success", "reflection_id": reflection_id}
        except Exception as e:
            logger.error(f"Error storing AI reflection: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_current_time(self) -> Dict:
        """Get the current server time in ISO format (UTC and local)"""
        try:
            from datetime import datetime, timezone
            import time
            
            utc_time = datetime.now(timezone.utc)
            local_time = datetime.now()
            timezone_name = time.tzname[0]
            
            return {
                "utc_time": utc_time.isoformat(),
                "local_time": local_time.isoformat(),
                "timezone": timezone_name,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_weather_open_meteo(self, latitude: float = None, longitude: float = None,
                                   timezone_str: str = None, force_refresh: bool = False,
                                   return_changes_only: bool = False, update_today: bool = True,
                                   severe_update: bool = False) -> Dict:
        """Open-Meteo forecast (no API key). Defaults to Motley, MN and caches once per local day."""
        try:
            import requests
            from datetime import datetime, timedelta
            import json
            import os
            
            # Default location (Motley, MN)
            lat = latitude if latitude is not None else 46.3436
            lon = longitude if longitude is not None else -94.6297
            tz = timezone_str if timezone_str is not None else "America/Chicago"
            
            # Create cache directory
            cache_dir = Path("weather_cache")
            cache_dir.mkdir(exist_ok=True)
            
            today = datetime.now().strftime("%Y-%m-%d")
            cache_file = cache_dir / f"weather_{today}.json"
            
            # Check cache unless forced refresh
            if not force_refresh and cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                if return_changes_only:
                    return {"status": "cached", "message": "Using cached weather data"}
                else:
                    return cached_data
            
            # Fetch from Open-Meteo API
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "timezone": tz,
                "current": ["temperature_2m", "relative_humidity_2m", "weather_code", "wind_speed_10m"],
                "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "forecast_days": 7
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            weather_data["cached_at"] = datetime.now().isoformat()
            weather_data["location"] = {"latitude": lat, "longitude": lon, "timezone": tz}
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(weather_data, f, indent=2)
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {"status": "error", "message": str(e)}


# =============================================================================
# MCP SERVER INTEGRATION (Optional - for Model Context Protocol support)
# =============================================================================

# The following code provides MCP server functionality when needed
# To use as MCP server, run: python ai_memory_core.py

async def main():
    """Main entry point - can be used for testing or as MCP server"""
    
    # Initialize the memory system
    memory = PersistentAIMemorySystem()
    
    # Example usage
    print("Persistent AI Memory System - Enhanced Version")
    print("=" * 50)
    
    # Test system health
    health = await memory.get_system_health()
    print(f"System Status: {health['status']}")
    print(f"Databases: {len(health['databases'])} active")
    
    # Test memory creation
    result = await memory.create_memory(
        "This is a test memory with high importance",
        memory_type="test",
        importance_level=8,
        tags=["test", "demo"]
    )
    print(f"Created memory: {result['memory_id']}")
    
    # Test search
    search_results = await memory.search_memories("test memory", limit=5)
    print(f"Found {search_results['count']} memories matching 'test memory'")
    
    print("\nMemory system is ready for use!")
    print("Features available:")
    print("   - 5 specialized databases")
    print("   - Vector semantic search")
    print("   - Real-time file monitoring")
    print("   - Schedule management")
    print("   - Project context tracking")
    print("   - MCP tool call logging")

if __name__ == "__main__":
    asyncio.run(main())
