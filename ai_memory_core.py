#!/usr/bin/env python3
"""
Persistent AI Memory System - Core Module

A comprehensive memory system for AI assistants with real-time conversation capture,
semantic search, and multi-database architecture.

Features:
- 5 specialized databases (conversations, AI memories, schedule, VS Code projects, MCP tool calls)
- Vector semantic search with embeddings
- Real-time conversation file monitoring
- AI self-reflection and tool usage analysis
- Appointment and reminder scheduling
- VS Code project context tracking
- Comprehensive system health monitoring
"""

import sqlite3
import json
import uuid
import hashlib
import asyncio
import aiohttp
import logging
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
        """Store a message and auto-manage sessions/conversations with duplicate detection"""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        message_id = str(uuid.uuid4())
        
        # Check for duplicate messages (same content, role, and session within recent time)
        if session_id:
            # Check if we already have this exact message in this session recently
            existing = await self.execute_query(
                """SELECT message_id FROM messages 
                   WHERE conversation_id IN (
                       SELECT conversation_id FROM conversations WHERE session_id = ?
                   ) AND role = ? AND content = ? 
                   AND datetime(timestamp) > datetime('now', '-1 hour')""",
                (session_id, role, content)
            )
            
            if existing:
                logger.debug(f"Skipping duplicate message in session {session_id}")
                return {
                    "message_id": existing[0]["message_id"],
                    "conversation_id": None,  # Don't return conversation_id for duplicates
                    "session_id": session_id,
                    "duplicate": True
                }
        
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


class ScheduleDatabase(DatabaseManager):
    """Manages appointments and reminders database"""
    
    def __init__(self, db_path: str = "schedule.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            # Appointments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    timestamp_created TEXT NOT NULL,
                    scheduled_datetime TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    location TEXT,
                    source_conversation_id TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Reminders table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    timestamp_created TEXT NOT NULL,
                    due_datetime TEXT NOT NULL,
                    content TEXT NOT NULL,
                    priority_level INTEGER DEFAULT 5,
                    completed INTEGER DEFAULT 0,
                    source_conversation_id TEXT,
                    embedding BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def create_appointment(self, title: str, scheduled_datetime: str, 
                               description: str = None, location: str = None,
                               source_conversation_id: str = None) -> str:
        """Create a new appointment"""
        
        appointment_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        await self.execute_update(
            """INSERT INTO appointments 
               (appointment_id, timestamp_created, scheduled_datetime, title, description, location, source_conversation_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (appointment_id, timestamp, scheduled_datetime, title, description, location, source_conversation_id)
        )
        
        return appointment_id
    
    async def create_reminder(self, content: str, due_datetime: str, 
                            priority_level: int = 5, source_conversation_id: str = None) -> str:
        """Create a new reminder"""
        
        reminder_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        await self.execute_update(
            """INSERT INTO reminders 
               (reminder_id, timestamp_created, due_datetime, content, priority_level, source_conversation_id) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (reminder_id, timestamp, due_datetime, content, priority_level, source_conversation_id)
        )
        
        return reminder_id
    
    async def get_upcoming_appointments(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming appointments within specified days"""
        
        future_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        rows = await self.execute_query(
            """SELECT * FROM appointments 
               WHERE scheduled_datetime >= ? AND scheduled_datetime <= ?
               ORDER BY scheduled_datetime ASC""",
            (datetime.now(timezone.utc).isoformat(), future_date.isoformat())
        )
        
        return [dict(row) for row in rows]
    
    async def get_active_reminders(self) -> List[Dict]:
        """Get all uncompleted reminders"""
        
        rows = await self.execute_query(
            "SELECT * FROM reminders WHERE completed = 0 ORDER BY due_datetime ASC"
        )
        
        return [dict(row) for row in rows]


class VSCodeProjectDatabase(DatabaseManager):
    """Manages VS Code project context and development sessions"""
    
    def __init__(self, db_path: str = "vscode_project.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            # Project sessions table
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
            
            # Project insights table
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
            
            # Code context table
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
            
            # Development conversations table
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
        timestamp = datetime.now(timezone.utc).isoformat()
        
        await self.execute_update(
            """INSERT INTO project_sessions 
               (session_id, start_timestamp, workspace_path, active_files, git_branch, session_summary) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, timestamp, workspace_path, 
             json.dumps(active_files) if active_files else None,
             git_branch, session_summary)
        )
        
        return session_id
    
    async def store_project_insight(self, content: str, insight_type: str = None,
                                  related_files: List[str] = None, importance_level: int = 5,
                                  source_conversation_id: str = None) -> str:
        """Store a project insight"""
        
        insight_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
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
    """Monitors files for conversation changes and auto-imports them"""
    
    def __init__(self, memory_system, watch_directories: List[str] = None):
        self.memory_system = memory_system
        self.watch_directories = watch_directories or []
        self.observer = None
        self.processed_files = set()  # Track processed files to avoid duplicates
        
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
        """Process a changed conversation file"""
        try:
            # Check if file is a conversation file (JSON, txt, etc.)
            if not any(file_path.endswith(ext) for ext in ['.json', '.txt', '.md', '.log']):
                return
            
            # Generate file hash to avoid duplicate processing
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                return
            
            self.processed_files.add(file_hash)
            
            # Read and parse conversation content
            conversations = await self._extract_conversations(file_path)
            
            # Store conversations in database
            for conv in conversations:
                await self.memory_system.store_conversation(
                    content=conv['content'],
                    role=conv['role'],
                    metadata={'source_file': file_path, 'imported_at': datetime.now(timezone.utc).isoformat()}
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
        """Extract conversations from various file formats"""
        conversations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle JSON format (ChatGPT exports, etc.)
            if file_path.endswith('.json'):
                try:
                    data = json.loads(content)
                    # Handle different JSON conversation formats
                    if 'mapping' in data:  # ChatGPT format
                        conversations.extend(self._parse_chatgpt_format(data))
                    elif isinstance(data, list):  # Simple conversation array
                        conversations.extend(self._parse_simple_array(data))
                except json.JSONDecodeError:
                    pass
            
            # Handle text formats
            elif file_path.endswith(('.txt', '.md', '.log')):
                conversations.extend(self._parse_text_format(content))
        
        except Exception as e:
            logger.error(f"Error extracting conversations from {file_path}: {e}")
        
        return conversations
    
    def _parse_chatgpt_format(self, data: Dict) -> List[Dict]:
        """Parse ChatGPT export format"""
        conversations = []
        
        try:
            if 'mapping' in data:
                for node_id, node in data['mapping'].items():
                    if node.get('message') and node['message'].get('content'):
                        content_parts = node['message']['content'].get('parts', [])
                        if content_parts:
                            conversations.append({
                                'role': node['message'].get('author', {}).get('role', 'unknown'),
                                'content': ' '.join(str(part) for part in content_parts if part)
                            })
        except Exception as e:
            logger.error(f"Error parsing ChatGPT format: {e}")
        
        return conversations
    
    def _parse_simple_array(self, data: List) -> List[Dict]:
        """Parse simple conversation array format"""
        conversations = []
        
        for item in data:
            if isinstance(item, dict) and 'content' in item:
                conversations.append({
                    'role': item.get('role', 'user'),
                    'content': str(item['content'])
                })
        
        return conversations
    
    def _parse_text_format(self, content: str) -> List[Dict]:
        """Parse text-based conversation formats"""
        conversations = []
        lines = content.split('\n')
        
        current_role = 'user'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect role markers
            if line.lower().startswith(('user:', 'human:', 'me:')):
                if current_content:
                    conversations.append({
                        'role': current_role,
                        'content': '\n'.join(current_content)
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
    """Main memory system that coordinates all databases and operations - FULL FEATURED VERSION"""
    
    def __init__(self, data_dir: str = "memory_data", enable_file_monitoring: bool = True, 
                 watch_directories: List[str] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize all 5 databases
        self.conversations_db = ConversationDatabase(str(self.data_dir / "conversations.db"))
        self.ai_memory_db = AIMemoryDatabase(str(self.data_dir / "ai_memories.db"))
        self.schedule_db = ScheduleDatabase(str(self.data_dir / "schedule.db"))
        self.vscode_db = VSCodeProjectDatabase(str(self.data_dir / "vscode_project.db"))
        self.mcp_db = MCPToolCallDatabase(str(self.data_dir / "mcp_tool_calls.db"))
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # Initialize file monitoring
        self.file_monitor = None
        if enable_file_monitoring:
            self.file_monitor = ConversationFileMonitor(self, watch_directories)
    
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

    # =============================================================================
    # ADVANCED SEARCH OPERATIONS
    # =============================================================================
    
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
# MCP SERVER INTEGRATION (Optional - for Model Context Protocol support)
# =============================================================================

# The following code provides MCP server functionality when needed
# To use as MCP server, run: python ai_memory_core.py

async def main():
    """Main entry point - can be used for testing or as MCP server"""
    
    # Initialize the memory system
    memory = PersistentAIMemorySystem()
    
    # Example usage
    print("🧠 Persistent AI Memory System - Enhanced Version")
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
    print(f"✅ Created memory: {result['memory_id']}")
    
    # Test search
    search_results = await memory.search_memories("test memory", limit=5)
    print(f"🔍 Found {search_results['count']} memories matching 'test memory'")
    
    print("\n✨ Memory system is ready for use!")
    print("📚 Features available:")
    print("   • 5 specialized databases")
    print("   • Vector semantic search")
    print("   • Real-time file monitoring")
    print("   • Schedule management")
    print("   • Project context tracking")
    print("   • MCP tool call logging")

if __name__ == "__main__":
    asyncio.run(main())
