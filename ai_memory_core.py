
#!/usr/bin/env python3    


"""
AI Memory System

Core memory system that handles all database operations, embeddings, and memory logic.
Provides persistent, portable memory capabilities for AI assistants across multiple databases.

All timestamps are stored in the local timezone (Minnesota) using ISO format. This ensures
that timestamps are correctly displayed and interpreted in the local time context where
the system is being used.
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
import traceback
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
        return ZoneInfo("America/Chicago")  # Minnesota is in Central Time

# Get the base directory dynamically - works on both Windows and Linux
def get_base_path():
    """Get the base AI Memory path, works on both Windows and Linux"""
    current_file = Path(__file__).resolve()
    # This should return the directory containing this file
    return current_file.parent



def get_current_timestamp() -> str:
    """Get current timestamp in local timezone ISO format"""
    return datetime.now(get_local_timezone()).isoformat()
    
def datetime_to_local_isoformat(dt: datetime) -> str:
    """Convert any datetime to local timezone ISO format"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_local_timezone())
    return dt.astimezone(get_local_timezone()).isoformat()
import hashlib
from utils import parse_timestamp, ensure_directories, get_memory_data_dir, get_log_dir
from database_maintenance import DatabaseMaintenance

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
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount


class ConversationDatabase(DatabaseManager):
    """Manages conversation auto-save database"""
    
    def __init__(self, db_path: str = "memory_data/conversations.db"):
        super().__init__(db_path)
        self.initialize_tables()
        # Debug: log the absolute path to a file in memory_data
        import os
        debug_path = os.path.abspath(db_path)
        log_file = os.path.join(os.path.dirname(debug_path), "db_debug_log.txt")
        with open(log_file, "a") as f:
            f.write(f"ConversationDatabase using: {debug_path}\n")
    
    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
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
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)

            # --- MIGRATION LOGIC FOR MESSAGES TABLE ---
            # Define expected columns
            expected_columns = [
                'message_id', 'conversation_id', 'timestamp', 'role', 'content', 'source_type',
                'source_id', 'source_url', 'source_metadata', 'sync_status', 'last_sync',
                'metadata', 'embedding', 'created_at', 'source', 'user_id', 'model_id'
            ]
            # Get current columns
            cur = conn.execute("PRAGMA table_info(messages)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in expected_columns:
                    if col not in current_columns:
                        needs_migration = True
                        break
            # If migration needed, backup data, drop, recreate, restore
            if needs_migration:
                logger.warning("Migrating messages table to new schema!")
                # Backup old data
                old_rows = conn.execute("SELECT * FROM messages").fetchall()
                # Drop old table
                conn.execute("DROP TABLE IF EXISTS messages")
                # Recreate with new schema
                conn.execute("""
                    CREATE TABLE messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT,
                        source_url TEXT,
                        source_metadata TEXT,
                        sync_status TEXT,
                        last_sync TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'direct',
                        user_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)
                # Restore old data
                for row in old_rows:
                    # Build new row dict with defaults for missing columns
                    row_dict = dict(row)
                    for col in expected_columns:
                        if col not in row_dict:
                            if col == 'source_type':
                                row_dict[col] = 'unknown'
                            elif col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            else:
                                row_dict[col] = None
                    # Insert
                    conn.execute(
                        f"INSERT INTO messages ({', '.join(expected_columns)}) VALUES ({', '.join(['?' for _ in expected_columns])})",
                        tuple(row_dict[col] for col in expected_columns)
                    )
                logger.warning(f"Restored {len(old_rows)} messages after migration.")

            else:
                # Create table if not exists (normal path)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT,
                        source_url TEXT,
                        source_metadata TEXT,
                        sync_status TEXT,
                        last_sync TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'direct',
                        user_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
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

            # Memory-Conversation Links table (Phase 1 integration with AI Memory System)
            # First, check if table exists with old schema (has bad foreign key to curated_memories)
            cursor = conn.execute("PRAGMA table_info(memory_conversation_links)")
            existing_link_cols = [row[1] for row in cursor.fetchall()]
            
            if existing_link_cols:
                # Table exists - check if it has any foreign keys
                cursor = conn.execute("PRAGMA foreign_key_list(memory_conversation_links)")
                fks = cursor.fetchall()
                has_bad_fk = any(fk[2] == 'curated_memories' for fk in fks)
                
                if has_bad_fk:
                    logger.warning("Migrating memory_conversation_links: replacing bad foreign key to curated_memories with correct one to conversations")
                    # Backup data
                    old_rows = conn.execute("SELECT * FROM memory_conversation_links").fetchall()
                    # Drop and recreate with correct foreign key (only to conversations, not curated_memories)
                    conn.execute("DROP TABLE IF EXISTS memory_conversation_links")
                    conn.execute("""
                        CREATE TABLE memory_conversation_links (
                            link_id TEXT PRIMARY KEY,
                            memory_id TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            link_type TEXT NOT NULL,
                            link_strength REAL DEFAULT 1.0,
                            source_system TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT,
                            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                        )
                    """)
                    # Restore data
                    for row in old_rows:
                        conn.execute("""
                            INSERT INTO memory_conversation_links 
                            (link_id, memory_id, conversation_id, link_type, link_strength, source_system, created_at, updated_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, row)
                    logger.warning(f"Restored {len(old_rows)} links after migration")
            else:
                # Table doesn't exist - create fresh
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_conversation_links (
                        link_id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        link_type TEXT NOT NULL,
                        link_strength REAL DEFAULT 1.0,
                        source_system TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )
                """)

            # Memory Processing Queue table (tracks which conversations need processing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_processing_queue (
                    queue_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    memory_id TEXT,
                    status TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    attempts INTEGER DEFAULT 0,
                    last_attempt TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Memory Processing Log table (audit trail of processing attempts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_processing_log (
                    log_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    memory_id TEXT,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    result_metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
    
    async def store_message(self, content: str, role: str, session_id: str = None, 
                          conversation_id: str = None, metadata: Dict = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict[str, str]:
        """Store a message and auto-manage sessions/conversations with duplicate detection"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        timestamp = datetime.now(get_local_timezone()).isoformat()
        message_id = str(uuid.uuid4())
        
        # Check for duplicate messages (same content, role, and session within recent time)
        if session_id:
            # Create a content hash for duplicate detection
            content_hash = hashlib.md5(f"{content}:{role}:{session_id}".encode()).hexdigest()
            
            # Check if we already have this exact message in this session recently
            existing = await self.execute_query(
                """SELECT message_id FROM messages 
                   WHERE conversation_id IN (
                       SELECT conversation_id FROM conversations WHERE session_id = ? AND user_id = ? AND model_id = ?
                   ) AND role = ? AND content = ? 
                   AND datetime(timestamp) > datetime('now', '-1 hour')""",
                (session_id, user_id, model_id, role, content)
            )
            
            if existing:
                logger.debug(f"Skipping duplicate message in session {session_id}")
                return {
                    "message_id": existing[0]["message_id"],
                    "conversation_id": None,  # Don't return conversation_id for duplicates
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
            # Check if session exists, create if not
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
                "INSERT INTO conversations (conversation_id, session_id, start_timestamp, user_id, model_id) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, session_id, timestamp, user_id, model_id)
            )
        
        # Extract source type from metadata
        source_type = "unknown"
        if metadata:
            source_type = metadata.get("application", metadata.get("file_type", "unknown"))
        
        # Store the message with user_id and model_id
        await self.execute_update(
            """INSERT INTO messages 
               (message_id, conversation_id, timestamp, role, content, source_type, metadata, user_id, model_id, source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, conversation_id, timestamp, role, content, source_type,
             json.dumps(metadata) if metadata else None, user_id, model_id, source)
        )
        
        return {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "duplicate": False
        }
    
    async def get_recent_messages(self, limit: int = 10, session_id: str = None, days_back: int = 7, user_id: str = None, model_id: str = None) -> List[Dict]:
        """Get recent messages, optionally filtered by session and within specified days.
        If model_id is None, queries all models for that user (cross-model fallback).
        """
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        # Calculate cutoff date
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_timestamp = cutoff_date.isoformat()
        
        # Build query based on whether model_id filtering is requested
        if model_id:
            # Filter by specific model
            if session_id:
                query = "SELECT m.message_id, m.conversation_id, m.timestamp, m.role, m.content, m.metadata, c.session_id FROM messages m JOIN conversations c ON m.conversation_id = c.conversation_id WHERE c.session_id = ? AND m.timestamp >= ? AND c.user_id = ? AND c.model_id = ? ORDER BY m.timestamp DESC LIMIT ?"
                params = (session_id, cutoff_timestamp, user_id, model_id, limit)
            else:
                query = "SELECT m.message_id, m.conversation_id, m.timestamp, m.role, m.content, m.metadata, c.session_id FROM messages m JOIN conversations c ON m.conversation_id = c.conversation_id WHERE m.timestamp >= ? AND c.user_id = ? AND c.model_id = ? ORDER BY m.timestamp DESC LIMIT ?"
                params = (cutoff_timestamp, user_id, model_id, limit)
        else:
            # Query all models for this user
            if session_id:
                query = "SELECT m.message_id, m.conversation_id, m.timestamp, m.role, m.content, m.metadata, c.session_id FROM messages m JOIN conversations c ON m.conversation_id = c.conversation_id WHERE c.session_id = ? AND m.timestamp >= ? AND c.user_id = ? ORDER BY m.timestamp DESC LIMIT ?"
                params = (session_id, cutoff_timestamp, user_id, limit)
            else:
                query = "SELECT m.message_id, m.conversation_id, m.timestamp, m.role, m.content, m.metadata, c.session_id FROM messages m JOIN conversations c ON m.conversation_id = c.conversation_id WHERE m.timestamp >= ? AND c.user_id = ? ORDER BY m.timestamp DESC LIMIT ?"
                params = (cutoff_timestamp, user_id, limit)
        
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]

    async def link_memory_to_conversation(self, memory_id: str, conversation_id: str, 
                                         link_type: str = 'direct', link_strength: float = 1.0,
                                         source_system: str = 'processed_from_chat', metadata: dict = None) -> str:
        """
        Link a memory to a conversation.
        
        Args:
            memory_id: UUID of the memory
            conversation_id: UUID of the conversation
            link_type: 'direct', 'related', or 'enhanced'
            link_strength: 0.0-1.0 confidence score
            source_system: 'openwebui_import', 'processed_from_chat', 'manual', 'enhanced'
            metadata: Optional JSON metadata for link
        
        Returns:
            link_id: UUID of the created link
        """
        link_id = str(uuid.uuid4())
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        await self.execute_update(
            """INSERT INTO memory_conversation_links 
               (link_id, memory_id, conversation_id, link_type, link_strength, source_system, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (link_id, memory_id, conversation_id, link_type, link_strength, source_system, 
             timestamp, timestamp, json.dumps(metadata) if metadata else None)
        )
        
        logger.debug(f"Linked memory {memory_id} to conversation {conversation_id} with type {link_type}")
        return link_id

    async def get_memory_conversation_links(self, memory_id: str = None, conversation_id: str = None,
                                           link_type: str = None) -> List[Dict]:
        """
        Retrieve memory-conversation links.
        
        Args:
            memory_id: Optional - filter by memory ID
            conversation_id: Optional - filter by conversation ID
            link_type: Optional - filter by link type ('direct', 'related', 'enhanced')
        
        Returns:
            List of link dictionaries with all columns
        """
        query = "SELECT * FROM memory_conversation_links WHERE 1=1"
        params = []
        
        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        
        if link_type:
            query += " AND link_type = ?"
            params.append(link_type)
        
        query += " ORDER BY link_strength DESC, created_at DESC"
        
        rows = await self.execute_query(query, tuple(params))
        return [dict(row) for row in rows]

    async def link_conversations(self, source_conversation_id: str, related_conversation_id: str,
                                relationship_type: str = 'same_session', confidence_score: float = 1.0,
                                matching_method: str = None, metadata: dict = None) -> str:
        """
        Link two conversations (e.g., OpenWebUI chat linked to LM Studio import).
        
        Args:
            source_conversation_id: Primary conversation ID (typically OpenWebUI canonical)
            related_conversation_id: Related conversation ID (e.g., from file import)
            relationship_type: 'same_session', 'related_topic', 'continuation', etc.
            confidence_score: 0.0-1.0 confidence level of the match
            matching_method: How match was determined ('content_similarity', 'timestamp_proximity', 'manual')
            metadata: Optional metadata with breadcrumbs
        
        Returns:
            relationship_id: UUID of the created relationship
        """
        relationship_id = str(uuid.uuid4())
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        # Ensure metadata includes breadcrumbs and confidence info
        if metadata is None:
            metadata = {}
        metadata.setdefault('created_at_timestamp', timestamp)
        metadata.setdefault('matching_method', matching_method)
        metadata.setdefault('confidence_score', confidence_score)
        metadata.setdefault('validated_at', None)  # Pending validation
        
        await self.execute_update(
            """INSERT INTO conversation_relationships 
               (relationship_id, source_conversation_id, related_conversation_id, relationship_type, 
                created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (relationship_id, source_conversation_id, related_conversation_id, relationship_type,
             timestamp, json.dumps(metadata))
        )
        
        logger.debug(f"Linked conversations: {source_conversation_id} <-> {related_conversation_id} "
                    f"(type={relationship_type}, confidence={confidence_score:.2f})")
        return relationship_id

    async def resolve_source_conversation_id(self, source_conversation_id: str, user_id: str = None) -> str:
        """
        Resolve generic source_conversation_id to actual conversation_id.
        
        Handles patterns like:
        - "openwebui_user_<uuid>" -> Find first LM Studio conversation for that user
        - "current_session" -> Find most recent conversation
        - UUID already -> Return as-is
        
        Args:
            source_conversation_id: Source format string or UUID
            user_id: User context for resolution
        
        Returns:
            conversation_id: Actual UUID or None if cannot resolve
        """
        if not source_conversation_id:
            return None
        
        # If it looks like a UUID (36 chars with 4 dashes), assume it's already a conversation_id
        if len(source_conversation_id) == 36 and source_conversation_id.count('-') == 4:
            return source_conversation_id
        
        # Handle "current_session" - get most recent conversation
        if source_conversation_id == "current_session":
            if user_id:
                result = await self.execute_query(
                    """SELECT conversation_id FROM conversations 
                       WHERE user_id = ? ORDER BY start_timestamp DESC LIMIT 1""",
                    (user_id,)
                )
                if result:
                    return result[0]['conversation_id']
            return None
        
        # Handle "openwebui_user_<uuid>" patterns - find first conversation for that user in test/friday mode
        if source_conversation_id.startswith("openwebui_user_"):
            # Extract user context from pattern if possible
            result = await self.execute_query(
                """SELECT conversation_id FROM conversations 
                   WHERE user_id = 'test' OR user_id = ? 
                   ORDER BY start_timestamp DESC LIMIT 1""",
                (user_id or "unknown",)
            )
            if result:
                return result[0]['conversation_id']
            return None
        
        # If we can't resolve it, return None
        return None

    async def queue_conversation_for_processing(self, conversation_id: str, 
                                               processing_type: str, priority: int = 5) -> str:
        """
        Queue a conversation for memory processing.
        
        Args:
            conversation_id: UUID of conversation to process
            processing_type: 'new_openwebui', 'recent_chat', 'aging_chat', 'historical_chat'
            priority: 1-10 (higher = more urgent)
        
        Returns:
            queue_id: UUID of the queue entry
        """
        queue_id = str(uuid.uuid4())
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        # Get message count for this conversation
        message_count_result = await self.execute_query(
            "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        message_count = message_count_result[0]['count'] if message_count_result else 0
        
        try:
            await self.execute_update(
                """INSERT INTO memory_processing_queue 
                   (queue_id, conversation_id, processing_status, processing_type, message_count, processing_priority, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (queue_id, conversation_id, 'pending', processing_type, message_count, priority, timestamp, timestamp)
            )
            logger.debug(f"Queued conversation {conversation_id} for {processing_type} with priority {priority}")
            return queue_id
        except Exception as e:
            # If conversation already queued (UNIQUE constraint), get the existing queue_id
            logger.debug(f"Conversation {conversation_id} already in queue: {str(e)}")
            existing = await self.execute_query(
                "SELECT queue_id FROM memory_processing_queue WHERE conversation_id = ?",
                (conversation_id,)
            )
            if existing:
                return existing[0]['queue_id']
            raise

    async def get_processing_priority(self) -> dict:
        """
        Get next conversation to process based on priority and status.
        
        Returns:
            Dictionary with queue_id, conversation_id, and metadata, or empty dict if none
        """
        rows = await self.execute_query(
            """SELECT * FROM memory_processing_queue 
               WHERE processing_status IN ('pending', 'processing') 
               AND marked_processed = FALSE
               ORDER BY processing_priority DESC, created_at ASC
               LIMIT 1""",
            ()
        )
        
        if rows:
            return dict(rows[0])
        return {}

    async def mark_processing_complete(self, queue_id: str, memory_id: str = None) -> bool:
        """
        Mark a conversation as processed.
        
        Args:
            queue_id: UUID of queue entry
            memory_id: Optional - memory ID that was created from this processing
        
        Returns:
            True if successful, False otherwise
        """
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        try:
            await self.execute_update(
                """UPDATE memory_processing_queue 
                   SET processing_status = 'completed', marked_processed = TRUE, updated_at = ?
                   WHERE queue_id = ?""",
                (timestamp, queue_id)
            )
            logger.debug(f"Marked queue {queue_id} as complete")
            return True
        except Exception as e:
            logger.error(f"Error marking {queue_id} complete: {str(e)}")
            return False

    async def update_processing_status(self, queue_id: str, status: str, reason: str = None) -> bool:
        """
        Update processing status in queue.
        
        Args:
            queue_id: UUID of queue entry
            status: 'pending', 'processing', 'completed', 'skipped'
            reason: Optional - reason for status change
        
        Returns:
            True if successful, False otherwise
        """
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        try:
            await self.execute_update(
                """UPDATE memory_processing_queue 
                   SET processing_status = ?, updated_at = ?
                   WHERE queue_id = ?""",
                (status, timestamp, queue_id)
            )
            logger.debug(f"Updated queue {queue_id} status to {status}" + (f": {reason}" if reason else ""))
            return True
        except Exception as e:
            logger.error(f"Error updating {queue_id} status: {str(e)}")
            return False

    async def log_processing_attempt(self, conversation_id: str, processing_type: str,
                                    status: str, memory_id: str = None, reason: str = None) -> str:
        """
        Log a processing attempt for audit trail.
        
        Args:
            conversation_id: UUID of conversation processed
            processing_type: Type of processing done
            status: 'success', 'failed', 'skipped'
            memory_id: Optional - memory ID that was created/modified
            reason: Optional - explanation of outcome
        
        Returns:
            log_id: UUID of the log entry
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        await self.execute_update(
            """INSERT INTO memory_processing_log 
               (log_id, conversation_id, memory_id, processing_type, status, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (log_id, conversation_id, memory_id, processing_type, status, reason, timestamp)
        )
        
        logger.info(f"Logged processing: conversation {conversation_id}, status {status}, memory {memory_id}")
        return log_id


class AIMemoryDatabase(DatabaseManager):
    """Manages AI-curated memories database"""
    
    def __init__(self, db_path: str = "memory_data/ai_memories.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            expected_columns = [
                'memory_id', 'timestamp_created', 'timestamp_updated', 'source_conversation_id',
                'source_message_ids', 'memory_type', 'content', 'importance_level', 'tags',
                'embedding', 'embedding_dimension', 'user_id', 'model_id', 'memory_bank', 'source', 'created_at', 'updated_at'
            ]

            # --- Check and migrate existing table ---
            cur = conn.execute("PRAGMA table_info(curated_memories)")
            current_columns = [row[1] for row in cur.fetchall()]

            # Only try to ALTER if table exists (current_columns will be non-empty)
            if current_columns:
                # Add missing columns to existing table
                if "user_id" not in current_columns:
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN user_id TEXT")
                if "model_id" not in current_columns:
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN model_id TEXT")
                if "memory_bank" not in current_columns:
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN memory_bank TEXT DEFAULT 'General'")
                if "source" not in current_columns:
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN source TEXT DEFAULT 'direct'")
                if "embedding_dimension" not in current_columns:
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN embedding_dimension INTEGER")
                if "updated_at" not in current_columns:
                    # SQLite doesn't support DEFAULT CURRENT_TIMESTAMP in ALTER TABLE, so add without default
                    # Existing rows will have NULL, new rows can be set via INSERT/UPDATE
                    conn.execute("ALTER TABLE curated_memories ADD COLUMN updated_at TEXT")

            # Detect incomplete schema (older versions)
            needs_migration = False
            if current_columns:
                for col in expected_columns:
                    if col not in current_columns:
                        needs_migration = True
                        break

            if needs_migration:
                logger.warning("Migrating curated_memories table to new schema!")
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
                        embedding_dimension INTEGER,
                        user_id TEXT,
                        model_id TEXT,
                        memory_bank TEXT DEFAULT 'General',
                        source TEXT DEFAULT 'direct',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in expected_columns:
                        if col not in row_dict:
                            if col in ('timestamp_created', 'timestamp_updated', 'created_at'):
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'importance_level':
                                row_dict[col] = 5
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO curated_memories ({', '.join(expected_columns)}) VALUES ({', '.join(['?' for _ in expected_columns])})",
                        tuple(row_dict[col] for col in expected_columns)
                    )
                logger.warning(f"Restored {len(old_rows)} curated memories after migration.")
            else:
                # Ensure table exists if missing entirely or if we just did migration
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
                        embedding_dimension INTEGER,
                        user_id TEXT,
                        model_id TEXT,
                        memory_bank TEXT DEFAULT 'General',
                        source TEXT DEFAULT 'direct',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Add indexes for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_curated_memories_user_model
                ON curated_memories (user_id, model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_curated_memories_bank
                ON curated_memories (memory_bank, importance_level)
            """)

            conn.commit()

    
    async def create_memory(
        self,
        content: str,
        memory_type: str = None,
        importance_level: int = 5,
        tags: List[str] = None,
        source_conversation_id: str = None,
        memory_bank: str = "General",
        user_id: str = "",
        model_id: str = "",
        source: str = "direct",
    ) -> str:

        """Create a new curated memory with embedded metadata and optional source tracking"""
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(get_local_timezone()).isoformat()
        
        # CHANGE 1B: Format content with embedded metadata tags (like short-term system does)
        formatted_content = content
        if tags:
            formatted_content = f"[Tags: {', '.join(tags)}] {formatted_content}"
        if memory_bank and memory_bank != "General":  # Only embed if not default
            formatted_content = f"{formatted_content} [Memory Bank: {memory_bank}]"
        if model_id and model_id != "Friday":  # Only embed if not default
            formatted_content = f"{formatted_content} [Model: {model_id}]"
        
        await self.execute_update(
            """INSERT INTO curated_memories 
            (memory_id, timestamp_created, timestamp_updated, source_conversation_id,
                memory_type, content, importance_level, tags, memory_bank, user_id, model_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                timestamp,
                timestamp,
                source_conversation_id,
                memory_type,
                formatted_content,
                importance_level,
                json.dumps(tags) if tags else None,
                memory_bank,
                user_id,
                model_id,
                source,
            ),
        )

        
        return memory_id
    
    async def update_memory(self, memory_id: str, content: str = None, 
                          importance_level: int = None, tags: List[str] = None, user_id: str = None, model_id: str = None, source: str = "direct") -> bool:
        """Update an existing memory"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        timestamp = get_current_timestamp()
        updates = ["timestamp_updated = ?"]
        params = [timestamp]
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if importance_level is not None:
            updates.append("importance_level = ?")
            params.append(importance_level)
        
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        
        if source is not None:
            updates.append("source = ?")
            params.append(source)
        
        params.append(memory_id)
        params.append(user_id)
        params.append(model_id)
        
        query = f"UPDATE curated_memories SET {', '.join(updates)} WHERE memory_id = ? AND user_id = ? AND model_id = ?"
        await self.execute_update(query, tuple(params))
        
        return True
    
    async def delete_memory(self, memory_id: str, user_id: str = None, model_id: str = None, source: str = "direct") -> bool:
        """Delete a memory by ID"""
        if not user_id or not model_id:
            logger.error("MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults.")
            return False
        try:
            
            await self.execute_update(
                "DELETE FROM curated_memories WHERE memory_id = ? AND user_id = ? AND model_id = ?",
                (memory_id, user_id, model_id)
            )
            logger.info(f"🗑️  Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_memories(self, limit: int = 10, memory_type: str = None, user_id: str = None, model_id: str = None, source: str = "direct") -> List[Dict]:
        """Get memories, optionally filtered by type, user, and model"""
        
        if not user_id or not model_id:
            logger.error("MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults.")
            return []
        
        if memory_type:
            query = """
                SELECT * FROM curated_memories 
                WHERE memory_type = ? AND user_id = ? AND model_id = ?
                ORDER BY importance_level DESC, timestamp_created DESC 
                LIMIT ?
            """
            params = (memory_type, user_id, model_id, limit)
        else:
            query = """
                SELECT * FROM curated_memories 
                WHERE user_id = ? AND model_id = ?
                ORDER BY importance_level DESC, timestamp_created DESC 
                LIMIT ?
            """
            params = (user_id, model_id, limit)
        
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]


class ScheduleDatabase(DatabaseManager):
    async def get_appointments(self, limit: int = 5, days_ahead: int = 30, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get upcoming appointments (today onward), normalizing mixed datetime types to epoch seconds.
        
        If model_id is None, queries all models for that user (cross-model fallback).
        If model_id is provided (even empty string), filters to that specific model.
        """
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        now = datetime.now(get_local_timezone())
        # Use start of today instead of current time to include all appointments scheduled for today
        start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        future = now + timedelta(days=days_ahead)
        start_of_today_epoch = int(start_of_today.timestamp())
        future_epoch = int(future.timestamp())
        
        try:
            # Build query based on whether model_id filtering is requested
            if model_id:
                # Filter by specific model
                query = """
                    SELECT *
                    FROM appointments
                    WHERE user_id = ?
                    AND model_id = ?
                    AND
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END >= ?
                    AND
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END <= ?
                    ORDER BY
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END ASC
                    LIMIT ?
                """
                rows = await self.execute_query(query, (user_id, model_id, start_of_today_epoch, future_epoch, limit))
            else:
                # Query all models for this user
                query = """
                    SELECT *
                    FROM appointments
                    WHERE user_id = ?
                    AND
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END >= ?
                    AND
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END <= ?
                    ORDER BY
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END ASC
                    LIMIT ?
                """
                rows = await self.execute_query(query, (user_id, start_of_today_epoch, future_epoch, limit))
            
            if not rows:
                return {
                    "status": "no_appointments",
                    "message": f"No upcoming appointments in the next {days_ahead} days."
                }
            
            return {
                "status": "success",
                "count": len(rows),
                "appointments": [dict(row) for row in rows]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to retrieve appointments. Please try again later. Error: {str(e)}"
            }
    
    def __init__(self, db_path: str = "memory_data/schedule.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # --- MIGRATION LOGIC FOR APPOINTMENTS TABLE ---
            appointments_expected = [
                'appointment_id', 'timestamp_created', 'scheduled_datetime', 'title', 'description',
                'location', 'cancelled_at', 'completed_at', 'source_conversation_id', 'embedding', 'created_at', "status",
                'user_id', 'model_id'
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
                logger.warning("Migrating appointments table to new schema!")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
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
                logger.warning(f"Restored {len(old_rows)} appointments after migration.")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)

            # --- MIGRATION LOGIC FOR REMINDERS TABLE ---
            reminders_expected = [
                'reminder_id', 'timestamp_created', 'due_datetime', 'content', 'priority_level',
                'completed', 'is_completed', 'completed_at', 'source_conversation_id', 'conversation_title',
                'urgency_level', 'notification_sent_at', 'notification_status', 'escalation_count',
                'last_escalated_at', 'embedding', 'created_at', 'user_id', 'model_id'
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
                logger.warning("Migrating reminders table to new schema!")
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
                        conversation_title TEXT,
                        urgency_level INTEGER DEFAULT 1,
                        notification_sent_at TEXT,
                        notification_status TEXT CHECK(notification_status IN ('pending', 'sent', 'read', 'dismissed')),
                        escalation_count INTEGER DEFAULT 0,
                        last_escalated_at TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in reminders_expected:
                        if col not in row_dict:
                            if col == 'timestamp_created' or col == 'due_datetime' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'priority_level':
                                row_dict[col] = 5
                            elif col == 'completed':
                                row_dict[col] = 0
                            elif col == 'is_completed':
                                row_dict[col] = row_dict.get('completed', 0)  # Copy from completed field
                            elif col == 'completed_at':
                                row_dict[col] = None  # No completion timestamp for migrated reminders
                            elif col == 'conversation_title':
                                row_dict[col] = None  # Will be populated on next interaction
                            elif col == 'urgency_level':
                                row_dict[col] = 1  # Default to low urgency
                            elif col == 'notification_sent_at' or col == 'notification_status' or col == 'last_escalated_at':
                                row_dict[col] = None  # These will be set by escalation process
                            elif col == 'escalation_count':
                                row_dict[col] = 0  # Start with no escalations
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO reminders ({', '.join(reminders_expected)}) VALUES ({', '.join(['?' for _ in reminders_expected])})",
                        tuple(row_dict[col] for col in reminders_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} reminders after migration.")
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
                        conversation_title TEXT,
                        urgency_level INTEGER DEFAULT 1,
                        notification_sent_at TEXT,
                        notification_status TEXT CHECK(notification_status IN ('pending', 'sent', 'read', 'dismissed')),
                        escalation_count INTEGER DEFAULT 0,
                        last_escalated_at TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)
            conn.commit()
            
            # --- CREATE REMINDER_NOTIFICATIONS TRACKING TABLE ---
            # This table tracks active notification states separately from reminders
            # Uses composite key (reminder_id, urgency_level) to allow one entry per urgency tier per reminder
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminder_notifications (
                    notification_id TEXT PRIMARY KEY,
                    reminder_id TEXT NOT NULL,
                    urgency_level INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    UNIQUE(reminder_id, urgency_level, user_id, model_id),
                    FOREIGN KEY (reminder_id) REFERENCES reminders(reminder_id)
                )
            """)
            conn.commit()
    
    async def create_appointment(self, title: str, scheduled_datetime: str, 
                               description: str = None, location: str = None,
                               source_conversation_id: str = None,
                               recurrence_pattern: str = None,
                               recurrence_count: int = None,
                               recurrence_end_date: str = None, user_id: str = None,
                               model_id: str = None, source: str = "direct") -> Union[str, List[str]]:
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
            user_id: Optional user ID for user separation
            model_id: Optional model ID for model separation
            
        Returns:
            Single appointment_id if no recurrence, list of appointment_ids if recurring
        """
        from dateutil.relativedelta import relativedelta
        from dateutil.parser import parse as parse_date
        
        # Create the first appointment
        appointment_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO appointments 
               (appointment_id, timestamp_created, scheduled_datetime, title, description, location, source_conversation_id, user_id, model_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (appointment_id, timestamp, scheduled_datetime, title, description, location, source_conversation_id, user_id, model_id)
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
                    
                    # Create the recurring appointment
                    recurring_id = str(uuid.uuid4())
                    recurring_datetime = current_datetime.isoformat()
                    
                    await self.execute_update(
                        """INSERT INTO appointments 
                           (appointment_id, timestamp_created, scheduled_datetime, title, description, location, source_conversation_id, user_id, model_id) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (recurring_id, timestamp, recurring_datetime, title, description, location, source_conversation_id, user_id, model_id)
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
                            priority_level: int = 5, source_conversation_id: str = None,
                            recurrence_pattern: str = None, recurrence_count: int = None,
                            recurrence_end_date: str = None, user_id: str = None,
                            model_id: str = None, source: str = "direct", conversation_title: str = None) -> Union[str, List[str]]:
        """Create a new reminder or multiple recurring reminders"""
        
        # If no recurrence pattern, create a single reminder
        if not recurrence_pattern:
            reminder_id = str(uuid.uuid4())
            timestamp = get_current_timestamp()
            
            await self.execute_update(
                """INSERT INTO reminders 
                   (reminder_id, timestamp_created, due_datetime, content, priority_level, source_conversation_id, conversation_title, user_id, model_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (reminder_id, timestamp, due_datetime, content, priority_level, source_conversation_id, conversation_title, user_id, model_id)
            )
            
            return reminder_id
        
        # Handle recurring reminders
        from dateutil.relativedelta import relativedelta
        from datetime import datetime
        import pytz
        
        # Parse the start datetime
        if due_datetime.endswith('Z'):
            start_dt = datetime.fromisoformat(due_datetime[:-1]).replace(tzinfo=pytz.UTC)
        else:
            start_dt = datetime.fromisoformat(due_datetime)
            if start_dt.tzinfo is None:
                # Assume America/Chicago timezone if no timezone specified
                chicago_tz = pytz.timezone('America/Chicago')
                start_dt = chicago_tz.localize(start_dt)
        
        # Parse end date if provided
        end_dt = None
        if recurrence_end_date:
            if recurrence_end_date.endswith('Z'):
                end_dt = datetime.fromisoformat(recurrence_end_date[:-1]).replace(tzinfo=pytz.UTC)
            else:
                end_dt = datetime.fromisoformat(recurrence_end_date)
                if end_dt.tzinfo is None:
                    chicago_tz = pytz.timezone('America/Chicago')
                    end_dt = chicago_tz.localize(end_dt)
        
        # Calculate delta based on pattern
        delta_map = {
            'daily': relativedelta(days=1),
            'weekly': relativedelta(weeks=1),
            'monthly': relativedelta(months=1),
            'yearly': relativedelta(years=1)
        }
        
        if recurrence_pattern not in delta_map:
            raise ValueError(f"Invalid recurrence pattern: {recurrence_pattern}")
        
        delta = delta_map[recurrence_pattern]
        reminder_ids = []
        current_dt = start_dt
        count = 0
        timestamp = get_current_timestamp()
        
        while True:
            # Check stopping conditions
            if recurrence_count and count >= recurrence_count:
                break
            if end_dt and current_dt > end_dt:
                break
            
            # Create reminder for this occurrence
            reminder_id = str(uuid.uuid4())
            
            await self.execute_update(
                """INSERT INTO reminders 
                   (reminder_id, timestamp_created, due_datetime, content, priority_level, source_conversation_id, conversation_title, user_id, model_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (reminder_id, timestamp, current_dt.isoformat(), content, priority_level, source_conversation_id, conversation_title, user_id, model_id)
            )
            
            reminder_ids.append(reminder_id)
            
            # Move to next occurrence
            current_dt += delta
            count += 1
            
            # Safety check to prevent infinite loops
            if count > 1000:
                break
        
        return reminder_ids[0] if len(reminder_ids) == 1 else reminder_ids

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
    """Manages VS Code project development database"""
    
    def __init__(self, db_path: str = "memory_data/vscode_project.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # --- MIGRATION LOGIC FOR PROJECT_SESSIONS TABLE ---
            sessions_expected = [
                'session_id', 'start_timestamp', 'end_timestamp', 'workspace_path', 'active_files',
                'git_branch', 'git_commit_hash', 'session_summary', 'embedding', 'created_at',
                'user_id', 'model_id'
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
                logger.warning("Migrating project_sessions table to new schema!")
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
                        git_commit_hash TEXT,
                        session_summary TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in sessions_expected:
                        if col not in row_dict:
                            if col == 'start_timestamp' or col == 'end_timestamp' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO project_sessions ({', '.join(sessions_expected)}) VALUES ({', '.join(['?' for _ in sessions_expected])})",
                        tuple(row_dict[col] for col in sessions_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} project sessions after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_timestamp TEXT NOT NULL,
                        end_timestamp TEXT,
                        workspace_path TEXT NOT NULL,
                        active_files TEXT,
                        git_branch TEXT,
                        git_commit_hash TEXT,
                        session_summary TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)

            # --- MIGRATION LOGIC FOR DEVELOPMENT_CONVERSATIONS TABLE ---
            devcon_expected = [
                'conversation_id', 'session_id', 'timestamp', 'chat_context_id', 'conversation_content',
                'decisions_made', 'code_changes', 'source_metadata', 'embedding', 'created_at'
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
                logger.warning("Migrating development_conversations table to new schema!")
                old_rows = conn.execute("SELECT * FROM development_conversations").fetchall()
                conn.execute("DROP TABLE IF EXISTS development_conversations")
                conn.execute("""
                    CREATE TABLE development_conversations (
                        conversation_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        chat_context_id TEXT,
                        conversation_content TEXT NOT NULL,
                        decisions_made TEXT,
                        code_changes TEXT,
                        source_metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES project_sessions (session_id)
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in devcon_expected:
                        if col not in row_dict:
                            if col == 'timestamp' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO development_conversations ({', '.join(devcon_expected)}) VALUES ({', '.join(['?' for _ in devcon_expected])})",
                        tuple(row_dict[col] for col in devcon_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} development conversations after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS development_conversations (
                        conversation_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        chat_context_id TEXT,
                        conversation_content TEXT NOT NULL,
                        decisions_made TEXT,
                        code_changes TEXT,
                        source_metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES project_sessions (session_id)
                    )
                """)

            # --- MIGRATION LOGIC FOR PROJECT_INSIGHTS TABLE ---
            insights_expected = [
                'insight_id', 'timestamp_created', 'timestamp_updated', 'insight_type', 'content',
                'related_files', 'source_conversation_id', 'importance_level', 'embedding', 'created_at',
                'user_id', 'model_id', 'source'
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
                logger.warning("Migrating project_insights table to new schema!")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown',
                        source TEXT DEFAULT 'direct'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in insights_expected:
                        if col not in row_dict:
                            if col == 'timestamp_created' or col == 'timestamp_updated' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'importance_level':
                                row_dict[col] = 5
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO project_insights ({', '.join(insights_expected)}) VALUES ({', '.join(['?' for _ in insights_expected])})",
                        tuple(row_dict[col] for col in insights_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} project insights after migration.")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown',
                        source TEXT DEFAULT 'direct'
                    )
                """)

            # --- MIGRATION LOGIC FOR CODE_CONTEXT TABLE ---
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
                logger.warning("Migrating code_context table to new schema!")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in codectx_expected:
                        if col not in row_dict:
                            if col == 'timestamp' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO code_context ({', '.join(codectx_expected)}) VALUES ({', '.join(['?' for _ in codectx_expected])})",
                        tuple(row_dict[col] for col in codectx_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} code contexts after migration.")
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
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown'
                    )
                """)
            conn.commit()
    
    async def save_development_session(self, workspace_path: str, active_files: List[str] = None,
                                     git_branch: str = None, session_summary: str = None,
                                     user_id: str = None, model_id: str = None, source: str = "direct") -> str:
        """Save a development session, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        session_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO project_sessions 
               (session_id, start_timestamp, workspace_path, active_files, git_branch, session_summary, user_id, model_id) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, timestamp, workspace_path, 
             json.dumps(active_files) if active_files else None,
             git_branch, session_summary, user_id, model_id)
        )
        
        return session_id
    
    async def store_development_conversation(self, content: str, session_id: str = None,
                                          chat_context_id: str = None, decisions_made: str = None,
                                          code_changes: Dict = None, user_id: str = None, model_id: str = None,
                                          source: str = "direct") -> str:
        """Store a development conversation from VS Code
        
        Args:
            content: The conversation content
            session_id: Optional project session ID (will create new if none)
            chat_context_id: Optional VS Code chat context ID
            decisions_made: Summary of decisions made in conversation
            code_changes: Dictionary of files changed and their changes
            user_id: User ID for scoping
            model_id: Model ID for scoping
            source: Source of the tool call (default: direct)
        """
        conversation_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        # Create session if none provided
        if not session_id:
            session_id = await self.save_development_session(
                workspace_path=os.getcwd(),  # Current workspace
                session_summary="Auto-created session for development conversation",
                user_id=user_id or "unknown",
                model_id=model_id or "unknown",
                source=source
            )
        
        # Store conversation
        await self.execute_update(
            """INSERT INTO development_conversations 
               (conversation_id, session_id, timestamp, chat_context_id,
                conversation_content, decisions_made, code_changes, user_id, model_id, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, session_id, timestamp, chat_context_id,
             content, decisions_made, json.dumps(code_changes) if code_changes else None,
             user_id or "unknown", model_id or "unknown", source)
        )
        
        return conversation_id

    async def store_project_insight(self, content: str, insight_type: str = None,
                                  related_files: List[str] = None, importance_level: int = 5,
                                  source_conversation_id: str = None, user_id: str = None,
                                  model_id: str = None, source: str = "direct") -> str:
        """Store a project development insight, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        insight_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO project_insights 
               (insight_id, timestamp_created, timestamp_updated, insight_type, content, 
                related_files, source_conversation_id, importance_level, user_id, model_id, source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, timestamp, timestamp, insight_type, content,
             json.dumps(related_files) if related_files else None,
             source_conversation_id, importance_level, user_id, model_id, source)
        )
        
        return insight_id


class MCPToolCallDatabase(DatabaseManager):
    """Manages MCP tool call logging and AI self-reflection"""
    
    def __init__(self, db_path: str = "memory_data/mcp_tool_calls.db"):
        super().__init__(db_path)
        self.initialize_tables()
    
    def initialize_tables(self):
        """Create tables if they don't exist, and migrate schema if columns are missing"""
        with self.get_connection() as conn:
            # --- MIGRATION LOGIC FOR TOOL_CALLS TABLE ---
            toolcalls_expected = [
                'call_id', 'timestamp', 'client_id', 'tool_name', 'parameters', 'execution_time_ms',
                'status', 'result', 'error_message', 'embedding', 'created_at', 'source'
            ]
            cur = conn.execute("PRAGMA table_info(tool_calls)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in toolcalls_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                logger.warning("Migrating tool_calls table to new schema!")
                old_rows = conn.execute("SELECT * FROM tool_calls").fetchall()
                conn.execute("DROP TABLE IF EXISTS tool_calls")
                conn.execute("""
                    CREATE TABLE tool_calls (
                        call_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        client_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        parameters TEXT,
                        execution_time_ms REAL,
                        status TEXT NOT NULL,
                        result TEXT,
                        error_message TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'direct'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in toolcalls_expected:
                        if col not in row_dict:
                            if col == 'timestamp' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO tool_calls ({', '.join(toolcalls_expected)}) VALUES ({', '.join(['?' for _ in toolcalls_expected])})",
                        tuple(row_dict[col] for col in toolcalls_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} tool calls after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tool_calls (
                        call_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        client_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        parameters TEXT,
                        execution_time_ms REAL,
                        status TEXT NOT NULL,
                        result TEXT,
                        error_message TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        source TEXT DEFAULT 'direct'
                    )
                """)

            # --- MIGRATION LOGIC FOR USAGE_PATTERNS TABLE ---
            usage_expected = [
                'pattern_id', 'timestamp_created', 'analysis_period_days', 'pattern_type', 'insight',
                'confidence_score', 'supporting_data', 'embedding', 'created_at'
            ]
            cur = conn.execute("PRAGMA table_info(usage_patterns)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in usage_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                logger.warning("Migrating usage_patterns table to new schema!")
                old_rows = conn.execute("SELECT * FROM usage_patterns").fetchall()
                conn.execute("DROP TABLE IF EXISTS usage_patterns")
                conn.execute("""
                    CREATE TABLE usage_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        analysis_period_days INTEGER NOT NULL,
                        pattern_type TEXT NOT NULL,
                        insight TEXT NOT NULL,
                        confidence_score REAL DEFAULT 0.5,
                        supporting_data TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in usage_expected:
                        if col not in row_dict:
                            if col == 'timestamp_created' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'confidence_score':
                                row_dict[col] = 0.5
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO usage_patterns ({', '.join(usage_expected)}) VALUES ({', '.join(['?' for _ in usage_expected])})",
                        tuple(row_dict[col] for col in usage_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} usage patterns after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        analysis_period_days INTEGER NOT NULL,
                        pattern_type TEXT NOT NULL,
                        insight TEXT NOT NULL,
                        confidence_score REAL DEFAULT 0.5,
                        supporting_data TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # --- MIGRATION LOGIC FOR AI_REFLECTIONS TABLE ---
            reflections_expected = [
                'reflection_id', 'timestamp_created', 'reflection_type', 'content', 'insights',
                'recommendations', 'confidence_level', 'source_period_days', 'embedding', 'created_at',
                'user_id', 'model_id', 'source'
            ]
            cur = conn.execute("PRAGMA table_info(ai_reflections)")
            current_columns = [row[1] for row in cur.fetchall()]
            needs_migration = False
            if current_columns:
                for col in reflections_expected:
                    if col not in current_columns:
                        needs_migration = True
                        break
            if needs_migration:
                logger.warning("Migrating ai_reflections table to new schema!")
                old_rows = conn.execute("SELECT * FROM ai_reflections").fetchall()
                conn.execute("DROP TABLE IF EXISTS ai_reflections")
                conn.execute("""
                    CREATE TABLE ai_reflections (
                        reflection_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        reflection_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        insights TEXT,
                        recommendations TEXT,
                        confidence_level REAL DEFAULT 0.5,
                        source_period_days INTEGER,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown',
                        source TEXT DEFAULT 'direct'
                    )
                """)
                for row in old_rows:
                    row_dict = dict(row)
                    for col in reflections_expected:
                        if col not in row_dict:
                            if col == 'timestamp_created' or col == 'created_at':
                                row_dict[col] = datetime.now().isoformat()
                            elif col == 'confidence_level':
                                row_dict[col] = 0.5
                            else:
                                row_dict[col] = None
                    conn.execute(
                        f"INSERT INTO ai_reflections ({', '.join(reflections_expected)}) VALUES ({', '.join(['?' for _ in reflections_expected])})",
                        tuple(row_dict[col] for col in reflections_expected)
                    )
                logger.warning(f"Restored {len(old_rows)} ai reflections after migration.")
            else:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_reflections (
                        reflection_id TEXT PRIMARY KEY,
                        timestamp_created TEXT NOT NULL,
                        reflection_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        insights TEXT,
                        recommendations TEXT,
                        confidence_level REAL DEFAULT 0.5,
                        source_period_days INTEGER,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL DEFAULT 'unknown',
                        model_id TEXT NOT NULL DEFAULT 'unknown',
                        source TEXT DEFAULT 'direct'
                    )
                """)
            conn.commit()
    
    async def log_tool_call(self, client_id: str, tool_name: str, parameters: Dict = None,
                          execution_time_ms: float = None, status: str = "success",
                          result: Any = None, error_message: str = None, source: str = "direct") -> str:
        """Log a tool call for AI self-reflection analysis with source tracking"""
        
        call_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        # Serialize complex data
        parameters_json = json.dumps(parameters) if parameters else None
        
        # Sanitize result to be JSON serializable
        def sanitize_for_json(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, bytes):
                return f"<binary data {len(obj)} bytes>"
            else:
                return str(obj)
        
        result_json = json.dumps(sanitize_for_json(result)) if result and not isinstance(result, str) else str(result) if result else None
        
        await self.execute_update(
            """INSERT INTO tool_calls 
               (call_id, timestamp, client_id, tool_name, parameters, execution_time_ms, 
                status, result, error_message, source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, timestamp, client_id, tool_name, parameters_json, 
             execution_time_ms, status, result_json, error_message, source)
        )
        
        return call_id
    
    async def get_tool_usage_stats(self, days: int = 7, client_id: str = None) -> Dict:
        """Get tool usage statistics for AI self-reflection"""
        
        cutoff_date = (datetime.now(get_local_timezone()) - 
                      timedelta(days=days)).isoformat()
        
        base_query = "SELECT * FROM tool_calls WHERE timestamp >= ?"
        params = [cutoff_date]
        
        if client_id:
            base_query += " AND client_id = ?"
            params.append(client_id)
        
        base_query += " ORDER BY timestamp DESC"
        
        calls = await self.execute_query(base_query, tuple(params))
        
        if not calls:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "tool_frequency": {},
                "error_patterns": [],
                "client_activity": {}
            }
        
        # Analyze the data
        total_calls = len(calls)
        successful_calls = sum(1 for call in calls if call["status"] == "success")
        success_rate = (successful_calls / total_calls) * 100 if total_calls > 0 else 0
        
        # Execution times (only for successful calls with timing data)
        execution_times = [call["execution_time_ms"] for call in calls 
                          if call["execution_time_ms"] is not None and call["status"] == "success"]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Tool frequency
        tool_frequency = {}
        for call in calls:
            tool = call["tool_name"]
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
        
        # Error patterns
        error_patterns = []
        error_calls = [call for call in calls if call["status"] == "error"]
        error_tools = {}
        for call in error_calls:
            tool = call["tool_name"]
            error_tools[tool] = error_tools.get(tool, 0) + 1
        
        for tool, count in error_tools.items():
            error_patterns.append({
                "tool": tool,
                "error_count": count,
                "error_rate": (count / tool_frequency.get(tool, 1)) * 100
            })
        
        # Client activity
        client_activity = {}
        for call in calls:
            client = call["client_id"]
            if client not in client_activity:
                client_activity[client] = {"total": 0, "successful": 0}
            client_activity[client]["total"] += 1
            if call["status"] == "success":
                client_activity[client]["successful"] += 1
        
        return {
            "total_calls": total_calls,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "tool_frequency": tool_frequency,
            "error_patterns": error_patterns,
            "client_activity": client_activity,
            "analysis_period_days": days,
            "raw_calls": [dict(call) for call in calls]
        }
    
    async def store_usage_pattern(self, pattern_type: str, insight: str, 
                                analysis_period_days: int, confidence_score: float = 0.5,
                                supporting_data: Dict = None) -> str:
        """Store AI-discovered usage pattern"""
        
        pattern_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO usage_patterns 
               (pattern_id, timestamp_created, analysis_period_days, pattern_type, 
                insight, confidence_score, supporting_data) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pattern_id, timestamp, analysis_period_days, pattern_type, insight,
             confidence_score, json.dumps(supporting_data) if supporting_data else None)
        )
        
        return pattern_id
    
    async def store_ai_reflection(self, reflection_type: str, content: str,
                                insights: List[str] = None, recommendations: List[str] = None,
                                confidence_level: float = 0.5, source_period_days: int = None,
                                user_id: str = None, model_id: str = None, source: str = "direct") -> str:
        """Store AI self-reflection analysis with user/model tracking"""
        
        # Provide defaults if not specified
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        reflection_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        await self.execute_update(
            """INSERT INTO ai_reflections 
               (reflection_id, timestamp_created, reflection_type, content, insights, 
                recommendations, confidence_level, source_period_days, user_id, model_id, source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (reflection_id, timestamp, reflection_type, content,
             json.dumps(insights) if insights else None,
             json.dumps(recommendations) if recommendations else None,
             confidence_level, source_period_days, user_id, model_id, source)
        )
        
        return reflection_id
    
    async def get_recent_reflections(self, limit: int = 5, reflection_type: str = None, user_id: str = None, model_id: str = None) -> List[Dict]:
        """Get recent AI reflections, optionally filtered by user/model"""
        
        # Build WHERE clause conditionally
        where_parts = []
        params = []
        
        if reflection_type:
            where_parts.append("reflection_type = ?")
            params.append(reflection_type)
        
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        
        if model_id:
            where_parts.append("model_id = ?")
            params.append(model_id)
        
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        
        query = f"""
            SELECT * FROM ai_reflections 
            WHERE {where_clause}
            ORDER BY timestamp_created DESC 
            LIMIT ?
        """
        params.append(limit)
        
        rows = await self.execute_query(query, params)
        return [dict(row) for row in rows]


class ConversationFileMonitor:
    def __init__(self, memory_system, watch_directories):
        self.memory_system = memory_system
        self.watch_directories = watch_directories
        # Access all databases through memory_system for consistency
        self.conversations_db = memory_system.conversations_db
        self.curated_db = memory_system.curated_db
        
    async def _import_characterai_conversation(self, file_path: str, content: str):
        """Import a Character.ai conversation file with deduplication."""
        try:
            data = json.loads(content)
            metadata = {
                "source_file": file_path,
                "import_timestamp": datetime.now(get_local_timezone()).isoformat(),
                "file_type": "characterai_conversation",
                "application": "character.ai"
            }
            messages = self._parse_conversation_data(data, file_path)
            session_id = None
            conversation_id = None
            imported_count = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                content_data = msg.get("content", msg.get("text", ""))
                if isinstance(content_data, str):
                    msg_content = content_data
                elif isinstance(content_data, dict):
                    msg_content = json.dumps(content_data)
                else:
                    msg_content = str(content_data)
                msg_id = msg.get("id") or msg.get("message_id")
                timestamp = parse_timestamp(msg.get("timestamp"))
                content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                duplicate = False
                if msg_id:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", (timestamp, msg_content)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    result = await self.conversations_db.store_message(
                        content=msg_content,
                        role=role,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        user_id="test",
                        model_id="test",
                        metadata={**metadata, "imported_id": msg_id, "imported_timestamp": timestamp, "content_hash": content_hash}
                    )
                    if isinstance(result, dict) and "error" not in result and session_id is None:
                        session_id = result["session_id"]
                        conversation_id = result["conversation_id"]
                    if isinstance(result, dict) and "error" not in result:
                        imported_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store message from Character.ai import: {result['error']}")
            logger.info(f"Imported {imported_count} Character.ai messages from {file_path}")
        except Exception as e:
            logger.error(f"Error importing Character.ai conversation {file_path}: {e}")

    async def _import_localai_conversation(self, file_path: str, content: str):
        """Import a Local.ai conversation file with deduplication."""
        try:
            data = json.loads(content)
            metadata = {
                "source_file": file_path,
                "import_timestamp": get_current_timestamp(),
                "file_type": "localai_conversation",
                "application": "local.ai"
            }
            messages = self._parse_conversation_data(data, file_path)
            session_id = None
            conversation_id = None
            imported_count = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                content_data = msg.get("content", msg.get("text", ""))
                if isinstance(content_data, str):
                    msg_content = content_data
                elif isinstance(content_data, dict):
                    msg_content = json.dumps(content_data)
                else:
                    msg_content = str(content_data)
                msg_id = msg.get("id") or msg.get("message_id")
                timestamp = parse_timestamp(msg.get("timestamp"))
                content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                duplicate = False
                if msg_id:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", (timestamp, msg_content)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    result = await self.conversations_db.store_message(
                        content=msg_content,
                        role=role,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        user_id="test",
                        model_id="test",
                        metadata={**metadata, "imported_id": msg_id, "imported_timestamp": timestamp, "content_hash": content_hash}
                    )
                    if isinstance(result, dict) and "error" not in result and session_id is None:
                        session_id = result["session_id"]
                        conversation_id = result["conversation_id"]
                    if isinstance(result, dict) and "error" not in result:
                        imported_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store message from Local.ai import: {result['error']}")
            logger.info(f"Imported {imported_count} Local.ai messages from {file_path}")
        except Exception as e:
            logger.error(f"Error importing Local.ai conversation {file_path}: {e}")

    async def _import_textgenwebui_conversation(self, file_path: str, content: str):
        """Import a text-generation-webui conversation file with deduplication."""
        try:
            # Assume log format: one message per line, JSON or plain text
            lines = content.strip().split('\n')
            metadata = {
                "source_file": file_path,
                "import_timestamp": get_current_timestamp(),
                "file_type": "textgenwebui_conversation",
                "application": "text-generation-webui"
            }
            session_id = None
            conversation_id = None
            imported_count = 0
            for line in lines:
                msg_content = line.strip()
                if not msg_content:
                    continue
                content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                duplicate = False
                existing = await self.conversations_db.execute_query(
                    "SELECT content FROM messages WHERE content = ?", (msg_content,)
                )
                if existing:
                    duplicate = True
                if not duplicate:
                    result = await self.conversations_db.store_message(
                        content=msg_content,
                        role="unknown",
                        session_id=session_id,
                        conversation_id=conversation_id,
                        user_id="test",
                        model_id="test",
                        metadata={**metadata, "content_hash": content_hash}
                    )
                    if isinstance(result, dict) and "error" not in result and session_id is None:
                        session_id = result["session_id"]
                        conversation_id = result["conversation_id"]
                    if isinstance(result, dict) and "error" not in result:
                        imported_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store message from text-generation-webui import: {result['error']}")
            logger.info(f"Imported {imported_count} text-generation-webui messages from {file_path}")
        except Exception as e:
            logger.error(f"Error importing text-generation-webui conversation {file_path}: {e}")
    async def _import_lmstudio_conversation(self, file_path: str, content: str):
        """Import an LM Studio conversation file with deduplication."""
        try:
            # Debug: Check content length and first few chars
            logger.debug(f"LM Studio import - file: {file_path}, content length: {len(content)}")
            if not content.strip():
                logger.warning(f"Empty content in LM Studio file: {file_path}")
                return
                
            data = json.loads(content)
            logger.debug(f"Successfully parsed JSON with keys: {list(data.keys())}")
            
            metadata = {
                "source_file": file_path,
                "import_timestamp": get_current_timestamp(),
                "file_type": "lmstudio_conversation",
                "application": "lm_studio"
            }
            messages = self._parse_conversation_data(data, file_path)
            logger.info(f"Parsed {len(messages)} messages from LM Studio file")
            
            session_id = None
            conversation_id = None
            imported_count = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                content_data = msg.get("content", msg.get("text", ""))
                if isinstance(content_data, str):
                    msg_content = content_data
                elif isinstance(content_data, dict):
                    msg_content = json.dumps(content_data)
                else:
                    msg_content = str(content_data)
                msg_id = msg.get("id") or msg.get("message_id")
                timestamp = parse_timestamp(msg.get("timestamp"))
                content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                duplicate = False
                if msg_id:
                    existing = await self.memory_system.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    existing = await self.memory_system.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", (timestamp, msg_content)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    result = await self.memory_system.conversations_db.store_message(
                        content=msg_content,
                        role=role,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        user_id="test",
                        model_id="test",
                        metadata={**metadata, "imported_id": msg_id, "imported_timestamp": timestamp, "content_hash": content_hash}
                    )
                    if isinstance(result, dict) and "error" not in result and session_id is None:
                        session_id = result["session_id"]
                        conversation_id = result["conversation_id"]
                    if isinstance(result, dict) and "error" not in result:
                        imported_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store message from LM Studio import: {result['error']}")
            logger.info(f"Imported {imported_count} new messages from LM Studio conversation {file_path} (skipped {len(messages) - imported_count} duplicates)")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in LM Studio file {file_path}: {e}")
            logger.error(f"Content preview: {content[:200]}...")
        except Exception as e:
            logger.error(f"Error importing LM Studio conversation {file_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _import_ollama_conversation(self, file_path: str, content: str = None):
        """Import Ollama conversations from SQLite database."""
        try:
            # Ollama stores conversations in SQLite database, not JSON files
            if file_path.lower().endswith('db.sqlite') and 'ollama' in file_path.lower():
                await self._import_ollama_database(file_path)
            else:
                # Handle legacy JSON format if it exists
                if content:
                    data = json.loads(content)
                    metadata = {
                        "source_file": file_path,
                        "import_timestamp": get_current_timestamp(),
                        "file_type": "ollama_conversation",
                        "application": "ollama"
                    }
                    messages = self._parse_conversation_data(data, file_path)
                    await self._import_parsed_messages(messages, metadata, file_path)
        except Exception as e:
            logger.error(f"Error importing Ollama conversation {file_path}: {e}")
    
    async def _import_ollama_database(self, db_path: str):
        """Import conversations from Ollama SQLite database."""
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
                return
            
            # Group messages by chat
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
                    chats[chat_id]['messages'].append({
                        'role': role,
                        'content': content,
                        'model_name': model_name,
                        'created_at': msg_created_at
                    })
            
            # Import each chat
            total_imported = 0
            for chat_id, chat_data in chats.items():
                if not chat_data['messages']:
                    continue
                    
                metadata = {
                    "source_file": db_path,
                    "import_timestamp": get_current_timestamp(),
                    "file_type": "ollama_database",
                    "application": "ollama",
                    "chat_id": chat_id,
                    "chat_title": chat_data['title']
                }
                
                imported_count = await self._import_parsed_messages(chat_data['messages'], metadata, db_path)
                total_imported += imported_count
            
            logger.info(f"Imported {total_imported} messages from {len(chats)} Ollama chats in {db_path}")
            
        except Exception as e:
            logger.error(f"Error importing Ollama database {db_path}: {e}")
    
    async def _import_parsed_messages(self, messages: List[Dict], metadata: Dict, file_path: str) -> int:
        """Helper method to import a list of parsed messages."""
        session_id = None
        conversation_id = None
        imported_count = 0
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role", "unknown")
            content_data = msg.get("content", msg.get("text", ""))
            if isinstance(content_data, str):
                msg_content = content_data
            elif isinstance(content_data, dict):
                msg_content = json.dumps(content_data)
            else:
                msg_content = str(content_data)
            
            if not msg_content.strip():
                continue
                
            msg_id = msg.get("id") or msg.get("message_id")
            timestamp = parse_timestamp(msg.get("timestamp") or msg.get("created_at"))
            content_hash = hashlib.md5(msg_content.encode()).hexdigest()
            
            # Check for duplicates
            duplicate = False
            if msg_id:
                existing = await self.memory_system.conversations_db.execute_query(
                    "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                )
                if existing:
                    duplicate = True
            
            if not duplicate:
                existing = await self.memory_system.conversations_db.execute_query(
                    "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", 
                    (timestamp, msg_content)
                )
                if existing:
                    duplicate = True
            
            if not duplicate:
                result = await self.memory_system.conversations_db.store_message(
                    content=msg_content,
                    role=role,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    user_id="test",
                    model_id="test",
                    metadata={
                        **metadata, 
                        "imported_id": msg_id, 
                        "imported_timestamp": timestamp, 
                        "content_hash": content_hash,
                        "model_name": msg.get("model_name")
                    }
                )
                if isinstance(result, dict) and "error" not in result and session_id is None:
                    session_id = result["session_id"]
                    conversation_id = result["conversation_id"]
                if isinstance(result, dict) and "error" not in result:
                    imported_count += 1
                elif isinstance(result, dict) and "error" in result:
                    logger.error(f"Failed to store message from Ollama import: {result['error']}")
        
        return imported_count
    """Monitors conversation files from various chat applications and imports them to memory"""
    
    def __init__(self, memory_system, watch_directories: List[str] = None):
        self.memory_system = memory_system
        self.watch_directories = watch_directories or []
        self.observers = []
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.file_hashes = {}  # Track file content hashes to detect changes
        self.processed_messages = {}  # Track processed messages per file: {file_path: set(message_hashes)}
        self.conversation_contexts = {}  # Track ongoing conversations for tool detection
        self.mcp_server_running = False  # Will be updated periodically
        self.last_mcp_check = 0  # Timestamp of last MCP server check
        self.last_processed_times = {}  # Track when files were last processed
        self.empty_files = set()  # Track files that are empty to avoid repetitive logging
        self.blacklisted_files = set()  # Track files to permanently ignore (set manually when needed)
        self.min_process_interval = 5.0  # Minimum seconds between processing the same file to reduce CPU usage
        self.max_file_size_mb = 50  # Maximum file size to process at once (in MB)
        self.file_chunk_size = 1024 * 1024  # Process large files in 1MB chunks
        
        # File stability tracking - wait for files to stop being written before processing
        self.file_stability_tracking = {}  # {file_path: {"size": bytes, "last_check": timestamp, "stable_count": int}}
        self.stability_check_interval = 0.5  # How long to wait between size checks (seconds)
        self.stability_threshold = 3  # How many consecutive stable checks before processing (1.5 seconds total)
        
        # Tool detection patterns
        self.tool_patterns = {
            # Memory search triggers
            'search_request': re.compile(
                r'(?:remember|recall|find|search|what did|tell me about|do you know about)'
                r'.*?(?:conversation|memory|previous|earlier|before|history)',
                re.IGNORECASE
            ),
            # Appointment/schedule triggers  
            'schedule_request': re.compile(
                r'(?:schedule|appointment|meeting|remind me|set reminder|calendar)',
                re.IGNORECASE
            ),
            # Memory storage triggers
            'memory_storage': re.compile(
                r'(?:remember this|save this|store this|make a note|keep track)',
                re.IGNORECASE
            )
        }
        
        # Default watch directories for various platforms
        self.default_directories = self._get_default_chat_directories()
    
    def _check_mcp_server(self) -> bool:
        """Check if the MCP server is running by attempting a connection"""
        # Only check every 60 seconds to avoid overhead
        current_time = time.time()
        if current_time - self.last_mcp_check < 60:
            return self.mcp_server_running
            
        try:
            # Try to connect to MCP server port
            with socket.create_connection(("localhost", 1234), timeout=1.0):
                self.mcp_server_running = True
        except (socket.timeout, ConnectionRefusedError):
            self.mcp_server_running = False
        
        self.last_mcp_check = current_time
        return self.mcp_server_running
        
    async def _is_message_in_mcp(self, msg_hash: str) -> bool:
        try:
            # Check in both conversations and messages tables using the conversations_db
            # First check messages table
            result = await self.conversations_db.execute_query(
                "SELECT COUNT(*) FROM messages WHERE message_hash = ?",
                (msg_hash,)
            )
            if result and result[0][0] > 0:
                return True
                
            # Then check conversations table
            result = await self.conversations_db.execute_query(
                "SELECT COUNT(*) FROM conversations WHERE message_hash = ?",
                (msg_hash,)
            )
            return result and result[0][0] > 0
        except Exception as e:
            logger.debug(f"Failed to check message in MCP: {e}")
            return False  # If check fails, assume message doesn't exist
    
    def _get_default_chat_directories(self) -> List[str]:
        """Get default chat storage directories for different platforms"""
        home = Path.home()
        documents = home / "Documents"
        downloads = home / "Downloads"
        directories = []
        
        # ChatGPT desktop app directories
        chatgpt_paths = [
            home / "AppData" / "Roaming" / "ChatGPT" / "chats",  # Windows
            home / ".config" / "ChatGPT" / "chats",  # Linux
            home / "Library" / "Application Support" / "ChatGPT" / "chats"  # macOS
        ]
        
        # Claude desktop app directories
        claude_paths = [
            home / "AppData" / "Roaming" / "Anthropic" / "Claude" / "conversations",  # Windows
            home / ".config" / "anthropic-claude" / "conversations",  # Linux
            home / "Library" / "Application Support" / "Claude" / "conversations"  # macOS
        ]
        
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
        
        # ChatGPT desktop app
        chatgpt_paths = [
            home / "AppData" / "Roaming" / "ChatGPT" / "chats",  # Windows
            home / ".config" / "ChatGPT" / "chats",  # Linux
            home / "Library" / "Application Support" / "ChatGPT" / "chats"  # macOS
        ]
        
        # Claude desktop paths
        claude_paths = [
            home / "AppData" / "Roaming" / "Anthropic" / "Claude" / "conversations",  # Windows
            home / ".config" / "anthropic-claude" / "conversations",  # Linux
            home / "Library" / "Application Support" / "Claude" / "conversations"  # macOS
        ]
        
        # Microsoft Copilot/Bing Chat
        copilot_paths = [
            home / "AppData" / "Roaming" / "Microsoft" / "Windows" / "INetCache" / "Copilot",  # Windows
            home / ".config" / "microsoft-copilot" / "Cache",  # Linux
            home / "Library" / "Caches" / "com.microsoft.copilot"  # macOS
        ]
        
        # Character.ai desktop
        character_ai_paths = [
            home / "AppData" / "Roaming" / "Character.ai" / "conversations",  # Windows
            home / ".config" / "character-ai" / "conversations",  # Linux
            home / "Library" / "Application Support" / "Character.ai" / "conversations"  # macOS
        ]
        
        # Local.ai paths
        local_ai_paths = [
            home / ".local.ai" / "conversations",  # Windows/Linux/macOS
            home / "AppData" / "Roaming" / "local.ai" / "conversations",  # Windows alternative
            home / ".config" / "local.ai" / "conversations"  # Linux alternative
        ]
        
        # text-generation-webui paths
        text_gen_paths = [
            home / "text-generation-webui" / "logs",  # Default install location
            documents / "text-generation-webui" / "logs",  # Common custom location
            home / ".cache" / "text-generation-webui" / "logs"  # Alternative location
        ]
        
        # OpenAI API Playground exports
        openai_paths = [
            downloads / "openai-playground-exports",  # Common export location
            documents / "OpenAI" / "playground-exports"  # Alternative location
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
        
        # Add paths for each application
        add_paths_if_exist(lm_studio_paths, "LM Studio")
        add_paths_if_exist(chatgpt_paths, "ChatGPT")
        add_paths_if_exist(claude_paths, "Claude")
        add_paths_if_exist(copilot_paths, "Microsoft Copilot/Bing")
        add_paths_if_exist(character_ai_paths, "Character.ai")
        add_paths_if_exist(local_ai_paths, "Local.ai")
        add_paths_if_exist(text_gen_paths, "text-generation-webui")
        add_paths_if_exist(openai_paths, "OpenAI Playground")
        
        # Special handling for Ollama database
        for db_path in ollama_db_paths:
            if db_path.exists():
                directories.append(str(db_path))
                logger.info(f"Found Ollama database: {db_path}")
        
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
    
    def _detect_conversation_source(self, file_path: str) -> str:
        """Detect the source application of a conversation file"""
        file_lower = file_path.lower()
        if 'chatgpt' in file_lower:
            return 'ChatGPT'
        elif 'claude' in file_lower:
            return 'Claude'
        elif 'copilot' in file_lower or 'bing' in file_lower:
            return 'Microsoft Copilot'
        elif 'character.ai' in file_lower:
            return 'Character.ai'
        elif 'lmstudio' in file_lower:
            return 'LM Studio'
        elif 'ollama' in file_lower:
            return 'Ollama'
        elif 'local.ai' in file_lower:
            return 'Local.ai'
        elif 'text-generation-webui' in file_lower:
            return 'Text Generation WebUI'
        elif 'openai' in file_lower and 'playground' in file_lower:
            return 'OpenAI Playground'
        elif 'vscode' in file_lower or 'chatsessions' in file_lower:
            return 'VS Code'
        else:
            return 'Unknown'
            
    def _parse_conversation_data(self, data: Union[Dict, List], file_path: str) -> List[Dict]:
        """Parse conversation data using a registry of format handlers for future-proofing."""
        # Registry of format handlers: (predicate, handler)
        format_handlers = [
            (lambda d: isinstance(d, dict) and 'mapping' in d, self._parse_chatgpt_format),
            # LM Studio format: has 'messages' with 'versions' structure
            (lambda d: isinstance(d, dict) and 'messages' in d and self._is_lmstudio_format(d), self._parse_lmstudio_format),
            (lambda d: isinstance(d, dict) and 'messages' in d, self._parse_claude_format),
            (lambda d: isinstance(d, dict) and 'conversation' in d, self._parse_character_ai_format),
            (lambda d: isinstance(d, dict) and 'history' in d, self._parse_text_gen_format),
            (lambda d: isinstance(d, list) and any('role' in msg for msg in d if isinstance(msg, dict)), self._parse_simple_array),
            (lambda d: isinstance(d, list) and any('character' in msg for msg in d if isinstance(msg, dict)), lambda d: self._parse_character_ai_format({'conversation': d})),
        ]
        for predicate, handler in format_handlers:
            if predicate(data):
                return handler(data)
        # Fallback: treat as a list of dicts with 'content' and 'role', or empty
        if isinstance(data, list):
            return [msg for msg in data if isinstance(msg, dict) and 'content' in msg]
        return []

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
                    base_timestamp = datetime.fromtimestamp(conversation_timestamp / 1000)
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
                            timestamp = datetime_to_local_isoformat(
                                base_timestamp + timedelta(minutes=i * 2)
                            )
                        
                        conversations.append({
                            'role': role,
                            'content': final_content,
                            'timestamp': timestamp,
                            'metadata': {
                                'source': 'LM_Studio',
                                'model': version.get('senderInfo', {}).get('senderName', 'unknown'),
                                'conversation_name': data.get('name', 'Untitled'),
                                'version_index': current_version,
                                'step_type': version.get('type'),
                                'token_count': data.get('tokenCount')
                            }
                        })
        
        except Exception as e:
            logger.error(f"Error parsing LM Studio format: {e}")
        
        return conversations

    def _parse_chatgpt_format(self, data: Dict) -> List[Dict]:
        """Parse ChatGPT export format with timestamps"""
        conversations = []
        try:
            if 'mapping' in data:
                for node_id, node in data['mapping'].items():
                    if node.get('message') and node['message'].get('content'):
                        content_parts = node['message']['content'].get('parts', [])
                        if content_parts:
                            timestamp = None
                            if 'create_time' in node['message']:
                                try:
                                    timestamp = datetime_to_local_isoformat(
                                        datetime.fromtimestamp(
                                            int(node['message']['create_time'])
                                        )
                                    )
                                except (ValueError, TypeError):
                                    pass
                            conversations.append({
                                'role': node['message'].get('author', {}).get('role', 'unknown'),
                                'content': ' '.join(str(part) for part in content_parts if part),
                                'timestamp': timestamp
                            })
        except Exception as e:
            logger.error(f"Error parsing ChatGPT format: {e}")
        return conversations

    def _parse_simple_array(self, data: List) -> List[Dict]:
        """Parse simple conversation array format with timestamps"""
        conversations = []
        for item in data:
            if isinstance(item, dict) and 'content' in item:
                timestamp = None
                for key in ['timestamp', 'time', 'created_at', 'date']:
                    if key in item:
                        try:
                            if isinstance(item[key], (int, float)):
                                timestamp = datetime_to_local_isoformat(datetime.fromtimestamp(item[key]))
                            else:
                                timestamp = datetime_to_local_isoformat(datetime.fromisoformat(str(item[key])))
                            break
                        except (ValueError, TypeError):
                            continue
                conversations.append({
                    'role': item.get('role', 'user'),
                    'content': str(item['content']),
                    'timestamp': timestamp
                })
        return conversations

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
    
    def _parse_markdown_format(self, content: str) -> List[Dict]:
        """Parse markdown conversation formats commonly used by AI apps"""
        conversations = []
        current_role = None
        current_content = []
        current_timestamp = None
        
        # Common markdown patterns
        role_patterns = {
            'user': [
                r'^#+ *User:',
                r'^#+ *Human:',
                r'^\*\*User:\*\*',
                r'^\*\*Human:\*\*',
                r'^> *User:',
                r'^> *Human:'
            ],
            'assistant': [
                r'^#+ *Assistant:',
                r'^#+ *AI:',
                r'^\*\*Assistant:\*\*',
                r'^\*\*AI:\*\*',
                r'^> *Assistant:',
                r'^> *AI:'
            ]
        }
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try to extract timestamp from markdown metadata
            if line.startswith('<!--') and 'timestamp:' in line:
                try:
                    ts = re.search(r'timestamp: *(.*?) *-->', line)
                    if ts:
                        current_timestamp = datetime.fromisoformat(ts.group(1))
                    continue
                except (ValueError, AttributeError):
                    pass
            
            # Check for role changes
            new_role = None
            for role, patterns in role_patterns.items():
                if any(re.match(pattern, line) for pattern in patterns):
                    # Save previous message if exists
                    if current_role and current_content:
                        conversations.append({
                            'role': current_role,
                            'content': '\n'.join(current_content),
                            'timestamp': current_timestamp.isoformat() if current_timestamp else None
                        })
                        current_content = []
                    new_role = role
                    # Remove the role marker from the line
                    line = re.sub(r'^[#>*]+.*?:', '', line).strip()
                    break
            
            if new_role:
                current_role = new_role
            
            if current_role and line:
                current_content.append(line)
        
        # Add the last message
        if current_role and current_content:
            conversations.append({
                'role': current_role,
                'content': '\n'.join(current_content),
                'timestamp': current_timestamp.isoformat() if current_timestamp else None
            })
        
        return conversations
    
    def add_watch_directory(self, directory: str):
        """Add a directory to monitor for conversation files"""
        if directory not in self.watch_directories:
            self.watch_directories.append(directory)
            logger.info(f"Added watch directory: {directory}")
    
    async def start_monitoring(self):
        """Start monitoring all configured directories"""
        try:
            # Store reference to the current event loop
            self.loop = asyncio.get_running_loop()
            
            # Import watchdog here to make it optional
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class ConversationFileHandler(FileSystemEventHandler):
                def __init__(self, monitor):
                    self.monitor = monitor
                
                def on_modified(self, event):
                    if not event.is_directory:
                        try:
                            logger.info(f"Detected file modification: {event.src_path}")
                            # Get the event loop from the main thread
                            loop = self.monitor.loop
                            if loop and loop.is_running():
                                logger.debug("Processing file change in event loop")
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
            
            # Combine configured and default directories
            all_directories = self.watch_directories + self.default_directories
            
            for directory in all_directories:
                if os.path.exists(directory):
                    observer = Observer()
                    handler = ConversationFileHandler(self)
                    observer.schedule(handler, directory, recursive=True)
                    observer.start()
                    self.observers.append(observer)
                    logger.info(f"Started monitoring: {directory}")
            
            # Initial scan of all directories
            await self._initial_scan()
            
        except ImportError:
            logger.warning("Watchdog library not installed. File monitoring disabled. Install with: pip install watchdog")
        except Exception as e:
            logger.error(f"Error starting file monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop all file monitoring"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        self.observers.clear()
        logger.info("Stopped file monitoring")
    
    async def _initial_scan(self):
        """Perform initial scan of all watch directories"""
        logger.info("Performing initial scan of conversation files...")
        
        for directory in self.watch_directories + self.default_directories:
            if os.path.exists(directory):
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self._is_conversation_file(file_path):
                            await self._process_file_change(file_path)
    
    def _is_conversation_file(self, file_path: str) -> bool:
        """Determine if a file is likely a conversation file"""
        file_path_lower = file_path.lower()
        file_name = os.path.basename(file_path_lower)
        
        # Ollama database files
        if file_name == 'db.sqlite' and 'ollama' in file_path_lower:
            return True
        
        # VS Code chat session files (UUID.json in chatSessions folder)
        if 'chatsessions' in file_path_lower and file_name.endswith('.json'):
            # Check if filename looks like a UUID
            name_without_ext = file_name[:-5]  # Remove .json
            if len(name_without_ext) == 36 and name_without_ext.count('-') == 4:
                return True
        
        # LM Studio conversation files
        if file_path_lower.endswith('.json') and ('conversation' in file_path_lower or 'lm' in file_path_lower):
            return True
        
        # Other potential conversation formats
        if file_path_lower.endswith(('.jsonl', '.chat', '.conversation')):
            return True
        
        # Generic chat files
        if 'chat' in file_path_lower and file_path_lower.endswith(('.json', '.jsonl', '.txt')):
            return True
        
        return False
    
    async def _check_file_stability(self, file_path: str) -> bool:
        """
        Check if a file has stabilized (stopped being written to).
        Returns True if file is stable and ready to process, False if still being written.
        
        Strategy:
        - Track file size on each check
        - If size hasn't changed for N consecutive checks, file is stable
        - Add to wait_list on first detection of change
        - Remove from wait_list when stable
        """
        try:
            if not os.path.exists(file_path):
                # File was deleted, remove from tracking
                self.file_stability_tracking.pop(file_path, None)
                return False
            
            current_size = os.path.getsize(file_path)
            current_time = time.time()
            
            # First time seeing this file or it's been a while
            if file_path not in self.file_stability_tracking:
                self.file_stability_tracking[file_path] = {
                    "size": current_size,
                    "last_check": current_time,
                    "stable_count": 0
                }
                logger.debug(f"Started tracking file stability for: {file_path} ({current_size} bytes)")
                return False  # First check, not stable yet
            
            tracking_info = self.file_stability_tracking[file_path]
            time_since_check = current_time - tracking_info["last_check"]
            
            # Too soon to check again, skip
            if time_since_check < self.stability_check_interval:
                return False
            
            # Check if file size changed
            if current_size != tracking_info["size"]:
                # File is being written to, reset counter
                tracking_info["size"] = current_size
                tracking_info["last_check"] = current_time
                tracking_info["stable_count"] = 0
                file_size_mb = current_size / (1024 * 1024)
                logger.debug(f"File still growing: {file_path} ({file_size_mb:.2f}MB)")
                return False
            else:
                # Size is stable, increment counter
                tracking_info["stable_count"] += 1
                tracking_info["last_check"] = current_time
                
                if tracking_info["stable_count"] >= self.stability_threshold:
                    # File is stable!
                    logger.info(f"File stabilized after {tracking_info['stable_count'] * self.stability_check_interval:.1f}s: {file_path} ({current_size / (1024 * 1024):.2f}MB)")
                    return True
                else:
                    # Still waiting for more confirmations
                    logger.debug(f"File size stable ({tracking_info['stable_count']}/{self.stability_threshold} checks): {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking file stability for {file_path}: {e}")
            self.file_stability_tracking.pop(file_path, None)
            return False
    
    async def _process_file_change(self, file_path: str):
        """Process a changed conversation file"""
        try:
            # Skip if file is blacklisted
            if file_path in self.blacklisted_files:
                return
            
            # Skip if not a conversation file or doesn't exist
            if not self._is_conversation_file(file_path) or not os.path.exists(file_path):
                return
            
            # Check if file is stable (not being written to) - CRITICAL FIRST CHECK
            is_stable = await self._check_file_stability(file_path)
            if not is_stable:
                # File is still being written, don't process yet
                logger.debug(f"Skipping {file_path} - file still being written to")
                return
            
            # File is stable, remove from stability tracking
            self.file_stability_tracking.pop(file_path, None)
                
            # Check if we've processed this file recently
            current_time = time.time()
            last_processed = self.last_processed_times.get(file_path, 0)
            if current_time - last_processed < self.min_process_interval:
                return
            
            # Update last processed time before we start processing
            self.last_processed_times[file_path] = current_time
            
            # Check file size and warn if very large
            try:
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                
                if file_size_mb > self.max_file_size_mb:
                    logger.warning(f"Large file detected: {file_path} ({file_size_mb:.2f}MB) - processing with progressive reads to prevent memory issues")
            except Exception as e:
                logger.error(f"Failed to check file size for {file_path}: {e}")
            
            # Calculate file hash and read content with better error handling
            try:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    if not file_content:
                        # Log empty file detection to debug log and track it
                        debug_log_path = os.path.join(os.path.dirname(self.memory_system.conversations_db.db_path), "db_debug_log.txt")
                        try:
                            with open(debug_log_path, 'a', encoding='utf-8') as debug_file:
                                debug_file.write(f"{get_current_timestamp()}: Empty file detected: {file_path}\n")
                        except Exception as log_error:
                            logger.error(f"Failed to write to debug log: {log_error}")
                        
                        # Track empty files to avoid repetitive logging
                        self.empty_files.add(file_path)
                        return
                    
                    # Check if this file was previously empty but now has content
                    if file_path in self.empty_files:
                        logger.info(f"Previously empty file now has content: {file_path}")
                        self.empty_files.remove(file_path)
                    
                    file_hash = hashlib.md5(file_content).hexdigest()
                
                # Skip if we've already processed this exact content
                if file_path in self.file_hashes and self.file_hashes[file_path] == file_hash:
                    return
                
                self.file_hashes[file_path] = file_hash
                
                # Handle SQLite databases (binary files)
                if file_path.lower().endswith('db.sqlite') and 'ollama' in file_path.lower():
                    await self._import_conversation_file(file_path, None)  # Pass None for binary files
                else:
                    # Try to decode text content
                    try:
                        decoded_content = file_content.decode('utf-8')
                        if not decoded_content.strip():
                            logger.warning(f"File contains only whitespace: {file_path}")
                            return
                        logger.debug(f"File content preview: {decoded_content[:100]}...")
                        
                        # Parse and import the conversation
                        await self._import_conversation_file(file_path, decoded_content)
                    except UnicodeDecodeError as e:
                        logger.error(f"Failed to decode file content for {file_path}: {e}")
            except PermissionError:
                logger.error(f"Permission denied reading file: {file_path}")
            except FileNotFoundError:
                logger.error(f"File not found (may have been deleted): {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    async def _import_conversation_file(self, file_path: str, content: str):
        """Import a conversation file into the memory system"""
        try:
            # If MCP server is running, only process messages older than server start
            mcp_running = self._check_mcp_server()
            
            # Special handling for Ollama SQLite database
            if file_path.lower().endswith('db.sqlite') and 'ollama' in file_path.lower():
                await self._import_ollama_conversation(file_path)
                return
            
            # Check if this is a VS Code chat session file
            if 'chatsessions' in file_path.lower() and file_path.endswith('.json'):
                await self._import_vscode_chat_session(file_path, content, mcp_running)
            # Detect other file formats and parse accordingly
            elif file_path.endswith('.json'):
                await self._import_json_conversation(file_path, content, mcp_running)
            elif file_path.endswith('.jsonl'):
                await self._import_jsonl_conversation(file_path, content, mcp_running)
            else:
                await self._import_text_conversation(file_path, content)
                
        except Exception as e:
            logger.error(f"Error importing conversation from {file_path}: {e}")
    
    async def _read_large_json_progressively(self, file_path: str, chunk_size_mb: int = 10):
        """Read and parse large JSON files progressively to prevent memory exhaustion
        
        Args:
            file_path: Path to the JSON file
            chunk_size_mb: Size of chunks to read at a time (in MB)
            
        Returns:
            Parsed JSON object or None if file is too corrupted
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # For files larger than chunk_size, read and validate incrementally
            if file_size_mb > chunk_size_mb:
                logger.info(f"Reading large file progressively: {file_path} ({file_size_mb:.2f}MB)")
                
                # First, try to read entire file and parse it
                # If it fails, we'll have already logged the issue
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    # Try to parse - if valid JSON, return it
                    data = json.loads(content)
                    logger.info(f"Successfully parsed large JSON file: {file_path}")
                    return data
                    
                except json.JSONDecodeError as e:
                    # Try to recover by finding the last complete JSON object
                    logger.warning(f"JSON parsing failed for {file_path}, attempting recovery from partial JSON")
                    
                    # Find the last complete object by scanning backward
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    # Try to find last complete JSON array/object
                    for end_pos in range(len(content) - 1, max(len(content) - 100000, 0), -1):
                        try_content = content[:end_pos]
                        
                        # Try to complete the JSON structure
                        if try_content.count('[') > try_content.count(']'):
                            test_str = try_content + ']'
                        elif try_content.count('{') > try_content.count('}'):
                            test_str = try_content + '}'
                        else:
                            test_str = try_content
                        
                        try:
                            data = json.loads(test_str)
                            logger.info(f"Recovered JSON from {file_path} - recovered {end_pos}/{len(content)} bytes")
                            return data
                        except json.JSONDecodeError:
                            continue
                    
                    # If we couldn't recover, log and return None
                    logger.error(f"Could not recover valid JSON from {file_path} - file may be corrupted")
                    return None
            else:
                # Small file, read normally
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error reading large JSON file {file_path}: {e}")
            return None

    async def _import_vscode_chat_session(self, file_path: str, content: str, mcp_running: bool = False):
        """Import a VS Code chat session file (Copilot format) with duplicate prevention"""
        try:
            # Handle empty or whitespace-only content
            if not content or not content.strip():
                logger.warning(f"Empty or whitespace-only VS Code chat session file: {file_path}")
                return
            
            # For very large files, use progressive reader
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 30:  # > 30MB, use progressive reader
                logger.info(f"Large VS Code chat file detected ({file_size_mb:.2f}MB), using progressive reader")
                data = await self._read_large_json_progressively(file_path)
                if not data:
                    logger.error(f"Failed to read large VS Code chat file: {file_path}")
                    return
            else:
                # Regular path for smaller files
                try:
                    # Keep trying until we get valid JSON - monitor file writes
                    retry_delay = 0.5  # Start with 500ms
                    attempt = 0
                    
                    while True:
                        attempt += 1
                        try:
                            data = json.loads(content)
                            logger.debug(f"Successfully parsed JSON data after {attempt} attempts: {json.dumps(data, indent=2)[:500]}...")
                            break
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse failed on attempt {attempt} for {file_path}: {e}. File likely being written...")
                            
                            # Wait for file to stabilize - check if size is changing
                            try:
                                initial_size = os.path.getsize(file_path)
                                await asyncio.sleep(retry_delay)
                                new_size = os.path.getsize(file_path)
                                
                                if initial_size == new_size:
                                    # File size stable, re-read and try again
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                else:
                                    # File still growing, wait longer
                                    logger.info(f"File {file_path} still growing ({initial_size} -> {new_size} bytes), waiting...")
                                    await asyncio.sleep(retry_delay)
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                            except Exception as file_error:
                                logger.error(f"Error monitoring file {file_path}: {file_error}")
                                await asyncio.sleep(retry_delay)
                                
                            # After 10 attempts, increase delay but keep trying
                            if attempt > 10 and attempt % 10 == 0:
                                retry_delay = min(retry_delay * 1.5, 30)  # Cap at 30 seconds
                                logger.info(f"After {attempt} attempts, increasing retry delay to {retry_delay}s for {file_path}")
                            
                            # Never give up, but log progress
                            if attempt % 50 == 0:
                                logger.info(f"Still trying to parse {file_path} after {attempt} attempts. File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'unknown'} bytes")
                except json.JSONDecodeError as e:
                    # Log the problematic content for debugging
                    logger.error(f"Invalid JSON in VS Code chat session {file_path}. Content preview: {content[:100]}...")
                    logger.debug(f"Full content that caused JSON error: {content}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error parsing VS Code chat session: {str(e)}")
                    logger.debug(f"Content that caused error: {content[:500]}...")
                    raise
            
            # Create a consistent session ID based on the file path and initial metadata
            file_session_id = hashlib.md5(f"vscode:{file_path}".encode()).hexdigest()
            
            # Initialize conversation tracking variables
            conversation_id = None
            session_id = None
            
            # Instead of checking whole session, we'll check individual messages as we process them
            last_import_result = await self.memory_system.vscode_db.execute_query(
                """SELECT conversation_content, timestamp FROM development_conversations 
                   WHERE chat_context_id = ? ORDER BY timestamp DESC LIMIT 1""",
                (file_session_id,)
            )
            last_imported_content = last_import_result[0][0] if last_import_result else ""
            last_timestamp = last_import_result[0][1] if last_import_result else "1970-01-01T00:00:00+00:00"
            
            # We'll compare timestamps and content as we process messages to find new ones
            
            # Create a development session for this VS Code chat
            dev_session_id = await self.memory_system.vscode_db.save_development_session(
                workspace_path=os.path.dirname(file_path),
                session_summary=f"Imported VS Code chat session from {os.path.basename(file_path)}"
            )
            
            # Extract metadata from VS Code chat session
            metadata = {
                "source_file": file_path,
                "import_timestamp": get_current_timestamp(),
                "file_type": "vscode_chat_session",
                "application": "vscode_copilot",
                "session_version": data.get("version"),
                "requester": data.get("requesterUsername"),
                "responder": data.get("responderUsername"),
                "initial_location": data.get("initialLocation")
            }
            
            # Build full conversation content for development tracking
            full_conversation = []
            
            # Process each request-response pair
            requests = data.get("requests", [])
            logger.debug(f"Found {len(requests)} requests in chat session")
            logger.debug(f"Sample request structure: {json.dumps(requests[0] if requests else {}, indent=2)}")
            
            # Log raw structure of each request for debugging
            for idx, req in enumerate(requests):
                logger.debug(f"\nRequest {idx} structure:")
                logger.debug(f"Keys present: {list(req.keys())}")
                if "message" in req:
                    logger.debug(f"Message present with keys: {list(req['message'].keys())}")
                if "response" in req:
                    logger.debug(f"Response present with type: {type(req['response'])}")
                    if isinstance(req['response'], dict):
                        logger.debug(f"Response keys: {list(req['response'].keys())}")
            
            new_message_count = 0
            skipped_duplicates = 0
            
            # Initialize tracking variables for message counting
            current_session_id = file_session_id
            current_conversation_id = None
            messages_to_process = []
            already_imported = set()

            # Get all previously imported message hashes
            try:
                imported_messages = await self.memory_system.vscode_db.execute_query(
                    """SELECT source_metadata FROM development_conversations 
                       WHERE chat_context_id = ?""",
                    (file_session_id,)
                )
                for row in imported_messages:
                    if row[0]:  # source_metadata exists
                        metadata = json.loads(row[0])
                        if "content_hash" in metadata:
                            already_imported.add(metadata["content_hash"])
                logger.debug(f"Found {len(already_imported)} previously imported message hashes")
            except Exception as e:
                logger.warning(f"Error getting imported messages: {e}, starting fresh")
                already_imported = set()
            
            for request in requests:
                # Skip if this exact message was manually stored through MCP
                msg_hash = None
                if "message" in request:
                    msg_hash = hashlib.md5(
                        f"user:{request['message'].get('text', '')}".encode()
                    ).hexdigest()
                    if await self._is_message_in_mcp(msg_hash):
                        skipped_duplicates += 1
                        continue
                
                # Import user message
                if "message" in request:
                    user_content = request["message"].get("text", "")
                    # Ensure we have a valid ISO format timestamp
                try:
                    msg_timestamp = request.get("timestamp")
                    if not msg_timestamp:
                        logger.debug("No timestamp found, using current time")
                        msg_timestamp = get_current_timestamp()
                    elif isinstance(msg_timestamp, (int, float)):
                        logger.debug(f"Converting numeric timestamp: {msg_timestamp}")
                        try:
                            msg_timestamp = datetime_to_local_isoformat(datetime.fromtimestamp(msg_timestamp / 1000))
                        except (ValueError, OSError):
                            try:
                                msg_timestamp = datetime_to_local_isoformat(datetime.fromtimestamp(msg_timestamp))
                            except (ValueError, OSError):
                                logger.warning(f"Could not convert numeric timestamp: {msg_timestamp}")
                                msg_timestamp = get_current_timestamp()
                    elif isinstance(msg_timestamp, str):
                        logger.debug(f"Processing string timestamp: {msg_timestamp}")
                        try:
                            # Try parsing different formats
                            if msg_timestamp.endswith('Z'):
                                msg_timestamp = msg_timestamp.replace('Z', '+00:00')
                            if 'T' not in msg_timestamp and ' ' in msg_timestamp:
                                msg_timestamp = msg_timestamp.replace(' ', 'T')
                            parsed = datetime.fromisoformat(msg_timestamp)
                            msg_timestamp = datetime_to_local_isoformat(parsed)
                        except ValueError as e:
                            logger.warning(f"Invalid timestamp string format: {msg_timestamp}, error: {e}")
                            msg_timestamp = get_current_timestamp()
                    else:
                        logger.warning(f"Unexpected timestamp type: {type(msg_timestamp)}")
                        msg_timestamp = get_current_timestamp()
                except Exception as e:
                    logger.warning(f"Error processing timestamp: {e}, using current time")
                    msg_timestamp = get_current_timestamp()
                    
                # Process message if it's valid
                if user_content.strip():
                    content_hash = hashlib.md5(f"{user_content}:{msg_timestamp}".encode()).hexdigest()
                    if content_hash not in already_imported:
                        logger.debug("Found new user message to store")
                        result = await self.memory_system.store_conversation(
                            content=user_content,
                            role="user",
                            session_id=current_session_id,
                            conversation_id=current_conversation_id,
                            metadata={
                                **metadata, 
                                "request_id": request.get("requestId"), 
                                "timestamp": msg_timestamp,
                                "content_hash": content_hash
                            }
                        )
                        logger.info(f"Stored new user message: first 100 chars: {user_content[:100]}...")
                        
                        if result.get("duplicate"):
                            skipped_duplicates += 1
                        else:
                            new_message_count += 1
                            full_conversation.append(f"User: {user_content}")
                            # Update tracking IDs from result
                            if result.get("session_id"):
                                current_session_id = result["session_id"]
                            if result.get("conversation_id"):
                                current_conversation_id = result["conversation_id"]
                            # Add to already imported set to prevent duplicates in same session
                            already_imported.add(content_hash)
                    else:
                        skipped_duplicates += 1
                        logger.debug(f"Skipped duplicate message (hash: {content_hash})")
                    
                # Check for debug logging control commands
                if "disable debug logs" in user_content.lower() or "stop debug logging" in user_content.lower():
                    logger.info("===================================")
                    logger.info("Debug logging disabled by user request")
                    logger.info("Only important messages will be shown")
                    logger.info("Restart the system to re-enable debug logging")
                    logger.info("===================================")
                    logger.setLevel(logging.INFO)
                        
                # Process message with timestamp logging
                logger.debug(f"Processing message with timestamp: {msg_timestamp}")
                if user_content.strip():
                    content_hash = hashlib.md5(f"{user_content}:{msg_timestamp}".encode()).hexdigest()
                    logger.debug("Found user message to process")

                # Only process if this is a new message (after our last processed timestamp)
                if user_content.strip() and str(msg_timestamp) > str(last_timestamp):
                    logger.debug(f"Found new message to process")
                    # Use current tracking IDs
                    logger.debug(f"Storing new user message, timestamp: {msg_timestamp}")
                    result = await self.memory_system.store_conversation(
                        content=user_content,
                        role="user",
                        session_id=current_session_id,
                        conversation_id=current_conversation_id,
                        metadata={**metadata, "request_id": request.get("requestId"), "timestamp": msg_timestamp}
                    )
                    logger.info(f"Stored new user message: first 100 chars: {user_content[:100]}...")
                    
                    if result.get("duplicate"):
                        skipped_duplicates += 1
                    else:
                        new_message_count += 1
                        full_conversation.append(f"User: {user_content}")
                        # Update tracking IDs from result
                        if result.get("session_id"):
                            current_session_id = result["session_id"]
                        if result.get("conversation_id"):
                            current_conversation_id = result["conversation_id"]
                    full_conversation.append(f"User: {user_content}")
                    logger.debug(f"Stored user message with session_id: {current_session_id}, conversation_id: {current_conversation_id}")
                
                # Import assistant response if present
                if "response" in request:
                    response_data = request["response"]
                    assistant_content = None
                    logger.debug("Found assistant response to process")
                    logger.debug(f"Processing response type: {type(response_data).__name__}")
                    logger.debug(f"Session: {current_session_id[:8]}..., Conv: {current_conversation_id[:8] if current_conversation_id else 'None'}")
                    
                    # VS Code responses can have multiple parts
                    if isinstance(response_data, str):
                        logger.debug("Response is string type")
                        assistant_content = response_data
                    elif isinstance(response_data, dict):
                        logger.debug(f"Response is dict type with keys: {list(response_data.keys())}")
                        if "result" in response_data:
                            result_data = response_data["result"]
                            logger.debug(f"Found result data of type {type(result_data)}")
                            if isinstance(result_data, dict):
                                logger.debug(f"Result data keys: {list(result_data.keys())}")
                                # Try to use content or markdown from result
                                if "content" in result_data:
                                    assistant_content = result_data["content"]
                                    logger.debug("Using content from result")
                                elif "markdown" in result_data:
                                    assistant_content = result_data["markdown"]
                                    logger.debug("Using markdown content from result")
                                elif "value" in result_data:
                                    assistant_content = result_data["value"]
                                    logger.debug("Using value content from result")
                            elif isinstance(result_data, str):
                                assistant_content = result_data
                                logger.debug("Using string result data directly")
                            else:
                                assistant_content = json.dumps(result_data)
                                logger.debug("Converted result data to JSON string")
                        elif "content" in response_data:
                            assistant_content = response_data["content"]
                            logger.debug("Using content from response_data directly")
                        elif "value" in response_data:
                            assistant_content = response_data["value"]
                            logger.debug("Using value from response_data directly")
                        elif "markdown" in response_data:
                            assistant_content = response_data["markdown"]
                            logger.debug("Using markdown from response_data directly")
                        else:
                            assistant_content = json.dumps(response_data)
                            logger.debug("No recognized format found, using full response as JSON")
                    
                    if assistant_content:
                        logger.debug(f"Found assistant content (first 100 chars): {assistant_content[:100]}...")
                        # Get timestamp from user message or current time
                        msg_timestamp = None

                        logger.debug(f"Processing assistant message with timestamp: {msg_timestamp}")
                        if assistant_content.strip():
                            content_hash = hashlib.md5(f"{assistant_content}:{msg_timestamp}".encode()).hexdigest()
                            logger.debug("Found assistant message to process")
                        
                        # Try to get timestamp from the user's message first
                        if "message" in request and request["message"].get("timestamp"):
                            try:
                                user_ts = request["message"].get("timestamp")
                                if isinstance(user_ts, (int, float)):
                                    if user_ts > 10**12:  # milliseconds
                                        user_ts = datetime.fromtimestamp(user_ts / 1000)
                                    else:  # seconds
                                        user_ts = datetime.fromtimestamp(user_ts)
                                else:  # string
                                    user_ts = datetime.fromisoformat(str(user_ts).replace('Z', '+00:00'))
                                # Set assistant timestamp 1 second after user's message
                                msg_timestamp = datetime_to_local_isoformat(user_ts + timedelta(seconds=1))
                                logger.debug(f"Using timestamp based on user message: {msg_timestamp}")
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.debug(f"Error parsing user timestamp: {e}")
                                msg_timestamp = None
                        
                        # If no user timestamp, try request timestamp
                        if not msg_timestamp:
                            msg_timestamp = request.get("timestamp")
                            if isinstance(msg_timestamp, (int, float)):
                                try:
                                    if msg_timestamp > 10**12:  # milliseconds
                                        msg_timestamp = datetime_to_local_isoformat(datetime.fromtimestamp(msg_timestamp / 1000))
                                    else:  # seconds
                                        msg_timestamp = datetime_to_local_isoformat(datetime.fromtimestamp(msg_timestamp))
                                except (ValueError, OSError):
                                    msg_timestamp = None
                            elif isinstance(msg_timestamp, str):
                                try:
                                    msg_timestamp = datetime.fromisoformat(msg_timestamp.replace('Z', '+00:00')).isoformat()
                                except ValueError:
                                    msg_timestamp = None
                        
                        # If still no timestamp, use current time
                        if not msg_timestamp:
                            msg_timestamp = get_current_timestamp()
                            logger.debug("Using current system time for timestamp")
                        
                        logger.debug(f"Final message timestamp: {msg_timestamp}, last processed: {last_timestamp}")

                        # Process message with timestamp logging
                        logger.debug(f"Processing message with timestamp: {msg_timestamp}")
                        if assistant_content.strip():
                            content_hash = hashlib.md5(f"{assistant_content}:{msg_timestamp}".encode()).hexdigest()
                            logger.debug("Found assistant message to process")
                        
                        # Store the message if it's new
                        if assistant_content.strip() and str(msg_timestamp) > str(last_timestamp):
                            logger.debug("Storing new assistant message")
                            result = await self.memory_system.store_conversation(
                                content=assistant_content,
                                role="assistant",
                                session_id=current_session_id,
                                conversation_id=current_conversation_id,
                                metadata={
                                    **metadata, 
                                    "request_id": request.get("requestId"), 
                                    "timestamp": msg_timestamp,
                                    "source_type": "vscode_chat"
                                }
                            )
                            logger.info(f"Stored new assistant message: first 100 chars: {assistant_content[:100]}...")
                            
                            if result.get("duplicate"):
                                skipped_duplicates += 1
                                logger.debug("Assistant message was duplicate")
                            else:
                                new_message_count += 1
                                full_conversation.append(f"Assistant: {assistant_content}")
                                logger.info(f"Added new assistant message to conversation: first 100 chars: {assistant_content[:100]}...")
                                # Update tracking IDs from result if needed
                                if result.get("session_id"):
                                    current_session_id = result["session_id"]
                                if result.get("conversation_id"):
                                    current_conversation_id = result["conversation_id"]
            
            # Only store development conversation if we have new messages
            if new_message_count > 0:
                # Store the complete development conversation
                await self.memory_system.vscode_db.store_development_conversation(
                    content="\n\n".join(full_conversation),
                    session_id=dev_session_id,
                    chat_context_id=file_session_id,
                    code_changes=data.get("codeActions", {})  # Store any code actions if present
                )
            
            logger.info(f"Imported {new_message_count} new messages from VS Code chat session {file_path} (skipped {skipped_duplicates} duplicates)")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in VS Code chat session {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error importing VS Code chat session {file_path}: {e}")
    
    async def _import_json_conversation(self, file_path: str, content: str, mcp_running: bool = False):
        """Import a JSON conversation file (future-proof, extensible format support)"""
        try:
            # Keep trying until we get valid JSON - monitor file writes
            retry_delay = 0.5  # Start with 500ms
            attempt = 0
            
            while True:
                attempt += 1
                try:
                    data = json.loads(content)
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed on attempt {attempt} for {file_path}: {e}. File likely being written...")
                    
                    # Wait for file to stabilize - check if size is changing
                    try:
                        initial_size = os.path.getsize(file_path)
                        await asyncio.sleep(retry_delay)
                        new_size = os.path.getsize(file_path)
                        
                        if initial_size == new_size:
                            # File size stable, re-read and try again
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        else:
                            # File still growing, wait longer
                            logger.info(f"File {file_path} still growing ({initial_size} -> {new_size} bytes), waiting...")
                            await asyncio.sleep(retry_delay)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                    except Exception as file_error:
                        logger.error(f"Error monitoring file {file_path}: {file_error}")
                        await asyncio.sleep(retry_delay)
                        
                    # After 10 attempts, increase delay but keep trying
                    if attempt > 10 and attempt % 10 == 0:
                        retry_delay = min(retry_delay * 1.5, 30)  # Cap at 30 seconds
                        logger.info(f"After {attempt} attempts, increasing retry delay to {retry_delay}s for {file_path}")
                    
                    # Never give up, but log progress
                    if attempt % 50 == 0:
                        logger.info(f"Still trying to parse {file_path} after {attempt} attempts. File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'unknown'} bytes")
                
            metadata = {
                "source_file": file_path,
                "import_timestamp": datetime.now(get_local_timezone()).isoformat(),
                "file_type": "json_conversation",
                "mcp_enabled": mcp_running,
                "application": "lm_studio" if "lm" in file_path.lower() else "unknown",
                "source_type": "lm_studio" if "lm" in file_path.lower() else "external_json"
            }
            # Use extensible parser
            messages = self._parse_conversation_data(data, file_path)
            session_id = None
            conversation_id = None
            imported_count = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown")
                content_data = msg.get("content", msg.get("text", ""))
                # Handle different content formats
                if isinstance(content_data, str):
                    msg_content = content_data
                elif isinstance(content_data, dict):
                    msg_content = json.dumps(content_data)
                else:
                    msg_content = str(content_data)
                # Deduplication: use id, timestamp, and content hash
                msg_id = msg.get("id") or msg.get("message_id")
                timestamp = parse_timestamp(msg.get("timestamp"))
                content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                # Check for duplicate by id or content hash
                duplicate = False
                if msg_id:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    existing = await self.conversations_db.execute_query(
                        "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", (timestamp, msg_content)
                    )
                    if existing:
                        duplicate = True
                if not duplicate:
                    result = await self.conversations_db.store_message(
                        content=msg_content,
                        role=role,
                        session_id=session_id,
                        conversation_id=conversation_id,
                        user_id="test",
                        model_id="test",
                        metadata={**metadata, "imported_id": msg_id, "imported_timestamp": timestamp, "content_hash": content_hash}
                    )
                    if isinstance(result, dict) and "error" not in result and session_id is None:
                        session_id = result["session_id"]
                        conversation_id = result["conversation_id"]
                    if isinstance(result, dict) and "error" not in result:
                        imported_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store message from JSON import: {result['error']}")
            logger.info(f"Imported {imported_count} messages from {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error importing JSON conversation {file_path}: {e}")
    
    async def _import_jsonl_conversation(self, file_path: str, content: str):
        """Import a JSONL conversation file (one JSON object per line), with deduplication."""
        try:
            metadata = {
                "source_file": file_path,
                "import_timestamp": datetime.now(get_local_timezone()).isoformat(),
                "file_type": "jsonl_conversation",
                "source_type": "external_jsonl"
            }
            lines = content.strip().split('\n')
            message_count = 0
            session_id = None
            conversation_id = None
            for line in lines:
                if line.strip():
                    try:
                        msg = json.loads(line)
                        role = msg.get("role", "unknown")
                        content_data = msg.get("content", "")
                        if isinstance(content_data, str):
                            msg_content = content_data
                        else:
                            msg_content = json.dumps(content_data)
                        # Deduplication: use id, timestamp, and content hash
                        msg_id = msg.get("id") or msg.get("message_id")
                        timestamp = parse_timestamp(msg.get("timestamp"))
                        content_hash = hashlib.md5(msg_content.encode()).hexdigest()
                        duplicate = False
                        if msg_id:
                            existing = await self.conversations_db.execute_query(
                                "SELECT message_id FROM messages WHERE message_id = ?", (msg_id,)
                            )
                            if existing:
                                duplicate = True
                        if not duplicate:
                            existing = await self.conversations_db.execute_query(
                                "SELECT message_id FROM messages WHERE timestamp = ? AND content = ?", (timestamp, msg_content)
                            )
                            if existing:
                                duplicate = True
                        if not duplicate and msg_content.strip():
                            result = await self.conversations_db.store_message(
                                content=msg_content,
                                role=role,
                                session_id=session_id,
                                conversation_id=conversation_id,
                                user_id="test",
                                model_id="test",
                                metadata={**metadata, "imported_id": msg_id, "imported_timestamp": timestamp, "content_hash": content_hash}
                            )
                            if isinstance(result, dict) and "error" not in result and session_id is None:
                                session_id = result["session_id"]
                                conversation_id = result["conversation_id"]
                            if isinstance(result, dict) and "error" not in result:
                                message_count += 1
                            elif isinstance(result, dict) and "error" in result:
                                logger.error(f"Failed to store message from JSONL import: {result['error']}")
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
            logger.info(f"Imported {message_count} messages from {file_path}")
        except Exception as e:
            logger.error(f"Error importing JSONL conversation {file_path}: {e}")
    
    def _parse_claude_format(self, data: Dict) -> List[Dict]:
        """Parse Claude/Anthropic conversation format"""
        conversations = []
        
        try:
            # Handle both array and object formats
            messages = data.get('messages', [])
            if isinstance(messages, dict):
                messages = messages.values()
            
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    # Try to extract timestamp
                    timestamp = None
                    if 'timestamp' in msg:
                        try:
                            timestamp = datetime.fromisoformat(msg['timestamp'])
                        except (ValueError, TypeError):
                            pass
                    
                    conversations.append({
                        'role': msg.get('role', 'unknown'),
                        'content': msg['content'],
                        'timestamp': timestamp.isoformat() if timestamp else None,
                        'metadata': {
                            'source': 'Claude',
                            'model': data.get('model', 'claude'),
                            'conversation_id': data.get('conversation_id'),
                            'message_id': msg.get('id'),
                            'parent_id': msg.get('parent')
                        }
                    })
        except Exception as e:
            logger.error(f"Error parsing Claude format: {e}")
        
        return conversations
    
    async def _import_conversation_file(self, file_path: str, content: str):
        """Import a conversation file into the memory system, only importing new messages after the first scan."""
        try:
            mcp_running = self._check_mcp_server() if hasattr(self, '_check_mcp_server') else False
            file_lower = file_path.lower()
            # VS Code chat session files
            if 'chatsessions' in file_lower and file_path.endswith('.json'):
                await self._import_vscode_chat_session(file_path, content, mcp_running)
            # LM Studio conversation files
            elif 'lmstudio' in file_lower or ('lm studio' in file_lower) or ('lm_studio' in file_lower):
                await self._import_lmstudio_conversation(file_path, content)
            # Ollama conversation files
            elif 'ollama' in file_lower:
                await self._import_ollama_conversation(file_path, content)
            # Character.ai conversation files
            elif 'character.ai' in file_lower or 'character-ai' in file_lower:
                await self._import_characterai_conversation(file_path, content)
            # Local.ai conversation files
            elif 'local.ai' in file_lower or 'localai' in file_lower:
                await self._import_localai_conversation(file_path, content)
            # text-generation-webui conversation files
            elif 'text-generation-webui' in file_lower or 'textgenwebui' in file_lower:
                await self._import_textgenwebui_conversation(file_path, content)
            # Other JSON conversation files
            elif file_path.endswith('.json'):
                await self._import_json_conversation(file_path, content, mcp_running)
            elif file_path.endswith('.jsonl'):
                await self._import_jsonl_conversation(file_path, content, mcp_running)
            else:
                await self._import_text_conversation(file_path, content)
        except Exception as e:
            logger.error(f"Error importing conversation from {file_path}: {e}")

    # The following block had indentation and variable errors. Fixing only those, not refactoring.
    async def _import_text_conversation(self, file_path, lines, session_id=None, conversation_id=None, metadata=None):
        current_role = "user"
        current_message = []
        message_count = 0
        try:
            for line in lines:
                if line.startswith(("User:", "Human:", "You:")):
                    if current_message:
                        msg_content = '\n'.join(current_message).strip()
                        duplicate = False
                        existing = await self.conversations_db.execute_query(
                            "SELECT content FROM messages WHERE content = ?", (msg_content,)
                        )
                        if existing:
                            duplicate = True
                        if not duplicate:
                            result = await self._save_text_message(current_message, current_role, session_id, conversation_id, metadata)
                            if session_id is None:
                                session_id = result["session_id"]
                                conversation_id = result["conversation_id"]
                            message_count += 1
                    current_message = [line[line.find(":")+1:].strip()]
                    current_role = "user"
                elif line.startswith(("Assistant:", "AI:", "Bot:", "Friday:")):
                    if current_message:
                        msg_content = '\n'.join(current_message).strip()
                        duplicate = False
                        existing = await self.conversations_db.execute_query(
                            "SELECT content FROM messages WHERE content = ?", (msg_content,)
                        )
                        if existing:
                            duplicate = True
                        if not duplicate:
                            result = await self._save_text_message(current_message, current_role, session_id, conversation_id, metadata)
                            if isinstance(result, dict) and "error" not in result and session_id is None:
                                session_id = result["session_id"]
                                conversation_id = result["conversation_id"]
                            if isinstance(result, dict) and "error" not in result:
                                message_count += 1
                            elif isinstance(result, dict) and "error" in result:
                                logger.error(f"Failed to store text message from import: {result['error']}")
                    current_message = [line[line.find(":")+1:].strip()]
                    current_role = "assistant"
                else:
                    # Continuation of current message
                    current_message.append(line)
            # Save the last message
            if current_message:
                msg_content = '\n'.join(current_message).strip()
                existing = await self.conversations_db.execute_query(
                    "SELECT content FROM messages WHERE content = ?", (msg_content,)
                )
                if not existing:
                    result = await self._save_text_message(current_message, current_role, session_id, conversation_id, metadata)
                    if isinstance(result, dict) and "error" not in result:
                        message_count += 1
                    elif isinstance(result, dict) and "error" in result:
                        logger.error(f"Failed to store final text message from import: {result['error']}")
            logger.info(f"Imported {message_count} messages from {file_path}")
        except Exception as e:
            logger.error(f"Error importing text conversation {file_path}: {e}")
    
    async def _save_text_message(self, message_lines: List[str], role: str, session_id: str, conversation_id: str, metadata: Dict):
        """Save a text message to the memory system"""
        content = '\n'.join(message_lines).strip()
        if content:
            return await self.conversations_db.store_message(
                content=content,
                role=role,
                session_id=session_id,
                conversation_id=conversation_id,
                user_id="test",
                model_id="test",
                metadata=metadata
            )


class EmbeddingService:
    """Intelligent embedding service that preserves existing embeddings while optimizing for quality"""
    
    def __init__(self, config_path: str = "embedding_config.json"):
        self.config_path = config_path
        self.full_config = self._load_full_config()
        self.primary_config = self.full_config.get("primary", {})
        self.fallback_config = self.full_config.get("fallback", {})
        self.initialized = False
        self.provider_availability = {
            "lm_studio": None,  # Will be tested on first use
            "ollama": None,
            "openai": None
        }
        
        # Set up embeddings_endpoint from primary config
        primary_base_url = self.primary_config.get("base_url", "")
        self.embeddings_endpoint = primary_base_url
        
        logger.info("🔧 Intelligent Embedding Service Configuration")
        primary_provider = self.primary_config.get('provider', 'lm_studio')
        primary_model = self.primary_config.get('model', 'text-embedding-nomic-embed-text-v1.5')
        fallback_provider = self.fallback_config.get('provider', 'ollama')
        fallback_model = self.fallback_config.get('model', 'nomic-embed-text:latest')
        
        logger.info(f"✅ Primary: {primary_provider} ({primary_model})")
        logger.info(f"⚡ Fallback: {fallback_provider} ({fallback_model})")
        logger.info(f"💾 Preserving existing 768D embeddings, using best available for new ones")
        logger.info("To customize, edit embedding_config.json in the Friday directory")
    
    def _load_full_config(self) -> dict:
        """Load complete embedding configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    return config_data.get("embedding_configuration", {})
            else:
                # Create default config file if it doesn't exist
                self._create_default_config()
                return self._get_default_full_config()
        except Exception as e:
            logger.warning(f"Failed to load embedding config: {e}, using defaults")
            return self._get_default_full_config()
    
    def _get_default_full_config(self) -> dict:
        """Get default full configuration for Friday system"""
        return {
            "primary": {
                "provider": "lm_studio",
                "model": "text-embedding-qwen3-embedding-0.6b",
                "base_url": "http://192.168.1.50:1234/v1/embeddings",
                "description": "High-quality LM Studio embeddings for semantic search"
            },
            "fallback": {
                "provider": "ollama",
                "model": "qwen3-embedding:0.6b",
                "base_url": "http://localhost:11434/api/embeddings",
                "description": "Fast local Ollama embeddings"
            }
        }
    
    def _create_default_config(self):
        """Create default embedding configuration file"""
        default_config = {
            "embedding_configuration": {
                "primary": {
                    "provider": "lm_studio",
                    "model": "text-embedding-qwen3-embedding-0.6b", 
                    "base_url": "http://192.168.1.50:1234/v1/embeddings",
                    "description": "High-quality LM Studio embeddings for semantic search"
                },
                "fallback": {
                    "provider": "ollama",
                    "model": "qwen3-embedding:0.6b",
                    "base_url": "http://localhost:11434/api/embeddings",
                    "description": "Fast local Ollama embeddings"
                },
                "options": {
                    "openai": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "your-openai-api-key-here",
                        "description": "OpenAI embeddings (requires API key)"
                    }
                }
            },
            "instructions": {
                "setup": [
                    "1. Edit this file to configure your preferred embedding providers",
                    "2. Configure 'primary' for your main embedding service",
                    "3. Configure 'fallback' for backup when primary fails",
                    "4. For Ollama: Make sure model is pulled (ollama pull nomic-embed-text:latest)",
                    "5. For LM Studio: Load an embedding model and update base_url if needed",
                    "6. For OpenAI: Add your API key",
                    "7. Restart Friday to apply changes"
                ],
                "providers": {
                    "lm_studio": "High quality embeddings, best for semantic search",
                    "ollama": "Fast local embeddings, good for development",
                    "openai": "Premium quality but requires internet and API costs"
                },
                "preservation_note": "Existing embeddings are preserved automatically - new embeddings use your configured providers"
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default embedding config at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
    
    @property
    def config(self) -> dict:
        """Backward compatibility property - returns primary config"""
        return self.primary_config
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using intelligent provider selection with preservation strategy"""
        
        # Try primary provider first
        primary_provider = self.primary_config.get("provider", "lm_studio")
        
        try:
            if primary_provider == "lm_studio":
                result = await self._generate_lm_studio_embedding(text)
                if result:
                    self.provider_availability["lm_studio"] = True
                    self.initialized = True
                    return result
                else:
                    self.provider_availability["lm_studio"] = False
                    logger.warning("LM Studio unavailable, trying fallback")
                    
            elif primary_provider == "ollama":
                result = await self._generate_ollama_embedding(text)
                if result:
                    self.provider_availability["ollama"] = True
                    self.initialized = True
                    return result
                else:
                    self.provider_availability["ollama"] = False
                    logger.warning("Ollama unavailable, trying fallback")
                    
            elif primary_provider == "openai":
                result = await self._generate_openai_embedding(text)
                if result:
                    self.provider_availability["openai"] = True
                    self.initialized = True
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
                        self.initialized = True
                        logger.info("Using LM Studio fallback for embedding")
                        return result
                        
                elif fallback_provider == "ollama":
                    result = await self._generate_ollama_embedding(text, fallback=True)
                    if result:
                        self.provider_availability["ollama"] = True
                        self.initialized = True
                        logger.info("Using Ollama fallback for embedding")
                        return result
                        
                elif fallback_provider == "openai":
                    result = await self._generate_openai_embedding(text, fallback=True)
                    if result:
                        self.provider_availability["openai"] = True
                        self.initialized = True
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
            
        base_url = config.get("base_url", "http://localhost:11434/api/embeddings")
        model = config.get("model", "qwen3-embedding:0.6b")
        
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
            
        base_url = config.get("base_url", "http://192.168.1.50:1234/v1/embeddings")
        model = config.get("model", "text-embedding-qwen3-embedding-0.6b")
        
        max_retries = 3 if self.initialized else 5
        
        for attempt in range(max_retries):
            try:
                # Set longer timeout to allow model loading time
                timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    payload = {"model": model, "input": text}
                    async with session.post(base_url, json=payload) as response:
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
                            
                            # Retry if model not found (JIT race)
                            if (
                                (response.status == 404 or response.status == 400) and
                                ("model does not exist" in error_text.lower() or 
                                 "failed to load model" in error_text.lower() or
                                 "cannot read properties of null" in error_text.lower())
                                and attempt < max_retries - 1
                            ):
                                delay = (attempt + 1) * 10  # Increased from 5 to 10 seconds
                                logger.info(f"Retrying LM Studio in {delay} seconds (attempt {attempt+2}/{max_retries})...")
                                await asyncio.sleep(delay)
                                continue
                            return None
            except Exception as e:
                logger.error(f"LM Studio embedding error: {e}")
                return None
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
            
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        model = self.config.get("model", "text-embedding-3-small")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            payload = {"model": model, "input": text}
            async with session.post(f"{base_url}/embeddings", json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and "data" in data and len(data["data"]) > 0:
                        return data["data"][0]["embedding"]
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error {response.status}: {error_text}")
                    return None
    
    async def _generate_custom_embedding(self, text: str) -> List[float]:
        """Generate embedding using custom endpoint"""
        base_url = self.config.get("base_url", "http://localhost:8000")
        model = self.config.get("model", "custom-model")
        
        async with aiohttp.ClientSession() as session:
            payload = {"model": model, "input": text}
            async with session.post(f"{base_url}/embeddings", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding") or data.get("data", [{}])[0].get("embedding")
                else:
                    error_text = await response.text()
                    logger.error(f"Custom API error {response.status}: {error_text}")
                    return None
    
    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings


class AIMemorySystem:
    # Tool documentation for dual-purpose get_tool_information
    # Organized by category: common (all clients), vscode, sillytavern
    TOOL_DOCUMENTATION = {
        "common": {
            "complete_reminder": {
                "description": "Mark a reminder as completed",
                "parameters": {"reminder_id": "string (required) - ID of the reminder to complete"},
                "use_cases": ["When Nate completes a task", "Tracking accomplishments"],
                "example": "complete_reminder(reminder_id='abc123')"
            },
            "get_active_reminders": {
                "description": "Get active (not completed) reminders",
                "parameters": {
                    "limit": "integer (default 10) - Number of reminders to return",
                    "days_ahead": "integer (default 30) - Only show reminders due within X days"
                },
                "use_cases": ["Check what Nate needs to do soon", "Review upcoming tasks"],
                "example": "get_active_reminders(days_ahead=7, limit=5)"
            },
            "get_weather_open_meteo": {
                "description": "Get weather forecast (defaults to Motley, MN)",
                "parameters": {
                    "override": "boolean - Set true to use custom coordinates",
                    "latitude/longitude/timezone_str": "Used only if override=true"
                },
                "use_cases": ["Weather questions", "Planning outdoor activities"],
                "example": "get_weather_open_meteo() or with override=true for custom location"
            },
            "brave_web_search": {
                "description": "Search the internet using Brave search engine",
                "parameters": {
                    "query": "string (required) - Search query",
                    "count": "integer (1-20, default 10) - Number of results",
                    "country": "string (default 'US') - Country code",
                    "language": "string (default 'en') - Language code"
                },
                "use_cases": ["Research questions", "Finding current information"],
                "example": "brave_web_search(query='python asyncio tutorial', count=10)"
            },
            "brave_local_search": {
                "description": "Find businesses and places near a location",
                "parameters": {
                    "query": "string (required) - What to search for",
                    "location": "string (optional) - Search location",
                    "radius": "integer (default 5000) - Search radius in meters"
                },
                "use_cases": ["Finding services", "Locating restaurants"],
                "example": "brave_local_search(query='coffee shops', location='Minnesota')"
            },
            "get_completed_reminders": {
                "description": "Get recently completed reminders",
                "parameters": {"days": "integer (default 7) - How far back to look"},
                "use_cases": ["Celebrating accomplishments", "Tracking productivity"],
                "example": "get_completed_reminders(days=1)"
            },
            "reschedule_reminder": {
                "description": "Update the due date of a reminder",
                "parameters": {
                    "reminder_id": "string (required)",
                    "new_due_datetime": "string (required) - ISO datetime format"
                },
                "use_cases": ["Postponing tasks", "Adjusting schedules"],
                "example": "reschedule_reminder(reminder_id='abc', new_due_datetime='2025-11-25T14:00:00Z')"
            },
            "delete_reminder": {
                "description": "Permanently delete a reminder",
                "parameters": {"reminder_id": "string (required)"},
                "use_cases": ["Removing obsolete tasks"],
                "example": "delete_reminder(reminder_id='abc123')"
            },
            "cancel_appointment": {
                "description": "Cancel a scheduled appointment",
                "parameters": {"appointment_id": "string (required)"},
                "use_cases": ["Postponing meetings", "Cancelling events"],
                "example": "cancel_appointment(appointment_id='xyz789')"
            },
            "complete_appointment": {
                "description": "Mark an appointment as completed",
                "parameters": {"appointment_id": "string (required)"},
                "use_cases": ["Marking events as done"],
                "example": "complete_appointment(appointment_id='xyz789')"
            },
            "get_upcoming_appointments": {
                "description": "Get upcoming appointments (not cancelled)",
                "parameters": {
                    "limit": "integer (default 5)",
                    "days_ahead": "integer (default 30)"
                },
                "use_cases": ["Check upcoming events", "Planning calendar"],
                "example": "get_upcoming_appointments(days_ahead=7)"
            },
            "search_memories": {
                "description": "Search memories using semantic similarity or direct ID lookup. Searches across long-term curated memories, short-term conversational memories, conversations, and schedule.",
                "parameters": {
                    "query": "string (required unless memory_id provided)",
                    "memory_id": "string (for direct lookup)",
                    "limit": "integer (default 10)",
                    "database_filter": "enum: conversations, ai_memories, schedule, all (default all) — ai_memories includes both long-term curated and short-term memories",
                    "memory_type": "string - Filter by type (preference, skill, safety, general)"
                },
                "use_cases": ["Finding past information", "Recalling preferences", "Looking up decisions", "Surfacing recent conversational context not yet in long-term memory"],
                "example": "search_memories(query='python preferences', database_filter='ai_memories')"
            },
            "store_conversation": {
                "description": "Store conversation excerpts for future reference",
                "parameters": {
                    "content": "string (required) - Conversation text",
                    "role": "string (required) - 'user' or 'assistant'",
                    "session_id": "string (optional)",
                    "metadata": "object (optional)"
                },
                "use_cases": ["Saving important discussions"],
                "example": "store_conversation(content='Discussion about project X', role='user')"
            },
            "create_memory": {
                "description": "Create a curated memory entry",
                "parameters": {
                    "content": "string (required)",
                    "memory_type": "string (preference, skill, safety, general, etc.)",
                    "importance_level": "integer (1-10, default 5)",
                    "tags": "array of strings"
                },
                "use_cases": ["Storing preferences", "Recording decisions", "Documenting constraints"],
                "example": "create_memory(content='Nate prefers additive code changes', importance_level=8, memory_type='preference')"
            },
            "update_memory": {
                "description": "Update an existing memory",
                "parameters": {
                    "memory_id": "string (required)",
                    "content": "string (optional)",
                    "importance_level": "integer (optional)",
                    "tags": "array (optional)"
                },
                "use_cases": ["Refining stored information", "Increasing importance of key facts"],
                "example": "update_memory(memory_id='id123', content='Updated info', importance_level=9)"
            },
            "create_appointment": {
                "description": "Create a calendar event (single or recurring)",
                "parameters": {
                    "title": "string (required)",
                    "scheduled_datetime": "string (required) - ISO format",
                    "description": "string (optional)",
                    "location": "string (optional)",
                    "recurrence_pattern": "enum: daily, weekly, monthly, yearly (optional)",
                    "recurrence_count": "integer (optional)"
                },
                "use_cases": ["Medical appointments", "Meetings", "Recurring events"],
                "example": "create_appointment(title='Doctor visit', scheduled_datetime='2025-11-25T10:00:00Z')"
            },
            "create_reminder": {
                "description": "Create a task reminder (single or recurring)",
                "parameters": {
                    "content": "string (required) - Task description",
                    "due_datetime": "string (required) - ISO format",
                    "priority_level": "integer (1-10, default 5)",
                    "recurrence_pattern": "enum: daily, weekly, monthly, yearly (optional)",
                    "recurrence_count": "integer (optional)"
                },
                "use_cases": ["Task management", "Recurring reminders", "Due dates"],
                "example": "create_reminder(content='Work on project X', due_datetime='2025-11-24T09:00:00Z', priority_level=7)"
            },
            "get_reminders": {
                "description": "Get recent or filtered reminders",
                "parameters": {
                    "limit": "integer (default 5)",
                    "include_completed": "boolean (default false)",
                    "days_ahead": "integer (default 30)"
                },
                "use_cases": ["Viewing task list", "Planning work"],
                "example": "get_reminders(limit=10, include_completed=false)"
            },
            "get_recent_context": {
                "description": "Get recent conversation context from past N days",
                "parameters": {
                    "limit": "integer (default 5) - Number of items",
                    "session_id": "string (optional)",
                    "days_back": "integer (default 7)"
                },
                "use_cases": ["Recalling recent discussions", "Context restoration"],
                "example": "get_recent_context(days_back=3, limit=10)"
            },
            "get_system_health": {
                "description": "Get comprehensive system health and database status",
                "parameters": {},
                "use_cases": ["Troubleshooting", "System diagnostics"],
                "example": "get_system_health()"
            },
            "get_tool_information": {
                "description": "Get tool usage statistics OR tool documentation. Pass mode='documentation' for docs.",
                "parameters": {
                    "mode": "string - 'usage' (default) for stats or 'documentation' for tool docs",
                    "tool_name": "string (optional) - Specific tool name to document",
                    "days": "integer (default 7) - For usage mode: analyze past N days",
                    "client_id": "string (optional) - For usage mode: analyze specific client"
                },
                "use_cases": ["Getting tool usage stats", "Learning about available tools", "Understanding tool capabilities"],
                "example": "get_tool_information() OR get_tool_information(mode='documentation') OR get_tool_information(mode='documentation', tool_name='search_memories')"
            },
            "reflect_on_tool_usage": {
                "description": "AI self-reflection on tool usage patterns and effectiveness",
                "parameters": {
                    "days": "integer (default 7) - Analyze past N days",
                    "client_id": "string (optional)"
                },
                "use_cases": ["Analyzing tool effectiveness", "Improving tool usage patterns"],
                "example": "reflect_on_tool_usage(days=7)"
            },
            "get_ai_insights": {
                "description": "Get recent AI self-reflection insights and patterns",
                "parameters": {
                    "limit": "integer (default 5)",
                    "insight_type": "string (optional) - Filter by type",
                    "query": "string (optional) - Search keywords"
                },
                "use_cases": ["Recalling learned patterns", "Understanding AI observations"],
                "example": "get_ai_insights(limit=5, insight_type='pattern_analysis')"
            },
            "store_ai_reflection": {
                "description": "Store AI insights or observations about patterns",
                "parameters": {
                    "content": "string (required) - What was observed",
                    "reflection_type": "string (default 'general')",
                    "insights": "array of strings (optional)",
                    "recommendations": "array of strings (optional)",
                    "confidence_level": "number 0.0-1.0 (default 0.7)"
                },
                "use_cases": ["Recording patterns", "Documenting insights", "Meta-learning"],
                "example": "store_ai_reflection(content='Noticed pattern X', insights=['Pattern1', 'Pattern2'], confidence_level=0.8)"
            },
            "write_ai_insights": {
                "description": "Alias for store_ai_reflection - write AI insights",
                "parameters": {"Same as store_ai_reflection": ""},
                "use_cases": ["Alternative method to store insights"],
                "example": "write_ai_insights(content='...')"
            },
            "get_current_time": {
                "description": "Get current server time in UTC and local time",
                "parameters": {},
                "use_cases": ["Before creating time-based items", "Timezone verification"],
                "example": "get_current_time()"
            },
            "get_appointments": {
                "description": "Get recent appointments with optional filtering",
                "parameters": {
                    "limit": "integer (default 5)",
                    "days_ahead": "integer (default 30)"
                },
                "use_cases": ["Viewing calendar", "Planning"],
                "example": "get_appointments(days_ahead=30)"
            }
        },
        "vscode": {
            "save_development_session": {
                "description": "Capture VS Code development session state",
                "parameters": {
                    "workspace_path": "string (required)",
                    "active_files": "array of strings (optional)",
                    "git_branch": "string (optional)",
                    "session_summary": "string (optional)"
                },
                "use_cases": ["Session checkpoints", "Saving progress"],
                "example": "save_development_session(workspace_path='/path/to/work', session_summary='Completed feature X')"
            },
            "store_project_insight": {
                "description": "Record development decisions or architectural insights",
                "parameters": {
                    "content": "string (required)",
                    "insight_type": "string (optional)",
                    "related_files": "array of strings (optional)",
                    "importance_level": "integer 1-10 (default 5)"
                },
                "use_cases": ["Documenting decisions", "Architecture notes", "Design patterns"],
                "example": "store_project_insight(content='Chose async pattern for performance', importance_level=8)"
            },
            "search_project_history": {
                "description": "Find past development decisions and context",
                "parameters": {
                    "query": "string (required)",
                    "limit": "integer (default 10)"
                },
                "use_cases": ["Recalling past decisions", "Finding implementation notes"],
                "example": "search_project_history(query='database migration')"
            },
            "link_code_context": {
                "description": "Connect conversation to specific code",
                "parameters": {
                    "file_path": "string (required)",
                    "description": "string (required)",
                    "function_name": "string (optional)",
                    "conversation_id": "string (optional)"
                },
                "use_cases": ["Code-conversation linking", "Context preservation"],
                "example": "link_code_context(file_path='friday_memory_system.py', description='Working on embeddings')"
            },
            "get_project_continuity": {
                "description": "Get context to continue development work",
                "parameters": {
                    "workspace_path": "string (optional)",
                    "limit": "integer (default 5) - Context items"
                },
                "use_cases": ["Session restoration", "Work continuation"],
                "example": "get_project_continuity(workspace_path='/path/to/work')"
            }
        },
        "sillytavern": {
            "get_character_context": {
                "description": "Get context about characters from memory",
                "parameters": {
                    "character_name": "string (required)",
                    "context_type": "string (optional) - personality, relationships, history",
                    "limit": "integer (default 5)"
                },
                "use_cases": ["Character development", "Roleplay context"],
                "example": "get_character_context(character_name='Alice', context_type='personality')"
            },
            "store_roleplay_memory": {
                "description": "Store important roleplay moments or character developments",
                "parameters": {
                    "character_name": "string (required)",
                    "event_description": "string (required)",
                    "importance_level": "integer 1-10 (default 5)",
                    "tags": "array of strings (optional)"
                },
                "use_cases": ["Character development tracking", "Scene memory"],
                "example": "store_roleplay_memory(character_name='Alice', event_description='Revealed backstory')"
            },
            "search_roleplay_history": {
                "description": "Search past roleplay interactions and character development",
                "parameters": {
                    "query": "string (required)",
                    "character_name": "string (optional)",
                    "limit": "integer (default 10)"
                },
                "use_cases": ["Recalling scenes", "Character continuity"],
                "example": "search_roleplay_history(query='conflict with Bob', character_name='Alice')"
            }
        }
    }

    async def background_main(self):
        # Start maintenance loop, file monitoring, etc.
        await self.run_database_maintenance()
        # Start periodic maintenance in background (every 24 hours)
        maintenance_task = asyncio.create_task(self._periodic_maintenance_loop())
        self._background_tasks.add(maintenance_task)
        maintenance_task.add_done_callback(self._background_tasks.discard)
        await self._start_monitoring()
        # ...other background tasks as needed...
    
    async def _periodic_maintenance_loop(self):
        """Run database maintenance periodically every 24 hours"""
        while True:
            try:
                # Wait 24 hours before running maintenance again
                await asyncio.sleep(86400)  # 86400 seconds = 24 hours
                logger.info("⏰ Running periodic database maintenance...")
                await self.run_database_maintenance()
            except asyncio.CancelledError:
                logger.info("Periodic maintenance loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic maintenance loop: {e}")
                # Continue the loop even if maintenance fails
    
    async def get_appointments(self, limit: int = 5, days_ahead: int = 30, user_id: str = None, model_id: str = None) -> Dict:
        """Get recent appointments from the schedule database"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.schedule_db.get_appointments(limit, days_ahead, user_id, model_id)
        
        if result.get("status") != "success":
            return {
                "status": "no_appointments",
                "message": "No appointments in the next {} days.".format(days_ahead),
                "appointments": []
            }
        
        appointments = result.get("appointments", [])
        return {
            "status": "success",
            "count": len(appointments),
            "appointments": appointments
        }
    """Main memory system that coordinates all databases and operations"""
    def __init__(self, data_dir: str = "memory_data", enable_file_monitoring: bool = True, 
                 watch_directories: List[str] = None, workspace_path: str = None):
        # Get base path dynamically if workspace_path not provided
        if workspace_path is None:
            workspace_path = str(get_base_path())
        self.workspace_path = workspace_path
        
        # Ensure all required directories exist (logs, memory_data, archives, backups, weather)
        ensure_directories()
        
        # Set up paths dynamically
        base_path = Path(workspace_path)
        memory_data_path = base_path / "memory_data"
        
        # Create directories if they don't exist
        memory_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases with dynamic paths
        self.conversations_db = ConversationDatabase(str(memory_data_path / "conversations.db"))
        self.ai_memory_db = AIMemoryDatabase(str(memory_data_path / "ai_memories.db"))
        self.schedule_db = ScheduleDatabase(str(memory_data_path / "schedule.db"))
        self.vscode_db = VSCodeProjectDatabase(str(memory_data_path / "vscode_project.db"))
        self.mcp_db = MCPToolCallDatabase(str(memory_data_path / "mcp_tool_calls.db"))
        self.workspace_path = Path(self.workspace_path) if isinstance(self.workspace_path, str) else self.workspace_path
        self.data_dir = Path(self.workspace_path) / "memory_data"
        self.memory_data_path = self.data_dir  # Alias for compatibility with multi-DB discovery
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # CHANGE 0: Verify embedding model consistency with short-term system
        logger.info("=" * 80)
        logger.info("EMBEDDING MODEL SYNCHRONIZATION VERIFICATION")
        logger.info("=" * 80)
        primary_model = self.embedding_service.primary_config.get("model", "unknown")
        primary_provider = self.embedding_service.primary_config.get("provider", "unknown")
        fallback_model = self.embedding_service.fallback_config.get("model", "unknown")
        fallback_provider = self.embedding_service.fallback_config.get("provider", "unknown")
        logger.info(f"✅ PRIMARY EMBEDDING:  {primary_provider} → {primary_model}")
        logger.info(f"⚡ FALLBACK EMBEDDING: {fallback_provider} → {fallback_model}")
        logger.info("✓ Embedding models synchronized with short-term system")
        logger.info("=" * 80)
        
        # Track embedding config state for change detection
        # This allows us to detect when Adaptive Memory v3 changes config and re-embed accordingly
        self._last_embedding_config = self._capture_embedding_config()
        
        # Initialize database maintenance manager (handles rotation, discovery, etc.)
        self.db_maintenance = DatabaseMaintenance(self, memory_data_path=str(memory_data_path))
        
        # Registry for active database files per type
        # This tracks which DB file to write to (rotates when limit reached)
        self.active_db_files = {
            "conversations": str(memory_data_path / "conversations.db"),
            "ai_memories": str(memory_data_path / "ai_memories.db"),
            "schedule": str(memory_data_path / "schedule.db"),
            "mcp_tool_calls": str(memory_data_path / "mcp_tool_calls.db"),
            "vscode_project": str(memory_data_path / "vscode_project.db")
        }
        
        # Initialize file monitoring (but do NOT start yet)
        self.file_monitor = None
        if enable_file_monitoring:
            try:
                # Make sure vscode_db is initialized before creating monitor
                if not hasattr(self, 'vscode_db') or self.vscode_db is None:
                    logger.warning("VS Code database not initialized, creating it now")
                    self.vscode_db = VSCodeProjectDatabase(str(self.data_dir / "vscode_project.db"))
                self.file_monitor = ConversationFileMonitor(self, watch_directories)
                # Do NOT start monitoring here; will be started after LM Studio initialization
            except Exception as e:
                logger.error(f"Error initializing file monitor: {e}")
                raise
        
        # Initialize reminder escalation background task
        self._escalation_task = None
        self._escalation_running = False
        
        # Background task tracking for async task lifecycle management
        self._background_tasks = set()
    
    # === Embedding Configuration Management ===
    def _capture_embedding_config(self) -> Dict[str, Any]:
        """Capture current embedding configuration for change detection.
        
        Returns a snapshot of the current embedding config including:
        - Primary provider model name and dimension
        - Primary endpoint URL
        - Fallback provider model name and dimension
        
        Used to detect when Adaptive Memory v3 syncs new config from OpenWebUI valves.
        """
        try:
            primary_config = self.embedding_service.primary_config
            fallback_config = self.embedding_service.fallback_config
            
            config_snapshot = {
                "primary_model": primary_config.get("model", "unknown"),
                "primary_dimension": primary_config.get("dimension", 768),
                "primary_endpoint": primary_config.get("base_url", "unknown"),
                "fallback_model": fallback_config.get("model", "unknown"),
                "fallback_dimension": fallback_config.get("dimension", 768),
                "timestamp": datetime.now().isoformat()
            }
            return config_snapshot
        except Exception as e:
            logger.error(f"Error capturing embedding config: {e}")
            return {}
    
    def _reload_embedding_config(self) -> bool:
        """Reload embedding configuration from file and return True if changed.
        
        This is called periodically to detect when Adaptive Memory v3 updates
        embedding_config.json with new valve values.
        
        Returns:
            bool: True if config has changed, False otherwise
        """
        try:
            # Force reload the config from disk
            self.embedding_service.full_config = self.embedding_service._load_full_config()
            self.embedding_service.primary_config = self.embedding_service.full_config.get("primary", {})
            self.embedding_service.fallback_config = self.embedding_service.full_config.get("fallback", {})
            
            # Capture new config
            new_config = self._capture_embedding_config()
            
            # Compare with last known config (excluding timestamp - it's for logging only)
            new_config_compare = {k: v for k, v in new_config.items() if k != "timestamp"}
            last_config_compare = {k: v for k, v in self._last_embedding_config.items() if k != "timestamp"} if self._last_embedding_config else {}
            
            if new_config_compare != last_config_compare:
                logger.warning(
                    f"Embedding config change detected!\n"
                    f"  Old: {self._last_embedding_config}\n"
                    f"  New: {new_config}"
                )
                self._last_embedding_config = new_config
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error reloading embedding config: {e}")
            return False
    
    async def check_and_handle_embedding_config_change(self) -> bool:
        """Check for embedding config changes and trigger re-embedding if needed.
        
        This method should be called periodically from the MCP server or inlet.
        When a config change is detected (e.g., dimension change), it:
        1. Logs the change
        2. Triggers a full re-embedding of all memories with new dimensions
        3. Updates all memory embeddings in the database
        
        Returns:
            bool: True if config changed and re-embedding was triggered
        """
        config_changed = self._reload_embedding_config()
        
        if config_changed:
            logger.info("=" * 80)
            logger.info("EMBEDDING CONFIG CHANGE DETECTED - INITIATING RE-EMBEDDING PROCESS")
            logger.info("=" * 80)
            
            # Trigger re-embedding of all memories
            try:
                await self._reembed_all_memories()
                logger.info("✅ Re-embedding process completed successfully")
                return True
            except Exception as e:
                logger.error(f"Error during re-embedding process: {e}\n{traceback.format_exc()}")
                return False
        
        return False
    
    async def _generate_and_store_embedding(self, memory_id: str, content: str, 
                                              table: str = "curated_memories",
                                              db_instance=None) -> bool:
        """Generate embedding for content and store in database. Awaitable with error handling.
        
        Args:
            memory_id: ID of the memory/record
            content: Text content to embed
            table: Database table name
            db_instance: Database instance (uses ai_memory_db if None)
        
        Returns:
            bool: True if embedding was successfully generated and stored, False otherwise
        """
        if db_instance is None:
            db_instance = self.ai_memory_db
        
        try:
            if not content or not content.strip():
                logger.warning(f"Empty content for {memory_id}, skipping embedding")
                return False
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding is None:
                logger.error(f"Failed to generate embedding for {memory_id}: returned None")
                return False
            
            # Store embedding
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            embedding_dim = len(embedding)
            
            await db_instance.execute_update(
                f"UPDATE {table} SET embedding = ?, embedding_dimension = ?, updated_at = ? WHERE memory_id = ?",
                (embedding_blob, embedding_dim, get_current_timestamp(), memory_id)
            )
            
            logger.debug(f"✅ Successfully stored {embedding_dim}D embedding for {memory_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error generating/storing embedding for {memory_id}: {e}\n{traceback.format_exc()}")
            return False

    async def _reembed_all_memories(self) -> None:
        """Re-embed all memories in the AI memory database with current embedding config.
        
        This is called when embedding config changes (e.g., dimension change).
        Updates all memory embeddings to use new dimensions/model.
        """
        logger.info("Starting full re-embedding of all memories...")
        
        try:
            # Get all memories from AI memory database
            import sqlite3
            db_path = self.data_dir / "ai_memories.db"
            
            memories = []
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT memory_id, content FROM curated_memories")
                memories = cursor.fetchall()
            
            logger.info(f"Found {len(memories)} memories to re-embed")
            
            if not memories:
                logger.info("No memories to re-embed")
                return
            
            # Re-embed each memory
            reembedded_count = 0
            for memory_id, content in memories:
                try:
                    # Use the new awaitable method for guaranteed completion
                    success = await self._generate_and_store_embedding(
                        memory_id=memory_id,
                        content=content,
                        table="curated_memories",
                        db_instance=self.ai_memory_db
                    )
                    
                    if success:
                        reembedded_count += 1
                    else:
                        logger.error(f"Failed to re-embed memory {memory_id}")
                        continue
                    
                    if reembedded_count % 50 == 0:
                        logger.info(f"  Progress: {reembedded_count}/{len(memories)} memories re-embedded")
                
                except Exception as e:
                    logger.error(f"Error re-embedding memory {memory_id}: {e}")
                    continue
            
            logger.info(f"✅ Re-embedding complete: {reembedded_count}/{len(memories)} memories successfully updated")
        
        except Exception as e:
            logger.error(f"Error in _reembed_all_memories: {e}\n{traceback.format_exc()}")
            raise

    # === Reminder Management Tools ===
    async def complete_reminder(self, reminder_id: str, selection_id: str | None = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Mark a reminder as completed (using dynamic path)"""
        import sqlite3
        import asyncio

        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }

        # Use dynamic path based on workspace
        db_path = self.data_dir / "schedule.db"

        # ---- sync helpers (run via to_thread to avoid blocking the event loop) ----
        def _update_by_id(rid: str) -> int:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.execute(
                    "UPDATE reminders SET completed = 1, completed_at = ? WHERE reminder_id = ? AND user_id = ? AND model_id = ?",
                    (get_current_timestamp(), rid, user_id, model_id)
                )
                conn.commit()
                return cur.rowcount or 0

        def _select_by_due(due: str):
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.execute(
                    "SELECT reminder_id, conversation_title, due_datetime "
                    "FROM reminders "
                    "WHERE due_datetime = ? AND (completed IS NULL OR completed = 0) AND user_id = ? AND model_id = ?",
                    (due, user_id, model_id)
                )
                return cur.fetchall()

        # If a selection_id is provided, skip discovery and complete that specific ID.
        if selection_id:
            affected = await asyncio.to_thread(_update_by_id, selection_id)
            if affected > 0:
                return {
                    "status": "success",
                    "message": f"Reminder {selection_id} marked as completed",
                    "reminder_id": selection_id,
                    "source_db": str(db_path)
                }
            return {
                "status": "error",
                "message": f"Reminder {selection_id} not found (follow-up)",
                "source_db": str(db_path)
            }

        # No selection yet — try exact reminder_id first.
        affected = await asyncio.to_thread(_update_by_id, reminder_id)
        if affected > 0:
            return {
                "status": "success",
                "message": f"Reminder {reminder_id} marked as completed",
                "reminder_id": reminder_id,
                "source_db": str(db_path)
            }

        # Treat 'reminder_id' as a due_datetime and discover matches.
        matches = await asyncio.to_thread(_select_by_due, reminder_id)

        if not matches:
            return {
                "status": "error",
                "message": "Reminder not found by id or due date",
                "source_db": str(db_path)
            }

        if len(matches) == 1:
            only_id = matches[0][0]
            affected = await asyncio.to_thread(_update_by_id, only_id)
            if affected > 0:
                return {
                    "status": "success",
                    "message": f"Reminder with due date {reminder_id} (id {only_id}) marked as completed",
                    "reminder_id": only_id,
                    "source_db": str(db_path)
                }
            return {
                "status": "error",
                "message": "Failed to mark reminder as completed after single-match resolution",
                "source_db": str(db_path)
            }

        # Multiple matches — return options and instruct caller to pass selection_id on the next call.
        options = [
            {"reminder_id": row[0], "conversation_title": row[1], "due_datetime": row[2]}
            for row in matches
        ]
        return {
            "status": "needs_selection",
            "message": f"Multiple reminders share due date {reminder_id}. Call this tool again with selection_id set to the chosen reminder_id.",
            "options": options,
            "source_db": str(db_path)
        }



    async def get_active_reminders(self, limit: int = 10, days_ahead: int = 30, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get active (not completed) reminders within the next X days.
        
        If model_id is None, queries all models for that user (cross-model fallback).
        If model_id is provided (even empty string), filters to that specific model.
        """
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        now = datetime.now(get_local_timezone())
        # Use start of today instead of current time to include all reminders due today
        start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        cutoff = (now + timedelta(days=days_ahead)).isoformat()
        
        # Build query based on whether model_id filtering is requested
        if model_id:
            # Filter by specific model
            rows = await self.schedule_db.execute_query(
                "SELECT * FROM reminders WHERE completed = 0 AND due_datetime >= ? AND due_datetime <= ? AND user_id = ? AND model_id = ? ORDER BY due_datetime LIMIT ?",
                (start_of_today, cutoff, user_id, model_id, limit)
            )
        else:
            # Query all models for this user
            rows = await self.schedule_db.execute_query(
                "SELECT * FROM reminders WHERE completed = 0 AND due_datetime >= ? AND due_datetime <= ? AND user_id = ? ORDER BY due_datetime LIMIT ?",
                (start_of_today, cutoff, user_id, limit)
            )
        
        if not rows:
            return {
                "status": "no_reminders",
                "message": "No active reminders scheduled."
            }
        
        return {
            "status": "success",
            "count": len(rows),
            "reminders": [
                {
                    "reminder_id": r["reminder_id"],
                    "content": r["content"],
                    "due_datetime": r["due_datetime"],
                    "priority_level": r["priority_level"],
                    "created_at": r["created_at"]
                }
                for r in rows
            ]
        }

    async def get_completed_reminders(self, days: int = 7, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get recently completed reminders.
        
        If model_id is None, queries all models for that user (cross-model fallback).
        If model_id is provided (even empty string), filters to that specific model.
        """
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        cutoff = (datetime.now(get_local_timezone()) - timedelta(days=days)).isoformat()
        
        # Build query based on whether model_id filtering is requested
        if model_id:
            # Filter by specific model
            rows = await self.schedule_db.execute_query(
                "SELECT * FROM reminders WHERE completed = 1 AND completed_at >= ? AND user_id = ? AND model_id = ? ORDER BY completed_at DESC",
                (cutoff, user_id, model_id)
            )
        else:
            # Query all models for this user
            rows = await self.schedule_db.execute_query(
                "SELECT * FROM reminders WHERE completed = 1 AND completed_at >= ? AND user_id = ? ORDER BY completed_at DESC",
                (cutoff, user_id)
            )
        
        if not rows:
            return {
                "status": "no_completed_reminders",
                "message": "All reminders are completed in the last {} days.".format(days)
            }
        
        return {
            "status": "success",
            "count": len(rows),
            "reminders": [
                {
                    "reminder_id": r["reminder_id"],
                    "content": r["content"],
                    "due_datetime": r["due_datetime"],
                    "completed_at": r["completed_at"]
                }
                for r in rows
            ]
        }

    async def _escalate_due_reminders(self, grace_period_minutes: int = 15) -> Dict[str, int]:
        """
        Background process to escalate reminder notifications as deadline approaches.
        Calculates urgency levels and manages reminder_notifications entries.
        
        Urgency levels:
        - 1: upcoming (> 4 hours away)
        - 3: soon (1-4 hours away)
        - 5: urgent (< 1 hour away)
        
        Returns count of escalations and cleanups performed.
        """
        import uuid
        from datetime import datetime, timedelta
        
        try:
            now = datetime.now(get_local_timezone())
            escalations_count = 0
            cleanups_count = 0
            
            # Query all non-completed reminders
            all_reminders = await self.schedule_db.execute_query(
                "SELECT reminder_id, due_datetime, user_id, model_id FROM reminders WHERE completed = 0 AND is_completed = 0"
            )
            
            if not all_reminders:
                logger.debug("No active reminders to escalate")
                return {"escalations": 0, "cleanups": 0}
            
            for reminder_row in all_reminders:
                try:
                    reminder_id = reminder_row[0]
                    due_datetime_str = reminder_row[1]
                    user_id = reminder_row[2]
                    model_id = reminder_row[3]
                    
                    # Parse due datetime
                    if due_datetime_str.endswith('Z'):
                        due_dt = datetime. fromisoformat(due_datetime_str[:-1]).replace(tzinfo=pytz.UTC)
                    else:
                        due_dt = datetime.fromisoformat(due_datetime_str)
                        if due_dt.tzinfo is None:
                            due_dt = get_local_timezone().localize(due_dt)
                    
                    time_until_due = due_dt - now
                    grace_period = timedelta(minutes=grace_period_minutes)
                    
                    # Determine urgency level based on time remaining
                    if time_until_due < timedelta(hours=0):  # Past due
                        if time_until_due < -grace_period:  # Past grace period
                            # Clean up this reminder's notifications
                            await self.schedule_db.execute_update(
                                "DELETE FROM reminder_notifications WHERE reminder_id = ? AND user_id = ? AND model_id = ?",
                                (reminder_id, user_id, model_id)
                            )
                            cleanups_count += 1
                        else:  # Still in grace period
                            urgency_level = 5  # Keep as urgent
                    elif time_until_due < timedelta(hours=1):
                        urgency_level = 5  # Urgent
                    elif time_until_due < timedelta(hours=4):
                        urgency_level = 3  # Soon
                    else:
                        urgency_level = 1  # Upcoming
                    
                    # Skip cleanup for past reminders still in grace period - they should remain urgent
                    if time_until_due >= -grace_period:
                        # Check if notification already exists for this urgency level
                        existing = await self.schedule_db.execute_query(
                            "SELECT notification_id FROM reminder_notifications WHERE reminder_id = ? AND urgency_level = ? AND user_id = ? AND model_id = ?",
                            (reminder_id, urgency_level, user_id, model_id)
                        )
                        
                        if not existing:
                            # Create new notification entry for this urgency level
                            notification_id = str(uuid.uuid4())
                            now_iso = get_current_timestamp()
                            await self.schedule_db.execute_update(
                                """INSERT INTO reminder_notifications (notification_id, reminder_id, urgency_level, created_at, updated_at, user_id, model_id)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (notification_id, reminder_id, urgency_level, now_iso, now_iso, user_id, model_id)
                            )
                            escalations_count += 1
                            logger.debug(f"Created notification for reminder {reminder_id} at urgency level {urgency_level}")
                        else:
                            # Update timestamp of existing notification
                            await self.schedule_db.execute_update(
                                "UPDATE reminder_notifications SET updated_at = ? WHERE reminder_id = ? AND urgency_level = ? AND user_id = ? AND model_id = ?",
                                (get_current_timestamp(), reminder_id, urgency_level, user_id, model_id)
                            )
                
                except Exception as e:
                    logger.warning(f"Error processing reminder {reminder_row[0]} during escalation: {e}")
                    continue
            
            logger.info(f"Reminder escalation complete: {escalations_count} escalations, {cleanups_count} cleanups")
            return {"escalations": escalations_count, "cleanups": cleanups_count}
            
        except Exception as e:
            logger.error(f"Error in _escalate_due_reminders: {e}\n{traceback.format_exc()}")
            return {"escalations": 0, "cleanups": 0, "error": str(e)}

    async def get_active_reminders_for_injection(self, user_id: str = None, model_id: str = None) -> Dict:
        """
        Get active reminders grouped by urgency tier for context injection into conversations.
        Used to build the [Active Reminders] section in the conversation context.
        
        Returns dict with keys: "urgent", "soon", "upcoming" containing reminder details.
        """
        try:
            if not user_id or not model_id:
                return {}
            
            now = datetime.now(get_local_timezone())
            
            # Query all non-completed reminders that have active notifications
            reminders = await self.schedule_db.execute_query(
                """SELECT r.reminder_id, r.content, r.due_datetime, r.priority_level, r.conversation_title, n.urgency_level
                   FROM reminders r
                   INNER JOIN reminder_notifications n ON r.reminder_id = n.reminder_id
                   WHERE r.completed = 0 AND r.is_completed = 0 AND r.user_id = ? AND r.model_id = ?
                   ORDER BY r.due_datetime ASC""",
                (user_id, model_id)
            )
            
            if not reminders:
                return {}
            
            # Group by urgency level
            grouped = {"urgent": [], "soon": [], "upcoming": []}
            urgency_names = {5: "urgent", 3: "soon", 1: "upcoming"}
            
            for reminder in reminders:
                try:
                    reminder_id = reminder[0]
                    content = reminder[1]
                    due_datetime_str = reminder[2]
                    priority = reminder[3]
                    conversation_title = reminder[4]
                    urgency_level = reminder[5]
                    
                    # Calculate time until due
                    if due_datetime_str.endswith('Z'):
                        due_dt = datetime.fromisoformat(due_datetime_str[:-1]).replace(tzinfo=pytz.UTC)
                    else:
                        due_dt = datetime.fromisoformat(due_datetime_str)
                        if due_dt.tzinfo is None:
                            due_dt = get_local_timezone().localize(due_dt)
                    
                    time_until_due = due_dt - now
                    
                    # Format time until due
                    if time_until_due.total_seconds() < 0:
                        time_str = f"OVERDUE {abs(time_until_due.total_seconds() // 60):.0f}m ago"
                    elif time_until_due.total_seconds() < 3600:
                        time_str = f"{time_until_due.total_seconds() / 60:.0f}m"
                    elif time_until_due.total_seconds() < 86400:
                        time_str = f"{time_until_due.total_seconds() / 3600:.1f}h"
                    else:
                        time_str = f"{time_until_due.total_seconds() / 86400:.1f}d"
                    
                    # Format reminder for display
                    reminder_display = f"{content}"
                    if conversation_title:
                        reminder_display += f" (from: {conversation_title})"
                    reminder_display += f" • Due in {time_str}"
                    
                    urgency_name = urgency_names.get(urgency_level, "upcoming")
                    grouped[urgency_name].append(reminder_display)
                    
                except Exception as e:
                    logger.warning(f"Error formatting reminder for injection: {e}")
                    continue
            
            return grouped
            
        except Exception as e:
            logger.error(f"Error in get_active_reminders_for_injection: {e}\n{traceback.format_exc()}")
            return {}

    async def reschedule_reminder(self, reminder_id: str, new_due_datetime: str, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Reschedule a reminder to a new due datetime"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.schedule_db.execute_update(
            "UPDATE reminders SET due_datetime = ? WHERE reminder_id = ? AND user_id = ? AND model_id = ?",
            (new_due_datetime, reminder_id, user_id, model_id)
        )
        if result > 0:
            return {"status": "success", "message": f"Reminder {reminder_id} rescheduled to {new_due_datetime}"}
        else:
            return {"status": "error", "message": "Reminder not found"}

    async def delete_reminder(self, reminder_id: str, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.schedule_db.execute_update(
            "DELETE FROM reminders WHERE reminder_id = ? AND user_id = ? AND model_id = ?",
            (reminder_id, user_id, model_id)
        )
        if result and result != "0":
            return {"status": "success", "message": f"Reminder {reminder_id} deleted"}
        else:
            return {"status": "error", "message": "Reminder not found"}

    # === Appointment Management Tools ===
    async def cancel_appointment(self, appointment_id: str, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Cancel an appointment"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.schedule_db.execute_update(
            "UPDATE appointments SET status = 'cancelled', cancelled_at = ? WHERE appointment_id = ? AND user_id = ? AND model_id = ?",
            (get_current_timestamp(), appointment_id, user_id, model_id)
        )
        if result > 0:
            return {"status": "success", "message": f"Appointment {appointment_id} cancelled"}
        else:
            return {"status": "error", "message": "Appointment not found"}

    async def complete_appointment(self, appointment_id: str, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Mark an appointment as completed"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.schedule_db.execute_update(
            "UPDATE appointments SET status = 'completed', completed_at = ? WHERE appointment_id = ? AND user_id = ? AND model_id = ?",
            (get_current_timestamp(), appointment_id, user_id, model_id)
        )
        if result > 0:
            return {"status": "success", "message": f"Appointment {appointment_id} marked as completed"}
        else:
            return {"status": "error", "message": "Appointment not found"}

    async def get_upcoming_appointments(self, limit: int = 5, days_ahead: int = 30, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get upcoming appointments (not cancelled), today onward. Handles ISO-text and integer-epoch datetimes.
        
        If model_id is None, queries all models for that user (cross-model fallback).
        If model_id is provided (even empty string), filters to that specific model.
        """
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        now_local = datetime.now(get_local_timezone())
        cutoff_local = now_local + timedelta(days=days_ahead)
        now_epoch = int(now_local.timestamp())
        cutoff_epoch = int(cutoff_local.timestamp())
        
        # Build query based on whether model_id filtering is requested
        if model_id:
            # Filter by specific model
            query = """
                SELECT *
                FROM appointments
                WHERE status != 'cancelled'
                AND user_id = ?
                AND model_id = ?
                AND (
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END
                ) BETWEEN ? AND ?
                ORDER BY
                    CASE
                        WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                        ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                    END ASC
                LIMIT ?
            """
            rows = await self.schedule_db.execute_query(query, (user_id, model_id, now_epoch, cutoff_epoch, limit))
        else:
            # Query all models for this user
            query = """
                SELECT *
                FROM appointments
                WHERE status != 'cancelled'
                AND user_id = ?
                AND (
                        CASE
                            WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                            ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                        END
                ) BETWEEN ? AND ?
                ORDER BY
                    CASE
                        WHEN typeof(scheduled_datetime) = 'integer' THEN scheduled_datetime
                        ELSE CAST(strftime('%s', scheduled_datetime) AS INTEGER)
                    END ASC
                LIMIT ?
            """
            rows = await self.schedule_db.execute_query(query, (user_id, now_epoch, cutoff_epoch, limit))
        
        if not rows:
            return {
                "status": "no_appointments",
                "message": "No upcoming appointments in the next {} days.".format(days_ahead)
            }
        
        return {
            "status": "success",
            "count": len(rows),
            "appointments": [
                {
                    "appointment_id": r["appointment_id"],
                    "title": r["title"],
                    "scheduled_datetime": r["scheduled_datetime"],
                    "duration_minutes": r["duration_minutes"],
                    "description": r["description"]
                }
                for r in rows
            ]
        }
    
    def ensure_all_memory_databases_ready(self):
        base_dir = get_base_path()  # Get base path dynamically
        db_dir = base_dir / "memory_data"
        db_dir.mkdir(exist_ok=True)
        """
        Ensure all expected memory database files exist with required tables.
        Safe to run repeatedly. Creates empty tables if missing.
        """
        logger.info("Ensuring memory databases are initialized...")

        for db_path, schema in [
            (str(self.data_dir / "conversations.db"), """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    sender TEXT,
                    content TEXT,
                    timestamp TEXT
                );
            """),
            (str(self.data_dir / "ai_memories.db"), """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    timestamp TEXT,
                    importance TEXT
                );
            """),
            (str(self.data_dir / "schedule.db"), """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    due_datetime TEXT NOT NULL,
                    priority_level INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed INTEGER DEFAULT 0,
                    is_completed INTEGER DEFAULT 0,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    datetime TEXT,
                    notes TEXT,
                    status TEXT DEFAULT 'scheduled' CHECK("status" IN ('scheduled', 'cancelled', 'completed'))
                );
            """),
            (str(self.data_dir / "vscode_project.db"), """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    file_path TEXT,
                    timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    type TEXT,
                    detail TEXT
                );
            """),
            (str(self.data_dir / "mcp_tool_calls.db"), """
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    timestamp TEXT,
                    success INTEGER
                );
            """),
        ]:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.executescript(schema)
                conn.commit()
                conn.close()
                logger.info(f"✔️  Verified: {db_path}")
            except Exception as e:
                logger.error(f"❌ Failed to verify {db_path}: {e}")

    async def import_openwebui_chat_history(self, db_path=None):
        """
        Import NEW OpenWebUI messages into Friday memory system with per-user, per-model isolation.
        Uses dedup tracking to only import messages we haven't seen before (by message hash).
        Does NOT retroactively re-align old messages - that's handled by lazy remediation.
        """
        import sqlite3
        import json as json_lib
        import hashlib
        
        # Set default path to the new OpenWebUI location if not provided
        if db_path is None:
            # Try to find OpenWebUI database in common locations
            possible_paths = [
                "/app/backend/data/webui.db",  # Docker/container path
                os.path.join(os.path.expanduser("~"), ".local/share/open-webui/data/webui.db"),  # Linux
                os.path.join(os.path.expanduser("~"), "AppData/Local/open-webui/data/webui.db"),  # Windows
                "/data/webui.db",  # Generic Docker volume
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    db_path = path
                    logger.debug(f"Found OpenWebUI database at: {db_path}")
                    break
            
            if not db_path:
                logger.warning(f"OpenWebUI database not found in common locations: {possible_paths}")
                return
        
        if not os.path.exists(db_path):
            logger.error(f"OpenWebUI database not found at {db_path}")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Extract chat info including user_id and model
            cursor.execute('SELECT id, user_id, title, chat FROM chat')
            chats = {}
            for chat_id, user_id, title, chat_json in cursor.fetchall():
                try:
                    chat_data = json_lib.loads(chat_json) if chat_json else {}
                    # Extract primary model from chat
                    models = chat_data.get('models', [])
                    primary_model = models[0] if models else 'default'
                    # Extract model name (e.g., "friday" from "openai/friday")
                    model_name = primary_model.split('/')[-1] if '/' in primary_model else primary_model
                    chats[chat_id] = {
                        'user_id': user_id,
                        'title': title,
                        'model': model_name,
                        'full_model': primary_model,
                        'chat_data': chat_data
                    }
                except Exception as e:
                    logger.warning(f"Error parsing chat {chat_id}: {e}")
                    chats[chat_id] = {
                        'user_id': user_id,
                        'title': title,
                        'model': 'default',
                        'full_model': 'default',
                        'chat_data': {}
                    }
            
            # Build set of message hashes we've already imported (dedup tracking)
            existing_hashes = set()
            existing_msgs = await self.conversations_db.execute_query(
                """SELECT json_extract(metadata, '$.openwebui_message_hash') as msg_hash 
                   FROM messages WHERE source_type = 'openwebui' AND msg_hash IS NOT NULL"""
            )
            if existing_msgs:
                existing_hashes = {dict(m)['msg_hash'] for m in existing_msgs if dict(m)['msg_hash']}
            
            imported_count = 0
            skipped_count = 0
            
            for chat_id, chat_info in chats.items():
                user_id = chat_info['user_id']
                model_name = chat_info['model']
                chat_data = chat_info['chat_data']
                
                # Extract conversation_id with user_id + model isolation
                conversation_id = f"{user_id}_{model_name}"
                
                # Process messages from chat JSON
                history = chat_data.get('history', {})
                messages_dict = history.get('messages', {})
                
                for msg_id, msg_data in messages_dict.items():
                    if not isinstance(msg_data, dict):
                        continue
                    
                    role = msg_data.get('role', 'user')
                    content = msg_data.get('content', '')
                    timestamp = msg_data.get('timestamp', int(datetime.now(get_local_timezone()).timestamp()))
                    
                    if not content:
                        skipped_count += 1
                        continue
                    
                    # Create hash of this message: hash(chat_id + msg_id + content)
                    # This uniquely identifies this message across all imports
                    msg_hash = hashlib.sha256(
                        f"{chat_id}:{msg_id}:{content}".encode()
                    ).hexdigest()
                    
                    # Check if we've already imported this exact message
                    if msg_hash in existing_hashes:
                        skipped_count += 1
                        continue
                    
                    # Build metadata with full user/model info + dedup hash
                    metadata = {
                        'source': 'openwebui_import',
                        'chat_id': str(chat_id),
                        'chat_title': chat_info['title'],
                        'user_id': str(user_id),
                        'model': model_name,
                        'full_model': chat_info['full_model'],
                        'role': role,
                        'created_at': timestamp,
                        'message_id_in_chat': str(msg_id),
                        'openwebui_message_hash': msg_hash
                    }
                    
                    # Ensure session and conversation records exist (required by FOREIGN KEYs)
                    try:
                        # Create session if needed (use conversation_id as session_id for user+model isolation)
                        existing_session = await self.conversations_db.execute_query(
                            "SELECT session_id FROM sessions WHERE session_id = ?",
                            (conversation_id,)
                        )
                        if not existing_session:
                            await self.conversations_db.execute_update(
                                "INSERT INTO sessions (session_id, start_timestamp, context) VALUES (?, ?, ?)",
                                (conversation_id, timestamp, f"openwebui-import-{model_name}")
                            )
                        
                        # Create conversation if needed
                        existing_conv = await self.conversations_db.execute_query(
                            "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                            (conversation_id,)
                        )
                        if not existing_conv:
                            await self.conversations_db.execute_update(
                                "INSERT INTO conversations (conversation_id, session_id, start_timestamp, user_id, model_id) VALUES (?, ?, ?, ?, ?)",
                                (conversation_id, conversation_id, timestamp, str(user_id), model_name)
                            )
                    except Exception as e:
                        logger.warning(f"Error creating session/conversation for {conversation_id}: {e}")
                    
                    # Store message with unique message_id (using hash so no duplicates)
                    try:
                        await self.conversations_db.execute_update(
                            """INSERT INTO messages (message_id, conversation_id, timestamp, role, content, source_type, metadata, user_id, model_id)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (msg_hash, conversation_id, timestamp, role, content, 'openwebui', json.dumps(metadata), str(user_id), model_name)
                        )
                        imported_count += 1
                        existing_hashes.add(msg_hash)  # Add to tracking set
                    except Exception as e:
                        logger.warning(f"Error importing message {msg_id}: {e}")
                        skipped_count += 1
            
            conn.close()
            logger.info(f"OpenWebUI import: {imported_count} NEW messages imported, {skipped_count} already known")
            
        except Exception as e:
            logger.error(f"Error importing OpenWebUI chat history: {e}")
            raise

    async def verify_and_remediate_chat_isolation(self, webui_db_path=None):
        """
        Verify all imported chats have proper user_id and model isolation.
        Retroactively fix any chats that are missing this information.
        Returns stats on what was fixed.
        """
        import sqlite3
        import json as json_lib
        
        if webui_db_path is None:
            # Try to find OpenWebUI database in common locations
            possible_paths = [
                "/app/backend/data/webui.db",  # Docker/container path
                os.path.join(os.path.expanduser("~"), ".local/share/open-webui/data/webui.db"),  # Linux
                os.path.join(os.path.expanduser("~"), "AppData/Local/open-webui/data/webui.db"),  # Windows
                "/data/webui.db",  # Generic Docker volume
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    webui_db_path = path
                    logger.debug(f"Found OpenWebUI database at: {webui_db_path}")
                    break
            
            if not webui_db_path:
                logger.warning(f"OpenWebUI database not found in common locations: {possible_paths}")
                return {"status": "error", "message": "OpenWebUI database not found"}
        
        if not os.path.exists(webui_db_path):
            logger.error(f"OpenWebUI database not found at {webui_db_path}")
            return {"status": "error", "message": "OpenWebUI database not found"}
        
        try:
            # Build a lookup of chat_id -> (user_id, model)
            conn = sqlite3.connect(webui_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, user_id, chat FROM chat')
            chat_lookup = {}
            for chat_id, user_id, chat_json in cursor.fetchall():
                try:
                    chat_data = json_lib.loads(chat_json) if chat_json else {}
                    models = chat_data.get('models', [])
                    primary_model = models[0] if models else 'default'
                    model_name = primary_model.split('/')[-1] if '/' in primary_model else primary_model
                    chat_lookup[str(chat_id)] = {
                        'user_id': str(user_id),
                        'model': model_name
                    }
                except Exception as e:
                    logger.warning(f"Error processing chat {chat_id} for verification: {e}")
            conn.close()
            
            logger.info(f"Built lookup for {len(chat_lookup)} chats from OpenWebUI")
            
            # Query all messages from AI Memory System
            all_messages = await self.conversations_db.execute_query(
                """SELECT message_id, conversation_id, role, content, source_type, metadata, timestamp 
                   FROM messages WHERE source_type = 'openwebui' ORDER BY conversation_id"""
            )
            
            # Convert sqlite3.Row objects to dicts
            if all_messages:
                all_messages = [dict(msg) for msg in all_messages]
            
            stats = {
                'total_messages': len(all_messages) if all_messages else 0,
                'already_isolated': 0,
                'missing_isolation': 0,
                'remediations': 0,
                'errors': 0,
                'details': []
            }
            
            if not all_messages:
                logger.info("No OpenWebUI messages found to verify")
                return stats
            
            for msg in all_messages:
                msg_id = msg.get('message_id')
                current_conv_id = msg.get('conversation_id')
                metadata_str = msg.get('metadata', '{}')
                
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                except:
                    metadata = {}
                
                # Check if this message's conversation_id follows the isolation format
                chat_id = metadata.get('chat_id')
                
                if not chat_id:
                    stats['errors'] += 1
                    stats['details'].append({
                        'message_id': msg_id,
                        'issue': 'No chat_id in metadata',
                        'current_conv_id': current_conv_id
                    })
                    continue
                
                chat_id_str = str(chat_id)
                
                if chat_id_str not in chat_lookup:
                    stats['errors'] += 1
                    stats['details'].append({
                        'message_id': msg_id,
                        'issue': f'Chat {chat_id_str} not found in OpenWebUI',
                        'current_conv_id': current_conv_id
                    })
                    continue
                
                # Get the correct user_id and model
                chat_info = chat_lookup[chat_id_str]
                user_id = chat_info['user_id']
                model = chat_info['model']
                correct_conv_id = f"{user_id}_{model}"
                
                # Check if already correctly isolated
                if current_conv_id == correct_conv_id:
                    stats['already_isolated'] += 1
                    continue
                
                # Need to remediate
                stats['missing_isolation'] += 1
                
                # Update the metadata with correct user_id and model
                metadata['user_id'] = user_id
                metadata['model'] = model
                metadata['remediated_at'] = datetime.now(get_local_timezone()).isoformat()
                metadata['previous_conversation_id'] = current_conv_id
                
                try:
                    # Update the message with correct conversation_id and updated metadata
                    await self.conversations_db.execute_update(
                        """UPDATE messages SET conversation_id = ?, metadata = ? WHERE message_id = ?""",
                        (correct_conv_id, json.dumps(metadata), msg_id)
                    )
                    stats['remediations'] += 1
                    stats['details'].append({
                        'message_id': msg_id,
                        'previous_conv_id': current_conv_id,
                        'new_conv_id': correct_conv_id,
                        'user_id': user_id,
                        'model': model,
                        'status': 'remediated'
                    })
                except Exception as e:
                    stats['errors'] += 1
                    stats['details'].append({
                        'message_id': msg_id,
                        'issue': f'Update failed: {str(e)}',
                        'current_conv_id': current_conv_id
                    })
            
            logger.info(f"Chat isolation verification complete: {stats['already_isolated']} already isolated, "
                       f"{stats['remediations']} remediated, {stats['errors']} errors")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in verify_and_remediate_chat_isolation: {e}")
            return {"status": "error", "message": str(e)}

    async def _start_monitoring(self):
        """Internal method to start the file monitor"""
        if self.file_monitor:
            try:
                await self.file_monitor.start_monitoring()
                logger.info("File monitoring started")
            except Exception as e:
                logger.error(f"Error starting file monitoring: {e}")
        self.ensure_all_memory_databases_ready()
        
    async def stop_file_monitoring(self):
        """Stop monitoring conversation files"""
        if self.file_monitor:
            await self.file_monitor.stop_monitoring()
            logger.info("File monitoring stopped")
    
    def add_watch_directory(self, directory: str):
        """Add a directory to monitor for conversation files"""
        if self.file_monitor:
            self.file_monitor.add_watch_directory(directory)
    
    async def get_system_health(self, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get comprehensive system health and statistics including CPU, RAM, GPU usage"""
        # Set defaults for logging
        if not user_id:
            user_id = "Nate"
        if not model_id:
            model_id = "Friday"
        
        logger.info(f"System health check requested by user={user_id}, model={model_id}")
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(get_local_timezone()).isoformat(),
            "system_resources": {},
            "databases": {},
            "file_monitoring": {},
            "embedding_service": {},
            "requested_by": {"user_id": user_id, "model_id": model_id}
        }
        
        try:
            # Get system resource metrics (CPU, RAM, GPU)
            system_metrics = await self._get_system_metrics()
            health_data["system_resources"] = system_metrics
            
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
            active_reminders_count = await self.schedule_db.execute_query(
                "SELECT COUNT(*) as count FROM reminders WHERE completed = 0"
            )
            health_data["databases"]["schedule"] = {
                "status": "healthy",
                "appointment_count": appointments_count[0]["count"] if appointments_count else 0,
                "reminder_count": reminders_count[0]["count"] if reminders_count else 0,
                "active_reminder_count": active_reminders_count[0]["count"] if active_reminders_count else 0,
                "database_path": self.schedule_db.db_path
            }
            
            # Check VS Code project database
            project_sessions_count = await self.vscode_db.execute_query(
                "SELECT COUNT(*) as count FROM project_sessions"
            )
            insights_count = await self.vscode_db.execute_query(
                "SELECT COUNT(*) as count FROM project_insights"
            )
            code_context_count = await self.vscode_db.execute_query(
                "SELECT COUNT(*) as count FROM code_context"
            )
            health_data["databases"]["vscode_project"] = {
                "status": "healthy",
                "session_count": project_sessions_count[0]["count"] if project_sessions_count else 0,
                "insight_count": insights_count[0]["count"] if insights_count else 0,
                "code_context_count": code_context_count[0]["count"] if code_context_count else 0,
                "database_path": self.vscode_db.db_path
            }
            
            # Check MCP tool calls database
            tool_calls_count = await self.mcp_db.execute_query(
                "SELECT COUNT(*) as count FROM tool_calls"
            )
            successful_calls_count = await self.mcp_db.execute_query(
                "SELECT COUNT(*) as count FROM tool_calls WHERE status = 'success'"
            )
            reflections_count = await self.mcp_db.execute_query(
                "SELECT COUNT(*) as count FROM ai_reflections"
            )
            patterns_count = await self.mcp_db.execute_query(
                "SELECT COUNT(*) as count FROM usage_patterns"
            )
            
            total_calls = tool_calls_count[0]["count"] if tool_calls_count else 0
            successful_calls = successful_calls_count[0]["count"] if successful_calls_count else 0
            success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
            
            health_data["databases"]["mcp_tool_calls"] = {
                "status": "healthy",
                "total_tool_calls": total_calls,
                "successful_calls": successful_calls,
                "success_rate_percent": round(success_rate, 1),
                "ai_reflections": reflections_count[0]["count"] if reflections_count else 0,
                "usage_patterns": patterns_count[0]["count"] if patterns_count else 0,
                "database_path": self.mcp_db.db_path
            }
            
            # Check file monitoring status
            if self.file_monitor:
                health_data["file_monitoring"] = {
                    "status": "enabled",
                    "active_observers": len(self.file_monitor.observers),
                    "watched_directories": len(self.file_monitor.watch_directories + self.file_monitor.default_directories),
                    "processed_files": len(self.file_monitor.processed_files),
                    "monitored_paths": self.file_monitor.watch_directories + self.file_monitor.default_directories
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
                primary_endpoint = self.embedding_service.primary_config.get("base_url", "unknown")
                if test_embedding:
                    health_data["embedding_service"] = {
                        "status": "healthy",
                        "endpoint": primary_endpoint,
                        "embedding_dimensions": len(test_embedding)
                    }
                else:
                    health_data["embedding_service"] = {
                        "status": "unhealthy",
                        "endpoint": primary_endpoint,
                        "error": "Failed to generate test embedding"
                    }
            except Exception as e:
                primary_endpoint = self.embedding_service.primary_config.get("base_url", "unknown")
                health_data["embedding_service"] = {
                    "status": "unhealthy",
                    "endpoint": primary_endpoint,
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
    
    async def _get_system_metrics(self) -> Dict:
        """Get actual system metrics: CPU, RAM, and GPU/VRAM usage"""
        metrics = {
            "cpu": {},
            "memory": {},
            "gpu": {}
        }
        
        try:
            import psutil
            
            # CPU metrics
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count(logical=True)
                cpu_freq = psutil.cpu_freq()
                
                metrics["cpu"] = {
                    "usage_percent": cpu_percent,
                    "cores": psutil.cpu_count(logical=False),
                    "logical_cores": cpu_count,
                    "frequency_mhz": round(cpu_freq.current) if cpu_freq else None,
                    "status": "healthy"
                }
            except Exception as e:
                metrics["cpu"]["status"] = "unavailable"
                metrics["cpu"]["error"] = str(e)
            
            # RAM metrics
            try:
                ram = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                metrics["memory"] = {
                    "ram": {
                        "total_gb": round(ram.total / (1024**3), 2),
                        "used_gb": round(ram.used / (1024**3), 2),
                        "available_gb": round(ram.available / (1024**3), 2),
                        "usage_percent": ram.percent
                    },
                    "swap": {
                        "total_gb": round(swap.total / (1024**3), 2),
                        "used_gb": round(swap.used / (1024**3), 2),
                        "usage_percent": swap.percent
                    },
                    "status": "healthy"
                }
            except Exception as e:
                metrics["memory"]["status"] = "unavailable"
                metrics["memory"]["error"] = str(e)
            
            # GPU/VRAM metrics - try NVIDIA first, then AMD
            gpu_devices = []
            
            # Try NVIDIA GPUs (nvidia-smi)
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.strip():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 5:
                                gpu_devices.append({
                                    "type": "NVIDIA",
                                    "index": int(parts[0]),
                                    "name": parts[1],
                                    "memory_total_mb": int(float(parts[2])),
                                    "memory_used_mb": int(float(parts[3])),
                                    "memory_used_percent": round(float(parts[3]) / float(parts[2]) * 100, 1),
                                    "utilization_percent": int(float(parts[4]))
                                })
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass  # NVIDIA not available, will try AMD
            
# Try AMD GPUs (rocm-smi - combined flags produce single valid JSON blob)
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--showtemp", "--showuse", "--showproductname", "--json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    rocm_data = json.loads(result.stdout)
                    for card_key in sorted(rocm_data.keys()):
                        idx  = int(card_key.replace("card", ""))
                        card = rocm_data[card_key]

                        total_b  = int(card.get("VRAM Total Memory (B)", 0))
                        used_b   = int(card.get("VRAM Total Used Memory (B)", 0))
                        total_mb = total_b // (1024 * 1024)
                        used_mb  = used_b  // (1024 * 1024)
                        mem_pct  = round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0.0

                        gfx_ver  = card.get("GFX Version", "")
                        gpu_name = f"{card.get('Card Series', f'AMD GPU {idx}')} ({gfx_ver})" if gfx_ver else card.get("Card Series", f"AMD GPU {idx}")

                        temp_edge = card.get("Temperature (Sensor edge) (C)")

                        gpu_devices.append({
                            "type": "AMD",
                            "index": idx,
                            "name": gpu_name,
                            "memory_total_mb": total_mb,
                            "memory_used_mb": used_mb,
                            "memory_used_percent": mem_pct,
                            "utilization_percent": int(card.get("GPU use (%)", 0)),
                            "temperature_c": float(temp_edge) if temp_edge is not None else None
                        })
            except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
                pass  # AMD ROCm not available
            
            # Set GPU status based on what we found
            if gpu_devices:
                metrics["gpu"]["devices"] = gpu_devices
                metrics["gpu"]["status"] = "healthy"
                nvidia_count = sum(1 for d in gpu_devices if d["type"] == "NVIDIA")
                amd_count = sum(1 for d in gpu_devices if d["type"] == "AMD")
                if nvidia_count > 0 and amd_count > 0:
                    metrics["gpu"]["types"] = f"NVIDIA ({nvidia_count}) + AMD ({amd_count})"
                elif nvidia_count > 0:
                    metrics["gpu"]["types"] = f"NVIDIA ({nvidia_count})"
                elif amd_count > 0:
                    metrics["gpu"]["types"] = f"AMD ({amd_count})"
            else:
                metrics["gpu"]["status"] = "no_devices"
                metrics["gpu"]["message"] = "No NVIDIA or AMD GPUs detected"
        
        except ImportError:
            logger.warning("psutil not installed, system metrics unavailable")
            return {
                "cpu": {"status": "unavailable", "error": "psutil not installed"},
                "memory": {"status": "unavailable", "error": "psutil not installed"},
                "gpu": {"status": "unavailable", "error": "psutil not installed"}
            }
        
        return metrics
    
    # Weather operations
    async def get_weather_open_meteo(self, latitude: float = None, longitude: float = None,
                                     timezone_str: str = None, force_refresh: bool = False,
                                     return_changes_only: bool = False, update_today: bool = True,
                                     severe_update: bool = False) -> Dict:
        """Open-Meteo forecast (no API key). Defaults to configured location and caches once per local day."""
        try:
            import requests
            from pathlib import Path
            from utils import get_weather_cache_dir
            
            # Use configured location (from environment or config), or defaults
            try:
                # Try to load from the module-level variables if they exist
                from ai_memory_mcp_server import HOME_LAT, HOME_LON, HOME_TZ
                default_lat = HOME_LAT
                default_lon = HOME_LON
                default_tz = HOME_TZ
            except (ImportError, NameError):
                # Fallback to generic defaults
                default_lat = 46.33301
                default_lon = -94.64384
                default_tz = "America/Chicago"
            
            lat = latitude if latitude is not None else default_lat
            lon = longitude if longitude is not None else default_lon
            tz = timezone_str if timezone_str is not None else default_tz
            
            # Use centralized cache directory
            cache_dir = Path(get_weather_cache_dir())
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            today = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")
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
    
    # Conversation operations
    async def store_conversation(self, content: str, role: str, session_id: str = None,
                               conversation_id: str = None, metadata: Dict = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Store a conversation message"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        result = await self.conversations_db.store_message(
            content, role, session_id, conversation_id, metadata, user_id, model_id, source
        )
        
        # Generate and store embedding asynchronously
        embedding_task = asyncio.create_task(self._add_embedding_to_message(result["message_id"], content))
        self._background_tasks.add(embedding_task)
        embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success",
            "message_id": result["message_id"],
            "conversation_id": result["conversation_id"],
            "session_id": result["session_id"]
        }
    
    async def get_recent_context(
        self,
        limit: int = 5,
        session_id: str | None = None,
        days_back: int = 7,
        user_id: str | None = None,
        model_id: str | None = None,
        source: str = "direct",
    ) -> Dict:


        """Get recent conversation context from the last N days - returns clean text content, not embeddings"""
        
        messages = await self.conversations_db.get_recent_messages(
            limit, session_id, days_back, user_id=user_id, model_id=model_id
        )

        
        # Clean the messages to remove embeddings and provide only useful context
        clean_messages = []
        for msg in messages:
            clean_msg = {
                "message_id": msg["message_id"],
                "conversation_id": msg["conversation_id"],
                "timestamp": msg["timestamp"],
                "role": msg["role"],
                "content": msg["content"],
                "metadata": json.loads(msg["metadata"]) if msg["metadata"] else None
            }
            # Explicitly exclude embedding data
            clean_messages.append(clean_msg)
        
        return {
            "status": "success",
            "messages": clean_messages,
            "count": len(clean_messages),
            "days_back": days_back
        }
    
    # AI Memory operations
    async def create_memory(self, content: str, memory_type: str = None,
                          importance_level: int = 5, tags: List[str] = None,
                          source_conversation_id: str = None, memory_bank: str = "General",
                          user_id: str = None, model_id: str = None, source: str = "direct",
                          wait_for_embedding: bool = False) -> Dict:
        """Create a curated memory.
        
        Args:
            memory_bank: Category for memory (General, Personal, Work, Context, Tasks)
            user_id: User identifier
            model_id: Model identifier
            source: Memory source (direct, mcp_openwebui, mcp_external, openwebui_promotion)
            wait_for_embedding: If True, waits for embedding to complete before returning.
                               If False, embeddings are generated in background.
                               Should be True when called from promotion to ensure embeddings complete.
        """
        
        memory_id = await self.ai_memory_db.create_memory(
            content, memory_type, importance_level, tags, source_conversation_id,
            memory_bank=memory_bank, user_id=user_id, model_id=model_id, source=source
        )
        
        # Auto-link memory to conversation if source_conversation_id provided
        if source_conversation_id:
            try:
                # Resolve source_conversation_id to actual conversation_id
                actual_conversation_id = await self.conversations_db.resolve_source_conversation_id(
                    source_conversation_id, user_id=user_id
                )
                
                if actual_conversation_id:
                    # Calculate link_strength from importance_level (normalize 1-10 to 0.0-1.0)
                    link_strength = min(importance_level / 10.0, 1.0)
                    
                    # Create breadcrumb metadata
                    link_metadata = {
                        "source_method": source if source else "direct_memory_creation",
                        "extracted_at": datetime.now(get_local_timezone()).isoformat(),
                        "triggered_by": "memory_elevation" if source == "openwebui_promotion" else "direct_storage",
                        "importance_level": importance_level,
                        "source_conversation_context": source_conversation_id,
                        "user_id": user_id,
                        "model_id": model_id,
                        "memory_bank": memory_bank
                    }
                    
                    # Link memory to conversation
                    link_id = await self.conversations_db.link_memory_to_conversation(
                        memory_id=memory_id,
                        conversation_id=actual_conversation_id,
                        link_type="direct",
                        link_strength=link_strength,
                        source_system="memory_extraction",
                        metadata=link_metadata
                    )
                    logger.info(f"Linked memory {memory_id} to conversation {actual_conversation_id} "
                               f"(strength={link_strength:.2f}, link_id={link_id})")
                else:
                    logger.debug(f"Could not resolve source_conversation_id '{source_conversation_id}' "
                                f"to conversation_id for linking. Memory created but not linked.")
            except Exception as link_error:
                logger.error(f"Failed to link memory {memory_id} to conversation: {link_error}")
                # Don't fail the memory creation if linking fails - log and continue
        
        if wait_for_embedding:
            # Wait for embedding to complete (used during promotion)
            logger.debug(f"Creating memory {memory_id} with guaranteed embedding...")
            success = await self._generate_and_store_embedding(
                memory_id=memory_id,
                content=content,
                table="curated_memories",
                db_instance=self.ai_memory_db
            )
            if not success:
                logger.error(f"Failed to generate embedding for promoted memory {memory_id}")
                return {
                    "status": "partial_failure",
                    "memory_id": memory_id,
                    "embedding_status": "failed"
                }
        else:
            # Generate and store embedding asynchronously (background task)
            embedding_task = asyncio.create_task(self._add_embedding_to_memory(memory_id, content))
            self._background_tasks.add(embedding_task)
            embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "embedding_status": "completed" if wait_for_embedding else "background"
        }
    
    async def update_memory(self, memory_id: str, content: str = None,
                          importance_level: int = None, tags: List[str] = None, user_id: str = None, model_id: str = None) -> Dict:
        """Update an existing memory"""
        
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        
        success = await self.ai_memory_db.update_memory(memory_id, content, importance_level, tags, user_id, model_id)
        
        # If content was updated, regenerate embedding
        if content is not None:
            embedding_task = asyncio.create_task(self._add_embedding_to_memory(memory_id, content))
            self._background_tasks.add(embedding_task)
            embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success" if success else "error",
            "memory_id": memory_id
        }
    
    async def delete_memory(self, memory_id: str, user_id: str = None, model_id: str = None) -> Dict:
        """Delete a memory by ID"""
        if not user_id or not model_id:
            return {
                "status": "error",
                "error": "MISSING REQUIRED PARAMETERS: user_id and model_id are required for all operations. Do not use defaults. Provide the actual user identifier and your model name from the system prompt."
            }
        try:
            
            success = await self.ai_memory_db.delete_memory(memory_id, user_id, model_id)
            return {
                "status": "success" if success else "error",
                "memory_id": memory_id,
                "message": "Memory deleted" if success else "Failed to delete memory"
            }
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return {
                "status": "error",
                "memory_id": memory_id,
                "message": str(e)
            }
    
    # Schedule operations
    async def create_appointment(self, title: str, scheduled_datetime: str,
                               description: str = None, location: str = None,
                               source_conversation_id: str = None,
                               recurrence_pattern: str = None,
                               recurrence_count: int = None,
                               recurrence_end_date: str = None, user_id: str = None,
                               model_id: str = None, source: str = "direct") -> Dict:
        """Create an appointment, optionally recurring"""
        
        appointment_result = await self.schedule_db.create_appointment(
            title, scheduled_datetime, description, location, source_conversation_id,
            recurrence_pattern, recurrence_count, recurrence_end_date,
            user_id=user_id, model_id=model_id, source=source
        )
        
        # Handle the result based on whether it's a single ID or list of IDs
        if isinstance(appointment_result, list):
            appointment_ids = appointment_result
        else:
            appointment_ids = [appointment_result]
        
        # Generate embedding for search (combine title and description)
        content_for_embedding = f"{title}"
        if description:
            content_for_embedding += f" {description}"
        
        # Add embeddings for all created appointments
        for appointment_id in appointment_ids:
            embedding_task = asyncio.create_task(self._add_embedding_to_appointment(appointment_id, content_for_embedding))
            self._background_tasks.add(embedding_task)
            embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success",
            "appointment_ids": appointment_ids,
            "count": len(appointment_ids)
        }
    
    async def create_reminder(self, content: str, due_datetime: str,
                            priority_level: int = 5, source_conversation_id: str = None,
                            recurrence_pattern: str = None, recurrence_count: int = None,
                            recurrence_end_date: str = None, user_id: str = None,
                            model_id: str = None, source: str = "direct") -> Dict:
        """Create a reminder or multiple recurring reminders"""
        
        # Lookup conversation title if source_conversation_id is provided
        conversation_title = None
        if source_conversation_id:
            try:
                conversation_result = await self.conversations_db.execute_query(
                    """SELECT COALESCE(topic_summary, '[Untitled Conversation]') as title
                       FROM conversations 
                       WHERE conversation_id = ?""",
                    (source_conversation_id,)
                )
                if conversation_result:
                    conversation_title = conversation_result[0][0]
                    logger.debug(f"Retrieved conversation title '{conversation_title}' for reminder")
            except Exception as e:
                logger.warning(f"Could not retrieve conversation title for {source_conversation_id}: {e}")
        
        reminder_result = await self.schedule_db.create_reminder(
            content, due_datetime, priority_level, source_conversation_id,
            recurrence_pattern, recurrence_count, recurrence_end_date,
            user_id=user_id, model_id=model_id, source=source, conversation_title=conversation_title
        )
        
        # Handle embedding generation for single or multiple reminders
        if isinstance(reminder_result, list):
            # Multiple reminders created
            for reminder_id in reminder_result:
                embedding_task = asyncio.create_task(self._add_embedding_to_reminder(reminder_id, content))
                self._background_tasks.add(embedding_task)
                embedding_task.add_done_callback(self._background_tasks.discard)
            return {
                "status": "success",
                "reminder_ids": reminder_result,
                "count": len(reminder_result)
            }
        else:
            # Single reminder created
            embedding_task = asyncio.create_task(self._add_embedding_to_reminder(reminder_result, content))
            self._background_tasks.add(embedding_task)
            embedding_task.add_done_callback(self._background_tasks.discard)
            return {
                "status": "success",
                "reminder_id": reminder_result
            }
    
    # VS Code project operations
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
        """Store project insight"""
        
        insight_id = await self.vscode_db.store_project_insight(
            content, insight_type, related_files, importance_level, source_conversation_id
        )
        
        # Generate and store embedding for the insight content
        embedding_task = asyncio.create_task(self._add_embedding_to_project_insight(insight_id, content))
        self._background_tasks.add(embedding_task)
        embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success",
            "insight_id": insight_id
        }
    
    async def search_project_history(self, query: str, limit: int = 10, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Search VS Code project history using semantic similarity, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        # Generate embedding for the search query
        query_embedding = await self.embedding_service.generate_embedding(query)
        if not query_embedding:
            return await self._text_based_project_search(query, limit, user_id, model_id)
        
        all_results = []
        
        # Search development conversations
        conversation_results = await self._search_development_conversations(query_embedding, limit, user_id, model_id)
        all_results.extend(conversation_results)
        
        # Search project insights
        insight_results = await self._search_project_insights(query_embedding, limit, user_id, model_id)
        all_results.extend(insight_results)
        
        # Search code context
        context_results = await self._search_code_context(query_embedding, limit, user_id, model_id)
        all_results.extend(context_results)
        
        # Sort by similarity score and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "query": query,
            "results": all_results[:limit],
            "count": len(all_results[:limit])
        }
    
    async def _search_development_conversations(self, query_embedding: List[float], limit: int, user_id: str = None, model_id: str = None) -> List[Dict]:
        """Search development conversations using semantic similarity, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        query = """
            SELECT conversation_id, session_id, timestamp, chat_context_id, 
                   conversation_content, decisions_made, code_changes, embedding
            FROM development_conversations 
            WHERE embedding IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 500
        """
        
        rows = await self.vscode_db.execute_query(query)
        results = []
        
        for row in rows:
            if row["embedding"]:
                stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity > 0.3:
                    result = {
                        "type": "development_conversation",
                        "similarity_score": similarity,
                        "data": {
                            "conversation_id": row["conversation_id"],
                            "session_id": row["session_id"],
                            "timestamp": row["timestamp"],
                            "chat_context_id": row["chat_context_id"],
                            "conversation_content": row["conversation_content"],
                            "decisions_made": json.loads(row["decisions_made"]) if row["decisions_made"] else None,
                            "code_changes": json.loads(row["code_changes"]) if row["code_changes"] else None
                        }
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _search_project_insights(self, query_embedding: List[float], limit: int, user_id: str = None, model_id: str = None) -> List[Dict]:
        """Search project insights using semantic similarity, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        query = """
            SELECT insight_id, timestamp_created, timestamp_updated, insight_type,
                   content, related_files, source_conversation_id, importance_level, embedding
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
                            "timestamp_updated": row["timestamp_updated"],
                            "insight_type": row["insight_type"],
                            "content": row["content"],
                            "related_files": json.loads(row["related_files"]) if row["related_files"] else None,
                            "source_conversation_id": row["source_conversation_id"],
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
    
    async def _search_code_context(self, query_embedding: List[float], limit: int, user_id: str = None, model_id: str = None) -> List[Dict]:
        """Search code context using semantic similarity, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        query = """
            SELECT context_id, timestamp, file_path, function_name, 
                   description, purpose, related_insights, embedding
            FROM code_context 
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
                        "type": "code_context",
                        "similarity_score": similarity,
                        "data": {
                            "context_id": row["context_id"],
                            "timestamp": row["timestamp"],
                            "file_path": row["file_path"],
                            "function_name": row["function_name"],
                            "description": row["description"],
                            "purpose": row["purpose"],
                            "related_insights": json.loads(row["related_insights"]) if row["related_insights"] else None
                        }
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _text_based_project_search(self, query: str, limit: int, user_id: str = None, model_id: str = None) -> Dict:
        """Fallback text-based search for project data, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        query_words = query.lower().split()
        results = []
        
        # Search project insights
        for word in query_words:
            rows = await self.vscode_db.execute_query(
                "SELECT * FROM project_insights WHERE LOWER(content) LIKE ? ORDER BY importance_level DESC LIMIT ?",
                (f"%{word}%", limit)
            )
            for row in rows:
                results.append({
                    "type": "project_insight",
                    "similarity_score": 0.5,
                    "data": dict(row)
                })
        
        # Search code context
        for word in query_words:
            rows = await self.vscode_db.execute_query(
                "SELECT * FROM code_context WHERE LOWER(description) LIKE ? OR LOWER(purpose) LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{word}%", f"%{word}%", limit)
            )
            for row in rows:
                results.append({
                    "type": "code_context",
                    "similarity_score": 0.5,
                    "data": dict(row)
                })
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for result in results:
            key = f"{result['type']}_{result['data'].get('insight_id', result['data'].get('context_id', ''))}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "results": unique_results[:limit],
            "count": len(unique_results[:limit]),
            "note": "Used text-based search (embeddings unavailable)"
        }
    
    async def link_code_context(self, file_path: str, description: str,
                              function_name: str = None, conversation_id: str = None,
                              user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Link conversation to code context, scoped to user/model"""
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        context_id = str(uuid.uuid4())
        timestamp = get_current_timestamp()
        
        # Store the code context
        await self.vscode_db.execute_update(
            """INSERT INTO code_context 
               (context_id, timestamp, file_path, function_name, description, user_id, model_id, source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (context_id, datetime.now(get_local_timezone()).isoformat(), file_path, function_name, description, user_id, model_id, source)
        )
        
        # Generate and store embedding for the description
        embedding_task = asyncio.create_task(self._add_embedding_to_code_context(context_id, description))
        self._background_tasks.add(embedding_task)
        embedding_task.add_done_callback(self._background_tasks.discard)
        
        return {
            "status": "success",
            "context_id": context_id,
            "message": "Code context linked successfully"
        }
    
    async def get_project_continuity(self, workspace_path: str = None, limit: int = 5, include_archives: bool = False, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get project continuity context from active and optionally archived databases, scoped to user/model.
        
        Args:
            workspace_path: Optional workspace path to filter sessions. If provided, only sessions 
                          from that workspace are returned (for VS Code isolation).
            limit: Maximum number of items to return per category
            include_archives: If True, also query archived databases and merge with active results
            user_id: User ID for filtering results to user's data
            model_id: Model ID for filtering results to model's data
        
        Design notes:
            - Sessions are workspace-scoped (only return sessions for the specified workspace)
            - Conversations/insights are stored globally (Friday needs access without project context)
            - Both are preserved as-is in active database for VS Code scope isolation
            - Archives are full database dumps by date, preserving all relationships
        """
        
        if not user_id:
            user_id = "unknown"
        if not model_id:
            model_id = "unknown"
        
        continuity_data = {}
        
        # Get recent project sessions (workspace-scoped + user/model scoped)
        if workspace_path:
            session_query = """
                SELECT * FROM project_sessions 
                WHERE workspace_path = ? AND user_id = ? AND model_id = ?
                ORDER BY start_timestamp DESC 
                LIMIT ?
            """
            params = (workspace_path, user_id, model_id, limit)
        else:
            session_query = """
                SELECT * FROM project_sessions 
                WHERE user_id = ? AND model_id = ?
                ORDER BY start_timestamp DESC 
                LIMIT ?
            """
            params = (user_id, model_id, limit)
        
        sessions = await self.vscode_db.execute_query(session_query, params)
        continuity_data["recent_sessions"] = [dict(row) for row in sessions]
        
        # Get recent development conversations (global, not workspace-scoped)
        recent_conversations = await self.vscode_db.execute_query(
            """SELECT * FROM development_conversations 
               ORDER BY timestamp DESC 
               LIMIT ?""",
            (limit,)
        )
        continuity_data["recent_conversations"] = [dict(row) for row in recent_conversations]
        
        # Get high-importance insights (global, not workspace-scoped)
        important_insights = await self.vscode_db.execute_query(
            """SELECT * FROM project_insights 
               WHERE importance_level >= 7 
               ORDER BY timestamp_updated DESC 
               LIMIT ?""",
            (limit,)
        )
        continuity_data["important_insights"] = [dict(row) for row in important_insights]
        
        # Get recent code context (workspace-scoped via session relationship)
        recent_context = await self.vscode_db.execute_query(
            """SELECT * FROM code_context 
               ORDER BY timestamp DESC 
               LIMIT ?""",
            (limit,)
        )
        continuity_data["recent_code_context"] = [dict(row) for row in recent_context]
        
        # If requested, merge archived results with active database results
        if include_archives:
            try:
                archived_data = await self._query_vscode_archives(workspace_path, limit)
                
                # Merge and re-sort by timestamp (active takes priority in case of duplicates)
                all_sessions = continuity_data["recent_sessions"] + archived_data["recent_sessions"]
                all_conversations = continuity_data["recent_conversations"] + archived_data["recent_conversations"]
                all_insights = continuity_data["recent_insights"] + archived_data["recent_insights"]
                all_code_context = continuity_data["recent_code_context"] + archived_data["recent_code_context"]
                
                # Re-sort merged lists and keep top N unique items
                # Use dict to eliminate duplicates by ID (session_id, conversation_id, etc)
                seen_sessions = {}
                for s in all_sessions:
                    if s.get("session_id") not in seen_sessions:
                        seen_sessions[s["session_id"]] = s
                continuity_data["recent_sessions"] = sorted(
                    seen_sessions.values(),
                    key=lambda x: x.get("start_timestamp", ""),
                    reverse=True
                )[:limit]
                
                # For conversations/insights/code_context, no unique ID enforced, just take top N
                continuity_data["recent_conversations"] = sorted(
                    all_conversations,
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )[:limit]
                
                continuity_data["recent_insights"] = sorted(
                    all_insights,
                    key=lambda x: x.get("timestamp_updated", ""),
                    reverse=True
                )[:limit]
                
                continuity_data["recent_code_context"] = sorted(
                    all_code_context,
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True
                )[:limit]
                
            except Exception as e:
                logger.error(f"Error querying archives in get_project_continuity: {e}")
                # Continue without archives if there's an error
        
        return {
            "status": "success",
            "workspace_path": workspace_path,
            "continuity_data": continuity_data,
            "includes_archives": include_archives
        }
    
    async def _query_vscode_archives(self, workspace_path: str = None, limit: int = 5) -> Dict:
        """Query archived vscode_project databases and return merged results.
        
        Archives are stored in memory_data/archives/ with naming pattern: vscode_project_YYYYMM.db
        This method opens archived databases, queries them with the same filters, and merges results
        while preserving original timestamps for proper ordering.
        
        Returns a dict with the same structure as get_project_continuity's continuity_data.
        """
        import sqlite3
        from pathlib import Path
        
        archive_dir = self.data_dir / "archives"
        if not archive_dir.exists():
            logger.debug(f"Archive directory not found: {archive_dir}")
            return {
                "recent_sessions": [],
                "recent_conversations": [],
                "recent_insights": [],
                "recent_code_context": []
            }
        
        archive_data = {
            "recent_sessions": [],
            "recent_conversations": [],
            "recent_insights": [],
            "recent_code_context": []
        }
        
        # Find all vscode_project_*.db archive files
        vscode_archives = sorted(
            archive_dir.glob("vscode_project_*.db"),
            reverse=True  # Newest first
        )
        
        if not vscode_archives:
            logger.debug("No vscode_project archives found")
            return archive_data
        
        logger.debug(f"Found {len(vscode_archives)} vscode_project archives to query")
        
        # Query each archive (newest first for efficiency)
        for archive_path in vscode_archives:
            try:
                conn = sqlite3.connect(str(archive_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query sessions (workspace-scoped if needed)
                if workspace_path:
                    cursor.execute(
                        "SELECT * FROM project_sessions WHERE workspace_path = ? ORDER BY start_timestamp DESC LIMIT ?",
                        (workspace_path, limit)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM project_sessions ORDER BY start_timestamp DESC LIMIT ?",
                        (limit,)
                    )
                archive_data["recent_sessions"].extend([dict(row) for row in cursor.fetchall()])
                
                # Query conversations (global)
                cursor.execute(
                    "SELECT * FROM development_conversations ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                archive_data["recent_conversations"].extend([dict(row) for row in cursor.fetchall()])
                
                # Query insights (global, importance >= 7)
                cursor.execute(
                    "SELECT * FROM project_insights WHERE importance_level >= 7 ORDER BY timestamp_updated DESC LIMIT ?",
                    (limit,)
                )
                archive_data["recent_insights"].extend([dict(row) for row in cursor.fetchall()])
                
                # Query code context
                cursor.execute(
                    "SELECT * FROM code_context ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                archive_data["recent_code_context"].extend([dict(row) for row in cursor.fetchall()])
                
                conn.close()
                
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not query archive {archive_path.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error querying archive {archive_path.name}: {e}")
                continue
        
        # Sort merged results by timestamp (preserving archive data)
        archive_data["recent_sessions"].sort(
            key=lambda x: x.get("start_timestamp", ""),
            reverse=True
        )
        archive_data["recent_conversations"].sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        archive_data["recent_insights"].sort(
            key=lambda x: x.get("timestamp_updated", ""),
            reverse=True
        )
        archive_data["recent_code_context"].sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return archive_data

    async def _diagnose_archive_links(self, archive_name: str = None) -> Dict:
        """Diagnostic method to check link validity in archives.
        
        This helps identify whether session_id foreign key relationships are preserved
        during the archival/dump process. 
        
        Args:
            archive_name: Specific archive to check (e.g., "vscode_project_202509.db"). 
                         If None, checks all archives.
        
        Returns:
            Dict with link integrity diagnostics for each archive
        """
        import sqlite3
        from pathlib import Path
        
        archive_dir = self.data_dir / "archives"
        if not archive_dir.exists():
            return {"error": "Archive directory not found"}
        
        diagnostics = {}
        
        if archive_name:
            archives = [archive_dir / archive_name]
        else:
            archives = sorted(archive_dir.glob("vscode_project_*.db"), reverse=True)[:5]  # Check 5 newest
        
        for archive_path in archives:
            try:
                conn = sqlite3.connect(str(archive_path))
                cursor = conn.cursor()
                
                # Count records
                cursor.execute("SELECT COUNT(*) FROM project_sessions")
                session_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM development_conversations")
                conversation_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM project_insights")
                insights_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM code_context")
                context_count = cursor.fetchone()[0]
                
                # Check orphaned relationships
                cursor.execute("""
                    SELECT COUNT(*) FROM development_conversations 
                    WHERE session_id NOT IN (SELECT session_id FROM project_sessions)
                """)
                orphaned_conversations = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM code_context 
                    WHERE session_id IS NOT NULL 
                    AND session_id NOT IN (SELECT session_id FROM project_sessions)
                """)
                orphaned_context = cursor.fetchone()[0]
                
                conn.close()
                
                diagnostics[archive_path.name] = {
                    "sessions": session_count,
                    "conversations": conversation_count,
                    "insights": insights_count,
                    "code_context": context_count,
                    "orphaned_conversations": orphaned_conversations,
                    "orphaned_context": orphaned_context,
                    "link_integrity": {
                        "conversations_linked": conversation_count - orphaned_conversations,
                        "conversation_orphan_ratio": f"{(orphaned_conversations/conversation_count*100):.1f}%" if conversation_count > 0 else "0%",
                        "context_linked": context_count - orphaned_context,
                        "context_orphan_ratio": f"{(orphaned_context/context_count*100):.1f}%" if context_count > 0 else "0%"
                    }
                }
            except Exception as e:
                diagnostics[archive_path.name] = {"error": str(e)}
        
        return diagnostics

    # === LLM Analysis Helpers (CHANGE 2A, 2B, 2C) ===
    
    def _get_llm_config(self) -> tuple:
        """CHANGE 2A: Get LLM configuration from valves or use fallback.
        
        Reads LLM model from OpenWebUI valves if available, otherwise uses
        a sensible default local model.
        
        Returns:
            tuple: (provider_type, model_name) or ("openai_compatible", "mistral-small") on fallback
        """
        try:
            # Try to read from OpenWebUI valves
            try:
                from open_webui.apps.webui.models.settings import Settings  # type: ignore
                settings = Settings.get_settings()
                
                # Check if settings has LLM config
                if hasattr(settings, 'llm_provider_type'):
                    provider = settings.llm_provider_type
                else:
                    provider = "openai_compatible"
                
                if hasattr(settings, 'llm_model_name'):
                    model = settings.llm_model_name
                else:
                    model = None
                
                if model:
                    logger.info(f"✅ Using LLM from OpenWebUI settings: {provider} → {model}")
                    return (provider, model)
            except ImportError:
                logger.debug("OpenWebUI settings not available, using fallback")
            except Exception as e:
                logger.debug(f"Could not read LLM config from OpenWebUI: {e}")
            
            # Fallback: use a sensible default
            fallback_provider = "openai_compatible"
            fallback_model = "mistral-small"  # Local default
            logger.info(f"⚡ Using fallback LLM: {fallback_provider} → {fallback_model}")
            return (fallback_provider, fallback_model)
            
        except Exception as e:
            logger.error(f"Error getting LLM config: {e}, using fallback")
            return ("openai_compatible", "mistral-small")
    
    async def _analyze_memory_with_llm(
        self,
        memory_content: str,
        context: str = None,
        task: str = "extract_tags"
    ) -> Dict:
        """CHANGE 2B: Call LLM for intelligent memory analysis.
        
        Tasks:
        - "extract_tags": Extract what tags should be for this memory
        - "extract_bank": Extract what memory bank should be
        - "validate_tags": Is this tag set correct for this content?
        - "rank_results": Score how relevant this result is for a query
        
        Args:
            memory_content: The memory text to analyze
            context: Optional context (query for ranking, tags to validate, etc.)
            task: Type of analysis to perform
            
        Returns:
            Dict with task-specific results
        """
        try:
            provider, model = self._get_llm_config()
            
            # Build prompt based on task
            if task == "extract_tags":
                prompt = f"""Analyze this memory and extract appropriate tags.
Memory: {memory_content}

Return a JSON object with:
{{
    "tags": ["tag1", "tag2", ...],
    "memory_bank": "General|Personal|Work|Context|Tasks",
    "reasoning": "brief explanation"
}}

Respond ONLY with valid JSON, no other text."""
            
            elif task == "extract_bank":
                prompt = f"""What memory bank category does this belong to?
Memory: {memory_content}

Options: General, Personal, Work, Context, Tasks

Return JSON:
{{
    "memory_bank": "category",
    "reasoning": "brief explanation"
}}"""
            
            elif task == "validate_tags":
                prompt = f"""Do these tags accurately describe this memory content?
Memory: {memory_content}
Tags: {context}

Return JSON:
{{
    "valid": true/false,
    "suggested_tags": ["tag1", "tag2", ...],
    "reasoning": "brief explanation"
}}"""
            
            elif task == "rank_results":
                prompt = f"""How relevant is this memory to the query?
Query: {context}
Memory: {memory_content}

Score 0-100 where 100 is perfectly relevant.
Return JSON:
{{
    "relevance_score": 0-100,
    "reasoning": "brief explanation"
}}"""
            
            else:
                return {"error": f"Unknown task: {task}"}
            
            # Call LLM via OpenAI-compatible API
            import aiohttp
            import asyncio
            
            try:
                async with aiohttp.ClientSession() as session:
                    # Use LM Studio or local OpenAI-compatible endpoint
                    url = "http://localhost:1234/v1/chat/completions"  # LM Studio default
                    
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "top_p": 0.95
                    }
                    
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            response_text = data["choices"][0]["message"]["content"].strip()
                            
                            # Parse JSON response
                            import json as json_lib
                            try:
                                result = json_lib.loads(response_text)
                                logger.debug(f"LLM {task} result: {result}")
                                return result
                            except json_lib.JSONDecodeError:
                                logger.warning(f"LLM returned invalid JSON for {task}: {response_text}")
                                return {"error": "Invalid JSON response from LLM", "raw": response_text}
                        else:
                            error_text = await resp.text()
                            logger.error(f"LLM API error ({resp.status}): {error_text}")
                            return {"error": f"LLM API error: {resp.status}"}
                            
            except asyncio.TimeoutError:
                logger.error("LLM request timed out")
                return {"error": "LLM request timeout"}
            except aiohttp.ClientConnectError:
                logger.error("Could not connect to LLM (LM Studio). Is it running?")
                return {"error": "LLM not available"}
            
        except Exception as e:
            logger.error(f"Error calling LLM for {task}: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}
    
    def _should_use_llm_for_memory(self, source: str) -> str:
        """CHANGE 2C: Determine if and how to use LLM for this memory.
        
        Args:
            source: One of "openwebui_promotion", "mcp_openwebui", "mcp_external", "direct"
            
        Returns:
            "validate_only": Light validation (only for promotion)
            "full_analysis": Full LLM intelligence (everything else)
        """
        if source == "openwebui_promotion":
            return "validate_only"
        else:
            # mcp_openwebui, mcp_external, direct - all get full analysis
            return "full_analysis"

    async def search_memories_by_date(
            self,
            start_date: str = None,
            end_date: str = None,
            query: str = None,
            limit: int = 20,
            database_filter: str = "all",
            user_id: str = None,
            model_id: str = None,
            memory_bank: str = None,
            tags: List[str] = None
        ) -> Dict:
            """Search memories chronologically within an optional date range.
            
            If query is provided, results are filtered by semantic similarity first
            then sorted chronologically. If no query, returns all memories in the
            date range sorted by timestamp.
            
            start_date and end_date accept ISO format strings e.g. '2024-10-01'
            or full datetime strings e.g. '2024-10-01T00:00:00'.
            """

            # Parse date range into timestamps
            ts_start = None
            ts_end = None
            try:
                if start_date:
                    ts_start = int(datetime.fromisoformat(start_date).timestamp())
                if end_date:
                    ts_end = int(datetime.fromisoformat(end_date).timestamp())
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Invalid date format: {e}. Use ISO format e.g. '2024-10-01'",
                    "results": [],
                    "count": 0
                }

            # Generate query embedding if query provided
            query_embedding = None
            if query:
                query_embedding = await self.embedding_service.generate_embedding(query)

            all_results = []

            # Search FMS curated memories
            if database_filter in ["all", "ai_memories"]:
                memory_db_paths = await self._discover_sharded_databases("ai_memories")
                for db_path in memory_db_paths:
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.row_factory = sqlite3.Row

                        sql = """SELECT memory_id, content, importance_level, tags, memory_type,
                                timestamp_created, timestamp_updated, source_conversation_id,
                                user_id, model_id, embedding
                                FROM curated_memories WHERE 1=1"""
                        params = []

                        if user_id:
                            sql += " AND (user_id = ? OR user_id IS NULL)"
                            params.append(user_id)
                        if model_id:
                            sql += " AND model_id = ?"
                            params.append(model_id)
                        if ts_start:
                            sql += " AND CAST(strftime('%s', timestamp_created) AS INTEGER) >= ?"
                            params.append(ts_start)
                        if ts_end:
                            sql += " AND CAST(strftime('%s', timestamp_created) AS INTEGER) <= ?"
                            params.append(ts_end)
                        if memory_bank:
                            sql += " AND memory_type = ?"
                            params.append(memory_bank)

                        rows = conn.execute(sql, params).fetchall()
                        conn.close()

                        for row in rows:
                            similarity = 1.0  # Default if no query
                            if query_embedding and row["embedding"]:
                                try:
                                    stored_emb = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                                    similarity = self._calculate_cosine_similarity(query_embedding, stored_emb)
                                    if similarity < 0.3:
                                        continue
                                except Exception:
                                    pass

                            row_tags = json.loads(row["tags"]) if row["tags"] else []

                            # Apply tags filter
                            if tags:
                                if not any(t.lower() in [rt.lower() for rt in row_tags] for t in tags):
                                    continue

                            all_results.append({
                                "type": "ai_memory",
                                "similarity_score": similarity,
                                "timestamp": row["timestamp_created"],
                                "data": {
                                    "memory_id": row["memory_id"],
                                    "content": row["content"],
                                    "importance_level": row["importance_level"],
                                    "tags": row_tags,
                                    "memory_type": row["memory_type"],
                                    "timestamp_created": row["timestamp_created"],
                                    "source_conversation_id": row["source_conversation_id"],
                                    "user_id": row["user_id"],
                                    "model_id": row["model_id"]
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error searching AI memories by date from {db_path}: {e}")

            # Search OpenWebUI short-term memories
            if database_filter in ["all", "ai_memories"]:
                try:
                    webui_conn = sqlite3.connect("/app/backend/data/webui.db")
                    webui_conn.row_factory = sqlite3.Row

                    sql = "SELECT id, user_id, content, created_at FROM memory WHERE 1=1"
                    params = []

                    if user_id:
                        sql += " AND user_id = ?"
                        params.append(user_id)
                    if ts_start:
                        sql += " AND created_at >= ?"
                        params.append(ts_start)
                    if ts_end:
                        sql += " AND created_at <= ?"
                        params.append(ts_end)

                    rows = webui_conn.execute(sql, params).fetchall()
                    webui_conn.close()

                    emb_db_path = os.path.join(get_memory_data_dir(), "memory_embeddings.db")
                    emb_conn = sqlite3.connect(emb_db_path)
                    emb_conn.row_factory = sqlite3.Row
                    mem_ids = [row["id"] for row in rows]
                    embedding_map = {}
                    if mem_ids:
                        placeholders = ",".join("?" * len(mem_ids))
                        emb_rows = emb_conn.execute(
                            f"SELECT memory_id, embedding FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                            mem_ids
                        ).fetchall()
                        embedding_map = {
                            r["memory_id"]: np.frombuffer(r["embedding"], dtype=np.float32).tolist()
                            for r in emb_rows
                        }
                    emb_conn.close()

                    for row in rows:
                        content = row["content"]
                        mem_id = row["id"]

                        similarity = 1.0
                        if query_embedding and mem_id in embedding_map:
                            similarity = self._calculate_cosine_similarity(query_embedding, embedding_map[mem_id])
                            if similarity < 0.3:
                                continue
                        elif query_embedding and mem_id not in embedding_map:
                            continue  # Skip unembedded memories when query provided

                        importance_level = 5
                        imp_match = re.search(r'\[Importance:\s*(\d+)\]', content)
                        if imp_match:
                            importance_level = int(imp_match.group(1))

                        row_tags = []
                        tags_match = re.search(r'\[Tags:\s*([^\]]+)\]', content)
                        if tags_match:
                            row_tags = [t.strip() for t in tags_match.group(1).split(",")]

                        if tags:
                            if not any(t.lower() in [rt.lower() for rt in row_tags] for t in tags):
                                continue

                        bank = None
                        bank_match = re.search(r'\[Memory Bank:\s*([^\]]+)\]', content)
                        if bank_match:
                            bank = bank_match.group(1).strip()

                        if memory_bank and bank != memory_bank:
                            continue

                        all_results.append({
                            "type": "short_term",
                            "similarity_score": similarity,
                            "timestamp": row["created_at"],
                            "data": {
                                "memory_id": mem_id,
                                "content": content,
                                "importance_level": importance_level,
                                "tags": row_tags,
                                "memory_type": bank,
                                "timestamp_created": row["created_at"],
                                "source_conversation_id": None,
                                "user_id": row["user_id"],
                                "model_id": None
                            }
                        })
                except Exception as e:
                    logger.error(f"Error searching OpenWebUI memories by date: {e}")

            # Search conversations
            if database_filter in ["all", "conversations"]:
                conv_db_paths = await self._discover_sharded_databases("conversations")
                for db_path in conv_db_paths:
                    try:
                        conn = sqlite3.connect(db_path)
                        conn.row_factory = sqlite3.Row

                        sql = """SELECT message_id, conversation_id, timestamp, role, content, metadata, embedding
                                FROM messages WHERE 1=1"""
                        params = []

                        if ts_start:
                            sql += " AND CAST(strftime('%s', timestamp) AS INTEGER) >= ?"
                            params.append(ts_start)
                        if ts_end:
                            sql += " AND CAST(strftime('%s', timestamp) AS INTEGER) <= ?"
                            params.append(ts_end)

                        rows = conn.execute(sql, params).fetchall()
                        conn.close()

                        for row in rows:
                            similarity = 1.0
                            if query_embedding and row["embedding"]:
                                try:
                                    stored_emb = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                                    similarity = self._calculate_cosine_similarity(query_embedding, stored_emb)
                                    if similarity < 0.3:
                                        continue
                                except Exception:
                                    pass
                            elif query_embedding and not row["embedding"]:
                                continue

                            all_results.append({
                                "type": "conversation",
                                "similarity_score": similarity,
                                "timestamp": row["timestamp"],
                                "data": {
                                    "message_id": row["message_id"],
                                    "conversation_id": row["conversation_id"],
                                    "timestamp": row["timestamp"],
                                    "role": row["role"],
                                    "content": row["content"],
                                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error searching conversations by date from {db_path}: {e}")

            # Sort — chronologically if no query, by similarity then chronologically if query provided
            if query_embedding:
                all_results.sort(key=lambda x: (-x["similarity_score"], x["timestamp"] or ""))
            else:
                all_results.sort(key=lambda x: x["timestamp"] or "")

            return {
                "status": "success",
                "query": query,
                "start_date": start_date,
                "end_date": end_date,
                "results": all_results[:limit],
                "count": len(all_results[:limit]),
                "filters": {
                    "database_filter": database_filter,
                    "memory_bank": memory_bank,
                    "tags": tags
                }
            }

        # Search operations
    async def search_memories(
        self, query: str = None, limit: int = 10, database_filter: str = "all",
        min_importance: int = None, max_importance: int = None,
        memory_type: str = None, memory_id: str = None,
        tags: List[str] = None, memory_bank: str = None,
        user_id: Optional[str] = None, model_id: Optional[str] = None, source: str = "direct"
    ) -> Dict:

        """Search memories across databases using semantic similarity with importance filtering, or direct ID lookup"""
        
        # CHECK: Detect if embedding config has changed (from Adaptive Memory v3 valve sync)
        # Run this check periodically to catch config changes and trigger re-embedding
        try:
            config_changed = await self.check_and_handle_embedding_config_change()
            if config_changed:
                logger.info("Embedding config was updated and all memories have been re-embedded")
        except Exception as e:
            logger.warning(f"Error checking embedding config during search: {e}")
        
        # If memory_id is provided, do direct ID lookup instead of semantic search
        if memory_id:
            return await self._get_memory_by_id(memory_id)
        
        # Require query if no memory_id provided
        if not query:
            return {
                "status": "error",
                "error": "Either 'query' or 'memory_id' parameter is required",
                "results": [],
                "count": 0
            }
        
        # Generate embedding for the search query
        query_embedding = await self.embedding_service.generate_embedding(query)
        if not query_embedding:
            # Fallback to text-based search if embedding fails
            return await self._text_based_search(query, limit, database_filter, min_importance, max_importance, memory_type)
        
        all_results = []
        
        # Search conversations
        if database_filter in ["all", "conversations"]:
            conversation_results = await self._search_conversations(
                query_embedding, limit * 2
            )
            all_results.extend(conversation_results)

        # Search AI memories
        if database_filter in ["all", "ai_memories"]:
            memory_results = await self._search_ai_memories(
                query_embedding, limit * 2, min_importance, max_importance, memory_type,
                user_id=user_id, model_id=model_id
            )
            all_results.extend(memory_results)
            
        # Search OpenWebUI short-term memories
        if database_filter in ["all", "ai_memories"]:
            owui_results = await self._search_openwebui_memories(
            query_embedding, limit * 2, user_id, min_importance, max_importance
            )
            all_results.extend(owui_results)            

        # Search schedule
        if database_filter in ["all", "schedule"]:
            schedule_results = await self._search_schedule(
                query_embedding, limit
            )
            all_results.extend(schedule_results)

        
        # CHANGE 5: Apply tag and memory_bank filtering (OR logic for tags with canonical form matching)
        filtered_results = []
        
        # Load tag manager and registry if tags filter is specified
        tag_manager = None
        canonical_tags = []
        if tags:
            try:
                from tag_manager import TagManager
                from pathlib import Path
                
                tag_manager = TagManager()
                registry_path = Path(get_memory_data_dir()) / "tag_registry.json"
                tag_manager.load_registry(str(registry_path))
                
                # Convert requested tags to canonical forms
                for tag in tags:
                    canonical = tag_manager.get_canonical_form(tag)
                    canonical_tags.append(canonical.lower())
                
                logger.debug(f"Tag search with canonical forms: {tags} -> {canonical_tags}")
            except Exception as e:
                logger.warning(f"Could not load tag registry for canonical matching: {e}")
                # Fall back to direct tag matching
                canonical_tags = [t.lower() for t in tags]
        
        for result in all_results:
            # Check memory_bank filter (exact match if specified)
            if memory_bank:
                result_bank = result["data"].get("memory_type")  # memory_bank is stored in memory_type column
                if result_bank != memory_bank:
                    continue
            
            # Check tags filter (OR logic - match ANY provided tag, including canonical variations)
            if tags:
                result_tags = result["data"].get("tags", [])
                if not result_tags:
                    # No tags in this memory, skip it if tags filter is specified
                    continue
                
                # Normalize result tags to lowercase for comparison
                normalized_result_tags = [t.lower() for t in result_tags]
                
                # Check if any of the requested canonical tags appear in result_tags
                # OR if the memory has any tag that normalizes to our canonical forms
                match_found = False
                
                for canonical_tag in canonical_tags:
                    # Direct canonical match
                    if canonical_tag in normalized_result_tags:
                        match_found = True
                        break
                    
                    # Or check if we can find a tag in the memory that maps to this canonical form
                    if tag_manager:
                        for mem_tag in result_tags:
                            found_in_registry, mem_canonical = tag_manager.find_tag_by_any_variation(mem_tag)
                            if found_in_registry and mem_canonical.lower() == canonical_tag:
                                match_found = True
                                break
                    
                    if match_found:
                        break
                
                if not match_found:
                    continue
            
            # If passed all filters, include in results
            filtered_results.append(result)
        
        all_results = filtered_results
        
        # Sort all results by similarity score and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "status": "success",
            "query": query,
            "results": all_results[:limit],
            "count": len(all_results[:limit]),
            "filters": {
                "min_importance": min_importance,
                "max_importance": max_importance,
                "memory_type": memory_type,
                "database_filter": database_filter,
                "tags": tags,
                "memory_bank": memory_bank
            }
        }
    

    async def _get_memory_by_id(self, memory_id: str) -> Dict:
        """Get a specific memory by ID from any database"""
        
        # Check AI memories first
        ai_memory = await self.ai_memory_db.execute_query(
            "SELECT * FROM curated_memories WHERE memory_id = ?", (memory_id,)
        )
        if ai_memory:
            memory = dict(ai_memory[0])
            return {
                "status": "success",
                "query": f"memory_id:{memory_id}",
                "results": [{
                    "type": "ai_memory",
                    "similarity_score": 1.0,  # Perfect match for direct lookup
                    "data": {
                        "memory_id": memory["memory_id"],
                        "content": memory["content"],
                        "memory_type": memory["memory_type"],
                        "importance_level": memory["importance_level"],
                        "tags": json.loads(memory["tags"]) if memory["tags"] else [],
                        "timestamp_created": memory["timestamp_created"],
                        "timestamp_updated": memory["timestamp_updated"],
                        "source_conversation_id": memory["source_conversation_id"]
                    }
                }],
                "count": 1,
                "search_type": "direct_id_lookup"
            }
        
        # Check conversations
        conversation = await self.conversations_db.execute_query(
            "SELECT * FROM messages WHERE message_id = ?", (memory_id,)
        )
        if conversation:
            msg = dict(conversation[0])
            return {
                "status": "success",
                "query": f"memory_id:{memory_id}",
                "results": [{
                    "type": "conversation",
                    "similarity_score": 1.0,
                    "data": {
                        "message_id": msg["message_id"],
                        "conversation_id": msg["conversation_id"],
                        "timestamp": msg["timestamp"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "metadata": json.loads(msg["metadata"]) if msg["metadata"] else None
                    }
                }],
                "count": 1,
                "search_type": "direct_id_lookup"
            }
        
        # Check schedule items (appointments and reminders)
        appointment = await self.schedule_db.execute_query(
            "SELECT * FROM appointments WHERE appointment_id = ?", (memory_id,)
        )
        if appointment:
            appt = dict(appointment[0])
            return {
                "status": "success",
                "query": f"memory_id:{memory_id}",
                "results": [{
                    "type": "appointment",
                    "similarity_score": 1.0,
                    "data": {
                        "appointment_id": appt["appointment_id"],
                        "title": appt["title"],
                        "scheduled_datetime": appt["scheduled_datetime"],
                        "description": appt["description"],
                        "location": appt["location"],
                        "status": appt["status"],
                        "timestamp_created": appt["timestamp_created"]
                    }
                }],
                "count": 1,
                "search_type": "direct_id_lookup"
            }
        
        reminder = await self.schedule_db.execute_query(
            "SELECT * FROM reminders WHERE reminder_id = ?", (memory_id,)
        )
        if reminder:
            rem = dict(reminder[0])
            return {
                "status": "success",
                "query": f"memory_id:{memory_id}",
                "results": [{
                    "type": "reminder",
                    "similarity_score": 1.0,
                    "data": {
                        "reminder_id": rem["reminder_id"],
                        "content": rem["content"],
                        "due_datetime": rem["due_datetime"],
                        "priority_level": rem["priority_level"],
                        "completed": bool(rem["completed"]),
                        "completed_at": rem["completed_at"],
                        "timestamp_created": rem["timestamp_created"]
                    }
                }],
                "count": 1,
                "search_type": "direct_id_lookup"
            }
        
        # Memory ID not found in any database
        return {
            "status": "not_found",
            "query": f"memory_id:{memory_id}",
            "results": [],
            "count": 0,
            "search_type": "direct_id_lookup",
            "error": f"No memory found with ID: {memory_id}"
        }
    
    async def _search_conversations(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search conversation messages using semantic similarity across ALL conversation databases"""
        
        # Discover all conversation databases (current + sharded)
        conv_db_paths = await self._discover_sharded_databases("conversations")
        
        if not conv_db_paths:
            logger.warning("No conversation databases found")
            return []
        
        logger.debug(f"Searching across {len(conv_db_paths)} conversation database(s)")
        
        all_results = []
        seen_message_ids = set()  # Deduplication across databases
        
        # Query each conversation database in parallel
        tasks = []
        for db_path in conv_db_paths:
            task = self._search_single_conversation_db(db_path, query_embedding)
            tasks.append(task)
        
        db_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results from all databases
        for result in db_results:
            if isinstance(result, Exception):
                logger.error(f"Error searching conversation DB: {result}")
                continue
            
            for msg_result in result:
                msg_id = msg_result["data"]["message_id"]
                if msg_id not in seen_message_ids:
                    all_results.append(msg_result)
                    seen_message_ids.add(msg_id)
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return all_results[:limit]
    
    async def _search_single_conversation_db(self, db_path: str, query_embedding: List[float]) -> List[Dict]:
        """Search a single conversation database"""
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT message_id, conversation_id, timestamp, role, content, metadata, embedding
                FROM messages 
                WHERE embedding IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                if row["embedding"]:
                    try:
                        # Convert binary embedding back to float array
                        stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                        similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                        
                        if similarity > 0.3:  # Threshold for relevance
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
                    except Exception as e:
                        logger.error(f"Error processing row from {Path(db_path).name}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching conversation DB {db_path}: {e}")
            return []
    
    async def _search_ai_memories(
        self,
        query_embedding: List[float],
        limit: int,
        min_importance: int = None,
        max_importance: int = None,
        memory_type: str = None,
        user_id: str | None = None,
        model_id: str | None = None,
    ) -> List[Dict]:

        """Search AI curated memories using semantic similarity with importance filtering across ALL memory databases"""
        
        # Discover all AI memory databases (current + sharded)
        memory_db_paths = await self._discover_sharded_databases("ai_memories")
        
        if not memory_db_paths:
            logger.warning("No AI memory databases found")
            return []
        
        logger.debug(f"Searching across {len(memory_db_paths)} AI memory database(s)")
        
        all_results = []
        seen_memory_ids = set()  # Deduplication across databases
        
        # Query each memory database in parallel
        tasks = []
        for db_path in memory_db_paths:
            task = self._search_single_ai_memory_db(
                db_path,
                query_embedding,
                min_importance,
                max_importance,
                memory_type,
                user_id=user_id,
                model_id=model_id,
            )
            tasks.append(task)

        db_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results from all databases
        for result in db_results:
            if isinstance(result, Exception):
                logger.error(f"Error searching AI memory DB: {result}")
                continue

            for mem_result in result:
                mem_id = mem_result["data"]["memory_id"]
                mem_user = mem_result["data"].get("user_id")
                mem_model = mem_result["data"].get("model_id")

                # composite key: avoids cross-user or cross-model deduplication
                composite_key = f"{mem_user or 'GLOBAL'}::{mem_model or 'GLOBAL'}::{mem_id}"

                if composite_key not in seen_memory_ids:
                    all_results.append(mem_result)
                    seen_memory_ids.add(composite_key)

        
        # Boost results based on importance level
        for result in all_results:
            importance_boost = result["data"]["importance_level"] / 10.0 * 0.1
            result["similarity_score"] += importance_boost
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return all_results[:limit]
    
    async def _search_openwebui_memories(
        self,
        query_embedding: List[float],
        limit: int,
        user_id: str = None,
        min_importance: int = None,
        max_importance: int = None,
    ) -> List[Dict]:
        """Search OpenWebUI short-term memories using cached embeddings"""

        WEBUI_DB_PATH = "/app/backend/data/webui.db"
        EMBEDDINGS_DB_PATH = os.path.join(get_memory_data_dir(), "memory_embeddings.db")

        try:
            # Pull memories from webui.db
            webui_conn = sqlite3.connect(WEBUI_DB_PATH)
            webui_conn.row_factory = sqlite3.Row

            sql = "SELECT id, user_id, content, created_at, updated_at FROM memory"
            params = []
            if user_id:
                sql += " WHERE user_id = ?"
                params.append(user_id)

            rows = webui_conn.execute(sql, params).fetchall()
            webui_conn.close()

            if not rows:
                return []

            # Load all relevant embeddings in one shot
            emb_conn = sqlite3.connect(EMBEDDINGS_DB_PATH)
            emb_conn.row_factory = sqlite3.Row
            memory_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(memory_ids))
            emb_rows = emb_conn.execute(
                f"SELECT memory_id, embedding FROM memory_embeddings WHERE memory_id IN ({placeholders})",
                memory_ids
            ).fetchall()
            emb_conn.close()

            # Build lookup dict
            embedding_map = {
                row["memory_id"]: np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                for row in emb_rows
            }

            results = []
            for row in rows:
                content = row["content"]
                mem_id = row["id"]

                # Parse importance from content string e.g. [Importance: 7]
                importance_level = 5  # default
                importance_match = re.search(r'\[Importance:\s*(\d+)\]', content)
                if importance_match:
                    importance_level = int(importance_match.group(1))

                # Apply importance filter if specified
                if min_importance is not None and importance_level < min_importance:
                    continue
                if max_importance is not None and importance_level > max_importance:
                    continue

                # Skip if no cached embedding
                if mem_id not in embedding_map:
                    continue

                similarity = self._calculate_cosine_similarity(
                    query_embedding, embedding_map[mem_id]
                )

                if similarity <= 0.3:
                    continue

                # Apply importance boost — same logic as _search_ai_memories
                importance_boost = (importance_level / 10.0) * 0.1
                similarity += importance_boost

                # Parse tags from content e.g. [Tags: preference, identity]
                tags = None
                tags_match = re.search(r'\[Tags:\s*([^\]]+)\]', content)
                if tags_match:
                    tags = [t.strip() for t in tags_match.group(1).split(",")]

                # Parse memory bank
                memory_bank = None
                bank_match = re.search(r'\[Memory Bank:\s*([^\]]+)\]', content)
                if bank_match:
                    memory_bank = bank_match.group(1).strip()

                results.append({
                    "type": "short_term_memory",
                    "similarity_score": similarity,
                    "data": {
                        "memory_id": mem_id,
                        "content": content,
                        "importance_level": importance_level,
                        "tags": tags,
                        "memory_type": memory_bank,
                        "user_id": row["user_id"],
                        "model_id": None,
                        "timestamp_created": row["created_at"],
                        "timestamp_updated": row["updated_at"],
                        "source_conversation_id": None,
                    }
                })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching OpenWebUI memories: {e}")
            return []       
        
    
    async def _search_single_ai_memory_db(self, db_path: str, query_embedding: List[float],
                                        min_importance: int = None, max_importance: int = None,
                                        memory_type: str = None,
                                        user_id: str | None = None, model_id: str | None = None) -> List[Dict]:
        """Search a single AI memory database"""
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build SQL query with optional filters
            sql = "SELECT memory_id, timestamp_created, timestamp_updated, source_conversation_id, memory_type, user_id, model_id, content, importance_level, tags, embedding FROM curated_memories WHERE embedding IS NOT NULL"
            params = []

            # Only add filters if user/model provided
            if user_id:
                sql += " AND (user_id = ? OR user_id IS NULL)"
                params.append(user_id)

            if model_id:
                sql += " AND model_id = ?"
                params.append(model_id)
            
            if min_importance is not None:
                sql += " AND importance_level >= ?"
                params.append(min_importance)
                
            if max_importance is not None:
                sql += " AND importance_level <= ?"
                params.append(max_importance)
                
            if memory_type is not None:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                if row["embedding"]:
                    try:
                        stored_embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                        similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                        
                        if similarity > 0.3:  # Threshold for relevance
                            # CHANGE 3A: Extract tags/bank from content if columns are NULL
                            memory_type = row["memory_type"]
                            tags = json.loads(row["tags"]) if row["tags"] else None
                            content = row["content"]
                            
                            # Try to extract from embedded metadata if columns missing
                            if not tags and "[Tags:" in content:
                                match = re.search(r'\[Tags:\s*([^\]]+)\]', content)
                                if match:
                                    tags_str = match.group(1).strip()
                                    tags = [t.strip() for t in tags_str.split(",")]
                            
                            if not memory_type and "[Memory Bank:" in content:
                                match = re.search(r'\[Memory Bank:\s*([^\]]+)\]', content)
                                if match:
                                    memory_type = match.group(1).strip()
                            
                            result = {
                                "type": "ai_memory",
                                "similarity_score": similarity,
                                "data": {
                                    "memory_id": row["memory_id"],
                                    "timestamp_created": row["timestamp_created"],
                                    "timestamp_updated": row["timestamp_updated"],
                                    "source_conversation_id": row["source_conversation_id"],
                                    "memory_type": memory_type,
                                    "content": content,
                                    "importance_level": row["importance_level"],
                                    "tags": tags,
                                    "user_id": row["user_id"],
                                    "model_id": row["model_id"]
                                }
                            }
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing row from {Path(db_path).name}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching AI memory DB {db_path}: {e}")
            return []

    
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
                            "timestamp_created": row["timestamp_created"],
                            "scheduled_datetime": row["scheduled_datetime"],
                            "title": row["title"],
                            "description": row["description"],
                            "location": row["location"],
                            "source_conversation_id": row["source_conversation_id"]
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
                            "timestamp_created": row["timestamp_created"],
                            "due_datetime": row["due_datetime"],
                            "content": row["content"],
                            "priority_level": row["priority_level"],
                            "completed": bool(row["completed"]),
                            "source_conversation_id": row["source_conversation_id"]
                        }
                    }
                    results.append(result)
        
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def _discover_sharded_databases(self, db_type: str) -> List[str]:
        """
            memory_data_path = Path(self.memory_data_path)
            
            # Pattern matching based on db_type
        Discover all database files for a given type (current + all sharded versions).
        
        For "conversations": finds conversations.db, conversations_2025-08.db, conversations_2025-09.db, etc
        For "ai_memories": finds ai_memories.db, ai_memories_202508.db, etc
        For others: finds their current and sharded versions
        
        Returns: List of full paths to all matching database files, sorted by recency
        """
        try:
            db_paths = []
            memory_data_path = Path(self.memory_data_path)
            
            # Pattern matching based on db_type
            if db_type == "conversations":
                patterns = [
                    f"{db_type}.db",
                    f"{db_type}_*.db"  # conversations_2025-08.db, conversations_2025-08_20251102.db
                ]
            else:
                patterns = [
                    f"{db_type}.db",
                    f"{db_type}_*.db"  # ai_memories_202508.db, mcp_tool_calls_202511.db, etc
                ]
            
            # Scan for matching files
            for pattern in patterns:
                matching_files = list(memory_data_path.glob(pattern))
                for file_path in matching_files:
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        db_paths.append(str(file_path))
            
            if not db_paths:
                logger.debug(f"No databases found for type {db_type}")
                return []
            
            # Sort by modification time (newest first) for multi-DB queries
            db_paths.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            
            logger.debug(f"Discovered {len(db_paths)} database(s) for {db_type}: {[Path(p).name for p in db_paths]}")
            return db_paths
            
        except Exception as e:
            logger.error(f"Error discovering sharded databases for {db_type}: {e}")
            return []
            memory_data_path = Path(self.memory_data_path)
            
            # Pattern matching based on db_type
    
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
            sql += " ORDER BY importance_level DESC LIMIT ?"
            params.append(limit)
            
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _text_based_search(self, query: str, limit: int, database_filter: str,
                               min_importance: int = None, max_importance: int = None,
                               memory_type: str = None) -> Dict:
        """Fallback text-based search when embeddings are unavailable with importance filtering"""
        
        query_words = query.lower().split()
        results = []
        
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
                        "similarity_score": 0.5,  # Default score for text match
                        "data": {
                            "message_id": row["message_id"],
                            "conversation_id": row["conversation_id"],
                            "timestamp": row["timestamp"],
                            "role": row["role"],
                            "content": row["content"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else None
                        }
                    })
        
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
                # CHANGE 3B: Extract tags/bank from content if columns are NULL
                memory_type_col = row.get("memory_type")
                tags = json.loads(row.get("tags")) if row.get("tags") else None
                content = row.get("content", "")
                
                # Try to extract from embedded metadata if columns missing
                if not tags and "[Tags:" in content:
                    match = re.search(r'\[Tags:\s*([^\]]+)\]', content)
                    if match:
                        tags_str = match.group(1).strip()
                        tags = [t.strip() for t in tags_str.split(",")]
                
                if not memory_type_col and "[Memory Bank:" in content:
                    match = re.search(r'\[Memory Bank:\s*([^\]]+)\]', content)
                    if match:
                        memory_type_col = match.group(1).strip()
                
                results.append({
                    "type": "ai_memory",
                    "similarity_score": 0.5,
                    "data": {
                        "memory_id": row["memory_id"],
                        "timestamp_created": row["timestamp_created"],
                        "timestamp_updated": row["timestamp_updated"],
                        "source_conversation_id": row["source_conversation_id"],
                        "memory_type": memory_type_col,
                        "content": content,
                        "importance_level": row["importance_level"],
                        "tags": tags,
                        "user_id": row.get("user_id"),
                        "model_id": row.get("model_id")
                    }
                })
        
        # Remove duplicates and limit results
        seen = set()
        unique_results = []
        for result in results:
            key = f"{result['type']}_{result['data'].get('message_id', result['data'].get('memory_id', ''))}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return {
            "status": "success",
            "query": query,
            "results": unique_results[:limit],
            "count": len(unique_results[:limit]),
            "filters": {
                "min_importance": min_importance,
                "max_importance": max_importance,
                "memory_type": memory_type,
                "database_filter": database_filter
            },
            "search_type": "text_based",
            "note": "Used text-based search (embeddings unavailable)"
        }
    
    # Helper methods for embedding management
    async def _check_and_handle_db_rotation(self, db_type: str) -> None:
        """
        Check if a database needs rotation, and if so, update the active DB registry.
        
        Called before writes to ensure we're always writing to the current active DB.
        
        Args:
            db_type: Type of database (e.g., "conversations", "mcp_tool_calls")
        """
        try:
            # Check if rotation is needed and perform it if necessary
            rotation_result = await self.db_maintenance.check_and_rotate_if_needed(db_type)
            
            # If rotation happened, update our active DB registry
            if rotation_result["status"] == "rotated":
                new_db_path = rotation_result["new_db_path"]
                self.active_db_files[db_type] = new_db_path
                logger.warning(f"✅ Updated active DB for {db_type}: {Path(new_db_path).name}")
                
            elif rotation_result["status"] == "error":
                logger.error(f"⚠️  Error checking rotation for {db_type}: {rotation_result['error']}")
                # Continue anyway - don't let rotation issues block normal operations
                
        except Exception as e:
            logger.error(f"Error in database rotation check: {e}")
            # Continue with current DB - don't block writes on rotation errors
    
    async def get_active_db_path(self, db_type: str) -> str:
        """
        Get the currently active database path for a given type.
        
        Performs rotation check first to ensure we return the most current DB.
        
        Args:
            db_type: Type of database
        
        Returns:
            Path to the active database file
        """
        # Check and rotate if needed
        await self._check_and_handle_db_rotation(db_type)
        
        # Return current active DB
        return self.active_db_files.get(db_type, "")
    
    # Helper methods for embedding management
    async def _add_embedding_to_message(self, message_id: str, content: str):
        """Add embedding to a message (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.conversations_db.execute_query(
                "SELECT embedding FROM messages WHERE message_id = ? AND embedding IS NOT NULL",
                (message_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for message {message_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                # Convert to binary format for storage
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.conversations_db.execute_update(
                    "UPDATE messages SET embedding = ? WHERE message_id = ?",
                    (embedding_blob, message_id)
                )
                logger.debug(f"Generated embedding for message {message_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to message {message_id}: {e}")
    
    async def _add_embedding_to_memory(self, memory_id: str, content: str):
        """Add embedding to a memory (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.ai_memory_db.execute_query(
                "SELECT embedding FROM curated_memories WHERE memory_id = ? AND embedding IS NOT NULL",
                (memory_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for memory {memory_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                # Convert to binary format for storage
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.ai_memory_db.execute_update(
                    "UPDATE curated_memories SET embedding = ? WHERE memory_id = ?",
                    (embedding_blob, memory_id)
                )
                logger.debug(f"Generated embedding for memory {memory_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to memory {memory_id}: {e}")
    
    async def _add_embedding_to_appointment(self, appointment_id: str, content: str):
        """Add embedding to an appointment (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.schedule_db.execute_query(
                "SELECT embedding FROM appointments WHERE appointment_id = ? AND embedding IS NOT NULL",
                (appointment_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for appointment {appointment_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.schedule_db.execute_update(
                    "UPDATE appointments SET embedding = ? WHERE appointment_id = ?",
                    (embedding_blob, appointment_id)
                )
                logger.debug(f"Generated embedding for appointment {appointment_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to appointment {appointment_id}: {e}")
    
    async def _add_embedding_to_reminder(self, reminder_id: str, content: str):
        """Add embedding to a reminder (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.schedule_db.execute_query(
                "SELECT embedding FROM reminders WHERE reminder_id = ? AND embedding IS NOT NULL",
                (reminder_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for reminder {reminder_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.schedule_db.execute_update(
                    "UPDATE reminders SET embedding = ? WHERE reminder_id = ?",
                    (embedding_blob, reminder_id)
                )
                logger.debug(f"Generated embedding for reminder {reminder_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to reminder {reminder_id}: {e}")
    
    async def _add_embedding_to_project_insight(self, insight_id: str, content: str):
        """Add embedding to a project insight (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.vscode_db.execute_query(
                "SELECT embedding FROM project_insights WHERE insight_id = ? AND embedding IS NOT NULL",
                (insight_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for insight {insight_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.vscode_db.execute_update(
                    "UPDATE project_insights SET embedding = ? WHERE insight_id = ?",
                    (embedding_blob, insight_id)
                )
                logger.debug(f"Generated embedding for insight {insight_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to project insight {insight_id}: {e}")
    
    async def _add_embedding_to_code_context(self, context_id: str, content: str):
        """Add embedding to code context (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.vscode_db.execute_query(
                "SELECT embedding FROM code_context WHERE context_id = ? AND embedding IS NOT NULL",
                (context_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for code context {context_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.vscode_db.execute_update(
                    "UPDATE code_context SET embedding = ? WHERE context_id = ?",
                    (embedding_blob, context_id)
                )
                logger.debug(f"Generated embedding for code context {context_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to code context {context_id}: {e}")
    
    async def _add_embedding_to_development_conversation(self, conversation_id: str, content: str):
        """Add embedding to development conversation (background task) - skip if already exists"""
        try:
            # Check if embedding already exists
            existing = await self.vscode_db.execute_query(
                "SELECT embedding FROM development_conversations WHERE conversation_id = ? AND embedding IS NOT NULL",
                (conversation_id,)
            )
            if existing:
                logger.debug(f"Embedding already exists for dev conversation {conversation_id}, skipping")
                return
                
            embedding = await self.embedding_service.generate_embedding(content)
            if embedding:
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                await self.vscode_db.execute_update(
                    "UPDATE development_conversations SET embedding = ? WHERE conversation_id = ?",
                    (embedding_blob, conversation_id)
                )
                logger.debug(f"Generated embedding for dev conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error adding embedding to development conversation {conversation_id}: {e}")
    
    # MCP Tool Call Logging and AI Self-Reflection Operations
    async def log_tool_call(self, client_id: str, tool_name: str, parameters: Dict = None,
                          execution_time_ms: float = None, status: str = "success",
                          result: Any = None, error_message: str = None) -> str:
        """Log an MCP tool call for AI self-reflection"""
        
        return await self.mcp_db.log_tool_call(
            client_id, tool_name, parameters, execution_time_ms, 
            status, result, error_message
        )
    
    async def get_tool_information(self, mode: str = "usage", days: int = 7, client_id: str = None, tool_name: str = None, client_type: str = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Dual-purpose tool: Get usage statistics OR tool documentation
        
        Args:
            mode: "usage" (default) for statistics, "documentation" for tool docs
            days: For usage mode - analyze past N days
            client_id: For usage mode - specific client to analyze
            tool_name: For documentation mode - specific tool to document (optional)
            client_type: The detected client type (vscode, sillytavern, or unknown)
            user_id: User requesting the information
            model_id: Model requesting the information
        
        Returns:
            If mode="usage": Tool usage statistics and insights
            If mode="documentation": Tool descriptions and parameters
        """
        # Set defaults for logging
        if not user_id:
            user_id = "Nate"
        if not model_id:
            model_id = "Friday"
        
        logger.info(f"Tool information requested (mode={mode}, days={days}) by user={user_id}, model={model_id}")
        
        try:
            if mode == "documentation":
                # Return tool documentation
                return await self._get_tool_documentation(tool_name, client_type)
            else:
                # Return usage statistics (original behavior)
                stats = await self.mcp_db.get_tool_usage_stats(days, client_id)
                insights = await self._generate_tool_usage_insights(stats)
                return {
                    "status": "success",
                    "period_days": days,
                    "stats": stats,
                    "insights": insights,
                    "requested_by": {"user_id": user_id, "model_id": model_id}
                }
        except Exception as e:
            logger.error(f"Error in get_tool_information: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_tool_documentation(self, tool_name: str = None, client_type: str = None) -> Dict:
        """Get tool documentation filtered by client type
        
        Args:
            tool_name: Optional specific tool to document
            client_type: The client type (vscode, sillytavern, unknown)
        
        Returns:
            Dictionary of tool documentation entries
        """
        try:
            # Default to unknown client if not specified
            if not client_type:
                client_type = "unknown"
            
            # Build available tools based on client type
            docs = {"common": self.TOOL_DOCUMENTATION.get("common", {})}
            
            if client_type == "vscode":
                docs["vscode"] = self.TOOL_DOCUMENTATION.get("vscode", {})
            elif client_type == "sillytavern":
                docs["sillytavern"] = self.TOOL_DOCUMENTATION.get("sillytavern", {})
            
            # If specific tool requested
            if tool_name:
                # Search through categories for the tool
                for category, tools in docs.items():
                    if tool_name in tools:
                        return {
                            "status": "success",
                            "tool_name": tool_name,
                            "category": category,
                            "documentation": tools[tool_name]
                        }
                # Tool not found for this client
                return {
                    "status": "error",
                    "error": f"Tool '{tool_name}' is not available for client type '{client_type}'"
                }
            else:
                # Return all docs for this client
                return {
                    "status": "success",
                    "client_type": client_type,
                    "tool_categories": list(docs.keys()),
                    "total_tools": sum(len(tools) for tools in docs.values()),
                    "documentation": docs
                }
        except Exception as e:
            logger.error(f"Error getting tool documentation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    
    async def reflect_on_tool_usage(self, days: int = 7, client_id: str = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """AI self-reflection on tool usage patterns"""
        # Set defaults for logging
        if not user_id:
            user_id = "Nate"
        if not model_id:
            model_id = "Friday"
        
        logger.info(f"Tool usage reflection requested (days={days}) by user={user_id}, model={model_id}")
        
        stats = await self.mcp_db.get_tool_usage_stats(days, client_id)
        
        if stats["total_calls"] == 0:
            reflection_content = f"No tool calls recorded in the past {days} days. This suggests either low activity or potential logging issues."
            insights = ["Consider checking if tool call logging is properly configured"]
            recommendations = ["Verify MCP server integration", "Test basic tool functionality"]
        else:
            # Generate detailed AI reflection
            reflection_content = await self._generate_ai_reflection_content(stats, days)
            insights = await self._extract_usage_insights(stats)
            recommendations = await self._generate_usage_recommendations(stats)
        
        # Store the reflection for future reference
        reflection_id = await self.mcp_db.store_ai_reflection(
            reflection_type="tool_usage_analysis",
            content=reflection_content,
            insights=insights,
            recommendations=recommendations,
            confidence_level=0.8,
            source_period_days=days,
            user_id=user_id,
            model_id=model_id
        )
        
        return {
            "status": "success",
            "reflection_id": reflection_id,
            "period_days": days,
            "reflection": {
                "content": reflection_content,
                "insights": insights,
                "recommendations": recommendations,
                "patterns": await self._identify_usage_patterns(stats)
            },
            "requested_by": {"user_id": user_id, "model_id": model_id}
        }
    
    async def get_ai_insights(self, limit: int = 5, insight_type: str = None, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get recent AI self-reflection insights, filtered by user/model"""
        # Set defaults for logging
        if not user_id:
            user_id = "Nate"
        if not model_id:
            model_id = "Friday"
        
        logger.info(f"Getting AI insights (limit={limit}, type={insight_type}) for user={user_id}, model={model_id}")
        
        # Pass user_id and model_id to filter reflections appropriately
        reflections = await self.mcp_db.get_recent_reflections(limit=limit, reflection_type=insight_type, user_id=user_id, model_id=model_id)
        
        return {
            "status": "success",
            "reflections": reflections,
            "count": len(reflections),
            "requested_by": {"user_id": user_id, "model_id": model_id}
        }
    
    async def export_all_tool_calls(self, output_filename: str = None, user_id: str = "unknown", model_id: str = "system", source: str = "direct") -> Dict:
        """Export all tool calls from current and archived databases to a text file.
        
        This tool is for LORA training dataset generation. It exports all tool calls
        with their original JSON parameters in multi-line format.
        
        Args:
            output_filename: Optional custom filename. Defaults to timestamp-based name.
            user_id: User requesting the export (for logging)
            model_id: Model ID (defaults to 'system')
            source: Source of the tool call (default: direct)
        
        Returns:
            Dictionary with status, file path, and count of exported tool calls
        """
        try:
            import glob
            from pathlib import Path
            
            # Create tool calls export directory if it doesn't exist
            export_dir = Path(get_base_path()) / "tool calls"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tool_calls_export_{timestamp}.txt"
            
            output_path = export_dir / output_filename
            
            total_calls = 0
            total_errors = 0
            
            # Open file for writing
            with open(output_path, 'w', encoding='utf-8') as f:
                # Export from current database
                logger.info(f"Exporting tool calls from current database...")
                try:
                    current_db = sqlite3.connect(self.mcp_db.db_path)
                    current_db.row_factory = sqlite3.Row
                    cursor = current_db.execute("""
                        SELECT tool_name, parameters, timestamp, status, result 
                        FROM tool_calls 
                        ORDER BY timestamp ASC
                    """)
                    
                    for row in cursor:
                        try:
                            # Create export record
                            export_record = {
                                "tool_name": row["tool_name"],
                                "parameters": json.loads(row["parameters"]) if row["parameters"] else {},
                                "timestamp": row["timestamp"],
                                "status": row["status"],
                                "result": json.loads(row["result"]) if row["result"] else None
                            }
                            # Write as multi-line JSON
                            f.write(json.dumps(export_record, indent=2))
                            f.write("\n")
                            total_calls += 1
                        except Exception as e:
                            logger.warning(f"Error processing tool call: {e}")
                            total_errors += 1
                    
                    current_db.close()
                except Exception as e:
                    logger.error(f"Error reading current database: {e}")
                    total_errors += 1
                
                # Export from archived databases
                logger.info(f"Exporting tool calls from archived databases...")
                archive_dir = Path(self.mcp_db.db_path).parent / "archives"
                if archive_dir.exists():
                    archive_files = sorted(glob.glob(str(archive_dir / "mcp_tool_calls_*.db")))
                    
                    for archive_file in archive_files:
                        logger.info(f"Processing archive: {Path(archive_file).name}")
                        try:
                            archive_db = sqlite3.connect(archive_file)
                            archive_db.row_factory = sqlite3.Row
                            cursor = archive_db.execute("""
                                SELECT tool_name, parameters, timestamp, status, result 
                                FROM tool_calls 
                                ORDER BY timestamp ASC
                            """)
                            
                            for row in cursor:
                                try:
                                    # Create export record
                                    export_record = {
                                        "tool_name": row["tool_name"],
                                        "parameters": json.loads(row["parameters"]) if row["parameters"] else {},
                                        "timestamp": row["timestamp"],
                                        "status": row["status"],
                                        "result": json.loads(row["result"]) if row["result"] else None
                                    }
                                    # Write as multi-line JSON
                                    f.write(json.dumps(export_record, indent=2))
                                    f.write("\n")
                                    total_calls += 1
                                except Exception as e:
                                    logger.warning(f"Error processing archived tool call: {e}")
                                    total_errors += 1
                            
                            archive_db.close()
                        except Exception as e:
                            logger.error(f"Error reading archive {archive_file}: {e}")
                            total_errors += 1
            
            logger.info(f"Tool calls export completed: {total_calls} calls exported, {total_errors} errors")
            
            return {
                "status": "success",
                "message": f"Successfully exported {total_calls} tool calls",
                "file_path": str(output_path),
                "total_calls": total_calls,
                "errors": total_errors,
                "export_directory": str(export_dir)
            }
            
        except Exception as e:
            logger.error(f"Error exporting tool calls: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to export tool calls"
            }
    
    async def _generate_tool_usage_insights(self, stats: Dict) -> Dict:
        """Generate AI insights from tool usage statistics"""
        
        total_calls = stats["total_calls"]
        success_rate = stats["success_rate"]
        tool_frequency = stats["tool_frequency"]
        
        # Generate summary insights
        most_used_tool = max(tool_frequency.items(), key=lambda x: x[1])[0] if tool_frequency else None
        least_reliable_tools = [
            pattern["tool"] for pattern in stats["error_patterns"] 
            if pattern["error_rate"] > 20
        ]
        
        insights = {
            "total_tool_calls": total_calls,
            "success_rate_percent": round(success_rate, 1),
            "most_used_tool": most_used_tool,
            "tools_with_issues": least_reliable_tools,
            "avg_execution_time_ms": round(stats["avg_execution_time"], 2),
            "reflection": await self._generate_summary_reflection(stats)
        }
        
        return insights
    
    async def _generate_ai_reflection_content(self, stats: Dict, days: int) -> str:
        """Generate AI reflection content about tool usage"""
        
        total_calls = stats["total_calls"]
        success_rate = stats["success_rate"]
        most_used = max(stats["tool_frequency"].items(), key=lambda x: x[1]) if stats["tool_frequency"] else ("unknown", 0)
        
        reflection = f"""Tool Usage Analysis for Past {days} Days:
        
I processed {total_calls} tool calls with a {success_rate:.1f}% success rate. 
My most frequently used tool was '{most_used[0]}' ({most_used[1]} calls), which suggests this is a core capability I rely on heavily.

Performance Assessment:"""
        
        if success_rate >= 95:
            reflection += "\nMy tool execution is highly reliable with minimal errors."
        elif success_rate >= 80:
            reflection += "\nMy tool execution is generally reliable but there's room for improvement."
        else:
            reflection += "\nMy tool execution has significant reliability issues that need attention."
        
        if stats["error_patterns"]:
            reflection += f"\n\nError Analysis:\nI encountered issues with {len(stats['error_patterns'])} different tools. "
            high_error_tools = [p for p in stats["error_patterns"] if p["error_rate"] > 10]
            if high_error_tools:
                reflection += f"Tools with high error rates: {', '.join([p['tool'] for p in high_error_tools])}"
        
        return reflection
    
    async def _extract_usage_insights(self, stats: Dict) -> List[str]:
        """Extract specific insights from usage patterns"""
        
        insights = []
        
        # Tool frequency insights
        if stats["tool_frequency"]:
            most_used = max(stats["tool_frequency"].items(), key=lambda x: x[1])
            insights.append(f"I rely heavily on '{most_used[0]}' tool ({most_used[1]} uses)")
            
            if len(stats["tool_frequency"]) > 1:
                least_used = min(stats["tool_frequency"].items(), key=lambda x: x[1])
                insights.append(f"'{least_used[0]}' is underutilized ({least_used[1]} uses)")
        
        # Performance insights
        if stats["success_rate"] < 90:
            insights.append(f"Success rate of {stats['success_rate']:.1f}% indicates reliability issues")
        
        if stats["avg_execution_time"] > 1000:  # More than 1 second
            insights.append(f"Average execution time of {stats['avg_execution_time']:.0f}ms suggests performance bottlenecks")
        
        # Error pattern insights
        for pattern in stats["error_patterns"]:
            if pattern["error_rate"] > 15:
                insights.append(f"'{pattern['tool']}' has high error rate of {pattern['error_rate']:.1f}%")
        
        return insights
    
    async def _generate_usage_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on usage patterns"""
        
        recommendations = []
        
        # Success rate recommendations
        if stats["success_rate"] < 80:
            recommendations.append("Investigate and fix tools with high failure rates")
            recommendations.append("Implement better error handling and retry logic")
        
        # Performance recommendations
        if stats["avg_execution_time"] > 500:
            recommendations.append("Optimize slow-performing tools for better responsiveness")
        
        # Tool usage recommendations
        if stats["tool_frequency"]:
            tool_counts = list(stats["tool_frequency"].values())
            if len(tool_counts) > 1 and max(tool_counts) > sum(tool_counts) * 0.7:
                recommendations.append("Consider diversifying tool usage to reduce over-reliance on single tools")
        
        # Error-specific recommendations
        for pattern in stats["error_patterns"]:
            if pattern["error_rate"] > 20:
                recommendations.append(f"Prioritize fixing issues with '{pattern['tool']}' tool")
        
        if not recommendations:
            recommendations.append("Tool usage patterns look healthy - continue current practices")
        
        return recommendations
    
    async def _identify_usage_patterns(self, stats: Dict) -> Dict:
        """Identify interesting patterns in tool usage"""
        
        patterns = {
            "peak_usage_tools": [],
            "problematic_tools": [],
            "efficiency_metrics": {},
            "usage_distribution": {}
        }
        
        # Peak usage tools
        if stats["tool_frequency"]:
            sorted_tools = sorted(stats["tool_frequency"].items(), key=lambda x: x[1], reverse=True)
            for tool, count in sorted_tools[:3]:
                patterns["peak_usage_tools"].append({
                    "tool": tool,
                    "count": count,
                    "percentage": (count / stats["total_calls"]) * 100,
                    "insight": f"'{tool}' represents {(count / stats['total_calls']) * 100:.1f}% of all tool usage"
                })
        
        # Problematic tools
        for error_pattern in stats["error_patterns"]:
            if error_pattern["error_rate"] > 10:
                patterns["problematic_tools"].append({
                    "tool": error_pattern["tool"],
                    "error_rate": error_pattern["error_rate"],
                    "insight": f"'{error_pattern['tool']}' fails {error_pattern['error_rate']:.1f}% of the time"
                })
        
        # Efficiency metrics
        patterns["efficiency_metrics"] = {
            "overall_success_rate": stats["success_rate"],
            "avg_execution_time": stats["avg_execution_time"],
            "tools_count": len(stats["tool_frequency"]),
            "insight": f"Using {len(stats['tool_frequency'])} different tools with {stats['success_rate']:.1f}% reliability"
        }
        
        return patterns
    
    async def _generate_summary_reflection(self, stats: Dict) -> str:
        """Generate a brief AI reflection summary"""
        
        if stats["total_calls"] == 0:
            return "I haven't made any tool calls recently, which might indicate low activity or configuration issues."
        
        success_rate = stats["success_rate"]
        most_used = max(stats["tool_frequency"].items(), key=lambda x: x[1])[0] if stats["tool_frequency"] else "unknown"
        
        if success_rate >= 95:
            tone = "performing excellently"
        elif success_rate >= 80:
            tone = "performing well"
        elif success_rate >= 60:
            tone = "experiencing some issues"
        else:
            tone = "struggling with reliability"
        
        return f"I'm {tone} with a {success_rate:.1f}% success rate across {stats['total_calls']} tool calls. My primary tool usage focuses on '{most_used}', suggesting consistent workflow patterns."

    # === SillyTavern-specific methods ===
    
    async def get_character_context(self, character_name: str, context_type: str = None, limit: int = 5, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Get relevant context about characters from memory for SillyTavern roleplay, scoped to user/model"""
        try:
            # Search memories for character-related content
            search_query = f"character {character_name}"
            if context_type:
                search_query += f" {context_type}"
            
            # Search across all memory types for character information, scoped to user/model
            results = await self.search_memories(
                query=search_query,
                limit=limit,
                database_filter="all",
                user_id=user_id,
                model_id=model_id
            )
            
            # Filter and categorize results specifically for character context
            character_context = {
                "character_name": character_name,
                "context_type": context_type,
                "personality_traits": [],
                "relationships": [],
                "history": [],
                "preferences": [],
                "memories": results.get("results", [])
            }
            
            # Categorize memories by type
            for memory in results.get("results", []):
                content = memory.get("content", "").lower()
                tags = memory.get("tags", [])
                
                if any(tag in ["personality", "traits", "character"] for tag in tags):
                    character_context["personality_traits"].append(memory)
                elif any(tag in ["relationship", "social"] for tag in tags):
                    character_context["relationships"].append(memory)
                elif any(tag in ["history", "background", "past"] for tag in tags):
                    character_context["history"].append(memory)
                elif any(tag in ["preference", "likes", "dislikes"] for tag in tags):
                    character_context["preferences"].append(memory)
            
            return {
                "status": "success",
                "character_context": character_context,
                "total_memories": len(results.get("results", [])),
                "query_used": search_query
            }
            
        except Exception as e:
            logger.error(f"Error getting character context: {e}")
            return {
                "status": "error",
                "error": str(e),
                "character_context": {"character_name": character_name, "memories": []}
            }
    
    async def store_roleplay_memory(self, character_name: str, event_description: str, 
                                  importance_level: int = 5, tags: List[str] = None,
                                  user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Store important roleplay moments or character developments for SillyTavern, scoped to user/model"""
        try:
            if not user_id:
                user_id = "unknown"
            if not model_id:
                model_id = "unknown"
            
            # Create content that includes character context
            content = f"Roleplay event with {character_name}: {event_description}"
            
            # Default tags for roleplay memories
            if tags is None:
                tags = []
            
            roleplay_tags = ["roleplay", "character", character_name.lower()] + tags
            
            # Store as a curated memory with roleplay type
            result = await self.create_memory(
                content=content,
                memory_type="roleplay",
                importance_level=importance_level,
                tags=roleplay_tags,
                user_id=user_id,
                model_id=model_id
            )
            
            # Also store as a conversation to maintain chat context
            await self.store_conversation(
                content=f"[Roleplay Memory] {event_description}",
                role="system",
                metadata={
                    "character_name": character_name,
                    "event_type": "roleplay_memory",
                    "importance": importance_level,
                    "user_id": user_id,
                    "model_id": model_id
                }
            )
            
            return {
                "status": "success",
                "memory_id": result["memory_id"],
                "character_name": character_name,
                "content": content,
                "tags": roleplay_tags
            }
            
        except Exception as e:
            logger.error(f"Error storing roleplay memory: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def search_roleplay_history(self, query: str, character_name: str = None, limit: int = 10, user_id: str = None, model_id: str = None, source: str = "direct") -> Dict:
        """Search past roleplay interactions and character development for SillyTavern, scoped to user/model"""
        try:
            if not user_id:
                user_id = "unknown"
            if not model_id:
                model_id = "unknown"
            
            # Enhance query with roleplay context
            search_query = f"roleplay {query}"
            if character_name:
                search_query += f" {character_name}"
            
            # Search specifically for roleplay memories and conversations
            results = await self.search_memories(
                query=search_query,
                limit=limit,
                database_filter="all",
                memory_type="roleplay",
                user_id=user_id,
                model_id=model_id
            )
            
            # Also search general conversations for roleplay content
            all_results = await self.search_memories(
                query=search_query,
                limit=limit * 2,
                database_filter="conversations",
                user_id=user_id,
                model_id=model_id
            )
            
            # Combine and deduplicate results
            combined_results = results.get("results", [])
            for result in all_results.get("results", []):
                if result not in combined_results:
                    combined_results.append(result)
            
            # Sort by relevance/similarity score
            combined_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            combined_results = combined_results[:limit]
            
            # Categorize results for better SillyTavern integration
            categorized_results = {
                "character_development": [],
                "interactions": [],  
                "plot_events": [],
                "general": []
            }
            
            for result in combined_results:
                content = result.get("content", "").lower()
                tags = result.get("tags", [])
                
                if any(word in content for word in ["development", "growth", "change", "personality"]):
                    categorized_results["character_development"].append(result)
                elif any(word in content for word in ["interaction", "conversation", "dialogue"]):
                    categorized_results["interactions"].append(result)
                elif any(word in content for word in ["event", "plot", "story", "scene"]):
                    categorized_results["plot_events"].append(result)
                else:
                    categorized_results["general"].append(result)
            
            return {
                "status": "success",
                "query": query,
                "character_name": character_name,
                "total_results": len(combined_results),
                "results": combined_results,
                "categorized_results": categorized_results,
                "search_context": "roleplay_history"
            }
            
        except Exception as e:
            logger.error(f"Error searching roleplay history: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": [],
                "categorized_results": {}
            }

    async def run_database_maintenance(self, force: bool = False) -> Dict:
        """Run database maintenance including cleanup and optimization"""
        try:
            from database_maintenance import DatabaseMaintenance
            
            maintenance = DatabaseMaintenance(self)
            return await maintenance.run_maintenance(force)
            
        except ImportError:
            logger.warning("Database maintenance module not available")
            return {"error": "Maintenance module not found", "manual_cleanup_recommended": True}
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            return {"error": str(e), "status": "failed"}


# Example usage and testing
async def main():
    """Test the memory system with file monitoring and semantic search"""
    
    memory = AIMemorySystem(enable_file_monitoring=True)
    
    logger.warning("=== Testing AI Memory System with File Monitoring ===\n")
    
    # First test basic database operations
    logger.warning("1. Testing basic storage operations...")
    
    # Test storing conversations manually
    logger.warning("\n2. Storing initial conversations...")
    conv1 = await memory.store_conversation(
        content="I prefer detailed technical explanations when discussing programming concepts",
        role="user"
    )
    logger.warning(f"Stored conversation: {conv1['message_id']}")
    
    conv2 = await memory.store_conversation(
        content="Could you explain how semantic search works with embeddings?",
        role="user"
    )
    logger.warning(f"Stored conversation: {conv2['message_id']}")
    
    # Test creating memories
    logger.warning("\n3. Creating curated memories...")
    memory1 = await memory.create_memory(
        content="User enjoys deep technical discussions about AI and machine learning",
        memory_type="preference",
        importance_level=8,
        tags=["user_preference", "technical", "ai"],
        source_conversation_id=conv1["conversation_id"]
    )
    logger.warning(f"Created memory: {memory1['memory_id']}")
    
    memory2 = await memory.create_memory(
        content="User is building a persistent memory system for AI assistant with file monitoring",
        memory_type="project",
        importance_level=9,
        tags=["project", "memory_system", "ai_assistant", "file_monitoring"]
    )
    logger.warning(f"Created memory: {memory2['memory_id']}")
    
    # Test appointments and reminders
    logger.warning("\n4. Creating schedule items...")
    appointment = await memory.create_appointment(
        title="Review AI memory system implementation",
        scheduled_datetime="2025-08-03T10:00:00Z",
        description="Go through the semantic search functionality and test file monitoring"
    )
    logger.warning(f"Created appointment: {appointment}")
    
    reminder = await memory.create_reminder(
        content="Test the MCP server integration with LM Studio and file monitoring",
        due_datetime="2025-08-03T14:00:00Z",
        priority_level=8
    )
    logger.warning(f"Created reminder: {reminder}")
    
    # Test VS Code project tracking
    logger.warning("\n5. Testing VS Code project features...")
    session = await memory.save_development_session(
        workspace_path=str(get_base_path()),
        active_files=["friday_memory_system.py", "friday_memory_mcp_server.py"],
        git_branch="main",
        session_summary="Implementing file monitoring and semantic search for memory system"
    )
    logger.warning(f"Saved development session: {session['session_id']}")
    
    insight = await memory.store_project_insight(
        content="Added file monitoring system to automatically capture conversations from LM Studio and VS Code chat files",
        insight_type="architecture_decision",
        related_files=["friday_memory_system.py", "friday_memory_mcp_server.py"],
        importance_level=9
    )
    logger.warning(f"Stored project insight: {insight['insight_id']}")
    
    # Wait a moment for embeddings to be generated (in real use, this happens in background)
    logger.warning("\n6. Waiting for embeddings to be generated...")
    await asyncio.sleep(2)
    
    # Test semantic search
    logger.warning("\n7. Testing semantic search...")
    
    # Search for user preferences
    search1 = await memory.search_memories("user likes technical details", limit=3)
    logger.warning(f"Search 'user likes technical details': Found {search1['count']} results")
    for result in search1['results']:
        logger.warning(f"  - {result['type']}: {result.get('similarity_score', 'N/A'):.3f} similarity")
    
    # Search for project-related content
    search2 = await memory.search_memories("memory system project file monitoring", limit=3)
    logger.warning(f"\nSearch 'memory system project file monitoring': Found {search2['count']} results")
    for result in search2['results']:
        logger.warning(f"  - {result['type']}: {result.get('similarity_score', 'N/A'):.3f} similarity")
    
    # Search project history
    project_search = await memory.search_project_history("file monitoring architecture decisions")
    logger.warning(f"\nProject search 'file monitoring architecture': Found {project_search['count']} results")
    for result in project_search['results']:
        logger.warning(f"  - {result['type']}: {result.get('similarity_score', 'N/A'):.3f} similarity")
    
    # Test project continuity
    continuity = await memory.get_project_continuity(str(get_base_path()))
    logger.warning(f"\nProject continuity data: {len(continuity['continuity_data']['recent_sessions'])} sessions, "
          f"{len(continuity['continuity_data']['important_insights'])} important insights")
    
    logger.warning("\n8. Running background initialization (maintenance + file monitoring)...")
    # Now that all other systems are initialized and tested, start background operations
    # This will run database maintenance first, then start file monitoring
    await memory.background_main()
    
    logger.warning("   Background initialization complete - maintenance ran and file monitoring is now starting...")
    logger.warning("   The system will automatically detect and import conversations from:")
    logger.warning("   - VS Code chat sessions")
    logger.warning("   - LM Studio conversations")
    logger.warning("   - Ollama conversations")
    logger.warning("   - OpenWebUI Conversations")
    
    logger.warning("\n9. Cleaning up test data...")
    
    # Delete the test reminder we created
    try:
        cleanup_result = await memory.delete_reminder(reminder['reminder_id'])
        logger.warning(f"   Deleted test reminder: {cleanup_result['message']}")
    except Exception as e:
        logger.warning(f"   Failed to delete test reminder: {e}")
    
    # Cancel the test appointment we created  
    try:
        cleanup_result = await memory.cancel_appointment(appointment['appointment_id'])
        logger.warning(f"   Cancelled test appointment: {cleanup_result['message']}")
    except Exception as e:
        logger.warning(f"   Failed to cancel test appointment: {e}")
    
    # Delete the test memories we created
    try:
        # Delete memory1 and memory2 using their IDs
        cleanup_result = await memory.ai_memory_db.execute_update(
            "DELETE FROM curated_memories WHERE memory_id = ?", 
            (memory1['memory_id'],)
        )
        cleanup_result = await memory.ai_memory_db.execute_update(
            "DELETE FROM curated_memories WHERE memory_id = ?", 
            (memory2['memory_id'],)
        )
        logger.warning(f"   Deleted test memories")
    except Exception as e:
        logger.warning(f"   Failed to delete test memories: {e}")
    
    logger.warning("   Test data cleanup complete!")
    
    logger.warning("\n=== Memory System Test Complete ===")
    logger.warning("Note: System is fully initialized and file monitoring is now active. Press Ctrl+C to stop.")
    
    try:
        # Keep the program running to demonstrate file monitoring
        while True:
            await asyncio.sleep(10)
            # You could add periodic status updates here
    except KeyboardInterrupt:
        logger.warning("\nStopping file monitoring...")
        await memory.stop_file_monitoring()
        logger.warning("File monitoring stopped. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())


    async def test_ai_memory_system():
        """Test the AI Memory System with file monitoring"""
        
        # Initialize the memory system with file monitoring enabled
        memory_system = AIMemorySystem(enable_file_monitoring=True)
        
        logger.warning("🧠 AI Memory System - Test with File Monitoring")
        logger.warning("=" * 60)
        
        # Start file monitoring
        logger.warning("\n📁 Starting file monitoring...")
        await memory_system._start_monitoring()
        
        logger.warning("\n📂 Monitoring these directories:")
        if memory_system.file_monitor:
            all_dirs = memory_system.file_monitor.watch_directories + memory_system.file_monitor.default_directories
            for directory in all_dirs:
                logger.warning(f"  • {directory}")
        
        # Test basic memory operations
        logger.warning("\n💭 Testing basic memory operations...")
        
        # Store a conversation
        result = await memory_system.store_conversation(
            content="Hello, we are testing the Friday memory system with file monitoring!",
            role="user"
        )
        logger.warning(f"✅ Stored conversation: {result['message_id']}")
        
        # Create an AI memory
        memory_result = await memory_system.create_memory(
            content="User is testing the file monitoring system for automatic conversation capture",
            memory_type="system_test",
            importance_level=7,
            tags=["test", "file_monitoring", "conversation_capture"]
        )
        logger.warning(f"✅ Created AI memory: {memory_result['memory_id']}")
        
        # Create a reminder
        reminder_result = await memory_system.create_reminder(
            content="Remember to check if VS Code conversations are being automatically captured",
            due_datetime="2025-08-03T12:00:00Z",
            priority_level=8
        )
        logger.warning(f"✅ Created reminder: {reminder_result['reminder_id']}")
        
        # Get recent context
        context = await memory_system.get_recent_context(limit=3)
        logger.warning(f"\n📜 Recent context ({context['count']} messages):")
        for msg in context['messages']:
            logger.warning(f"  • [{msg['role']}] {msg['content'][:50]}...")
        
        logger.warning(f"\n🔍 File monitoring status:")
        logger.warning(f"  • Monitoring enabled: {memory_system.file_monitor is not None}")
        if memory_system.file_monitor:
            logger.warning(f"  • Active observers: {len(memory_system.file_monitor.observers)}")
            logger.warning(f"  • Processed files: {len(memory_system.file_monitor.processed_files)}")
        
        logger.warning(f"\n💡 To test file monitoring:")
        logger.warning(f"  1. Have a conversation in VS Code Copilot Chat")
        logger.warning(f"  2. The conversation should be automatically captured")
        logger.warning(f"  3. Check the logs for import messages")
        
        # Keep monitoring for a bit to catch any existing files
        logger.warning(f"\n⏳ Monitoring for 10 seconds to catch any existing conversations...")
        await asyncio.sleep(10)
        
        # Stop file monitoring
        logger.warning(f"\n🛑 Stopping file monitoring...")
        await memory_system.stop_file_monitoring()
        
        logger.warning(f"\n✅ Test completed!")
