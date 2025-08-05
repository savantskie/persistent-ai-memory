#!/usr/bin/env python3
"""
Friday Memory System - Database Maintenance Module

Provides automated cleanup, optimization, and retention policies for the memory system.
"""

import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseMaintenance:
    """Handles automated database cleanup and optimization"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.retention_policies = {
            "conversations": {
                "max_age_days": 90,  # Keep conversations for 3 months
                "max_count": 10000,  # Keep max 10k conversations
                "preserve_important": True  # Keep high-importance items
            },
            "curated_memories": {
                "max_age_days": 365,  # Keep memories for 1 year
                "max_count": 5000,   # Keep max 5k memories
                "preserve_important": True
            },
            "schedule": {
                "max_age_days": 30,  # Keep old appointments/reminders for 1 month
                "cleanup_completed": True  # Remove completed items
            },
            "mcp_tool_calls": {
                "max_age_days": 30,  # Keep tool call logs for 1 month
                "max_count": 50000   # Keep max 50k tool calls
            }
        }
    
    async def run_maintenance(self, force: bool = False) -> Dict:
        """Run full database maintenance"""
        logger.info("🧹 Starting database maintenance...")
        
        results = {
            "maintenance_timestamp": datetime.now(timezone.utc).isoformat(),
            "cleanup_results": {},
            "optimization_results": {},
            "statistics": {},
            "schema_upgrades": []
        }
        
        try:
            # 0. Apply any needed schema upgrades
            logger.info("🔄 Checking and applying schema upgrades...")
            schema_upgrades = await self._upgrade_schemas()
            results["schema_upgrades"] = schema_upgrades
            
            # 1. Clean up old data based on retention policies
            logger.info("📅 Applying retention policies...")
            results["cleanup_results"] = await self._apply_retention_policies(force)
            
            # 2. Remove duplicate entries (shouldn't be many with our new system)
            logger.info("🔍 Removing any remaining duplicates...")
            results["cleanup_results"]["duplicates"] = await self._remove_duplicates()
            
            # 3. Optimize database performance
            logger.info("⚡ Optimizing database performance...")
            results["optimization_results"] = await self._optimize_databases()
            
            # 4. Collect post-cleanup statistics
            logger.info("📊 Collecting statistics...")
            results["statistics"] = await self._collect_statistics()
            
            logger.info("✅ Database maintenance completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Database maintenance failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _apply_retention_policies(self, force: bool = False) -> Dict:
        """Apply retention policies to remove old data"""
        cleanup_results = {}
        
        # Clean conversations
        cleanup_results["conversations"] = await self._cleanup_conversations()
        
        # Clean AI memories (more conservative)
        cleanup_results["curated_memories"] = await self._cleanup_ai_memories()
        
        # Clean old schedule items
        cleanup_results["schedule"] = await self._cleanup_schedule()
        
        # Clean old tool call logs
        cleanup_results["mcp_tool_calls"] = await self._cleanup_tool_calls()
        
        return cleanup_results
    
    async def _cleanup_conversations(self) -> Dict:
        """Clean up old conversation data"""
        policy = self.retention_policies["conversations"]
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy["max_age_days"])
        
        # Get conversation statistics before cleanup
        before_stats = await self._get_conversation_stats()
        
        # Delete old conversations (but preserve important ones)
        if policy.get("preserve_important"):
            # Keep conversations with high engagement or marked as important
            delete_query = """
                DELETE FROM conversations 
                WHERE start_timestamp < ? 
                AND conversation_id NOT IN (
                    SELECT DISTINCT conversation_id FROM messages 
                    WHERE json_extract(metadata, '$.importance_level') >= 7
                    OR json_extract(metadata, '$.preserve') = 'true'
                )
                AND conversation_id NOT IN (
                    SELECT conversation_id FROM conversations c
                    WHERE (
                        SELECT COUNT(*) FROM messages m 
                        WHERE m.conversation_id = c.conversation_id
                    ) >= 10  -- Keep conversations with 10+ messages
                )
            """
        else:
            delete_query = "DELETE FROM conversations WHERE start_timestamp < ?"
        
        # Execute cleanup
        deleted_conversations = await self.memory_system.conversations_db.execute_update(
            delete_query, (cutoff_date.isoformat(),)
        )
        
        # Clean up orphaned messages
        await self.memory_system.conversations_db.execute_update(
            "DELETE FROM messages WHERE conversation_id NOT IN (SELECT conversation_id FROM conversations)"
        )
        
        # Get statistics after cleanup
        after_stats = await self._get_conversation_stats()
        
        return {
            "policy_applied": policy,
            "cutoff_date": cutoff_date.isoformat(),
            "conversations_before": before_stats["conversation_count"],
            "conversations_after": after_stats["conversation_count"],
            "conversations_deleted": before_stats["conversation_count"] - after_stats["conversation_count"],
            "messages_before": before_stats["message_count"],
            "messages_after": after_stats["message_count"],
            "messages_deleted": before_stats["message_count"] - after_stats["message_count"]
        }
    
    async def _cleanup_ai_memories(self) -> Dict:
        """Clean up old AI memory data (more conservative)"""
        policy = self.retention_policies["curated_memories"]
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy["max_age_days"])
        
        before_count = len(await self.memory_system.ai_memory_db.execute_query(
            "SELECT memory_id FROM curated_memories", ()
        ))
        
        # Only delete low-importance, old memories
        deleted = await self.memory_system.ai_memory_db.execute_update(
            """DELETE FROM curated_memories 
               WHERE created_at < ? 
               AND importance_level < 5 
               AND memory_type NOT IN ('safety', 'critical', 'preference')""",
            (cutoff_date.isoformat(),)
        )
        
        after_count = len(await self.memory_system.ai_memory_db.execute_query(
            "SELECT memory_id FROM curated_memories", ()
        ))
        
        return {
            "policy_applied": policy,
            "cutoff_date": cutoff_date.isoformat(),
            "memories_before": before_count,
            "memories_after": after_count,
            "memories_deleted": before_count - after_count
        }
    
    async def _cleanup_schedule(self) -> Dict:
        """Clean up completed and old schedule items"""
        policy = self.retention_policies["schedule"]
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy["max_age_days"])
        
        # Clean old completed appointments
        old_appointments = await self.memory_system.schedule_db.execute_update(
            "DELETE FROM appointments WHERE scheduled_datetime < ?",
            (cutoff_date.isoformat(),)
        )
        
        # Clean old completed reminders
        old_reminders = await self.memory_system.schedule_db.execute_update(
            "DELETE FROM reminders WHERE due_datetime < ? AND completed = 1",
            (cutoff_date.isoformat(),)
        )
        
        return {
            "policy_applied": policy,
            "cutoff_date": cutoff_date.isoformat(),
            "old_appointments_deleted": old_appointments,
            "old_reminders_deleted": old_reminders
        }
    
    async def _cleanup_tool_calls(self) -> Dict:
        """Clean up old tool call logs"""
        policy = self.retention_policies["mcp_tool_calls"]
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy["max_age_days"])
        
        before_count = len(await self.memory_system.mcp_db.execute_query(
            "SELECT call_id FROM tool_calls", ()
        ))
        
        # Delete old tool calls
        deleted = await self.memory_system.mcp_db.execute_update(
            "DELETE FROM tool_calls WHERE timestamp < ?",
            (cutoff_date.isoformat(),)
        )
        
        after_count = len(await self.memory_system.mcp_db.execute_query(
            "SELECT call_id FROM tool_calls", ()
        ))
        
        return {
            "policy_applied": policy,
            "cutoff_date": cutoff_date.isoformat(),
            "tool_calls_before": before_count,
            "tool_calls_after": after_count,
            "tool_calls_deleted": before_count - after_count
        }
    
    async def _remove_duplicates(self) -> Dict:
        """Remove any remaining duplicate entries"""
        # This should be minimal with our new duplicate prevention
        results = {}
        
        # Remove duplicate messages (same content, role, timestamp within 1 minute)
        duplicate_messages = await self.memory_system.conversations_db.execute_update("""
            DELETE FROM messages 
            WHERE message_id NOT IN (
                SELECT MIN(message_id) 
                FROM messages 
                GROUP BY content, role, conversation_id, 
                         datetime(timestamp, 'start of minute')
            )
        """)
        
        results["duplicate_messages_removed"] = duplicate_messages
        return results
    
    async def _upgrade_messages_schema(self) -> List[str]:
        """Upgrade messages table schema if needed"""
        upgrades_applied = []
        
        try:
            # Use raw connection for schema modification
            conn = self.memory_system.conversations_db.get_connection()
            try:
                # First check if we need to modify the source_type column
                cursor = conn.execute("""SELECT sql FROM sqlite_master 
                                      WHERE type='table' AND name='messages'""")
                table_sql = cursor.fetchone()[0]
                
                if 'source_type TEXT NOT NULL' in table_sql:
                    # We need to modify the constraint
                    logger.info("Updating messages table source_type constraint")
                    
                    # Create new table with modified schema
                    conn.execute("""CREATE TABLE messages_new (
                        message_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_type TEXT DEFAULT 'unknown',
                        source_id TEXT,
                        source_url TEXT,
                        source_metadata TEXT,
                        sync_status TEXT,
                        last_sync TEXT,
                        metadata TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                    )""")
                    
                    # Copy data
                    conn.execute("""INSERT INTO messages_new 
                                  SELECT * FROM messages""")
                    
                    # Drop old table and rename new one
                    conn.execute("DROP TABLE messages")
                    conn.execute("ALTER TABLE messages_new RENAME TO messages")
                    conn.commit()
                    
                    upgrades_applied.append("updated_messages_source_type")
                    logger.info("Successfully updated messages table schema")
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Error upgrading messages schema: {e}")
            
        return upgrades_applied

    async def _optimize_databases(self) -> Dict:
        """Optimize database performance"""
        results = {}
        
        # List of database paths to optimize
        db_paths = [
            self.memory_system.conversations_db.db_path,
            self.memory_system.ai_memory_db.db_path,
            self.memory_system.schedule_db.db_path,
            self.memory_system.vscode_db.db_path,
            self.memory_system.mcp_db.db_path
        ]
        
        for db_path in db_paths:
            db_name = Path(db_path).stem
            try:
                # Get database size before optimization
                size_before = Path(db_path).stat().st_size if Path(db_path).exists() else 0
                
                # Optimize database
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")  # Reclaim space and defragment
                conn.execute("REINDEX")  # Rebuild indexes for better performance
                conn.execute("ANALYZE")  # Update query planner statistics
                conn.close()
                
                # Get size after optimization
                size_after = Path(db_path).stat().st_size if Path(db_path).exists() else 0
                
                results[db_name] = {
                    "size_before_mb": round(size_before / 1024 / 1024, 2),
                    "size_after_mb": round(size_after / 1024 / 1024, 2),
                    "space_saved_mb": round((size_before - size_after) / 1024 / 1024, 2),
                    "optimized": True
                }
                
            except Exception as e:
                results[db_name] = {"error": str(e), "optimized": False}
        
        return results
    
    async def _upgrade_schemas(self) -> List[str]:
        """Apply any needed schema upgrades"""
        upgrades_applied = []
        
        try:
            # 1. Upgrade development_conversations table
            logger.info("Checking development_conversations schema...")
            conn = self.memory_system.vscode_db.get_connection()
            try:
                # Check if column exists
                cursor = conn.execute("""SELECT COUNT(*) 
                                      FROM pragma_table_info('development_conversations') 
                                      WHERE name='source_metadata'""")
                has_column = cursor.fetchone()[0] > 0
                
                if not has_column:
                    logger.info("Adding source_metadata column to development_conversations table")
                    # Add the column
                    conn.execute("""ALTER TABLE development_conversations 
                                  ADD COLUMN source_metadata TEXT""")
                    conn.commit()
                    upgrades_applied.append("added_source_metadata_column")
                    logger.info("Successfully added source_metadata column")
            finally:
                conn.close()
                
            # 2. Upgrade messages table schema
            logger.info("Checking messages table schema...")
            messages_upgrades = await self._upgrade_messages_schema()
            upgrades_applied.extend(messages_upgrades)
                
        except Exception as e:
            logger.error(f"Error during schema upgrades: {e}")
            
        return upgrades_applied

    async def _get_conversation_stats(self) -> Dict:
        """Get current conversation statistics"""
        conversations = await self.memory_system.conversations_db.execute_query(
            "SELECT COUNT(*) as count FROM conversations", ()
        )
        messages = await self.memory_system.conversations_db.execute_query(
            "SELECT COUNT(*) as count FROM messages", ()
        )
        
        return {
            "conversation_count": conversations[0]["count"],
            "message_count": messages[0]["count"]
        }
    
    async def _collect_statistics(self) -> Dict:
        """Collect database statistics after maintenance"""
        stats = {}
        
        # Get system health for current statistics
        health = await self.memory_system.get_system_health()
        
        for db_name, db_info in health["databases"].items():
            if "count" in str(db_info):
                # Extract counts from the database info
                stats[db_name] = {
                    key: value for key, value in db_info.items() 
                    if "count" in key or "size" in key
                }
        
        return stats


# Integration function to add to PersistentAIMemorySystem
async def run_database_maintenance(memory_system, force: bool = False) -> Dict:
    """Convenience function to run database maintenance"""
    maintenance = DatabaseMaintenance(memory_system)
    return await maintenance.run_maintenance(force)
