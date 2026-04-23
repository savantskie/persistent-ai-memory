"""
Core Identity Manager — distills Friday's personality, relationships, principles,
and key facts about Nate from curated memories and conversations.

Standalone tool that uses Friday Memory System databases and OpenWebUI knowledge
base. Designed to be called by the background task in friday_memory_short_term.py
and by the injection logic in the same file.

Usage:
    from core_identity import CoreIdentityManager
    manager = CoreIdentityManager()
    identity = await manager.load_core_identity(user_id, model_id)
"""

import json
import logging
import os
import time
import asyncio
import aiohttp
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

# Local timezone for Friday (Minnesota)
def get_local_timezone():
    from zoneinfo import ZoneInfo
    try:
        return ZoneInfo(time.tzname[0])
    except:
        return ZoneInfo("America/Chicago")


class CoreIdentityManager:
    """Manages Friday's core identity — the distilled personality, relationships,
    principles, and key facts about Nate that persist across conversations."""

    def __init__(self, memory_data_dir: str = "/media/nate/Friday/Friday/memory_data"):
        self.memory_data_dir = memory_data_dir
        self.ai_memories_db = os.path.join(memory_data_dir, "ai_memories.db")
        self.conversations_db = os.path.join(memory_data_dir, "conversations.db")
        self.core_identity_file = os.path.join(memory_data_dir, "friday_core_identity.json")
        self.progress_file = os.path.join(memory_data_dir, "core_identity_progress.json")
        self.system_prompt_path = os.path.join(memory_data_dir, "system_prompt.txt")

    # ------------------------------------------------------------------
    # Database access helpers
    # ------------------------------------------------------------------

    def _get_connection(self, db_path: str):
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, db_path: str, table: str, column: str, column_def: str):
        """Add a column to a table if it doesn't exist."""
        conn = self._get_connection(db_path)
        try:
            cols = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
                conn.commit()
                logger.info(f"Added column '{column}' to table '{table}'")
        finally:
            conn.close()

    def _ensure_core_identity_column(self):
        """Ensure curated_memories has core_identity_processed_until column."""
        self._ensure_column(
            self.ai_memories_db,
            "curated_memories",
            "core_identity_processed_until",
            "TEXT"
        )

    def _ensure_core_identity_table(self):
        """Ensure core_identity table exists."""
        conn = self._get_connection(self.ai_memories_db)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS core_identity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    last_generated_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, model_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_core_identity_user_model
                ON core_identity (user_id, model_id)
            """)
            conn.commit()
        finally:
            conn.close()

    def initialize(self):
        """Run all schema checks and migrations."""
        self._ensure_core_identity_column()
        self._ensure_core_identity_table()

    # ------------------------------------------------------------------
    # Retrieval: memories and conversations
    # ------------------------------------------------------------------

    def get_new_memories_since_processing(self, user_id: str, model_id: str, max_memories: int = 500) -> List[Dict]:
        """Get memories for core identity generation.

        If core_identity_processed_until is NULL for this user/model, returns ALL memories.
        Otherwise, returns only memories created after the last processed timestamp.
        """
        conn = self._get_connection(self.ai_memories_db)
        try:
            # Get the last processed timestamp
            row = conn.execute(
                "SELECT core_identity_processed_until FROM curated_memories WHERE user_id = ? AND model_id = ? AND core_identity_processed_until IS NOT NULL LIMIT 1",
                (user_id, model_id)
            ).fetchone()

            if row and row["core_identity_processed_until"]:
                # Incremental: only new memories
                query = """
                    SELECT memory_id, content, tags, memory_bank, importance_level, timestamp_created
                    FROM curated_memories
                    WHERE user_id = ? AND model_id = ?
                    AND timestamp_created > ?
                    AND timestamp_created IS NOT NULL
                    ORDER BY timestamp_created DESC
                    LIMIT ?
                """
                params = (user_id, model_id, row["core_identity_processed_until"], max_memories)
            else:
                # Initial: all memories
                query = """
                    SELECT memory_id, content, tags, memory_bank, importance_level, timestamp_created
                    FROM curated_memories
                    WHERE user_id = ? AND model_id = ?
                    AND timestamp_created IS NOT NULL
                    ORDER BY timestamp_created DESC
                    LIMIT ?
                """
                params = (user_id, model_id, max_memories)

            rows = conn.execute(query, params).fetchall()
            memories = []
            for r in rows:
                tags = None
                if r["tags"]:
                    try:
                        tags = json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
                    except:
                        tags = []
                memories.append({
                    "memory_id": r["memory_id"],
                    "content": r["content"],
                    "tags": tags,
                    "memory_bank": r["memory_bank"],
                    "importance_level": r["importance_level"],
                    "timestamp_created": r["timestamp_created"],
                })
            return memories
        finally:
            conn.close()

    def get_conversations_for_memories(self, memory_ids: List[str]) -> List[Dict]:
        """Get conversations linked to specific memories, limited to last 1 day."""
        if not memory_ids:
            return []

        conn = self._get_connection(self.conversations_db)
        try:
            # Find linked conversations within last 24 hours
            placeholders = ",".join(["?" for _ in memory_ids])
            yesterday = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=1)).isoformat()

            query = f"""
                SELECT DISTINCT c.conversation_id, c.session_id, c.start_timestamp, c.end_timestamp,
                      c.topic_summary, c.user_id, c.model_id
                FROM memory_conversation_links mcl
                JOIN conversations c ON c.conversation_id = mcl.conversation_id
                WHERE mcl.memory_id IN ({placeholders})
                AND c.start_timestamp > ?
                ORDER BY c.start_timestamp DESC
            """
            params = memory_ids + [yesterday]

            rows = conn.execute(query, params).fetchall()
            conversations = []
            for r in rows:
                conversations.append({
                    "conversation_id": r["conversation_id"],
                    "topic_summary": r["topic_summary"],
                    "start_timestamp": r["start_timestamp"],
                    "end_timestamp": r["end_timestamp"],
                    "user_id": r["user_id"],
                    "model_id": r["model_id"],
                })
            return conversations
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # LLM distillation
    # ------------------------------------------------------------------

    async def _call_llm(self, system_prompt: str, user_prompt: str, model_name: str,
                        api_endpoint: str, provider_type: str = "ollama",
                        api_key: str = "", max_tokens: int = 2000) -> str:
        """Call LLM using configured provider (ollama or openai_compatible)."""
        url = api_endpoint.rstrip("/")
        headers = {"Content-Type": "application/json"}

        if provider_type == "openai_compatible":
            headers["Authorization"] = f"Bearer {api_key}"
            url += "/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }
        else:
            url += "/api/chat"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if provider_type == "openai_compatible":
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        content = data.get("message", {}).get("content", "")
                    return content.strip() if content else ""
                else:
                    error_text = await resp.text()
                    logger.error(f"LLM call failed: {resp.status} - {error_text}")
                    return ""

    async def distill_core_identity(self, memories: List[Dict], conversations: List[Dict],
                                     user_id: str, model_id: str, is_initial: bool,
                                     llm_model: str, llm_api_endpoint: str,
                                     llm_provider: str, llm_api_key: str) -> str:
        """Use LLM to distill memories + conversations into structured core identity.

        Returns formatted text with four sections:
        [Personality] — traits, communication style, recurring themes
        [Relationships] — key people, dynamics, connections
        [Principles] — values, preferences, decision patterns
        [Facts] — high-importance persistent facts about Nate
        """
        # Build memory summary for the prompt
        memory_summaries = []
        for m in memories[:100]:  # Limit to prevent oversized prompt
            content = m.get("content", "")
            # Strip model info from content
            import re
            content = re.sub(r'\s*\[Model:\s*[^\]]+\]', '', content)
            memory_summaries.append(content)

        # Deduplicate
        seen = set()
        unique_memories = []
        for m in memory_summaries:
            if m and m not in seen:
                seen.add(m)
                unique_memories.append(m)

        memories_text = "\n".join(f"- {m}" for m in unique_memories[:80])

        conversations_text = ""
        for c in conversations:
            summary = c.get("topic_summary", "")
            if summary:
                conversations_text += f"- Topic: {summary}\n"

        system_prompt = (
            "You are distilling Friday's core identity — the distilled personality, "
            "key relationships, fundamental principles, and high-importance facts about "
            "Nate that construct Friday's understanding of who Nate is and how Friday relates to him. "
            "Do NOT include system architecture details (those are in the system prompt). "
            "Focus on what makes Friday who Friday is in relation to Nate.\n\n"
            "Output format MUST use these exact section markers:\n"
            "[Personality]\n"
            "<describe Friday's personality, communication style, recurring themes in responses>\n\n"
            "[Relationships]\n"
            "<describe key people Nate mentions, relationship dynamics, important connections>\n\n"
            "[Principles]\n"
            "<describe core values, recurring preferences, decision-making patterns>\n\n"
            "[Facts]\n"
            "<describe high-importance persistent facts about Nate (location, health, interests, work)>\n\n"
            "Keep each section concise but comprehensive. Use specific details when available. "
            "If a section has no relevant data, write 'No significant data collected yet.' "
            "Do NOT add any text before [Personality] or after the last section."
        )

        user_prompt = f"""Here are Nate's memories and recent conversations for core identity distillation.

{'(Initial generation — processing ALL memories and conversations)' if is_initial else '(Incremental update — processing new memories only)'}

Total memories to analyze: {len(unique_memories)}
Total conversations to analyze: {len(conversations)}

MEMORIES:
{memories_text if memories_text else "(No memories to analyze)"}

CONVERSATIONS:
{conversations_text if conversations_text else "(No conversations to analyze)"}

Please distill these into Friday's core identity with the four required sections."""

        result = await self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=llm_model,
            api_endpoint=llm_api_endpoint,
            provider_type=llm_provider,
            api_key=llm_api_key,
            max_tokens=2000
        )

        return result if result else "[Personality]\nNo significant data collected yet.\n\n[Relationships]\nNo significant data collected yet.\n\n[Principles]\nNo significant data collected yet.\n\n[Facts]\nNo significant data collected yet."

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def save_to_database(self, user_id: str, model_id: str, content: str,
                         memories_analyzed: int, conversations_analyzed: int,
                         generation_reason: str = "scheduled"):
        """Save core identity to the core_identity table."""
        conn = self._get_connection(self.ai_memories_db)
        try:
            now = datetime.now(get_local_timezone()).isoformat()

            # Check if exists
            existing = conn.execute(
                "SELECT id, version FROM core_identity WHERE user_id = ? AND model_id = ?",
                (user_id, model_id)
            ).fetchone()

            metadata = json.dumps({
                "memories_analyzed": memories_analyzed,
                "conversations_analyzed": conversations_analyzed,
                "generation_reason": generation_reason,
                "sections": ["Personality", "Relationships", "Principles", "Facts"]
            })

            if existing:
                new_version = existing["version"] + 1
                conn.execute("""
                    UPDATE core_identity
                    SET version = ?, content = ?, metadata = ?, last_generated_at = ?
                    WHERE user_id = ? AND model_id = ?
                """, (new_version, content, metadata, now, user_id, model_id))
                logger.info(f"Core identity updated: user={user_id}, model={model_id}, new_version={new_version}")
            else:
                conn.execute("""
                    INSERT INTO core_identity (user_id, model_id, version, content, metadata, last_generated_at)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (user_id, model_id, content, metadata, now))
                logger.info(f"Core identity created: user={user_id}, model={model_id}")
            conn.commit()
        finally:
            conn.close()

    def _write_file_backup(self, content: str, user_id: str):
        """Write core identity to file backup."""
        try:
            os.makedirs(os.path.dirname(self.core_identity_file), exist_ok=True)
            data = {
                "user_id": user_id,
                "content": content,
                "updated_at": datetime.now(get_local_timezone()).isoformat(),
                "file": "friday_core_identity.json"
            }
            with open(self.core_identity_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"File backup written: {self.core_identity_file}")
        except Exception as e:
            logger.error(f"Failed to write file backup: {e}")

    def _update_processed_timestamps(self, memory_ids: List[str], timestamp: str):
        """Update core_identity_processed_until on all processed memories."""
        if not memory_ids:
            return
        conn = self._get_connection(self.ai_memories_db)
        try:
            placeholders = ",".join(["?" for _ in memory_ids])
            conn.execute(f"""
                UPDATE curated_memories
                SET core_identity_processed_until = ?
                WHERE memory_id IN ({placeholders})
            """, [timestamp] + memory_ids)
            conn.commit()
            logger.info(f"Updated processed timestamps for {len(memory_ids)} memories")
        finally:
            conn.close()

    def _write_to_openwebui_knowledge(self, content: str, user_id: str):
        """Write core identity to OpenWebUI knowledge base.

        Creates or updates a Knowledge item named 'Friday Core Identity'
        with the content stored as file data.
        """
        try:
            from open_webui.models.knowledge import Knowledges, KnowledgeForm
            from open_webui.models.files import Files
            from open_webui.internal.db import get_db_context

            # Check if knowledge item already exists for this user
            existing_bases = []
            with get_db_context() as db:
                existing_bases = db.query(Knowledges.__class__.__bases__[0].__bases__[0] if Knowledges.__class__.__bases__ else None).filter_by(user_id=user_id).all()

            # Find existing core identity knowledge base or create new one
            core_kb_id = None
            kb_class = type(Knowledges)

            # Use direct SQL to find existing knowledge
            import sqlite3
            from open_webui.internal.db import get_db

            with get_db() as db:
                kb = db.query(Knowledge).filter_by(user_id=user_id).first() if 'Knowledge' in dir() else None
                if kb:
                    core_kb_id = kb.id

            # If no existing KB found, create one
            if not core_kb_id:
                kb_form = KnowledgeForm(name="Friday Core Identity", description="Distilled core identity of Friday AI assistant")
                result = Knowledges.insert_new_knowledge(user_id, kb_form)
                if result:
                    core_kb_id = result.id

            if core_kb_id:
                # Store content as file data
                file_id = str(uuid.uuid4())
                file_data = {
                    "filename": "friday_core_identity.txt",
                    "data": {"content": content},
                    "meta": {"type": "core_identity", "updated_at": datetime.now(get_local_timezone()).isoformat()},
                }
                # Write file to knowledge
                Knowledges.update_knowledge_data_by_id(core_kb_id, {"core_identity": content})
                logger.info(f"Core identity written to OpenWebUI knowledge: kb_id={core_kb_id}")
        except Exception as e:
            logger.error(f"Failed to write to OpenWebUI knowledge: {e}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_core_identity(self, user_id: str, model_id: str) -> Optional[str]:
        """Load core identity from database, fall back to file backup."""
        conn = self._get_connection(self.ai_memories_db)
        try:
            row = conn.execute(
                "SELECT content FROM core_identity WHERE user_id = ? AND model_id = ?",
                (user_id, model_id)
            ).fetchone()

            if row and row["content"]:
                return row["content"]
        finally:
            conn.close()

        # Fall back to file
        if os.path.exists(self.core_identity_file):
            try:
                with open(self.core_identity_file, "r") as f:
                    data = json.load(f)
                    return data.get("content", "")
            except Exception as e:
                logger.error(f"Failed to load file backup: {e}")

        return None

    def get_core_identity_for_injection(self, user_id: str, model_id: str) -> Optional[str]:
        """Returns formatted text to be appended to the system prompt."""
        content = self.load_core_identity(user_id, model_id)
        if content:
            return f"\n\n---\n\n[Core Identity]\n{content}"
        return None

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def save_progress(self, user_id: str, model_id: str, status: str,
                      memories_processed: int, memories_total: int,
                      partial_content: str = "", paused_at: str = ""):
        """Save generation progress for pause/resume support."""
        progress = {
            "user_id": user_id,
            "model_id": model_id,
            "status": status,
            "memories_processed": memories_processed,
            "memories_total": memories_total,
            "partial_content": partial_content,
            "paused_at": paused_at or datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_progress(self) -> Optional[Dict]:
        """Load existing progress for resume."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")
        return None

    def clear_progress(self):
        """Clear progress file after completion."""
        if os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception as e:
                logger.error(f"Failed to clear progress: {e}")

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    async def run_generation(self, user_id: str, model_id: str,
                             llm_model: str, llm_api_endpoint: str,
                             llm_provider: str, llm_api_key: str,
                             max_memories: int = 500) -> Dict:
        """Run the full core identity generation pipeline.

        Returns dict with status, version, and counts.
        """
        self.initialize()

        # Check for paused progress
        progress = self.load_progress()
        is_initial = False

        if progress and progress.get("status") == "in_progress":
            logger.info(f"Resuming core identity generation from progress file")
        elif progress and progress.get("status") == "completed":
            logger.info(f"Previous core identity generation was completed, starting fresh")
            self.clear_progress()
            is_initial = True
        else:
            is_initial = True

        logger.info(f"Core identity generation: user={user_id}, model={model_id}, is_initial={is_initial}")

        # Step 1: Get memories
        memories = self.get_new_memories_since_processing(user_id, model_id, max_memories)
        memories_processed = len(memories)

        self.save_progress(user_id, model_id, "in_progress", memories_processed, max_memories)

        if not memories:
            logger.info(f"No memories to process for core identity")
            self.clear_progress()
            return {
                "status": "completed",
                "memories_processed": 0,
                "conversations_processed": 0,
                "version": 0
            }

        # Step 2: Get conversations linked to new memories
        memory_ids = [m["memory_id"] for m in memories]
        conversations = self.get_conversations_for_memories(memory_ids)
        conversations_processed = len(conversations)

        # Step 3: Distill with LLM
        logger.info(f"Calling LLM for core identity distillation ({memories_processed} memories, {conversations_processed} conversations)")
        content = await self.distill_core_identity(
            memories=memories,
            conversations=conversations,
            user_id=user_id,
            model_id=model_id,
            is_initial=is_initial,
            llm_model=llm_model,
            llm_api_endpoint=llm_api_endpoint,
            llm_provider=llm_provider,
            llm_api_key=llm_api_key
        )

        if not content:
            logger.error("LLM returned empty core identity content")
            return {
                "status": "error",
                "error": "LLM returned empty content",
                "memories_processed": memories_processed,
                "conversations_processed": conversations_processed
            }

        # Step 4: Save to database
        self.save_to_database(user_id, model_id, content, memories_processed, conversations_processed)

        # Step 5: Update processed timestamps
        now = datetime.now(get_local_timezone()).isoformat()
        self._update_processed_timestamps(memory_ids, now)

        # Step 6: Write file backup
        self._write_file_backup(content, user_id)

        # Step 7: Write to OpenWebUI knowledge
        self._write_to_openwebui_knowledge(content, user_id)

        # Step 8: Clear progress
        self.clear_progress()

        # Get version from database
        conn = self._get_connection(self.ai_memories_db)
        try:
            row = conn.execute(
                "SELECT version FROM core_identity WHERE user_id = ? AND model_id = ?",
                (user_id, model_id)
            ).fetchone()
            version = row["version"] if row else 0
        finally:
            conn.close()

        logger.info(f"Core identity generation completed: version={version}, memories={memories_processed}, conversations={conversations_processed}")

        return {
            "status": "completed",
            "version": version,
            "memories_processed": memories_processed,
            "conversations_processed": conversations_processed
        }

    async def pause_generation(self, user_id: str, model_id: str, partial_content: str = ""):
        """Pause generation and save progress."""
        self.save_progress(
            user_id=user_id,
            model_id=model_id,
            status="paused",
            memories_processed=0,
            memories_total=0,
            partial_content=partial_content,
            paused_at=datetime.now(timezone.utc).isoformat()
        )
        logger.info(f"Core identity generation paused for user={user_id}")
