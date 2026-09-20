"""
Core Identity Manager -- distills an AI companion's personality, relationships, principles,
and key facts about the user from curated memories and conversations.

Standalone tool that uses AI Memory System databases. Designed to be called by the
background task in ai_memory_short_term.py and by the injection logic in the same file.

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
import glob
import uuid
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from asyncio import to_thread

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

def get_local_timezone():
    from zoneinfo import ZoneInfo
    try:
        return ZoneInfo(time.tzname[0])
    except:
        return ZoneInfo("America/Chicago")

BATCH_SIZE = 50


class CoreIdentityManager:
    """Manages the AI companion's core identity."""

    def __init__(self, memory_data_dir: str = None, request=None):
        if memory_data_dir is None:
            memory_data_dir = os.getenv("AI_MEMORY_DATA_DIR", "./memory_data")
        self.memory_data_dir = memory_data_dir
        self.request = request
        self.ai_memories_db = os.path.join(memory_data_dir, "ai_memories.db")
        self.conversations_db = os.path.join(memory_data_dir, "conversations.db")
        self.archives_dir = os.path.join(memory_data_dir, "archives")
        self.tracking_file = os.path.join(memory_data_dir, "core_identity_tracking.json")
        self.openwebui_db = os.getenv("AI_MEMORY_WEBUI_DB", os.path.join(memory_data_dir, "webui.db"))
        self.core_identity_file = os.path.join(memory_data_dir, "core_identity.json")
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
        self._ensure_column(
            self.ai_memories_db, "curated_memories",
            "core_identity_processed_until", "TEXT"
        )

    def _ensure_core_identity_table(self):
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
        self._ensure_core_identity_column()
        self._ensure_core_identity_table()

    # ------------------------------------------------------------------
    # File format migration helpers
    # ------------------------------------------------------------------

    def _is_plain_uuid(self, s: str) -> bool:
        return s.count("-") == 4 and "_" not in s

    def _backup_file_for_user(self, base_name: str, user_id: str) -> str:
        return os.path.join(self.memory_data_dir, f"{base_name}_{user_id}.json")

    def _backup_file_for_user_and_model(self, base_name: str, user_id: str, model_id: str) -> str:
        safe_model = model_id.replace("/", "-").replace(":", "-").replace(" ", "-")
        return os.path.join(self.memory_data_dir, f"{base_name}_{user_id}_{safe_model}.json")

    def _progress_file_for_user(self, base_name: str, user_id: str) -> str:
        return os.path.join(self.memory_data_dir, f"{base_name}_{user_id}.json")

    def _progress_file_for_user_and_model(self, base_name: str, user_id: str, model_id: str) -> str:
        safe_model = model_id.replace("/", "-").replace(":", "-").replace(" ", "-")
        return os.path.join(self.memory_data_dir, f"{base_name}_{user_id}_{safe_model}.json")

    # ------------------------------------------------------------------
    # Async-safe wrappers for blocking I/O operations
    # ------------------------------------------------------------------

    async def _write_file_backup_async(self, content: str, user_id: str, model_id: str):
        await to_thread(self._write_file_backup, content, user_id, model_id)

    async def _save_progress_async(self, user_id: str, model_id: str, status: str,
                                    memories_processed: int, memories_total: int,
                                    partial_content: str = "", paused_at: str = "",
                                    batch_index: int = 0, batches_total: int = 0,
                                    all_memory_ids: Optional[List] = None):
        await to_thread(
            self.save_progress, user_id, model_id, status,
            memories_processed, memories_total, partial_content,
            paused_at, batch_index, batches_total, all_memory_ids
        )

    async def load_core_identity_async(self, user_id: str, model_id: str) -> Optional[str]:
        return await to_thread(self.load_core_identity, user_id, model_id)

    # ------------------------------------------------------------------
    # Memory retrieval enhancements
    # ------------------------------------------------------------------

    def _conversation_ids_for_memories(self, memory_ids: List[str]) -> Dict[str, str]:
        if not memory_ids:
            return {}
        conn = self._get_connection(self.ai_memories_db)
        try:
            placeholders = ",".join("?" for _ in memory_ids)
            rows = conn.execute(
                f"""SELECT memory_id, source_conversation_id
                    FROM curated_memories
                    WHERE memory_id IN ({placeholders})
                      AND source_conversation_id IS NOT NULL
                      AND TRIM(source_conversation_id) != ''
                      AND source_conversation_id NOT LIKE '%_pruned'""",
                memory_ids
            ).fetchall()
            return {row["memory_id"]: row["source_conversation_id"] for row in rows}
        finally:
            conn.close()

    def get_conversations_for_memories(self, memory_ids: List[str]) -> List[Dict]:
        conv_map = self._conversation_ids_for_memories(memory_ids)
        conv_ids = list(set(conv_map.values()))
        if not conv_ids:
            return []
        conn = self._get_connection(self.conversations_db)
        try:
            placeholders = ",".join("?" for _ in conv_ids)
            rows = conn.execute(
                f"""SELECT conversation_id, session_id, start_timestamp, end_timestamp,
                           topic_summary, user_id, model_id
                    FROM conversations
                    WHERE conversation_id IN ({placeholders})
                    ORDER BY start_timestamp DESC""",
                conv_ids
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _build_memory_conversation_map(self, memory_ids: List[str]) -> Dict[str, List[str]]:
        conv_map = self._conversation_ids_for_memories(memory_ids)
        conv_ids = list(set(conv_map.values()))
        if not conv_ids:
            return {mid: [] for mid in memory_ids}
        conn = self._get_connection(self.conversations_db)
        try:
            placeholders = ",".join("?" for _ in conv_ids)
            conv_topics = conn.execute(
                f"SELECT conversation_id, topic_summary FROM conversations WHERE conversation_id IN ({placeholders})",
                conv_ids
            ).fetchall()
            topic_by_conv = {r["conversation_id"]: r["topic_summary"] for r in conv_topics}
            result = {}
            for mid in memory_ids:
                cid = conv_map.get(mid)
                if cid and cid in topic_by_conv and topic_by_conv[cid]:
                    result[mid] = [topic_by_conv[cid]]
                else:
                    result[mid] = []
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Retrieval: memories and conversations
    # ------------------------------------------------------------------

    def get_new_memories_since_processing(self, user_id: str, model_id: str, max_memories: int = 500) -> List[Dict]:
        conn = self._get_connection(self.ai_memories_db)
        try:
            row = conn.execute(
                "SELECT core_identity_processed_until FROM curated_memeries WHERE user_id = ? AND LOWER(model_id) = ? AND core_identity_processed_until IS NOT NULL LIMIT 1",
                (user_id, model_id.lower())
            ).fetchone()
            if row and row["core_identity_processed_until"]:
                memories = conn.execute(
                    """SELECT memory_id, content, tags, memory_bank, importance_level, timestamp_created
                       FROM curated_memories
                       WHERE user_id = ? AND LOWER(model_id) = ?
                       AND timestamp_created > ?
                       AND timestamp_created IS NOT NULL
                       AND (core_identity_processed_until IS NULL OR core_identity_processed_until = '')
                       ORDER BY timestamp_created ASC LIMIT ?""",
                    (user_id, model_id.lower(), row["core_identity_processed_until"], max_memories)
                ).fetchall()
            else:
                memories = conn.execute(
                    """SELECT memory_id, content, tags, memory_bank, importance_level, timestamp_created
                       FROM curated_memories
                       WHERE user_id = ? AND LOWER(model_id) = ?
                       AND (core_identity_processed_until IS NULL OR core_identity_processed_until = '')
                       AND timestamp_created IS NOT NULL
                       ORDER BY timestamp_created ASC LIMIT ?""",
                    (user_id, model_id.lower(), max_memories)
                ).fetchall()
            result = []
            for r in memories:
                tags = None
                if r["tags"]:
                    try:
                        tags = json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
                    except:
                        tags = []
                result.append({
                    "memory_id": r["memory_id"], "content": r["content"],
                    "tags": tags, "memory_bank": r["memory_bank"],
                    "importance_level": r["importance_level"],
                    "timestamp_created": r["timestamp_created"],
                })
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Multi-source memory retrieval
    # ------------------------------------------------------------------

    def _resolve_owu_user_id(self, user_id: str) -> str:
        if user_id.count("-") == 4 and "_" not in user_id:
            return user_id
        if not os.path.exists(self.openwebui_db):
            return user_id
        try:
            conn = self._get_connection(self.openwebui_db)
            row = conn.execute(
                "SELECT id FROM user WHERE LOWER(name) = LOWER(?) OR LOWER(email) = LOWER(?)",
                (user_id, user_id)
            ).fetchone()
            conn.close()
            if row:
                return row["id"]
        except Exception:
            pass
        return user_id

    def _resolve_canonical_model_id(self, model_id: str) -> str:
        return model_id.strip().lower() if model_id else ""

    def _get_webui_memories(self, user_id: str, tracking: Dict, max_memories: int = 500) -> List[Dict]:
        if not os.path.exists(self.openwebui_db):
            return []
        memories = []
        try:
            conn = self._get_connection(self.openwebui_db)
            webui_last = tracking.get("webui_last_processed_at", 0)
            rows = conn.execute(
                """SELECT id, content, created_at, updated_at
                   FROM memory WHERE user_id = ? AND created_at > ?
                   ORDER BY created_at ASC LIMIT ?""",
                (user_id, webui_last, max_memories)
            ).fetchall()
            conn.close()
            for r in rows:
                memories.append({
                    "memory_id": r["id"],
                    "content": r["content"],
                    "tags": [],
                    "memory_bank": "General",
                    "importance_level": 5,
                    "timestamp_created": datetime.fromtimestamp(r["created_at"], tz=timezone.utc).isoformat() if isinstance(r["created_at"], (int, float)) else r["created_at"],
                    "source": "webui",
                })
        except Exception as e:
            logger.error(f"Failed to get webui memories: {e}")
        return memories

    def _get_archived_memories(self, user_id: str, model_id: str, tracking: Dict, max_memories: int = 500) -> List[Dict]:
        if not os.path.exists(self.archives_dir):
            return []
        memories = []
        archive_current = tracking.get("archive_processing", "")
        archived_list = tracking.get("processed_archives", [])
        cursor_ts = tracking.get("archive_cursor_ts", "")
        cursor_id = tracking.get("archive_cursor_id", "")

        archives = sorted(
            glob.glob(os.path.join(self.archives_dir, "ai_memories_*.db")),
            reverse=True,
        )
        archives = [a for a in archives if os.path.basename(a) not in archived_list]

        if archive_current:
            archives = [a for a in archives if os.path.basename(a) == archive_current] + \
                       [a for a in archives if os.path.basename(a) != archive_current]

        for archive_path in archives:
            if len(memories) >= max_memories:
                break
            archive_name = os.path.basename(archive_path)
            try:
                conn = self._get_connection(archive_path)
                has_table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='curated_memories'"
                ).fetchone()
                if not has_table:
                    conn.close()
                    continue
                query = """SELECT memory_id, content, tags, memory_bank, importance_level, timestamp_created
                           FROM curated_memories WHERE (user_id = ? AND LOWER(model_id) = ?)
                           OR (user_id IS NULL AND model_id IS NULL)
                           OR (user_id = '' AND model_id IS NULL)
                           OR (user_id = '' AND model_id = '')"""
                params = [user_id, model_id.lower()]
                if cursor_ts and archive_name == archive_current:
                    query += " AND (timestamp_created > ? OR (timestamp_created = ? AND memory_id > ?))"
                    params.extend([cursor_ts, cursor_ts, cursor_id])
                query += " ORDER BY timestamp_created ASC, memory_id ASC LIMIT ?"
                params.append(max_memories + 1)
                rows = conn.execute(query, params).fetchall()
                conn.close()
                for r in rows:
                    if len(memories) >= max_memories:
                        break
                    tags = None
                    if r["tags"]:
                        try:
                            tags = json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
                        except:
                            tags = []
                    memories.append({
                        "memory_id": f"archive:{archive_name}:{r['memory_id']}",
                        "content": r["content"],
                        "tags": tags,
                        "memory_bank": r["memory_bank"],
                        "importance_level": r["importance_level"],
                        "timestamp_created": r["timestamp_created"],
                        "source": "archive",
                        "_archive_name": archive_name,
                    })
                if len(rows) > max_memories:
                    tracking["archive_processing"] = archive_name
                    tracking["archive_cursor_ts"] = rows[-2]["timestamp_created"]
                    tracking["archive_cursor_id"] = rows[-2]["memory_id"]
                else:
                    if archive_name not in tracking.get("processed_archives", []):
                        tracking.setdefault("processed_archives", []).append(archive_name)
            except Exception as e:
                logger.error(f"Failed to get archived memories from {archive_name}: {e}")
        return memories

    def _get_openwebui_conversations(self, user_id: str, limit: int = 100) -> List[Dict]:
        if not os.path.exists(self.openwebui_db):
            return []
        conversations = []
        try:
            conn = self._get_connection(self.openwebui_db)
            rows = conn.execute(
                """SELECT id, title, summary, updated_at
                   FROM chat WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit)
            ).fetchall()
            conn.close()
            for r in rows:
                conv = {"conversation_id": r["id"], "topic_summary": r["summary"] or r["title"] or ""}
                if conv["topic_summary"]:
                    conv["topic_summary"] = f"[OpenWebUI] {conv['topic_summary']}"
                conversations.append(conv)
        except Exception as e:
            logger.error(f"Failed to get OpenWebUI conversations: {e}")
        return conversations

    # ------------------------------------------------------------------
    # LLM distillation
    # ------------------------------------------------------------------

    async def _call_llm(self, system_prompt: str, user_prompt: str, model_name: str,
                        api_endpoint: str, provider_type: str = "ollama",
                        api_key: str = "", max_tokens: int = 2000) -> str:
        url = api_endpoint.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if provider_type == "openai_compatible":
            headers["Authorization"] = f"Bearer {api_key}"
            url += "/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
            }
        else:
            url += "/api/chat"
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "stream": False,
            }
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if provider_type == "openai_compatible":
                        message = data.get("choices", [{}])[0].get("message", {})
                        content = message.get("content") or message.get("reasoning_content", "")
                    else:
                        content = data.get("message", {}).get("content", "")
                    if content:
                        import re
                        content = re.sub(r'<\s*/?\s*(?:think|thinking)\s*>.*?</\s*(?:think|thinking)\s*>\s*', '', content, flags=re.DOTALL)
                        return content.strip()
                    return ""
                else:
                    error_text = await resp.text()
                    logger.error(f"LLM call failed: {resp.status} - {error_text}")
                    return ""

    async def distill_core_identity(self, memories: List[Dict], conversations: List[Dict],
                                     user_id: str, model_id: str, is_initial: bool,
                                     llm_model: str, llm_api_endpoint: str,
                                     llm_provider: str, llm_api_key: str,
                                     user_name: str = "User", model_name: str = "Assistant",
                                     accumulated_identity: str = "",
                                     batch_number: int = 0, total_batches: int = 0,
                                     memory_conversation_map: Optional[Dict[str, List[str]]] = None) -> str:
        import re
        memory_summaries = []
        for m in memories[:100]:
            content = m.get("content", "")
            content = re.sub(r'\s*\[Model:\s*[^\]]+\]', '', content)
            memory_summaries.append(content)
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

        if accumulated_identity:
            system_prompt = (
                f"You are updating an AI companion's core identity. You will receive the CURRENT identity "
                f"followed by NEW memories and conversations. Merge new information into the existing identity.\n\n"
                f"Output format MUST use these exact section markers:\n"
                f"[Personality]\n"
                f"[Relationship]\n"
                f"[Principles]\n"
                f"[Facts About {user_name}]\n"
                f"[Historical Context]\n"
                f"[Mistakes and Lessons Learned - {user_name}]\n"
                f"[Mistakes and Lessons Learned - {model_name}]\n\n"
                f"Preserve all existing content. Append or refine sections with new information. "
                f"If a section has no new data, write 'Still learning'."
            )
        else:
            system_prompt = (
                "You are distilling an AI companion's core identity. "
                "Output format MUST use these exact section markers:\n\n"
                "[Personality]\n"
                "[Relationship]\n"
                "[Principles]\n"
                f"[Facts About {user_name}]\n"
                "[Historical Context]\n"
                f"[Mistakes and Lessons Learned - {user_name}]\n"
                f"[Mistakes and Lessons Learned - {model_name}]\n\n"
                "Keep each section concise. If a section has no relevant data, write 'Still learning'."
            )

        batch_info = f"\n(Batch {batch_number + 1} of {total_batches})" if total_batches > 0 else ""
        user_prompt = (
            f"Core identity distillation{batch_info}\n\n"
            f"{'INITIAL' if is_initial else 'INCREMENTAL'} generation.\n"
            f"Total memories: {len(unique_memories)}\n"
            f"Total conversations: {len(conversations)}\n\n"
        )
        if accumulated_identity:
            user_prompt += f"CURRENT IDENTITY:\n{accumulated_identity}\n\n---\n\n"
        user_prompt += (
            f"MEMORIES:\n{memories_text if memories_text else '(No memories)'}\n\n"
            f"CONVERSATIONS:\n{conversations_text if conversations_text else '(No conversations)'}"
        )
        if memory_conversation_map:
            entries = []
            for mid, topics in list(memory_conversation_map.items())[:20]:
                if topics:
                    entries.append(f"Memory {mid[:8]} came from: {', '.join(topics)}")
            if entries:
                user_prompt += "\n\nMEMORY-CONVERSATION ASSOCIATIONS:\n" + "\n".join(entries)

        result = await self._call_llm(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model_name=llm_model, api_endpoint=llm_api_endpoint,
            provider_type=llm_provider, api_key=llm_api_key, max_tokens=2000
        )
        if result:
            return result
        return (
            f"[Personality]\nStill learning\n\n[Relationship]\nStill learning\n\n"
            f"[Principles]\nStill learning\n\n[Facts About {user_name}]\nStill learning\n\n"
            f"[Historical Context]\nStill learning\n\n"
            f"[Mistakes and Lessons Learned - {user_name}]\nStill learning\n\n"
            f"[Mistakes and Lessons Learned - {model_name}]\nStill learning"
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def save_to_database(self, user_id: str, model_id: str, content: str,
                          memories_analyzed: int, conversations_analyzed: int,
                          generation_reason: str = "scheduled") -> int:
        conn = self._get_connection(self.ai_memories_db)
        try:
            now = datetime.now(get_local_timezone()).isoformat()
            existing = conn.execute(
                "SELECT id, version FROM core_identity WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_id.lower())
            ).fetchone()
            metadata = json.dumps({
                "memories_analyzed": memories_analyzed,
                "conversations_analyzed": conversations_analyzed,
                "generation_reason": generation_reason,
                "sections": ["Personality", "Relationship", "Principles",
                             f"Facts About {user_id}", "Historical Context"]
            })
            if existing:
                new_version = existing["version"] + 1
                conn.execute("""
                    UPDATE core_identity SET version = ?, content = ?, metadata = ?, last_generated_at = ?
                    WHERE user_id = ? AND LOWER(model_id) = ?
                """, (new_version, content, metadata, now, user_id, model_id.lower()))
                logger.info(f"Core identity updated: version={new_version}")
                return new_version
            else:
                conn.execute("""
                    INSERT INTO core_identity (user_id, model_id, version, content, metadata, last_generated_at)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (user_id, model_id, content, metadata, now))
                logger.info("Core identity created")
                return 1
        finally:
            conn.close()

    def _write_file_backup(self, content: str, user_id: str, model_id: str):
        try:
            filepath = self._backup_file_for_user_and_model("core_identity", user_id, model_id)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = {
                "user_id": user_id, "model_id": model_id,
                "content": content, "file": os.path.basename(filepath),
                "updated_at": datetime.now(get_local_timezone()).isoformat(),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"File backup written: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write file backup: {e}")

    def _update_processed_timestamps(self, memory_ids: List[str], timestamp: str):
        if not memory_ids:
            return
        conn = self._get_connection(self.ai_memories_db)
        try:
            placeholders = ",".join("?" for _ in memory_ids)
            conn.execute(
                f"UPDATE curated_memories SET core_identity_processed_until = ? WHERE memory_id IN ({placeholders})",
                [timestamp] + memory_ids
            )
            conn.commit()
            logger.info(f"Updated timestamps for {len(memory_ids)} memories")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_core_identity(self, user_id: str, model_id: str) -> Optional[str]:
        conn = self._get_connection(self.ai_memories_db)
        try:
            row = conn.execute(
                "SELECT content FROM core_identity WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_id.lower())
            ).fetchone()
            if row and row["content"]:
                return row["content"]
        finally:
            conn.close()
        filepath = self._backup_file_for_user_and_model("core_identity", user_id, model_id)
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    return data.get("content", "")
            except Exception as e:
                logger.error(f"Failed to load file backup: {e}")
        legacy = self._backup_file_for_user("core_identity", user_id)
        if os.path.exists(legacy):
            try:
                with open(legacy) as f:
                    data = json.load(f)
                    return data.get("content", "")
            except Exception:
                pass
        return None

    def get_core_identity_for_injection(self, user_id: str, model_id: str) -> Optional[str]:
        import asyncio
        loop = asyncio.get_event_loop()
        content = loop.run_until_complete(self.load_core_identity_async(user_id, model_id))
        if content:
            return f"\n\n---\n\n[Core Identity]\n{content}"
        return None

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def save_progress(self, user_id: str, model_id: str, status: str,
                      memories_processed: int, memories_total: int,
                      partial_content: str = "", paused_at: str = "",
                      batch_index: int = 0, batches_total: int = 0,
                      all_memory_ids: Optional[List] = None):
        progress = {
            "user_id": user_id, "model_id": model_id,
            "status": status, "memories_processed": memories_processed,
            "memories_total": memories_total, "partial_content": partial_content,
            "paused_at": paused_at or datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "batch_index": batch_index, "batches_total": batches_total,
            "all_memory_ids": all_memory_ids or [],
        }
        try:
            filepath = self._progress_file_for_user_and_model("core_identity_progress", user_id, model_id)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def load_progress(self, user_id: str, model_id: str) -> Optional[Dict]:
        filepath = self._progress_file_for_user_and_model("core_identity_progress", user_id, model_id)
        if os.path.exists(filepath):
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")
        legacy = self._progress_file_for_user("core_identity_progress", user_id)
        if os.path.exists(legacy):
            try:
                with open(legacy) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def clear_progress(self, user_id: str, model_id: str):
        filepath = self._progress_file_for_user_and_model("core_identity_progress", user_id, model_id)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Failed to clear progress: {e}")
        legacy = self._progress_file_for_user("core_identity_progress", user_id)
        if os.path.exists(legacy):
            try:
                os.remove(legacy)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Tracking system
    # ------------------------------------------------------------------

    def _load_tracking(self, user_id: str) -> Dict:
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file) as f:
                    all_data = json.load(f)
                    return all_data.get(user_id, {
                        "webui_last_processed_at": 0,
                        "webui_last_processed_id": "",
                        "archive_processing": "",
                        "archive_cursor_ts": "",
                        "archive_cursor_id": "",
                        "processed_archives": [],
                        "needs_rescan": False,
                    })
        except Exception as e:
            logger.warning(f"Could not load tracking: {e}")
        return {
            "webui_last_processed_at": 0,
            "webui_last_processed_id": "",
            "archive_processing": "",
            "archive_cursor_ts": "",
            "archive_cursor_id": "",
            "processed_archives": [],
            "needs_rescan": False,
        }

    def _save_tracking(self, user_id: str, data: Dict):
        try:
            all_data = {}
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file) as f:
                    all_data = json.load(f)
            all_data[user_id] = data
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            with open(self.tracking_file, "w") as f:
                json.dump(all_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tracking: {e}")

    def _update_tracking_after_generation(self, user_id: str, model_id: str,
                                           webui_newest_ts: int = 0,
                                           webui_newest_id: str = "",
                                           archive_hit_end: bool = False,
                                           archive_current: str = ""):
        tracking = self._load_tracking(user_id)
        if webui_newest_ts:
            tracking["webui_last_processed_at"] = webui_newest_ts
            tracking["webui_last_processed_id"] = webui_newest_id
        if archive_hit_end and archive_current:
            if archive_current not in tracking["processed_archives"]:
                tracking["processed_archives"].append(archive_current)
            tracking["archive_processing"] = ""
            tracking["archive_cursor_ts"] = ""
            tracking["archive_cursor_id"] = ""
        elif archive_current:
            tracking["archive_processing"] = archive_current
        self._save_tracking(user_id, tracking)

    def _set_rescan_flag(self, user_id: str):
        tracking = self._load_tracking(user_id)
        tracking["needs_rescan"] = True
        self._save_tracking(user_id, tracking)

    def _identity_changed_significantly(self, old_content: str, new_content: str) -> bool:
        import re
        sections = re.findall(r'\[([^\]]+)\]', old_content)
        for section in sections:
            old_part = re.search(rf'\[{re.escape(section)}\](.*?)(?=\n\[|$)', old_content, re.DOTALL)
            new_part = re.search(rf'\[{re.escape(section)}\](.*?)(?=\n\[|$)', new_content, re.DOTALL)
            if old_part and new_part:
                old_text = old_part.group(1).strip()
                new_text = new_part.group(1).strip()
                if "Still learning" in old_text and "Still learning" not in new_text:
                    return True
        return False

    def _check_and_handle_rescan(self, user_id: str):
        tracking = self._load_tracking(user_id)
        if tracking.get("needs_rescan") and tracking.get("processed_archives"):
            tracking["processed_archives"] = []
            tracking["archive_processing"] = ""
            tracking["archive_cursor_ts"] = ""
            tracking["archive_cursor_id"] = ""
            tracking["needs_rescan"] = False
            self._save_tracking(user_id, tracking)
            logger.info(f"Rescan triggered for user {user_id}")

    # ------------------------------------------------------------------
    # Main generation entry point
    # ------------------------------------------------------------------

    async def run_generation(self, user_id: str, model_id: str,
                             llm_model: str, llm_api_endpoint: str,
                             llm_provider: str, llm_api_key: str,
                             max_memories: int = 500,
                             batch_size: int = 50,
                             user_name: str = "User",
                             model_name: str = "Assistant",
                             idle_check_callback=None,
                             request=None) -> Dict:
        self.initialize()
        self.request = request

        progress = self.load_progress(user_id, model_id)
        is_initial = False
        if progress and progress.get("status") == "in_progress":
            logger.info("Resuming core identity generation")
        elif progress and progress.get("status") == "completed":
            self.clear_progress(user_id, model_id)
            is_initial = True
        else:
            is_initial = True

        accumulated = ""
        all_memory_ids = []
        if progress and progress.get("partial_content"):
            accumulated = progress.get("partial_content", "")
            all_memory_ids = progress.get("all_memory_ids", [])

        memories = self.get_new_memories_since_processing(user_id, model_id, max_memories)

        # Gather from all three sources with tracking
        tracking = self._load_tracking(user_id) if hasattr(self, "_load_tracking") else {}
        curated = self.get_new_memories_since_processing(user_id, model_id, max_memories)
        webui = self._get_webui_memories(user_id, tracking, max_memories)
        archived = self._get_archived_memories(user_id, model_id, tracking, max_memories)

        # Deduplicate by content across all sources
        seen = set()
        all_from_sources = []
        for src in [curated, webui, archived]:
            for m in src:
                content = m.get("content", "")
                if content and content not in seen:
                    seen.add(content)
                    all_from_sources.append(m)

        memories = all_from_sources[:max_memories]
        if not memories and not accumulated:
            logger.info("No memories to process")
            self.clear_progress(user_id, model_id)
            return {"status": "completed", "memories_processed": 0, "conversations_processed": 0, "version": 0}

        memory_ids = list(dict.fromkeys(all_memory_ids + [m["memory_id"] for m in memories]))

        batches = [memories[i:i + batch_size] for i in range(0, len(memories), batch_size)]
        total_batches = len(batches)
        total_conversations = 0
        current_batch = 0

        for batch_idx, batch in enumerate(batches):
            if idle_check_callback and idle_check_callback():
                await self._save_progress_async(
                    user_id, model_id, "paused", len(all_memory_ids), max_memories,
                    partial_content=accumulated, batch_index=batch_idx,
                    batches_total=total_batches, all_memory_ids=memory_ids
                )
                return {"status": "paused", "memories_processed": len(all_memory_ids)}

            batch_ids = [m["memory_id"] for m in batch]
            conversations = self.get_conversations_for_memories(batch_ids)
            total_conversations += len(conversations)

            mem_conv_map = self._build_memory_conversation_map(batch_ids)

            content = await self.distill_core_identity(
                memories=batch, conversations=conversations,
                user_id=user_id, model_id=model_id, is_initial=is_initial and batch_idx == 0,
                llm_model=llm_model, llm_api_endpoint=llm_api_endpoint,
                llm_provider=llm_provider, llm_api_key=llm_api_key,
                user_name=user_name, model_name=model_name,
                accumulated_identity=accumulated,
                batch_number=batch_idx, total_batches=total_batches,
                memory_conversation_map=mem_conv_map
            )

            if not content:
                await self._save_progress_async(
                    user_id, model_id, "error", len(all_memory_ids), max_memories,
                    partial_content=accumulated, batch_index=batch_idx,
                    batches_total=total_batches, all_memory_ids=memory_ids
                )
                return {"status": "error", "error": "LLM returned empty content"}

            accumulated = content
            all_memory_ids = list(dict.fromkeys(all_memory_ids + batch_ids))
            current_batch = batch_idx + 1

            await self._save_progress_async(
                user_id, model_id, "in_progress", len(all_memory_ids), max_memories,
                partial_content=accumulated, batch_index=current_batch,
                batches_total=total_batches, all_memory_ids=memory_ids
            )

        version = self.save_to_database(
            user_id, model_id, accumulated,
            len(all_memory_ids), total_conversations
        )

        now = datetime.now(get_local_timezone()).isoformat()
        self._update_processed_timestamps(all_memory_ids, now)

        await self._write_file_backup_async(accumulated, user_id, model_id)

        self.clear_progress(user_id, model_id)

        logger.info(f"Core identity completed: version={version}, memories={len(all_memory_ids)}")
        return {
            "status": "completed", "version": version,
            "memories_processed": len(all_memory_ids),
            "conversations_processed": total_conversations,
            "batches": current_batch,
        }

    async def pause_generation(self, user_id: str, model_id: str, partial_content: str = ""):
        await self._save_progress_async(
            user_id=user_id, model_id=model_id,
            status="paused", memories_processed=0, memories_total=0,
            partial_content=partial_content,
            paused_at=datetime.now(timezone.utc).isoformat()
        )
        logger.info(f"Core identity generation paused for user={user_id}")