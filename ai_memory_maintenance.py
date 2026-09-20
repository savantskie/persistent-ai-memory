"""
Persistent AI Memory - Long-Term Memory Maintenance Module

Provides LLM-powered maintenance operations for long-term (curated) memories:
- Format reformatting: Rewrites old-style memories to match current format
- Contradiction/update scanning: Detects new info superseding old memories
- Link assistance: Helps reconnect unlinked memories to source conversations

Uses the same memory format style as the short-term extraction prompt.
Never culls or deletes memories -- only appends update notes and creates links.
"""

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)

# Default LLM endpoint for maintenance tasks
DEFAULT_LLM_ENDPOINT = os.getenv("AI_MEMORY_LLM_ENDPOINT", "http://localhost:8080/v1/chat/completions")
DEFAULT_LLM_MODEL = os.getenv("AI_MEMORY_LLM_MODEL", "qwen-3")

KNOWN_BANKS = {
    "General", "Personal", "Work", "Projects", "Technical",
    "Tasks", "Research", "Context", "Patterns", "Preferences",
    "Temporary", "Character", "Character_Interaction"
}

CONTRADICTION_BATCH_SIZE = 10
MAX_SCAN_PER_RUN = 200

# Backlog processor constants
BACKLOG_PROGRESS_FILE = "backlog_progress.json"
BACKLOG_SNAPSHOT_DIR = "backlog_snapshots"
PRUNED_SCI_MARKERS = ("_pruned",)
DEFAULT_CANONICAL_USER_ID = os.environ.get(
    "AI_MEMORY_CANONICAL_USER_ID", "default_user"
)
# Importance re-rank constants
RERANK_BATCH_SIZE = 15
RERANK_STALENESS_DAYS = 30
RERANK_TRANSCRIPT_CHARS = 4000
RERANK_SIBLINGS = 8
RERANK_LLM_TIMEOUT = 180.0
RERANK_SAFETY_BUFFER = 512
RERANK_MAX_RESPONSE_TOKENS = 1024


def _normalize_model_id(model_id: Optional[str]) -> Optional[str]:
    if model_id is None:
        return None
    m = re.sub(r"\s+", "-", model_id.strip().lower())
    return m or None


class LongTermMemoryMaintenance:
    """
    LLM-powered maintenance operations for long-term (curated) memories.
    Designed to be called from DatabaseMaintenance.run_maintenance().
    """

    def __init__(self, memory_system, llm_endpoint: str = None, llm_model: str = None):
        self.memory_system = memory_system
        self.llm_endpoint = llm_endpoint or DEFAULT_LLM_ENDPOINT
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self._model_ctx_size: Optional[int] = None

    # Idle gate: the short-term plugin touches this on every inlet()
    IDLE_HEARTBEAT_FILE = os.environ.get(
        "AI_MEMORY_INLET_HEARTBEAT",
        os.path.join(os.environ.get("AI_MEMORY_DATA_DIR", "./memory_data"), "last_inlet.touch"),
    )
    IDLE_GRACE_SECONDS = 600

    def _user_active(self) -> bool:
        try:
            return (time.time() - os.path.getmtime(self.IDLE_HEARTBEAT_FILE)) < self.IDLE_GRACE_SECONDS
        except OSError:
            return False

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        temperature: float = 0.1, timeout: float = 60.0) -> Optional[str]:
        if httpx is None:
            logger.error("httpx not available for LLM calls")
            return None
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.llm_endpoint,
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": 1024,
                        "stream": False,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    message = data.get("choices", [{}])[0].get("message", {})
                    content = message.get("content") or message.get("reasoning_content", "")
                    if content:
                        content = re.sub(
                            r'<\s*/?\s*(?:think|thinking)\s*>.*?</\s*(?:think|thinking)\s*>\s*',
                            '', content, flags=re.DOTALL
                        )
                        return content.strip()
                logger.warning(f"LLM returned status {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    # Context window helpers

    async def _get_model_context_size(self) -> int:
        if self._model_ctx_size is not None:
            return self._model_ctx_size
        try:
            base_url = self.llm_endpoint
            for suffix in ("/v1/chat/completions", "/chat/completions", "/completions"):
                base_url = base_url.removesuffix(suffix)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("data", []):
                        meta = model.get("meta", {})
                        n_ctx = meta.get("n_ctx")
                        if n_ctx:
                            self._model_ctx_size = int(n_ctx)
                            logger.info("Model context window: %d tokens", self._model_ctx_size)
                            return self._model_ctx_size
        except Exception as e:
            logger.warning(f"Could not query model context size: {e}")
        self._model_ctx_size = 32768
        logger.info("Model context size defaulting to %d", self._model_ctx_size)
        return self._model_ctx_size

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 3)

    def _needs_reformat(self, content: str) -> bool:
        if not content:
            return False
        has_bank = bool(re.search(r'\[Memory Bank:\s*([^\]]+)\]', content))
        has_tags = bool(re.search(r'\[Tags:\s*([^\]]+)\]', content))
        return not (has_bank and has_tags)

    def _get_reformat_prompt(self) -> str:
        return (
            "You are a memory formatting assistant. Your ONLY function is to reformat existing "
            "memory text into the proper memory system format.\n\n"

            "The memory system uses this format:\n"
            '- [Tags: tag1, tag2, ...] The memory content here [Memory Bank: BankName]\n\n'

            "Content rules:\n"
            "- About the user: 'User is...', 'User prefers...', 'User mentioned...'\n"
            "- About the assistant's own experiences: 'I noticed...', 'I found that...'\n"
            "- About characters: Use appropriate character perspective\n"
            "- Keep all factual content intact. Use clear, natural language.\n"
            "- Remove confusing temporal markers like 'previously'\n\n"

            f"Memory banks: {sorted(KNOWN_BANKS)}\n"
            "- Tags: comma-separated, lowercase, descriptive\n\n"

            "Output ONLY the reformatted memory text. No JSON. No markdown. No explanations."
        )

    async def reformat_memories(self, limit: int = 100) -> Dict:
        results = {"scanned": 0, "reformatted": 0, "skipped": 0, "errors": 0}
        try:
            memories = await self.memory_system.ai_memory_db.execute_query(
                """SELECT memory_id, content, user_id, model_id, memory_bank, tags, timestamp_created
                   FROM curated_memories
                   ORDER BY timestamp_created ASC LIMIT ?""",
                (limit,)
            )
            if not memories:
                return results
            results["scanned"] = len(memories)

            for mem in memories:
                mem_id = mem["memory_id"]
                content = mem.get("content", "") or ""
                if not self._needs_reformat(content):
                    results["skipped"] += 1
                    continue

                bare = re.sub(r'\[Tags:\s*[^\]]*\]\s*', '', content)
                bare = re.sub(r'\[Memory Bank:\s*[^\]]*\]\s*', '', bare)
                bare = re.sub(r'\[User:\s*[^\]]*\]\s*', '', bare)
                bare = re.sub(r'\[Model:\s*[^\]]*\]\s*', '', bare)
                bare = bare.strip()

                user_prompt = (
                    f"Reformat this memory:\n\n{bare}\n\n"
                    f"Context: user_id={mem.get('user_id', 'unknown')}, "
                    f"model_id={mem.get('model_id', 'unknown')}\n"
                    f"Current tags: {mem.get('tags', 'none')}\n"
                    f"Current bank: {mem.get('memory_bank', 'General')}"
                )
                reformatted = await self._call_llm(self._get_reformat_prompt(), user_prompt)
                if reformatted and reformatted != content:
                    await self.memory_system.ai_memory_db.execute_update(
                        """UPDATE curated_memories SET content = ?, timestamp_updated = ?, updated_at = ?
                           WHERE memory_id = ?""",
                        (reformatted, datetime.now(timezone.utc).isoformat(),
                         datetime.now(timezone.utc).isoformat(), mem_id)
                    )
                    results["reformatted"] += 1
                else:
                    results["skipped"] += 1
                await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"Error in memory reformatting: {e}")
            results["errors"] = results.get("errors", 0) + 1
        return results

    def _get_contradiction_prompt(self) -> str:
        return (
            "You are a memory analysis assistant. Compare memory texts and identify pairs "
            "covering the same topic with differing or updated information.\n\n"

            "For each pair determine:\n"
            "- older_index: index of the older/outdated memory\n"
            "- newer_index: index of the newer/updated memory\n"
            "- relationship_type: 'updated_by', 'complements', or 'related_to'\n"
            "- notes: brief explanation in memory format style ('User mentioned...', 'Updated detail...')\n\n"

            "Output ONLY a JSON object:\n"
            '{"pairs": [{"older_index": 0, "newer_index": 1, '
            '"relationship_type": "updated_by", "notes": "..."}, ...]}\n'
            'If no pairs: {"pairs": []}'
        )

    async def scan_for_updates(self, limit: int = MAX_SCAN_PER_RUN) -> Dict:
        results = {"scanned": 0, "pairs_found": 0, "updated": 0,
                    "linked": 0, "skipped_already_related": 0, "errors": 0}
        try:
            memories = await self.memory_system.ai_memory_db.execute_query(
                """SELECT memory_id, content, user_id, model_id, memory_bank,
                          tags, timestamp_created, importance_level
                   FROM curated_memories ORDER BY timestamp_created ASC LIMIT ?""",
                (limit,)
            )
            if not memories or len(memories) < 2:
                return results
            results["scanned"] = len(memories)

            existing = await self._table_exists("memory_relationships")
            existing_relations = []
            if existing:
                existing_relations = await self.memory_system.conversations_db.execute_query(
                    "SELECT DISTINCT source_memory_id, target_memory_id FROM memory_relationships", ()
                )
            already_linked = set()
            for rel in (existing_relations or []):
                already_linked.add((rel.get("source_memory_id", ""), rel.get("target_memory_id", "")))

            for batch_start in range(0, len(memories), CONTRADICTION_BATCH_SIZE):
                batch = memories[batch_start:batch_start + CONTRADICTION_BATCH_SIZE]
                batch_pairs = await self._analyze_batch(batch, already_linked)
                if not batch_pairs:
                    continue
                results["pairs_found"] += len(batch_pairs)

                for pair in batch_pairs:
                    older_mem = batch[pair["older_index"]]
                    newer_mem = batch[pair["newer_index"]]
                    rel_type = pair["relationship_type"]
                    notes = pair.get("notes", "")

                    if notes:
                        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        update_note = f" [Updated: {timestamp}: {notes}]"
                        new_content = older_mem["content"] + update_note
                        await self.memory_system.ai_memory_db.execute_update(
                            """UPDATE curated_memories SET content = ?, timestamp_updated = ?, updated_at = ?
                               WHERE memory_id = ?""",
                            (new_content, datetime.now(timezone.utc).isoformat(),
                             datetime.now(timezone.utc).isoformat(), older_mem["memory_id"])
                        )
                        results["updated"] += 1

                    await self._create_relationship(
                        source_memory_id=older_mem["memory_id"],
                        target_memory_id=newer_mem["memory_id"],
                        relationship_type=rel_type, notes=notes
                    )
                    results["linked"] += 1

                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Error scanning for updates: {e}")
            results["errors"] = results.get("errors", 0) + 1
        return results

    async def _analyze_batch(self, batch: List[Dict], already_linked: set) -> List[Dict]:
        if not batch or len(batch) < 2:
            return []
        batch_text = ""
        for idx, mem in enumerate(batch):
            content = mem.get("content", "") or ""
            core = re.sub(r'\s*\[Updated:\s*[^\]]*\]\s*', '', content).strip()
            batch_text += f"[{idx}] {core[:300]}\n---\n"

        response = await self._call_llm(
            self._get_contradiction_prompt(),
            f"Analyze these memories:\n\n{batch_text}",
            temperature=0.1
        )
        if not response:
            return []
        return self._parse_pairs_json(response, already_linked)

    def _parse_pairs_json(self, text: str, already_linked: set) -> List[Dict]:
        text = re.sub(r'```json|```', '', text).strip()
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1:
            return []
        text = text[brace_start:brace_end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        pairs = data.get("pairs", [])
        if not isinstance(pairs, list):
            return []

        filtered = []
        for pair in pairs:
            older_idx = pair.get("older_index", -1)
            newer_idx = pair.get("newer_index", -1)
            if older_idx < 0 or newer_idx < 0:
                continue
            fwd = (str(older_idx), str(newer_idx))
            rev = (str(newer_idx), str(older_idx))
            if fwd in already_linked or rev in already_linked:
                continue
            filtered.append({
                "older_index": older_idx,
                "newer_index": newer_idx,
                "relationship_type": pair.get("relationship_type", "related_to"),
                "notes": pair.get("notes", "")
            })
        return filtered

    async def _create_relationship(self, source_memory_id: str, target_memory_id: str,
                                    relationship_type: str, notes: str = ""):
        rel_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        await self.memory_system.conversations_db.execute_update("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """, ())
        await self.memory_system.conversations_db.execute_update(
            """INSERT OR IGNORE INTO memory_relationships
               (relationship_id, source_memory_id, target_memory_id,
                relationship_type, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rel_id, source_memory_id, target_memory_id,
             relationship_type, notes, timestamp)
        )

    async def _table_exists(self, table_name: str) -> bool:
        try:
            rows = await self.memory_system.conversations_db.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
            )
            return len(rows) > 0
        except Exception:
            return False

    async def assist_linking(self, limit: int = 50) -> Dict:
        results = {"scanned": 0, "llm_linked": 0, "no_match": 0, "errors": 0}
        try:
            memories = await self.memory_system.ai_memory_db.execute_query(
                """SELECT m.memory_id, m.content, m.user_id, m.model_id, m.timestamp_created
                   FROM curated_memories m
                   WHERE (m.source_conversation_id IS NULL OR m.source_conversation_id = '')
                   ORDER BY m.timestamp_created DESC LIMIT ?""",
                (limit,)
            )
            if not memories:
                return results

            linked_rows = await self.memory_system.conversations_db.execute_query(
                "SELECT DISTINCT memory_id FROM memory_conversation_links", ()
            )
            already_linked = {row["memory_id"] for row in (linked_rows or [])}

            for mem in memories:
                mem_id = mem["memory_id"]
                if mem_id in already_linked:
                    continue
                content = mem.get("content", "") or ""
                if len(content) < 20:
                    continue

                conversations = await self.memory_system.conversations_db.execute_query(
                    """SELECT c.conversation_id, c.topic_summary, m.content as msg_content
                       FROM conversations c
                       LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                       WHERE c.user_id = ? OR ? = ''
                       ORDER BY c.start_timestamp DESC LIMIT 200""",
                    (mem.get("user_id", ""), mem.get("user_id", ""))
                )
                if not conversations:
                    continue

                results["scanned"] += 1
                search_words = set(re.sub(r'\[[^\]]*\]\s*', '', content).lower().split())
                if len(search_words) < 3:
                    continue

                best_match = None
                best_score = 0.0
                for conv in conversations:
                    msg_content = conv.get("msg_content", "") or ""
                    msg_words = set(msg_content.lower().split())
                    if not msg_words:
                        continue
                    overlap = len(search_words & msg_words)
                    union = len(search_words | msg_words)
                    overlap_ratio = overlap / union if union > 0 else 0.0
                    if overlap_ratio > best_score:
                        best_score = overlap_ratio
                        best_match = conv

                if best_match and best_score >= 0.15:
                    conv_id = best_match["conversation_id"]
                    await self.memory_system.conversations_db.link_memory_to_conversation(
                        memory_id=mem_id,
                        conversation_id=conv_id,
                        link_type="related",
                        link_strength=min(best_score, 1.0),
                        source_system="retroactive_link",
                        metadata={"match_score": best_score,
                                   "matched_via": "llm_assisted_maintenance"}
                    )
                    results["llm_linked"] += 1
                else:
                    results["no_match"] += 1

                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in assisted linking: {e}")
            results["errors"] = results.get("errors", 0) + 1
        return results

    # ==================================================================
    # Backlog processor: dedupe -> identity/conv-id backfill -> re-rank
    # ==================================================================

    def _memory_data_dir(self) -> Path:
        return Path(self.memory_system.conversations_db.db_path).resolve().parent

    def _backlog_progress_path(self) -> Path:
        return self._memory_data_dir() / BACKLOG_PROGRESS_FILE

    def _load_backlog_progress(self) -> Dict:
        p = self._backlog_progress_path()
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read backlog progress ({e}); starting fresh")
        return {"dbs": {}, "updated_at": None}

    def _save_backlog_progress(self, prog: Dict) -> None:
        prog["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self._backlog_progress_path().write_text(
                json.dumps(prog, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Could not save backlog progress: {e}")

    def _list_backlog_dbs(self, include_archives: bool = True) -> List[Path]:
        md = self._memory_data_dir()
        dbs: List[Path] = []
        live = md / "ai_memories.db"
        if live.exists():
            dbs.append(live)
        if include_archives:
            arch = sorted(
                glob.glob(str(md / "archives" / "ai_memories_*.db")),
                reverse=True,
            )
            dbs.extend(Path(a) for a in arch)
        return dbs

    def _snapshot_db(self, db_path: Path, prog_db: Dict) -> Optional[str]:
        if prog_db.get("snapshot"):
            return prog_db["snapshot"]
        snap_dir = self._memory_data_dir() / BACKLOG_SNAPSHOT_DIR
        snap_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = snap_dir / f"{db_path.stem}.{stamp}.bak"
        try:
            shutil.copy2(str(db_path), str(dest))
            logger.warning(f"Snapshot before backlog changes: {dest.name}")
            return str(dest)
        except Exception as e:
            logger.error(f"Could not snapshot {db_path.name}: {e}")
            return None

    @staticmethod
    def _connect(path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _has_curated_table(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='curated_memories'"
        ).fetchone()
        return row is not None

    @staticmethod
    def _dedupe_curated_pass(conn: sqlite3.Connection) -> int:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_curated_memories_content "
            "ON curated_memories (content)"
        )
        cur = conn.execute("""
            DELETE FROM curated_memories
            WHERE memory_id IN (
                SELECT memory_id FROM (
                    SELECT memory_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY content, IFNULL(memory_type,''),
                                            IFNULL(memory_bank,''),
                                            IFNULL(user_id,''),
                                            LOWER(IFNULL(model_id,'')),
                                            IFNULL(source_conversation_id,'')
                               ORDER BY timestamp_created ASC, memory_id ASC
                           ) AS rn
                    FROM curated_memories
                )
                WHERE rn > 1
            )
        """)
        return cur.rowcount or 0

    @staticmethod
    def _identity_backfill_pass(conn: sqlite3.Connection, canonical_uid: str) -> Dict:
        stats = {"user_id_filled": 0, "model_id_normalized": 0}
        cur = conn.execute(
            "UPDATE curated_memories SET user_id = ? "
            "WHERE user_id IS NULL OR TRIM(user_id) = ''",
            (canonical_uid,),
        )
        stats["user_id_filled"] = cur.rowcount or 0
        rows = conn.execute(
            "SELECT DISTINCT model_id FROM curated_memories WHERE model_id IS NOT NULL"
        ).fetchall()
        for (mid,) in rows:
            norm = _normalize_model_id(mid)
            if norm and norm != mid:
                cur = conn.execute(
                    "UPDATE curated_memories SET model_id = ? WHERE model_id = ?",
                    (norm, mid),
                )
                stats["model_id_normalized"] += cur.rowcount or 0
        return stats

    @staticmethod
    def _iso_to_epoch(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    def _conv_id_backfill_pass(self, conn: sqlite3.Connection,
                                conversations_db_path: Optional[str]) -> Dict:
        stats = {"candidates": 0, "resolved": 0, "unmatched": 0}
        if not conversations_db_path or not Path(conversations_db_path).exists():
            return stats
        like_clauses = " OR ".join(
            ["source_conversation_id LIKE ?"] * len(PRUNED_SCI_MARKERS)
        )
        params = [f"%{m}%" for m in PRUNED_SCI_MARKERS]
        cand = conn.execute(
            f"""SELECT memory_id, user_id, timestamp_created
                FROM curated_memories
                WHERE source_conversation_id IS NULL
                   OR TRIM(source_conversation_id) = ''
                   OR {like_clauses}""",
            params,
        ).fetchall()
        stats["candidates"] = len(cand)
        if not cand:
            return stats
        cdb = self._connect(conversations_db_path)
        try:
            convs = cdb.execute(
                """SELECT conversation_id, user_id, start_timestamp, end_timestamp
                   FROM conversations"""
            ).fetchall()
        except Exception as e:
            logger.warning(f"conv-id backfill: cannot read conversations ({e})")
            cdb.close()
            return stats
        cdb.close()
        windows = []
        for cid, cuid, cs, ce in convs:
            s = self._iso_to_epoch(cs)
            e = self._iso_to_epoch(ce) or (s + 6 * 3600 if s else None)
            if s is None:
                continue
            windows.append((cid, cuid, s, e))
        updates = []
        for mem_id, muid, mts in cand:
            m_epoch = self._iso_to_epoch(mts)
            if m_epoch is None:
                stats["unmatched"] += 1
                continue
            best = None
            best_gap = None
            for cid, cuid, s, e in windows:
                if muid and cuid and muid != cuid:
                    continue
                if s - 3600 <= m_epoch <= (e or s) + 3600:
                    gap = abs(m_epoch - s)
                    if best_gap is None or gap < best_gap:
                        best, best_gap = cid, gap
            if best:
                updates.append((best, mem_id))
            else:
                stats["unmatched"] += 1
        for cid, mem_id in updates:
            conn.execute(
                "UPDATE curated_memories SET source_conversation_id = ? WHERE memory_id = ?",
                (cid, mem_id),
            )
        stats["resolved"] = len(updates)
        return stats

    @staticmethod
    def _ensure_curated_unique_index(conn: sqlite3.Connection) -> bool:
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_curated_memories_identity "
                "ON curated_memories (content, user_id, model_id, source_conversation_id)"
            )
            return True
        except sqlite3.OperationalError as e:
            logger.warning(f"UNIQUE index not created (duplicates still present?): {e}")
            return False

    def _process_one_db_sync(self, db_path: Path, canonical_uid_hint: str,
                               conversations_db_path: Optional[str],
                               prog_db: Dict) -> Dict:
        result = {"db": db_path.name, "skipped": False, "stages": {}}
        conn = self._connect(str(db_path))
        try:
            if not self._has_curated_table(conn):
                result["skipped"] = True
                return result
            before = conn.execute("SELECT COUNT(*) FROM curated_memories").fetchone()[0]
            result["rows_before"] = before
            row = conn.execute(
                "SELECT user_id, COUNT(*) c FROM curated_memories "
                "WHERE user_id IS NOT NULL AND TRIM(user_id) != '' "
                "GROUP BY user_id ORDER BY c DESC LIMIT 1"
            ).fetchone()
            canonical_uid = (row[0] if row else None) or canonical_uid_hint
            if prog_db.get("dedupe") != "done":
                d1 = self._dedupe_curated_pass(conn)
                conn.commit()
                result["stages"]["dedupe_1"] = d1
                prog_db["dedupe"] = "done"
            if prog_db.get("identity") != "done":
                result["stages"]["identity"] = self._identity_backfill_pass(conn, canonical_uid)
                conn.commit()
                result["stages"]["dedupe_2"] = self._dedupe_curated_pass(conn)
                conn.commit()
                prog_db["identity"] = "done"
            if prog_db.get("conv_ids") != "done":
                result["stages"]["conv_ids"] = self._conv_id_backfill_pass(
                    conn, conversations_db_path
                )
                conn.commit()
                result["stages"]["dedupe_3"] = self._dedupe_curated_pass(conn)
                conn.commit()
                prog_db["conv_ids"] = "done"
            if prog_db.get("unique_index") != "done":
                ok = self._ensure_curated_unique_index(conn)
                conn.commit()
                prog_db["unique_index"] = "done" if ok else "deferred"
                result["stages"]["unique_index"] = prog_db["unique_index"]
            after = conn.execute("SELECT COUNT(*) FROM curated_memories").fetchone()[0]
            result["rows_after"] = after
            result["rows_removed"] = before - after
            return result
        finally:
            conn.close()

    async def dedupe_and_backfill_backlog(self, deadline_ts: Optional[float] = None,
                                          include_archives: bool = True,
                                          snapshot: bool = True,
                                          yield_to_user: bool = True) -> Dict:
        prog = self._load_backlog_progress()
        summary = {"processed": [], "stopped_early": False}
        live_conv_db = str(self._memory_data_dir() / "conversations.db")
        all_dbs = self._list_backlog_dbs(include_archives=include_archives)
        logger.warning(
            "dedupe/backfill: %d DB(s) in queue: %s",
            len(all_dbs), [p.name for p in all_dbs],
        )
        for db_path in all_dbs:
            if deadline_ts and time.time() >= deadline_ts:
                summary["stopped_early"] = True
                logger.warning("dedupe/backfill: deadline reached, stopping")
                break
            if self._user_active() and yield_to_user:
                summary["stopped_early"] = True
                summary["yielded_to_user"] = True
                logger.warning("dedupe/backfill: user active, yielding")
                break
            key = db_path.name
            prog_db = prog["dbs"].setdefault(key, {})
            if prog_db.get("backfill_complete") or prog_db.get("dedupe_state"):
                logger.info("dedupe/backfill: %s already complete, skipping", key)
                continue
            logger.warning("dedupe/backfill: starting %s", key)
            if snapshot:
                snap = await asyncio.to_thread(self._snapshot_db, db_path, prog_db)
                if snap:
                    prog_db["snapshot"] = snap
            else:
                prog_db.setdefault("snapshot", "skipped-by-caller")
            self._save_backlog_progress(prog)
            is_live = db_path.name == "ai_memories.db"
            conv_db = live_conv_db if is_live else None
            try:
                res = await asyncio.to_thread(
                    self._process_one_db_sync, db_path,
                    DEFAULT_CANONICAL_USER_ID, conv_db, prog_db,
                )
                if not res.get("skipped"):
                    prog_db["backfill_complete"] = True
                summary["processed"].append(res)
                logger.warning(
                    "backlog[%s]: %d -> %d rows (%d removed) stages=%s",
                    key, res.get('rows_before'), res.get('rows_after'),
                    res.get('rows_removed'), res.get('stages'),
                )
            except Exception as e:
                logger.error(f"backlog processing failed for {key}: {e}")
                if ("no such column" in str(e)) or ("UNIQUE constraint failed" in str(e)):
                    prog_db["dedupe_state"] = "irrecoverable"
                summary["processed"].append({"db": key, "error": str(e)})
            finally:
                self._save_backlog_progress(prog)
        return summary

    async def run_backlog(self, deadline: Optional[str] = None,
                          max_seconds: Optional[int] = 1800,
                          include_archives: bool = True,
                          do_rerank: bool = True,
                          snapshot: bool = True,
                          yield_to_user: bool = True) -> Dict:
        deadline_ts = None
        if deadline:
            try:
                if len(deadline) <= 5 and ":" in deadline:
                    hh, mm = (int(x) for x in deadline.split(":"))
                    now = datetime.now()
                    tgt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if tgt <= now:
                        tgt = tgt + timedelta(days=1)
                    deadline_ts = tgt.timestamp()
                else:
                    deadline_ts = datetime.fromisoformat(
                        deadline.replace("Z", "+00:00")
                    ).timestamp()
            except Exception:
                logger.warning(f"Bad deadline {deadline!r}; using max_seconds")
        if deadline_ts is None and max_seconds:
            deadline_ts = time.time() + max_seconds
        started = time.time()
        eta = datetime.fromtimestamp(deadline_ts).isoformat() if deadline_ts else "none"
        logger.warning(
            "BACKLOG RUN START  deadline=%s  archives=%s  rerank=%s  snapshot=%s",
            eta, include_archives, do_rerank, snapshot,
        )
        out = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "deadline_ts": deadline_ts,
            "deadline_local": eta,
            "params": {"include_archives": include_archives, "do_rerank": do_rerank,
                       "snapshot": snapshot},
        }
        out["dedupe_backfill"] = await self.dedupe_and_backfill_backlog(
            deadline_ts=deadline_ts, include_archives=include_archives,
            snapshot=snapshot, yield_to_user=yield_to_user,
        )
        if do_rerank:
            out["rerank"] = await self.rerank_importance(
                deadline_ts=deadline_ts, include_archives=include_archives,
                yield_to_user=yield_to_user,
            )
        out["elapsed_seconds"] = round(time.time() - started, 1)
        out["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._write_run_summary(out)
        logger.warning(
            "BACKLOG RUN DONE  elapsed=%ss", out["elapsed_seconds"],
        )
        return out

    def _summarize_run(self, out: Dict) -> Dict:
        db = out.get("dedupe_backfill", {}) or {}
        rr = out.get("rerank", {}) or {}
        removed = sum(
            (p.get("rows_removed") or 0) for p in db.get("processed", [])
            if isinstance(p, dict)
        )
        scored = sum(
            (d.get("scored") or 0) for d in rr.get("dbs", []) if isinstance(d, dict)
        )
        return {
            "dbs_deduped": len(db.get("processed", [])),
            "rows_removed": removed,
            "dedupe_stopped_early": db.get("stopped_early", False),
            "rerank_dbs_touched": len(rr.get("dbs", [])),
            "memories_scored": scored,
            "rerank_stopped_early": rr.get("stopped_early", False),
        }

    def _write_run_summary(self, out: Dict) -> None:
        try:
            runs_dir = self._memory_data_dir() / "backlog_runs"
            runs_dir.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            (runs_dir / f"backlog_run_{stamp}.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8"
            )
            (self._memory_data_dir() / "backlog_last_run.json").write_text(
                json.dumps(out, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Could not write backlog run summary: {e}")

    # ==================================================================
    # Importance re-rank
    # ==================================================================

    def _rerank_system_prompt(self) -> str:
        return (
            "You are an AI assistant, assigning how important each memory is. "
            "For every memory you are given, return an integer 1-10:\n"
            "- 1-2: trivial, will not matter later\n"
            "- 3-4: minor detail, mildly useful context\n"
            "- 5-6: moderate, worth remembering\n"
            "- 7-8: important to how you relate to the user\n"
            "- 9-10: core identity, a hard constraint, or the relationship itself\n\n"
            "You are shown each memory with the conversation it came from and other "
            "memories from that same conversation, so you can judge its real weight.\n\n"
            'Output ONLY a JSON array: [{"index": 0, "score": 7}, {"index": 1, "score": 3}]'
        )

    @staticmethod
    def _ensure_last_reranked_column(conn: sqlite3.Connection) -> None:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(curated_memories)").fetchall()]
        if "last_reranked" not in cols:
            conn.execute("ALTER TABLE curated_memories ADD COLUMN last_reranked TEXT")
            conn.commit()

    def _rerank_select_batch(self, conn: sqlite3.Connection, staleness_iso: str,
                              limit: int) -> List[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            SELECT memory_id, content, source_conversation_id, importance_level, tags
            FROM curated_memories
            WHERE importance_level IS NULL
               OR last_reranked IS NULL
               OR last_reranked < ?
            ORDER BY (importance_level IS NULL) DESC,
                     COALESCE(last_reranked, '') ASC,
                     timestamp_created DESC
            LIMIT ?
            """,
            (staleness_iso, limit),
        ).fetchall()

    def _gather_rerank_context(self, mem: sqlite3.Row,
                                conversations_db_path: Optional[str],
                                curated_conn: sqlite3.Connection) -> str:
        parts = [f"MEMORY: {mem['content']}"]
        sci = mem["source_conversation_id"]
        if sci:
            try:
                sibs = curated_conn.execute(
                    "SELECT content FROM curated_memories "
                    "WHERE source_conversation_id = ? AND memory_id != ? LIMIT ?",
                    (sci, mem["memory_id"], RERANK_SIBLINGS),
                ).fetchall()
                if sibs:
                    parts.append(
                        "OTHER MEMORIES FROM THIS CONVERSATION:\n"
                        + "\n".join(f"- {s[0]}" for s in sibs)
                    )
            except Exception:
                pass
        if sci and conversations_db_path and Path(conversations_db_path).exists():
            try:
                cdb = self._connect(conversations_db_path)
                msgs = cdb.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? "
                    "ORDER BY timestamp ASC LIMIT 40",
                    (sci,),
                ).fetchall()
                cdb.close()
                if msgs:
                    tx = "\n".join(f"{r}: {c}" for r, c in msgs)
                    parts.append("SOURCE CONVERSATION:\n" + tx[:RERANK_TRANSCRIPT_CHARS])
            except Exception:
                pass
        return "\n\n".join(parts)

    @staticmethod
    def _parse_scores(text: str) -> Dict[int, int]:
        if not text:
            return {}
        text = re.sub(r"```json|```", "", text)
        a, b = text.find("["), text.rfind("]")
        if a == -1 or b == -1 or b < a:
            return {}
        try:
            arr = json.loads(text[a: b + 1])
        except Exception:
            return {}
        out = {}
        for item in arr if isinstance(arr, list) else []:
            try:
                idx = int(item["index"])
                sc = max(1, min(10, int(item["score"])))
                out[idx] = sc
            except Exception:
                continue
        return out

    def _apply_rerank_scores(self, conn: sqlite3.Connection, batch: List[sqlite3.Row],
                              scores: Dict[int, int]) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        applied = 0
        for i, mem in enumerate(batch):
            sc = scores.get(i)
            if sc is None:
                conn.execute(
                    "UPDATE curated_memories SET last_reranked = ? WHERE memory_id = ?",
                    (now_iso, mem["memory_id"]),
                )
                continue
            content = mem["content"] or ""
            if re.search(r"\[Importance:\s*\d+\]", content):
                new_content = re.sub(r"\[Importance:\s*\d+\]", f"[Importance: {sc}]", content)
            else:
                new_content = f"{content} [Importance: {sc}]"
            conn.execute(
                "UPDATE curated_memories SET importance_level = ?, content = ?, "
                "last_reranked = ?, timestamp_updated = ? WHERE memory_id = ?",
                (sc, new_content, now_iso, now_iso, mem["memory_id"]),
            )
            applied += 1
        conn.commit()
        return applied

    async def _rerank_one_db(self, db_path: Path, deadline_ts: Optional[float],
                              prog_db: Dict, yield_to_user: bool = True) -> Dict:
        res = {"db": db_path.name, "batches": 0, "scored": 0, "seen": 0}
        staleness_iso = (
            datetime.now(timezone.utc) - timedelta(days=RERANK_STALENESS_DAYS)
        ).isoformat()
        is_live = db_path.name == "ai_memories.db"
        conv_db = str(self._memory_data_dir() / "conversations.db") if is_live else None
        n_ctx = await self._get_model_context_size()
        max_prompt_tokens = n_ctx - RERANK_SAFETY_BUFFER - RERANK_MAX_RESPONSE_TOKENS
        system_prompt_text = self._rerank_system_prompt()
        system_prompt_tokens = self._estimate_tokens(system_prompt_text)
        conn = self._connect(str(db_path))
        try:
            if not self._has_curated_table(conn):
                res["skipped"] = True
                return res
            self._ensure_last_reranked_column(conn)
            pending = conn.execute(
                "SELECT COUNT(*) FROM curated_memories WHERE importance_level IS NULL "
                "OR last_reranked IS NULL OR last_reranked < ?", (staleness_iso,)
            ).fetchone()[0]
            res["pending_at_start"] = pending
            logger.warning("rerank %s: %d memories need scoring", db_path.name, pending)
            while True:
                if deadline_ts and time.time() >= deadline_ts:
                    res["stopped_early"] = True
                    logger.warning("rerank %s: deadline reached", db_path.name)
                    break
                if self._user_active() and yield_to_user:
                    res["stopped_early"] = True
                    res["yielded_to_user"] = True
                    logger.warning("rerank %s: user active, yielding", db_path.name)
                    break
                batch = await asyncio.to_thread(
                    self._rerank_select_batch, conn, staleness_iso, RERANK_BATCH_SIZE
                )
                if not batch:
                    prog_db["rerank_complete"] = True
                    logger.warning("rerank %s: complete", db_path.name)
                    break
                ctxs = [
                    f"[{i}]\n" + self._gather_rerank_context(m, conv_db, conn)
                    for i, m in enumerate(batch)
                ]
                cumulative = system_prompt_tokens + self._estimate_tokens(
                    "Score each memory 1-10 (importance).\n\n"
                    '\n\nOutput ONLY the JSON array of {"index","score"}.'
                )
                usable_count = 0
                for ctx in ctxs:
                    toks = self._estimate_tokens(ctx)
                    if cumulative + toks > max_prompt_tokens:
                        break
                    cumulative += toks
                    usable_count += 1
                if usable_count == 0:
                    logger.warning(
                        "rerank %s: even 1 memory exceeds context window", db_path.name
                    )
                    await asyncio.to_thread(self._apply_rerank_scores, conn, batch[:1], {})
                    res["seen"] += 1
                    continue
                if usable_count < len(ctxs):
                    ctxs = ctxs[:usable_count]
                    batch = batch[:usable_count]
                user_prompt = (
                    "Score each memory 1-10 (importance).\n\n"
                    + "\n\n=====\n\n".join(ctxs)
                    + '\n\nOutput ONLY the JSON array of {"index","score"}.'
                )
                text = await self._call_llm(system_prompt_text, user_prompt,
                                            temperature=0.2,
                                            timeout=RERANK_LLM_TIMEOUT)
                scores = self._parse_scores(text or "")
                applied = await asyncio.to_thread(
                    self._apply_rerank_scores, conn, batch, scores
                )
                res["batches"] += 1
                res["scored"] += applied
                res["seen"] += len(batch)
                prog_db["rerank_last_run"] = datetime.now(timezone.utc).isoformat()
                if res["batches"] % 10 == 0:
                    logger.warning(
                        "rerank %s: %d batches, %d scored, %d seen so far",
                        db_path.name, res["batches"], res["scored"], res["seen"],
                    )
                await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"rerank failed for {db_path.name}: {e}")
            res["error"] = str(e)
        finally:
            conn.close()
        logger.warning(
            "rerank %s: DONE  batches=%d scored=%d seen=%d%s",
            db_path.name, res["batches"], res["scored"], res["seen"],
            "  (stopped at deadline)" if res.get("stopped_early") else "",
        )
        return res

    async def rerank_importance(self, deadline_ts: Optional[float] = None,
                                include_archives: bool = True,
                                yield_to_user: bool = True) -> Dict:
        prog = self._load_backlog_progress()
        summary = {"dbs": []}
        dbs = self._list_backlog_dbs(include_archives=include_archives)
        if not dbs:
            logger.warning("rerank: no DBs in queue")
            return summary
        logger.warning("rerank: %d DB(s) in queue: %s", len(dbs), [p.name for p in dbs])
        sem = asyncio.Semaphore(4)
        save_lock = asyncio.Lock()

        async def _process_one(db_path: Path) -> Dict:
            async with sem:
                if deadline_ts and time.time() >= deadline_ts:
                    return {"db": db_path.name, "skipped": True, "stopped_early": True}
                if self._user_active() and yield_to_user:
                    return {"db": db_path.name, "skipped": True, "yielded_to_user": True}
                prog_db = prog["dbs"].setdefault(db_path.name, {})
                res = await self._rerank_one_db(db_path, deadline_ts, prog_db, yield_to_user=yield_to_user)
                async with save_lock:
                    self._save_backlog_progress(prog)
                return res

        results = await asyncio.gather(*[_process_one(p) for p in dbs])
        summary["dbs"] = [r for r in results if r and not r.get("skipped")]
        total = sum(d.get("scored", 0) for d in summary["dbs"])
        logger.warning("rerank: run complete, %d memories scored total", total)
        return summary

    async def run_all(self, limit_reformat: int = 100,
                      limit_scan: int = MAX_SCAN_PER_RUN,
                      limit_link: int = 50) -> Dict:
        results = {
            "reformat": await self.reformat_memories(limit_reformat),
            "updates": await self.scan_for_updates(limit_scan),
            "linking": await self.assist_linking(limit_link),
        }
        return results


# Standalone CLI entrypoint for backlog processor
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AI Memory backlog processor")
    parser.add_argument("--until", default=None,
                        help='wall-clock cutoff "HH:MM" (local) or ISO timestamp')
    parser.add_argument("--max-seconds", type=int, default=1800,
                        help="fallback time budget when --until is not given")
    parser.add_argument("--no-archives", action="store_true",
                        help="only process the live ai_memories.db")
    parser.add_argument("--dedupe-only", action="store_true",
                        help="dedupe + identity/conv-id backfill, skip the LLM re-rank")
    parser.add_argument("--rerank-only", action="store_true",
                        help="only the LLM importance re-rank")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="skip the per-DB file-copy snapshot")
    args = parser.parse_args()

    def _resolve_deadline_ts() -> Optional[float]:
        if args.until:
            try:
                if len(args.until) <= 5 and ":" in args.until:
                    hh, mm = (int(x) for x in args.until.split(":"))
                    now = datetime.now()
                    tgt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if tgt <= now:
                        tgt = tgt + timedelta(days=1)
                    return tgt.timestamp()
                return datetime.fromisoformat(args.until.replace("Z", "+00:00")).timestamp()
            except Exception:
                logger.warning("Bad --until; falling back to --max-seconds")
        return time.time() + args.max_seconds

    async def _main():
        from ai_memory_core import AIMemorySystem
        ms = AIMemorySystem(enable_file_monitoring=False)
        ltm = LongTermMemoryMaintenance(ms)
        if args.rerank_only:
            res = await ltm.rerank_importance(
                deadline_ts=_resolve_deadline_ts(),
                include_archives=not args.no_archives,
            )
        else:
            res = await ltm.run_backlog(
                deadline=args.until,
                max_seconds=args.max_seconds,
                include_archives=not args.no_archives,
                do_rerank=not args.dedupe_only,
                snapshot=not args.no_snapshot,
            )
        print(json.dumps(res, indent=2, default=str))

    asyncio.run(_main())
