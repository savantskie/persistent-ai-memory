#!/usr/bin/env python3
"""
PAM Model Memory Deletion Tool

Interactive CLI tool for discovering and deleting model-associated memories
from the AI Memory System. Menu-driven with dry-run safety.

Usage:
    python pam_delete_model_memories.py              # Interactive menu
    python pam_delete_model_memories.py --user-id <id>  # Override user ID detection
"""

import sqlite3
import glob
import re
import os
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Configuration with env var overrides
MEMORY_DATA_DIR = os.environ.get("AI_MEMORY_DATA_DIR", "./memory_data")
WEBUI_DB_PATH = os.environ.get("AI_MEMORY_WEBUI_DB", "./memory_data/webui.db")
EMBEDDINGS_DB_PATH = os.environ.get("AI_MEMORY_EMBEDDINGS_DB", "./memory_data/memory_embeddings.db")
LOG_FILE = os.path.join(os.environ.get("AI_MEMORY_LOG_DIR", "./logs"), "pam_delete_model.log")


def log_action(msg: str):
    """Log actions to the PAM log file."""
    try:
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass


def get_all_owui_users() -> Dict[str, Dict]:
    users = {}
    if not os.path.exists(WEBUI_DB_PATH):
        return users
    try:
        conn = sqlite3.connect(WEBUI_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, role FROM user ORDER BY created_at")
        for uid, name, email, role in cursor.fetchall():
            users[uid] = {"name": name, "email": email, "role": role}
        conn.close()
    except Exception:
        pass
    return users


def parse_model_from_content(content: str) -> Optional[str]:
    m = re.search(r'\[Model:\s*([^\]]+)\]', content)
    return m.group(1).strip().lower() if m else None


def discover_ai_memory_shards() -> List[str]:
    pattern = os.path.join(MEMORY_DATA_DIR, "ai_memories*.db")
    return sorted(glob.glob(pattern))


def discover_databases() -> Dict[str, str]:
    dbs = {}
    for shard in discover_ai_memory_shards():
        dbs[f"curated_memories:{os.path.basename(shard)}"] = shard
    dbs["core_identity"] = os.path.join(MEMORY_DATA_DIR, "ai_memories.db")
    dbs["conversations"] = os.path.join(MEMORY_DATA_DIR, "conversations.db")
    dbs["conversation_characters"] = os.path.join(MEMORY_DATA_DIR, "conversation_characters.db")
    dbs["owui_memory"] = WEBUI_DB_PATH
    dbs["embedding_cache"] = EMBEDDINGS_DB_PATH
    return dbs


def discover_models(user_id: str) -> Dict:
    results = {
        "model_ids": {},
        "character_names": {},
        "owui_models": {},
    }

    for shard in discover_ai_memory_shards():
        shard_name = os.path.basename(shard)
        try:
            conn = sqlite3.connect(shard)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT LOWER(model_id), COUNT(*) FROM curated_memories "
                "WHERE user_id = ? GROUP BY LOWER(model_id)",
                (user_id,)
            )
            for model_id, count in cursor.fetchall():
                model_id = model_id.strip()
                if model_id not in results["model_ids"]:
                    results["model_ids"][model_id] = {}
                results["model_ids"][model_id][f"curated_memories ({shard_name})"] = count
            conn.close()
        except Exception as e:
            print(f"  [WARN] Could not query {shard}: {e}")

    main_ai = os.path.join(MEMORY_DATA_DIR, "ai_memories.db")
    try:
        conn = sqlite3.connect(main_ai)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT LOWER(model_id), COUNT(*) FROM core_identity "
            "WHERE user_id = ? GROUP BY LOWER(model_id)",
            (user_id,)
        )
        for model_id, count in cursor.fetchall():
            model_id = model_id.strip()
            if model_id not in results["model_ids"]:
                results["model_ids"][model_id] = {}
            results["model_ids"][model_id]["core_identity"] = count
        conn.close()
    except Exception:
        pass

    conv_db = os.path.join(MEMORY_DATA_DIR, "conversations.db")
    if os.path.exists(conv_db):
        try:
            conn = sqlite3.connect(conv_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT LOWER(model_id), COUNT(*) FROM conversations "
                "WHERE user_id = ? GROUP BY LOWER(model_id)",
                (user_id,)
            )
            for model_id, count in cursor.fetchall():
                model_id = model_id.strip()
                if model_id not in results["model_ids"]:
                    results["model_ids"][model_id] = {}
                results["model_ids"][model_id]["conversations"] = count
            conn.close()
        except Exception:
            pass

    char_db = os.path.join(MEMORY_DATA_DIR, "conversation_characters.db")
    if os.path.exists(char_db):
        try:
            conn = sqlite3.connect(char_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT LOWER(character_name), COUNT(*) FROM conversation_characters "
                "GROUP BY LOWER(character_name)"
            )
            for char_name, count in cursor.fetchall():
                char_name = char_name.strip()
                if char_name not in results["character_names"]:
                    results["character_names"][char_name] = {}
                results["character_names"][char_name]["conversation_characters"] = count
            cursor.execute(
                "SELECT LOWER(model_card_name), COUNT(*) FROM conversation_characters "
                "WHERE model_card_name IS NOT NULL AND model_card_name != '' "
                "GROUP BY LOWER(model_card_name)"
            )
            for card_name, count in cursor.fetchall():
                card_name = card_name.strip()
                if card_name not in results["character_names"]:
                    results["character_names"][card_name] = {}
                results["character_names"][card_name]["conversation_characters (card)"] = count
            conn.close()
        except Exception:
            pass

    if os.path.exists(WEBUI_DB_PATH):
        try:
            conn = sqlite3.connect(WEBUI_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM memory WHERE user_id = ?", (user_id,))
            model_counts = {}
            for (content,) in cursor.fetchall():
                model_name = parse_model_from_content(content)
                if model_name:
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1
            for model_name, count in model_counts.items():
                if model_name not in results["model_ids"]:
                    results["model_ids"][model_name] = {}
                results["model_ids"][model_name]["owui_memory_table"] = count
            conn.close()
        except Exception:
            pass

    return results


def search_model(user_id: str, query: str) -> Dict:
    q = query.strip().lower()
    results = discover_models(user_id)
    filtered = {"model_ids": {}, "character_names": {}}
    for mid, targets in results["model_ids"].items():
        if q in mid:
            filtered["model_ids"][mid] = targets
    for cname, targets in results["character_names"].items():
        if q in cname:
            filtered["character_names"][cname] = targets
    return filtered


def count_by_model(user_id: str, model_name: str) -> Dict[str, int]:
    counts = {}
    model_lower = model_name.strip().lower()

    mem_count = 0
    for shard in discover_ai_memory_shards():
        try:
            conn = sqlite3.connect(shard)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM curated_memories "
                "WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            mem_count += cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
    if mem_count:
        counts["curated_memories"] = mem_count

    main_ai = os.path.join(MEMORY_DATA_DIR, "ai_memories.db")
    try:
        conn = sqlite3.connect(main_ai)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM core_identity "
            "WHERE user_id = ? AND LOWER(model_id) = ?",
            (user_id, model_lower)
        )
        ci_count = cursor.fetchone()[0]
        conn.close()
        if ci_count:
            counts["core_identity"] = ci_count
    except Exception:
        pass

    conv_db = os.path.join(MEMORY_DATA_DIR, "conversations.db")
    if os.path.exists(conv_db):
        try:
            conn = sqlite3.connect(conv_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM conversations "
                "WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            conv_count = cursor.fetchone()[0]
            if conv_count:
                counts["conversations"] = conv_count
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            msg_count = cursor.fetchone()[0]
            if msg_count:
                counts["messages"] = msg_count
            conn.close()
        except Exception:
            pass

    return counts


def delete_by_model_all(user_id: str, model_name: str, dry_run: bool = True) -> Dict[str, int]:
    deleted = {}
    model_lower = model_name.strip().lower()

    target_name = "curated_memories"
    total_del = 0
    for shard in discover_ai_memory_shards():
        try:
            conn = sqlite3.connect(shard)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM curated_memories "
                "WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            count = cursor.fetchone()[0]
            if count > 0:
                if not dry_run:
                    cursor.execute(
                        "DELETE FROM curated_memories "
                        "WHERE user_id = ? AND LOWER(model_id) = ?",
                        (user_id, model_lower)
                    )
                    conn.commit()
                total_del += count
            conn.close()
        except Exception:
            pass
    if total_del:
        deleted[target_name] = total_del

    main_ai = os.path.join(MEMORY_DATA_DIR, "ai_memories.db")
    try:
        conn = sqlite3.connect(main_ai)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM core_identity "
            "WHERE user_id = ? AND LOWER(model_id) = ?",
            (user_id, model_lower)
        )
        count = cursor.fetchone()[0]
        if count > 0:
            if not dry_run:
                cursor.execute(
                    "DELETE FROM core_identity "
                    "WHERE user_id = ? AND LOWER(model_id) = ?",
                    (user_id, model_lower)
                )
                conn.commit()
            deleted["core_identity"] = count
        conn.close()
    except Exception:
        pass

    conv_db = os.path.join(MEMORY_DATA_DIR, "conversations.db")
    if os.path.exists(conv_db):
        try:
            conn = sqlite3.connect(conv_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM conversations "
                "WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            conv_count = cursor.fetchone()[0]
            if conv_count > 0:
                if not dry_run:
                    cursor.execute(
                        "DELETE FROM conversations "
                        "WHERE user_id = ? AND LOWER(model_id) = ?",
                        (user_id, model_lower)
                    )
                    conn.commit()
                deleted["conversations"] = conv_count
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE user_id = ? AND LOWER(model_id) = ?",
                (user_id, model_lower)
            )
            msg_count = cursor.fetchone()[0]
            if msg_count > 0:
                if not dry_run:
                    cursor.execute(
                        "DELETE FROM messages WHERE user_id = ? AND LOWER(model_id) = ?",
                        (user_id, model_lower)
                    )
                    conn.commit()
                deleted["messages"] = msg_count
            conn.close()
        except Exception:
            pass

    return deleted


def show_discovered(user_id: str, results: Dict):
    print(f"\nDiscovered models for user {user_id}:")
    print("=" * 60)
    models = results.get("model_ids", {})
    chars = results.get("character_names", {})

    if models:
        print("\nModel IDs (found in curated_memories, core_identity, conversations, etc.):")
        sorted_models = sorted(models.items(), key=lambda x: sum(x[1].values()), reverse=True)
        for model_id, targets in sorted_models:
            total = sum(targets.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(targets.items()))
            print(f"  {model_id:<30} total={total:<5} [{detail}]")
    else:
        print("\nNo model IDs found.")

    if chars:
        print("\nCharacter names (found in conversation_characters):")
        sorted_chars = sorted(chars.items(), key=lambda x: sum(x[1].values()), reverse=True)
        for char_name, targets in sorted_chars:
            total = sum(targets.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(targets.items()))
            print(f"  {char_name:<30} total={total:<5} [{detail}]")
    else:
        print("\nNo character names found.")
    print("=" * 60)


def interactive_menu():
    print("\n" + "=" * 60)
    print("PAM MODEL MEMORY DELETION TOOL")
    print("=" * 60)
    print("\nThis tool discovers model-associated memories and allows")
    print("you to delete them with a dry-run preview.")

    users = get_all_owui_users()

    if users:
        print(f"\nDetected {len(users)} user(s) from the system:")
        for uid, info in users.items():
            print(f"  {uid}  ({info.get('name', '?')})")
    else:
        print("\nNo users found. You'll need to enter a user ID manually.")

    user_id = input("\nEnter user ID (or press Enter to use 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"

    print("\nDiscovering models... (this may take a moment)")
    all_results = discover_models(user_id)
    show_discovered(user_id, all_results)

    while True:
        print("\n" + "-" * 60)
        print("OPTIONS:")
        print("  1  Refresh discovery")
        print("  2  Search/filter models")
        print("  3  Dry-run delete for a model")
        print("  4  Actually delete a model (REAL deletion)")
        print("  5  Delete a character name")
        print("  6  Exit")
        print("-" * 60)

        choice = input("\nSelect option (1-6) [6]: ").strip() or "6"

        if choice == "1":
            all_results = discover_models(user_id)
            show_discovered(user_id, all_results)
        elif choice == "2":
            query = input("Search query: ").strip()
            if query:
                filtered = search_model(user_id, query)
                show_discovered(user_id, filtered)
        elif choice == "3":
            model_name = input("Model name to dry-run delete: ").strip()
            if model_name:
                counts = count_by_model(user_id, model_name)
                if counts:
                    print(f"\nDry-run: Would delete for model '{model_name}':")
                    total = 0
                    for target, count in sorted(counts.items()):
                        print(f"  {target}: {count}")
                        total += count
                    print(f"  TOTAL: {total} records")
                else:
                    print(f"No records found for model '{model_name}'")
        elif choice == "4":
            model_name = input("Model name to REALLY delete: ").strip()
            if model_name:
                counts = count_by_model(user_id, model_name)
                if not counts:
                    print(f"No records found for model '{model_name}'")
                    continue
                print(f"\nWARNING: This will permanently delete ALL records for '{model_name}':")
                total = 0
                for target, count in sorted(counts.items()):
                    print(f"  {target}: {count}")
                    total += count
                print(f"  TOTAL: {total} records")
                confirm = input(f"\nType the model name to confirm: ").strip()
                if confirm.lower() == model_name.lower():
                    deleted = delete_by_model_all(user_id, model_name, dry_run=False)
                    msg = f"Deleted {sum(deleted.values())} records for model '{model_name}': {deleted}"
                    print(f"\n{msg}")
                    log_action(msg)
                    all_results = discover_models(user_id)
                else:
                    print("Confirmation failed. No deletions performed.")
        elif choice == "5":
            print("This feature removes character records. Use with caution.")
            char_name = input("Character name to delete: ").strip()
            if char_name:
                char_lower = char_name.strip().lower()
                char_db = os.path.join(MEMORY_DATA_DIR, "conversation_characters.db")
                if os.path.exists(char_db):
                    try:
                        conn = sqlite3.connect(char_db)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM conversation_characters "
                            "WHERE LOWER(character_name) = ?",
                            (char_lower,)
                        )
                        count = cursor.fetchone()[0]
                        if count > 0:
                            print(f"\nFound {count} record(s) for character '{char_name}'")
                            confirm = input("Dry-run? (y/n) [y]: ").strip().lower() or "y"
                            if confirm == "y":
                                print(f"Would delete {count} record(s)")
                            else:
                                confirm2 = input(f"Type '{char_name}' to confirm deletion: ").strip()
                                if confirm2.lower() == char_lower:
                                    cursor.execute(
                                        "DELETE FROM conversation_characters "
                                        "WHERE LOWER(character_name) = ?",
                                        (char_lower,)
                                    )
                                    conn.commit()
                                    msg = f"Deleted {count} character records for '{char_name}'"
                                    print(f"\n{msg}")
                                    log_action(msg)
                                else:
                                    print("Confirmation failed.")
                        else:
                            print(f"No records found for character '{char_name}'")
                        conn.close()
                    except Exception as e:
                        print(f"Error: {e}")
        elif choice == "6":
            print("Exiting.")
            break


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--user-id":
        user_id = sys.argv[2] if len(sys.argv) > 2 else None
        if user_id:
            results = discover_models(user_id)
            show_discovered(user_id, results)
        else:
            interactive_menu()
    else:
        interactive_menu()