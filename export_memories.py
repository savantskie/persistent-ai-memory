#!/usr/bin/env python3
"""
Export AI Memory System Memories Tool

Exports memories from short-term memory to a formatted text file.
Preserves the exact formatting and structure as seen in the GUI.

Usage:
    python export_memories.py --count 50 --output my_memories.txt
    python export_memories.py --filter-days 7 --sort newest
    python export_memories.py --filter-date "2025-01-01" --user-id user123
"""

import sqlite3
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo


def get_local_timezone():
    try:
        import time
        return ZoneInfo(time.tzname[0])
    except:
        return ZoneInfo("America/Chicago")


class MemoryExporter:
    """Handles extraction and export of memories"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get(
                "AI_MEMORY_WEBUI_DB",
                os.path.join(os.environ.get("AI_MEMORY_DATA_DIR", "./memory_data"), "webui.db")
            )
        self.db_path = db_path
        self.tz = get_local_timezone()
        
    def _unix_to_datetime(self, unix_timestamp: int) -> datetime:
        dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
        return dt.astimezone(self.tz)
    
    def _format_datetime(self, unix_timestamp: int) -> str:
        dt = self._unix_to_datetime(unix_timestamp)
        return dt.strftime("%B %d, %Y %I:%M %p")
    
    def _get_days_ago(self, unix_timestamp: int) -> int:
        dt = self._unix_to_datetime(unix_timestamp)
        today = datetime.now(self.tz).date()
        memory_date = dt.date()
        delta = today - memory_date
        return delta.days
    
    def _parse_tags_from_content(self, content: str) -> Tuple[List[str], str]:
        if content.startswith("[Tags:"):
            end_bracket = content.find("]")
            if end_bracket != -1:
                tags_str = content[7:end_bracket].strip()
                tags = [t.strip() for t in tags_str.split(",")]
                remaining_content = content[end_bracket+1:].strip()
                return tags, remaining_content
        return [], content
    
    def _format_memory_preview(self, content: str, max_length: int = 70) -> str:
        preview = content.replace("\n", " ")
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        return preview
    
    def get_memories(
        self, 
        count: Optional[int] = None,
        filter_days: Optional[int] = None,
        filter_date: Optional[str] = None,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        sort_order: str = "newest"
    ) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT id, user_id, content, created_at, updated_at FROM memory WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if model_id:
                query += " AND content LIKE ?"
                params.append(f"%[Model: {model_id}]%")
            
            if not count:
                if filter_days is not None:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=filter_days)
                    cutoff_seconds = int(cutoff_date.timestamp())
                    query += " AND created_at >= ?"
                    params.append(cutoff_seconds)
                elif filter_date:
                    try:
                        target_date = datetime.fromisoformat(filter_date)
                        cutoff_seconds = int(target_date.timestamp())
                        query += " AND created_at >= ?"
                        params.append(cutoff_seconds)
                    except ValueError:
                        print(f"Invalid date format: {filter_date}. Use YYYY-MM-DD")
                        return []
            
            if count:
                query += " ORDER BY created_at DESC"
                query += f" LIMIT {count}"
            else:
                if sort_order == "oldest":
                    query += " ORDER BY created_at ASC"
                elif sort_order == "alpha":
                    query += " ORDER BY content ASC"
                else:
                    query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            memories = [dict(row) for row in rows]
            
            if count:
                if sort_order == "oldest":
                    memories.sort(key=lambda x: x['created_at'])
                elif sort_order == "alpha":
                    memories.sort(key=lambda x: x['content'])
            
            conn.close()
            return memories
            
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
    
    def export_to_file(
        self,
        memories: List[Dict],
        output_file: str = "exported_memories.txt",
        include_dates: bool = False
    ) -> bool:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EXPORTED MEMORIES\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total Memories: {len(memories)}\n")
                f.write("=" * 80 + "\n\n")
                
                if not memories:
                    f.write("No memories found.\n")
                    print(f"Exported 0 memories to {output_file}")
                    return True
                
                for idx, memory in enumerate(memories, 1):
                    content = memory['content']
                    tags, content_clean = self._parse_tags_from_content(content)
                    
                    if tags:
                        f.write(f"[Tags: {', '.join(tags)}] ")
                    
                    f.write(f"{content_clean}\n")
                    
                    if include_dates:
                        formatted_date = self._format_datetime(memory['created_at'])
                        f.write(f"\nDate: {formatted_date}\n")
                    
                    f.write("\n" + ("=" * 80) + "\n\n")
                
                print(f"Exported {len(memories)} memories to {output_file}")
                return True
                
        except IOError as e:
            print(f"File write error: {e}")
            return False
    
    def export(
        self,
        count: Optional[int] = None,
        filter_days: Optional[int] = None,
        filter_date: Optional[str] = None,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        sort_order: str = "newest",
        output_file: str = "exported_memories.txt",
        include_dates: bool = False
    ) -> bool:
        print(f"Querying memories from {self.db_path}...")
        memories = self.get_memories(
            count=count,
            filter_days=filter_days,
            filter_date=filter_date,
            user_id=user_id,
            model_id=model_id,
            sort_order=sort_order
        )
        
        if not memories:
            print("No memories found matching criteria.")
            return False
        
        print(f"Found {len(memories)} memories. Exporting...")
        return self.export_to_file(memories, output_file, include_dates)


def interactive_menu():
    print("\n" + "=" * 80)
    print("MEMORY EXPORTER")
    print("=" * 80)
    
    exporter = MemoryExporter()
    
    print("\nFiltering Options:")
    print("1. Export most recent memories (by count)")
    print("2. Export memories from last N days")
    print("3. Export memories from specific date onward")
    print("4. Export all memories")
    
    filter_choice = input("\nSelect filtering option (1-4) [1]: ").strip() or "1"
    
    count = None
    filter_days = None
    filter_date = None
    
    if filter_choice == "1":
        count_input = input("How many recent memories? [50]: ").strip() or "50"
        try:
            count = int(count_input)
        except ValueError:
            print("Invalid number, using default 50")
            count = 50
    elif filter_choice == "2":
        days_input = input("How many days back? [7]: ").strip() or "7"
        try:
            filter_days = int(days_input)
        except ValueError:
            print("Invalid number, using default 7")
            filter_days = 7
    elif filter_choice == "3":
        filter_date = input("Enter date (YYYY-MM-DD) [2026-01-01]: ").strip() or "2026-01-01"
    
    print("\nSort Order:")
    print("1. Newest first (default)")
    print("2. Oldest first")
    print("3. Alphabetical")
    
    sort_choice = input("\nSelect sort order (1-3) [1]: ").strip() or "1"
    sort_map = {"1": "newest", "2": "oldest", "3": "alpha"}
    sort_order = sort_map.get(sort_choice, "newest")
    
    output_file = input("\nOutput filename [exported_memories.txt]: ").strip() or "exported_memories.txt"
    
    include_dates_input = input("\nInclude dates in output? (y/n) [n]: ").strip().lower() or "n"
    include_dates = include_dates_input == "y"
    
    user_id = input("\nFilter by user ID (leave blank for all): ").strip() or None
    model_id = input("Filter by model ID (leave blank for all): ").strip() or None
    
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY:")
    if count:
        print(f"  Count: {count} most recent")
    elif filter_days:
        print(f"  Date Range: Last {filter_days} days")
    elif filter_date:
        print(f"  Date Range: From {filter_date} onward")
    else:
        print("  Count: All memories")
    print(f"  Sort: {sort_order.capitalize()}")
    print(f"  Include Dates: {'Yes' if include_dates else 'No'}")
    if user_id:
        print(f"  User ID Filter: {user_id}")
    if model_id:
        print(f"  Model ID Filter: {model_id}")
    print(f"  Output File: {output_file}")
    print("=" * 80)
    
    confirm = input("\nProceed with export? (y/n) [y]: ").strip().lower() or "y"
    if confirm != "y":
        print("Export cancelled.")
        return
    
    print("\nExporting...")
    success = exporter.export(
        count=count,
        filter_days=filter_days,
        filter_date=filter_date,
        user_id=user_id,
        model_id=model_id,
        sort_order=sort_order,
        output_file=output_file,
        include_dates=include_dates
    )
    
    if success:
        output_path = Path(output_file).resolve()
        print(f"\nFile saved to: {output_path}")
        print("\nWould you like to export again?")
        again = input("(y/n) [n]: ").strip().lower() or "n"
        if again == "y":
            interactive_menu()


def main():
    import sys
    
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Export memories to formatted text file"
        )
        
        parser.add_argument("--count", "-c", type=int, help="Number of most recent memories to export")
        parser.add_argument("--filter-days", "-d", type=int, metavar="N", help="Get memories from last N days")
        parser.add_argument("--filter-date", "-f", metavar="YYYY-MM-DD", help="Get memories from this date forward")
        parser.add_argument("--user-id", "-u", metavar="USER_ID", help="Filter by specific user ID")
        parser.add_argument("--model-id", "-m", metavar="MODEL_ID", help="Filter by specific model ID")
        parser.add_argument("--sort", "-s", choices=["newest", "oldest", "alpha"], default="newest", help="Sort order")
        parser.add_argument("--output", "-o", default="exported_memories.txt", help="Output file path")
        parser.add_argument("--dates", action="store_true", help="Include creation dates for each memory")
        
        args = parser.parse_args()
        
        exporter = MemoryExporter()
        success = exporter.export(
            count=args.count,
            filter_days=args.filter_days,
            filter_date=args.filter_date,
            user_id=args.user_id,
            model_id=args.model_id,
            sort_order=args.sort,
            output_file=args.output,
            include_dates=args.dates
        )
        
        if success:
            output_path = Path(args.output).resolve()
            print(f"File saved to: {output_path}")
        else:
            exit(1)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()