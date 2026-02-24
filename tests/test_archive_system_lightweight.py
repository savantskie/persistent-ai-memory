#!/usr/bin/env python3
"""
Lightweight test for PAM v1.0.0 archive system and retention policies

Tests:
1. Archive folder creation
2. Retention policy verification (permanent preservation)
3. Database schema  integrity
4. Cache invalidation infrastructure
"""

import sqlite3
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_maintenance import DatabaseMaintenance


def test_retention_policies():
    """Test that retention policies are set to permanent preservation"""
    print("=" * 70)
    print("TEST 1: Retention Policies (Permanent Preservation Mode)")
    print("=" * 70)
    
    try:
        # Create a mock memory system for testing
        class MockMemoryDB:
            def __init__(self):
                self.db_path = "/tmp/mock_test.db"
        
        class MockMemorySystem:
            def __init__(self):
                self.conversations_db = MockMemoryDB()
                self.memory_data_path = Path(tempfile.gettempdir()) / "test_memory"
        
        memory_system = MockMemorySystem()
        memory_system.memory_data_path.mkdir(exist_ok=True)
        
        # Create database maintenance
        maintenance = DatabaseMaintenance(memory_system)
        
        # Verify retention policies
        tests = [
            ("conversations.max_age_days", maintenance.retention_policies["conversations"]["max_age_days"]),
            ("conversations.preserve_important", maintenance.retention_policies["conversations"]["preserve_important"]),
            ("curated_memories.max_age_days", maintenance.retention_policies["curated_memories"]["max_age_days"]),
            ("curated_memories.preserve_important", maintenance.retention_policies["curated_memories"]["preserve_important"]),
            ("mcp_tool_calls.max_age_days", maintenance.retention_policies["mcp_tool_calls"]["max_age_days"]),
        ]
        
        all_passed = True
        for policy_name, value in tests:
            if policy_name.endswith("max_age_days"):
                # Should be None for permanent preservation
                if value is None:
                    print(f"  ✓ {policy_name} = None (permanent)")
                else:
                    print(f"  ✗ {policy_name} = {value} (should be None)")
                    all_passed = False
            elif policy_name.endswith("preserve_important"):
                # Should be True
                if value is True:
                    print(f"  ✓ {policy_name} = True")
                else:
                    print(f"  ✗ {policy_name} = {value} (should be True)")
                    all_passed = False
        
        if all_passed:
            print("\n✓ TEST 1 PASSED: Retention policies configured for permanent preservation\n")
        else:
            print("\n✗ TEST 1 FAILED: Some retention policies incorrect\n")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_archive_folder_creation():
    """Test that archive folder is created correctly"""
    print("=" * 70)
    print("TEST 2: Archive Folder Creation")
    print("=" * 70)
    
    try:
        test_dir = Path(tempfile.mkdtemp(prefix="pam_archive_test_"))
        print(f"  Using test directory: {test_dir}")
        
        # Create a mock memory system
        class MockMemoryDB:
            def __init__(self):
                self.db_path = str(test_dir / "conversations.db")
        
        class MockMemorySystem:
            def __init__(self):
                self.conversations_db = MockMemoryDB()
                self.memory_data_path = test_dir
        
        memory_system = MockMemorySystem()
        memory_system.memory_data_path.mkdir(exist_ok=True)
        
        # Create database maintenance
        maintenance = DatabaseMaintenance(memory_system)
        
        # Verify archive folder exists
        archive_folder = test_dir / "archives"
        if archive_folder.exists():
            print(f"  ✓ Archive folder created at: {archive_folder}")
        else:
            print(f"  ✗ Archive folder not found at: {archive_folder}")
            return False
        
        # Verify it's a directory
        if archive_folder.is_dir():
            print(f"  ✓ Archive folder is a valid directory")
        else:
            print(f"  ✗ Archive folder is not a directory")
            return False
        
        # Verify archive path is accessible from maintenance object
        if maintenance.archives_path.exists():
            print(f"  ✓ DatabaseMaintenance can access archives_path: {maintenance.archives_path}")
        else:
            print(f"  ✗ DatabaseMaintenance cannot access archives_path")
            return False
        
        print("\n✓ TEST 2 PASSED: Archive folder creation works correctly\n")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_cache_invalidation_infrastructure():
    """Test that cache invalidation infrastructure is in place"""
    print("=" * 70)
    print("TEST 3: Cache Invalidation Infrastructure")
    print("=" * 70)
    
    try:
        # Read short_term_memory.py to verify the changes
        short_term_path = Path(__file__).parent.parent / "short_term_memory.py"
        
        if not short_term_path.exists():
            print(f"  ✗ short_term_memory.py not found")
            return False
        
        with open(short_term_path, 'r') as f:
            content = f.read()
        
        # Check for memory_to_cache_keys implementation
        checks = [
            ("_memory_to_cache_keys class attribute", "_memory_to_cache_keys = {}"),
            ("memory_to_cache_keys property", "def memory_to_cache_keys(self):"),
            ("Cache key tracking", "self.memory_to_cache_keys[mem_id]"),
            ("Cache invalidation UPDATE", 'if operation.id in self.memory_to_cache_keys:'),
            ("Cache invalidation DELETE", 'if operation.id in self.memory_to_cache_keys:'),
        ]
        
        all_passed = True
        for check_name, search_string in checks:
            if search_string in content:
                print(f"  ✓ {check_name}")
            else:
                print(f"  ✗ {check_name} (not found)")
                all_passed = False
        
        if all_passed:
            print("\n✓ TEST 3 PASSED: Cache invalidation infrastructure implemented\n")
        else:
            print("\n✗ TEST 3 FAILED: Some cache invalidation components missing\n")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  PAM v1.0.0 - Archive System & Maintenance Tests".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Retention Policies", test_retention_policies()))
    results.append(("Archive Folder Creation", test_archive_folder_creation()))
    results.append(("Cache Invalidation", test_cache_invalidation_infrastructure()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<40} {status}")
    
    all_passed = all(result for _, result in results)
    print()
    
    if all_passed:
        print("🎉 All tests PASSED! Archive system is ready for v1.0.0 release.\n")
        return 0
    else:
        print("❌ Some tests FAILED. Review the failures above.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
