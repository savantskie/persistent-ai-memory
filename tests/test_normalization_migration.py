"""
Tests for the AI Memory Normalization Migration module.

This test suite verifies that the MemoryNormalizationMigration class correctly:
- Instantiates per user/model pair
- Detects and normalizes malformed tags
- Tracks migration completion per user/model combo
- Calls callbacks with proper user_id/model_id parameters
- Creates proper marker files
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_memory_normalization_migration import MemoryNormalizationMigration


class TestMemoryNormalizationMigrationInstantiation:
    """Test instantiation and initialization of migration instances."""

    def test_init_with_defaults(self):
        """Test initialization with default paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ai_memory_normalization_migration.get_memory_data_dir", return_value=tmpdir):
                with patch("ai_memory_normalization_migration.get_log_dir", return_value=tmpdir):
                    migration = MemoryNormalizationMigration(
                        user_id="test_user",
                        model_id="test_model"
                    )
                    
                    assert migration.user_id == "test_user"
                    assert migration.model_id == "test_model"
                    assert "test_user" in migration.migration_marker_path
                    assert "test_model" in migration.migration_marker_path

    def test_init_with_custom_marker_path(self):
        """Test initialization with custom marker path."""
        custom_path = "/tmp/custom_marker.json"
        migration = MemoryNormalizationMigration(
            user_id="test_user",
            model_id="test_model",
            migration_marker_path=custom_path
        )
        
        assert migration.migration_marker_path == custom_path
        assert migration.user_id == "test_user"
        assert migration.model_id == "test_model"

    def test_per_user_model_isolation(self):
        """Test that different user/model pairs get different marker paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ai_memory_normalization_migration.get_memory_data_dir", return_value=tmpdir):
                with patch("ai_memory_normalization_migration.get_log_dir", return_value=tmpdir):
                    migration1 = MemoryNormalizationMigration(
                        user_id="user1",
                        model_id="model1"
                    )
                    migration2 = MemoryNormalizationMigration(
                        user_id="user1",
                        model_id="model2"
                    )
                    migration3 = MemoryNormalizationMigration(
                        user_id="user2",
                        model_id="model1"
                    )
                    
                    # All should have different marker paths
                    paths = {migration1.migration_marker_path, migration2.migration_marker_path, migration3.migration_marker_path}
                    assert len(paths) == 3


class TestMigrationCompletion:
    """Test migration completion tracking."""

    def test_has_completed_false_initially(self):
        """Test that has_completed returns False for new instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed_test1_test2")
            migration = MemoryNormalizationMigration(
                user_id="test1",
                model_id="test2",
                migration_marker_path=marker_path
            )
            
            assert not migration.has_completed()

    def test_mark_completed_creates_file(self):
        """Test that mark_completed() creates marker file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed_test1_test2")
            migration = MemoryNormalizationMigration(
                user_id="test1",
                model_id="test2",
                migration_marker_path=marker_path
            )
            
            migration.mark_completed()
            
            assert Path(marker_path).exists()
            with open(marker_path) as f:
                data = json.load(f)
                assert "completed_at" in data

    def test_has_completed_true_after_marking(self):
        """Test that has_completed returns True after mark_completed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed_test1_test2")
            migration = MemoryNormalizationMigration(
                user_id="test1",
                model_id="test2",
                migration_marker_path=marker_path
            )
            
            migration.mark_completed()
            assert migration.has_completed()


class MockMemory:
    """Mock memory object for testing."""

    def __init__(self, memory_id, content, user_id="test_user"):
        self.id = memory_id
        self.content = content
        self.user_id = user_id


class TestNormalizationLogic:
    """Test the normalization logic."""

    @pytest.mark.asyncio
    async def test_normalize_model_tag(self):
        """Test normalizing Model tag from uppercase to lowercase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "This is a memory [Model: GPT4] about something"),
            ]
            
            updated_memories = {}
            
            async def query_func(uid, mid):
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                updated_memories[mem_id] = content
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 1
            assert "[Model: gpt4]" in updated_memories.get("1", "")

    @pytest.mark.asyncio
    async def test_normalize_user_tag(self):
        """Test normalizing User tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "This is for [User: JOHN] here"),
            ]
            
            updated_memories = {}
            
            async def query_func(uid, mid):
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                updated_memories[mem_id] = content
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 1
            assert "[User: john]" in updated_memories.get("1", "")

    @pytest.mark.asyncio
    async def test_normalize_bank_tag(self):
        """Test normalizing Memory Bank tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "File this in [Memory Bank: WORK] section"),
            ]
            
            updated_memories = {}
            
            async def query_func(uid, mid):
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                updated_memories[mem_id] = content
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 1
            assert "[Memory Bank: work]" in updated_memories.get("1", "")

    @pytest.mark.asyncio
    async def test_multiple_tags_normalized(self):
        """Test normalizing multiple tags in one memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "Info for [User: JOHN] [Model: GPT4] [Memory Bank: PERSONAL]"),
            ]
            
            updated_memories = {}
            
            async def query_func(uid, mid):
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                updated_memories[mem_id] = content
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 1
            content = updated_memories.get("1", "")
            assert "[User: john]" in content
            assert "[Model: gpt4]" in content
            assert "[Memory Bank: personal]" in content

    @pytest.mark.asyncio
    async def test_no_normalization_needed(self):
        """Test memories that don't need normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "Already normalized [User: john] [Model: gpt4]"),
            ]
            
            updated_memories = {}
            
            async def query_func(uid, mid):
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                updated_memories[mem_id] = content
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 0
            assert stats["unchanged"] == 1

    @pytest.mark.asyncio
    async def test_empty_memories(self):
        """Test migration with no memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            async def query_func(uid, mid):
                return []
            
            async def update_func(uid, mid, mem_id, content):
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="test_user",
                model_id="test_model",
                migration_marker_path=marker_path
            )
            
            stats = await migration.run_migration(query_func, update_func)
            
            assert stats["migrated"] == 0
            assert migration.has_completed()


class TestCallbackSignatures:
    """Test that callbacks receive correct parameters."""

    @pytest.mark.asyncio
    async def test_callback_receives_correct_user_model_params(self):
        """Test that update callback receives user_id and model_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "[Model: GPT4] test"),
            ]
            
            callback_params = {}
            
            async def query_func(uid, mid):
                callback_params["query_uid"] = uid
                callback_params["query_mid"] = mid
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                callback_params["update_uid"] = uid
                callback_params["update_mid"] = mid
                callback_params["update_mem_id"] = mem_id
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="alice",
                model_id="gpt4",
                migration_marker_path=marker_path
            )
            
            await migration.run_migration(query_func, update_func)
            
            assert callback_params.get("query_uid") == "alice"
            assert callback_params.get("query_mid") == "gpt4"
            assert callback_params.get("update_uid") == "alice"
            assert callback_params.get("update_mid") == "gpt4"
            assert callback_params.get("update_mem_id") == "1"

    @pytest.mark.asyncio
    async def test_migration_completes_only_once(self):
        """Test that migration doesn't run twice for same user/model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            marker_path = os.path.join(tmpdir, ".migration_completed")
            
            memories = [
                MockMemory("1", "[Model: GPT4] test"),
            ]
            
            query_count = {"count": 0}
            
            async def query_func(uid, mid):
                query_count["count"] += 1
                return memories
            
            async def update_func(uid, mid, mem_id, content):
                return True
            
            migration = MemoryNormalizationMigration(
                user_id="alice",
                model_id="gpt4",
                migration_marker_path=marker_path
            )
            
            # First run
            await migration.run_migration(query_func, update_func)
            
            # Create new instance with same user/model 
            migration2 = MemoryNormalizationMigration(
                user_id="alice",
                model_id="gpt4",
                migration_marker_path=marker_path
            )
            
            # Second run should skip
            await migration2.run_migration(query_func, update_func)
            
            # Should only query once
            assert query_count["count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
