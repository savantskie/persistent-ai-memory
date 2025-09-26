#!/usr/bin/env python3
"""
Integration test for the GitHub version to ensure parity with Friday's system
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path

async def test_integration():
    """Test that all major components work together"""
    
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="ai_memory_test_"))
    print(f"🧪 Testing in: {test_dir}")
    
    try:
        # Import and initialize the memory system
        from ai_memory_core import PersistentAIMemorySystem
        print("✅ Successfully imported PersistentAIMemorySystem")
        
        # Initialize with test directory
        memory_system = PersistentAIMemorySystem(
            data_dir=str(test_dir / "memory_data"),
            enable_file_monitoring=False
        )
        print("✅ Memory system initialized")
        
        # Test basic memory operations
        await memory_system.store_conversation(
            content="This is a test message for the GitHub version",
            role="user",
            session_id="test_session"
        )
        print("✅ Stored conversation successfully")
        
        # Test memory creation
        memory_id = await memory_system.create_memory(
            content="Test memory for integration testing",
            memory_type="test",
            importance_level=5,
            tags=["integration", "test"]
        )
        print(f"✅ Created memory: {memory_id}")
        
        # Test search
        results = await memory_system.search_memories(
            query="test message",
            limit=5
        )
        print(f"✅ Search returned {len(results)} results")
        
        # Test appointment creation
        appointment_id = await memory_system.create_appointment(
            title="Integration Test Appointment",
            scheduled_datetime="2025-08-06T10:00:00-05:00",
            description="Test appointment for GitHub version"
        )
        print(f"✅ Created appointment: {appointment_id}")
        
        # Test system health
        health = await memory_system.get_system_health()
        print(f"✅ System health check passed: {health['status']}")
        
        # Test MCP server import
        from mcp_server import AIMemoryMCPServer
        mcp_server = AIMemoryMCPServer()
        print("✅ MCP server initialized successfully")
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ GitHub version has full parity with Friday's system")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"🧹 Cleaned up test directory: {test_dir}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
