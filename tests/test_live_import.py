#!/usr/bin/env python3
"""Live test of file monitoring with conversation import"""

import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def test_live_import():
    print('🎯 Testing Live Conversation Import')
    print('=' * 50)
    
    # Initialize with file monitoring
    memory_system = PersistentAIMemorySystem(enable_file_monitoring=True)
    
    print('📊 Before import - checking current database...')
    health_before = await memory_system.get_system_health()
    messages_before = health_before["databases"]["conversations"]["message_count"]
    print(f'💬 Messages in database before: {messages_before}')
    
    print('\n📁 Starting file monitoring (this will import existing conversations)...')
    await memory_system.start_file_monitoring()
    
    # Wait a moment for initial scan to complete
    print('⏳ Waiting for initial scan to complete...')
    await asyncio.sleep(5)
    
    print('\n📊 After import - checking database...')
    health_after = await memory_system.get_system_health()
    messages_after = health_after["databases"]["conversations"]["message_count"]
    print(f'💬 Messages in database after: {messages_after}')
    print(f'📈 New messages imported: {messages_after - messages_before}')
    
    if messages_after > messages_before:
        print('\n🎉 SUCCESS! Conversations were imported!')
        print('📜 Getting recent imported messages...')
        recent = await memory_system.get_recent_context(limit=3)
        
        for i, msg in enumerate(recent["messages"]):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f'  {i+1}. {role_emoji} [{msg["role"]}]: {content_preview}')
            
            # Show metadata if available
            if msg.get("metadata"):
                import json
                metadata = json.loads(msg["metadata"]) if isinstance(msg["metadata"], str) else msg["metadata"]
                if "application" in metadata:
                    print(f'     📍 Source: {metadata["application"]} ({metadata.get("file_type", "unknown")})')
    else:
        print('\n🤔 No new conversations imported. This might mean:')
        print('   - Conversations were already imported previously')
        print('   - No conversation files found in monitored directories')
        print('   - Files were filtered out as non-conversation files')
    
    print(f'\n📁 Monitored directories:')
    for path in health_after["file_monitoring"]["monitored_paths"]:
        print(f'   • {path}')
    
    # Stop monitoring
    await memory_system.stop_file_monitoring()
    print(f'\n✅ Test completed!')

if __name__ == "__main__":
    asyncio.run(test_live_import())
