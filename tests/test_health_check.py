#!/usr/bin/env python3
"""Quick health check for the Persistent AI Memory System"""

import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def main():
    memory_system = PersistentAIMemorySystem()
    
    print("🔍 Persistent AI Memory System Health Check")
    print("=" * 50)
    
    try:
        health = await memory_system.get_system_health()
        
        print(f"🌟 Overall Status: {health.get('status', 'unknown')}")
        print()
        
        # Database stats
        if 'databases' in health:
            print("📊 Database Statistics:")
            db_stats = health['databases']
            for db_name, stats in db_stats.items():
                print(f"   💾 {db_name}:")
                for key, value in stats.items():
                    if key != 'database_path':
                        print(f"      └─ {key}: {value}")
            print()
        
        # Embedding service
        if 'embedding_service' in health:
            print("🔮 Embedding Service:")
            embedding = health['embedding_service']
            print(f"   📍 Status: {embedding.get('status', 'unknown')}")
            print(f"   📐 Dimensions: {embedding.get('embedding_dimensions', 'unknown')}")
            print(f"   🎯 Endpoint: {embedding.get('endpoint', 'unknown')}")
            print()
        
        # File monitoring
        if 'file_monitoring' in health:
            print("📁 File Monitoring:")
            monitoring = health['file_monitoring']
            print(f"   📊 Status: {monitoring.get('status', 'unknown')}")
            print(f"   📂 Paths being monitored: {len(monitoring.get('monitored_paths', []))}")
            for path in monitoring.get('monitored_paths', []):
                print(f"      └─ {path}")
        
        # Show raw health data for debugging
        print()
        print("🔧 Raw health data:")
        for key, value in health.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error during health check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
