#!/usr/bin/env python3
"""Test the new database maintenance system"""

import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def test_maintenance():
    print("🧹 Testing Database Maintenance System...")
    
    memory = PersistentAIMemorySystem()
    
    # Get system health before maintenance
    print("\n📊 System Health Before Maintenance:")
    health_before = await memory.get_system_health()
    for db_name, info in health_before["databases"].items():
        if "count" in str(info):
            print(f"  {db_name}: {info}")
    
    # Run maintenance
    print("\n🔧 Running Database Maintenance...")
    maintenance_result = await memory.run_database_maintenance(force=True)
    
    if "error" in maintenance_result:
        print(f"❌ Maintenance failed: {maintenance_result['error']}")
        if maintenance_result.get("manual_cleanup_recommended"):
            print("💡 Manual cleanup steps:")
            print("   1. Check database file sizes in memory_data/")
            print("   2. Consider manually deleting old records")
            print("   3. Run VACUUM on SQLite files if needed")
    else:
        print("✅ Maintenance completed successfully!")
        
        # Show cleanup results
        if "cleanup_results" in maintenance_result:
            print("\n🗑️ Cleanup Results:")
            for category, results in maintenance_result["cleanup_results"].items():
                if isinstance(results, dict) and "deleted" in str(results):
                    print(f"  {category}: {results}")
        
        # Show optimization results
        if "optimization_results" in maintenance_result:
            print("\n⚡ Optimization Results:")
            for db_name, results in maintenance_result["optimization_results"].items():
                if "space_saved_mb" in results:
                    print(f"  {db_name}: Saved {results['space_saved_mb']} MB")
    
    # Get system health after maintenance
    print("\n📊 System Health After Maintenance:")
    health_after = await memory.get_system_health()
    for db_name, info in health_after["databases"].items():
        if "count" in str(info):
            print(f"  {db_name}: {info}")

if __name__ == "__main__":
    asyncio.run(test_maintenance())
