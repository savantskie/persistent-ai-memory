#!/usr/bin/env python3
"""
Friday Automatic Database Maintenance

Standalone service for automatic database cleanup and optimization.
Can run independently or as part of the MCP server.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

from friday_memory_system import FridayMemorySystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FridayAutoMaintenance:
    """Automatic database maintenance service for Friday"""
    
    def __init__(self, maintenance_interval_hours: int = 3):
        self.memory_system = FridayMemorySystem()
        self.maintenance_interval = maintenance_interval_hours * 60 * 60  # Convert to seconds
        self.running = False
        self._task = None
        
    async def start(self):
        """Start the automatic maintenance service"""
        if self.running:
            logger.warning("Maintenance service is already running")
            return
            
        self.running = True
        logger.info(f"🔧 Starting Friday Auto Maintenance (interval: {self.maintenance_interval//3600}h)")
        
        # Wait a bit after startup before first maintenance
        initial_delay = 300  # 5 minutes
        logger.info(f"⏱️ Initial maintenance will run in {initial_delay//60} minutes")
        
        self._task = asyncio.create_task(self._maintenance_loop(initial_delay))
        
    async def stop(self):
        """Stop the automatic maintenance service"""
        if not self.running:
            return
            
        self.running = False
        logger.info("🛑 Stopping Friday Auto Maintenance...")
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        logger.info("✅ Friday Auto Maintenance stopped")
        
    async def _maintenance_loop(self, initial_delay: int = 0):
        """Main maintenance loop"""
        # Initial delay
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
            
        while self.running:
            try:
                await self._run_maintenance()
                
                # Wait for next maintenance cycle
                logger.info(f"⏰ Next maintenance in {self.maintenance_interval//3600} hours")
                await asyncio.sleep(self.maintenance_interval)
                
            except asyncio.CancelledError:
                logger.info("Maintenance loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in maintenance loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
                
    async def _run_maintenance(self):
        """Run a single maintenance cycle"""
        start_time = datetime.now()
        logger.info("🧹 Starting automatic database maintenance cycle...")
        
        try:
            # Run the maintenance
            result = await self.memory_system.run_database_maintenance()
            
            # Calculate duration
            duration = datetime.now() - start_time
            duration_str = f"{duration.total_seconds():.1f}s"
            
            if result.get("success", True):  # Default to success if key missing
                # Log summary
                optimization_results = result.get("optimization_results", {})
                cleanup_results = result.get("cleanup_results", {})
                
                total_space_saved = sum(
                    db_result.get("space_saved_mb", 0) 
                    for db_result in optimization_results.values()
                    if isinstance(db_result, dict)
                )
                
                total_items_cleaned = sum(
                    cleanup_result.get("conversations_deleted", 0) + 
                    cleanup_result.get("memories_deleted", 0) + 
                    cleanup_result.get("tool_calls_deleted", 0)
                    for cleanup_result in cleanup_results.values()
                    if isinstance(cleanup_result, dict)
                )
                
                logger.info(f"✅ Maintenance completed in {duration_str}")
                logger.info(f"   📦 Space saved: {total_space_saved:.2f} MB")
                logger.info(f"   🗑️ Items cleaned: {total_items_cleaned}")
                logger.info(f"   🗄️ Databases optimized: {len(optimization_results)}")
                
            else:
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"⚠️ Maintenance completed with issues in {duration_str}: {error_msg}")
                
        except Exception as e:
            duration = datetime.now() - start_time
            duration_str = f"{duration.total_seconds():.1f}s"
            logger.error(f"❌ Maintenance failed after {duration_str}: {e}")
            raise
    
    async def run_once(self):
        """Run maintenance once and exit (useful for manual runs or cron jobs)"""
        logger.info("🔧 Running one-time maintenance...")
        await self._run_maintenance()
        logger.info("✅ One-time maintenance completed")


async def main():
    """Main entry point for standalone maintenance service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Friday Automatic Database Maintenance")
    parser.add_argument("--interval", type=int, default=3, 
                       help="Maintenance interval in hours (default: 3)")
    parser.add_argument("--once", action="store_true",
                       help="Run maintenance once and exit")
    parser.add_argument("--immediate", action="store_true",
                       help="Run maintenance immediately (no initial delay)")
    
    args = parser.parse_args()
    
    # Create maintenance service
    if args.once:
        # Run once and exit
        maintenance = FridayAutoMaintenance()
        await maintenance.run_once()
        return
    
    # Run as a service
    maintenance = FridayAutoMaintenance(maintenance_interval_hours=args.interval)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(maintenance.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await maintenance.start()
        
        # Keep running until stopped
        while maintenance.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await maintenance.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)
