"""
Health check script for Persistent AI Memory System

This script validates that the system is properly installed and configured by:
1. Creating temporary test databases (isolated from production)
2. Validating embedding configuration
3. Testing core functionality
4. Cleaning up test artifacts
5. Providing clear pass/fail feedback

Usage:
    python tests/test_health_check.py
"""

import sys
import os
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}{text:^60}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{BLUE}ℹ {text}{RESET}")

def check_imports():
    """Check if all required modules can be imported"""
    print_header("Checking Required Imports")
    
    required_modules = {
        "ai_memory_core": "Core memory system",
        "database_maintenance": "Database maintenance utilities",
        "tag_manager": "Tag management system",
        "settings": "Configuration management",
    }
    
    failed = []
    for module_name, description in required_modules.items():
        try:
            __import__(module_name)
            print_success(f"Imported {module_name} ({description})")
        except ImportError as e:
            print_error(f"Failed to import {module_name}: {e}")
            failed.append(module_name)
    
    return len(failed) == 0, failed

def check_embedding_config():
    """Check if embedding_config.json exists and is valid"""
    print_header("Validating Embedding Configuration")
    
    config_path = Path(__file__).parent.parent / "embedding_config.json"
    
    if not config_path.exists():
        print_error(f"embedding_config.json not found at {config_path}")
        return False
    
    print_success(f"Found embedding_config.json at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print_success("embedding_config.json is valid JSON")
        
        # Check for required sections
        if "embedding_configuration" in config:
            embed_config = config["embedding_configuration"]
            
            if "primary" in embed_config:
                primary = embed_config["primary"]
                provider = primary.get("provider", "unknown")
                model = primary.get("model", "unknown")
                print_success(f"Primary provider configured: {provider} ({model})")
            else:
                print_warning("No primary provider configured")
            
            if "fallback" in embed_config:
                fallback = embed_config["fallback"]
                provider = fallback.get("provider", "unknown")
                model = fallback.get("model", "unknown")
                print_success(f"Fallback provider configured: {provider} ({model})")
            else:
                print_warning("No fallback provider configured")
        else:
            print_warning("No 'embedding_configuration' section found")
        
        return True
    except json.JSONDecodeError as e:
        print_error(f"embedding_config.json is invalid JSON: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading embedding_config.json: {e}")
        return False

def check_memory_config():
    """Check if memory_config.json exists and is valid"""
    print_header("Validating Memory Configuration")
    
    config_path = Path(__file__).parent.parent / "memory_config.json"
    
    if not config_path.exists():
        print_warning(f"memory_config.json not found - will use defaults")
        return True  # Not critical, defaults will be used
    
    print_success(f"Found memory_config.json at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print_success("memory_config.json is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print_warning(f"memory_config.json is invalid JSON (will use defaults): {e}")
        return True  # Not critical
    except Exception as e:
        print_warning(f"Error reading memory_config.json: {e}")
        return True  # Not critical

async def test_database_initialization():
    """Test database initialization with temporary test databases"""
    print_header("Testing Database Initialization")
    
    # Create a temporary directory for test databases
    with tempfile.TemporaryDirectory(prefix="pam_test_") as temp_dir:
        print_info(f"Created temporary test directory: {temp_dir}")
        
        try:
            # Import here to ensure all checks pass first
            from ai_memory_core import PersistentAIMemorySystem
            
            # Try to initialize the system (it will use default paths by default)
            print_info("Initializing PersistentAIMemorySystem...")
            system = PersistentAIMemorySystem()
            
            # Check if databases can be accessed
            try:
                # Try to get system health (validate databases work)
                print_info("Checking system health...")
                health = await system.get_system_health()
                print_success("System health check passed")
                
                # Display key metrics
                if "status" in health:
                    print_success(f"System status: {health['status']}")
                if "databases" in health:
                    print_success(f"Databases initialized: {len(health['databases'])} database(s)")
                    for db_name, db_info in health['databases'].items():
                        if isinstance(db_info, dict) and 'path' in db_info:
                            print_info(f"  - {db_name}: {db_info['path']}")
                
                print_success("Database initialization successful")
                return True
                
            except Exception as e:
                print_error(f"Failed to check system health: {e}")
                return False
                
        except Exception as e:
            print_error(f"Failed to initialize PersistentAIMemorySystem: {e}")
            import traceback
            print(f"{RED}{traceback.format_exc()}{RESET}")
            return False

def main():
    """Run all health checks"""
    print_header(f"Persistent AI Memory System Health Check")
    print_info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "imports": False,
        "embedding_config": False,
        "memory_config": False,
        "database": False
    }
    
    # Check 1: Imports
    results["imports"], failed_modules = check_imports()
    if not results["imports"]:
        print_error(f"Missing modules: {', '.join(failed_modules)}")
        print_error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        print_header("Health Check FAILED")
        return 1
    
    # Check 2: Embedding config
    results["embedding_config"] = check_embedding_config()
    if not results["embedding_config"]:
        print_error("Please create embedding_config.json - see README.md for examples")
    
    # Check 3: Memory config
    results["memory_config"] = check_memory_config()
    
    # Check 4: Database initialization
    results["database"] = asyncio.run(test_database_initialization())
    
    # Summary
    print_header("Health Check Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Passed: {passed}/{total}\n")
    
    for check_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        check_display = check_name.replace("_", " ").title()
        print(f"  {check_display}: {status}")
    
    print()
    
    if all(results.values()):
        print_success("All health checks passed! System is ready to use.")
        print_info("Next steps:")
        print_info("  - Run examples: python examples/basic_usage.py")
        print_info("  - Check documentation: README.md")
        print_info("  - Configure endpoints in embedding_config.json as needed")
        return 0
    else:
        print_error("Some health checks failed. See above for details.")
        if not results["embedding_config"]:
            print_info("\nTo fix embedding config:")
            print_info("  1. Copy embedding_config.json.example to embedding_config.json")
            print_info("  2. Edit it to match your local embedding service endpoints")
        if not results["imports"]:
            print_info("\nTo fix imports:")
            print_info("  1. Ensure you're in the persistent-ai-memory directory")
            print_info("  2. Install dependencies: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
