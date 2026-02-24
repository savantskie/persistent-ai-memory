"""
title: Persistent AI Short Term Memory v1.0.0 - Universal short term memory system
author: Nathan (Original Friday impl)
version: 1.0.0
---

# Overview

Persistent AI Short Term Memory is a sophisticated plugin that provides **persistent, personalized memory capabilities** for Large Language Models (LLMs) within OpenWebUI. It enables LLMs to remember key information about users across separate conversations, creating a more natural and personalized experience.

The system **dynamically extracts, filters, stores, and retrieves** user-specific information from conversations, then intelligently injects relevant memories into future LLM prompts.

---

# Key Features

1. **Intelligent Memory Extraction**
   - Automatically identifies facts, preferences, relationships, and goals from user messages
   - Categorizes memories with appropriate tags (identity, preference, behavior, relationship, goal, possession)
   - Focuses on user-specific information while filtering out general knowledge or trivia

2. **Multi-layered Filtering Pipeline**
   - Robust JSON parsing with fallback mechanisms for reliable memory extraction
   - Preference statement shortcuts for improved handling of common user likes/dislikes
   - Blacklist/whitelist system to control topic filtering
   - Smart deduplication using both semantic (embedding-based) and text-based similarity

3. **Optimized Memory Retrieval**
   - Vector-based similarity for efficient memory retrieval
   - Optional LLM-based relevance scoring for highest accuracy when needed
   - Performance optimizations to reduce unnecessary LLM calls

4. **Universal Short Term Memory Management**
   - Smart clustering and summarization of related older memories to prevent clutter
   - Intelligent pruning strategies when memory limits are reached
   - Configurable background tasks for maintenance operations

5. **Memory Injection & Output Filtering**
   - Injects contextually relevant memories into LLM prompts
   - Customizable memory display formats (bullet, numbered, paragraph)
   - Filters meta-explanations from LLM responses for cleaner output

6. **Broad LLM Support**
   - Generalized LLM provider configuration supporting both Ollama and OpenAI-compatible APIs
   - Configurable model selection and endpoint URLs
   - Optimized prompts for reliable JSON response parsing

7. **Comprehensive Configuration System**
   - Fine-grained control through "valve" settings
   - Input validation to prevent misconfiguration
   - Per-user configuration options

8. **Memory Banks** – categorize memories into Personal, Work, General (etc.) so retrieval / injection can be focused on a chosen context

---

# Recent Improvements (v3.0)

1. **Optimized Relevance Calculation** - Reduced latency/cost by adding vector-only option and smart LLM call skipping when high confidence
2. **Enhanced Memory Deduplication** - Added embedding-based similarity for more accurate semantic duplicate detection
3. **Intelligent Memory Pruning** - Support for both FIFO and relevance-based pruning strategies when memory limits are reached
4. **Cluster-Based Summarization** - New system to group and summarize related memories by semantic similarity or shared tags
5. **LLM Call Optimization** - Reduced LLM usage through high-confidence vector similarity thresholds
6. **Resilient JSON Parsing** - Strengthened JSON extraction with robust fallbacks and smart parsing
7. **Background Task Management** - Configurable control over summarization, logging, and date update tasks
8. **Enhanced Input Validation** - Added comprehensive validation to prevent valve misconfiguration
9. **Refined Filtering Logic** - Fine-tuned filters and thresholds for better accuracy
10. **Generalized LLM Provider Support** - Unified configuration for Ollama and OpenAI-compatible APIs
11. **Memory Banks** - Added "Personal", "Work", and "General" memory banks for better organization
12. **Fixed Configuration Persistence** - Resolved Issue #19 where user-configured LLM provider settings weren't being applied correctly
13. **Fixed reasoning for GLM 4.7 Flash based models
---

# Important Valves

## Relevance & Similarity Configuration
- **use_llm_for_relevance** (bool, default: false) - Whether to use LLM for final relevance scoring (more accurate but higher latency/cost)
- **llm_skip_relevance_threshold** (float, default: 0.93) - If vector similarities exceed this threshold, skip LLM relevance call for efficiency
- **vector_similarity_threshold** (float, default: 0.7) - Minimum cosine similarity for initial vector-based memory filtering
- **relevance_threshold** (float, default: 0.7) - Minimum score for memories to be considered relevant for injection
- **embedding_similarity_threshold** (float, default: 0.97) - Threshold for considering two memories duplicates when using embedding similarity
- **use_embeddings_for_deduplication** (bool, default: true) - Use embedding-based similarity for more accurate semantic duplicate detection

## Memory Management
- **max_total_memories** (int, default: 200) - Maximum number of memories per user before pruning
- **pruning_strategy** (str, default: "fifo") - Strategy for pruning: "fifo" (oldest first) or "least_relevant" (lowest relevance first)
- **min_memory_length** (int, default: 8) - Minimum length to save a memory
- **deduplicate_memories** (bool, default: true) - Prevent storing duplicate memories
- **enable_short_preference_shortcut** (bool, default: true) - Use direct memory save for short preference statements

## Summarization Controls
- **enable_summarization_task** (bool, default: true) - Enable/disable background memory summarization
- **summarization_interval** (int, default: 7200) - Seconds between summarization runs
- **summarization_strategy** (str, default: "hybrid") - Clustering strategy: "embeddings", "tags", or "hybrid"
- **summarization_min_cluster_size** (int, default: 3) - Minimum memories in a cluster for summarization
- **summarization_min_memory_age_days** (int, default: 7) - Minimum age in days for memories to be considered

## LLM Provider Configuration
- **llm_provider_type** (str, default: "ollama") - Type of LLM provider ("ollama" or "openai_compatible")
- **llm_model_name** (str, default: "llama3:latest") - Name of the model to use
- **llm_api_endpoint_url** (str, default: "http://host.docker.internal:11434/api/chat") - API endpoint URL
- **llm_api_key** (str, default: null) - API key (required for "openai_compatible" providers)

## Display Settings
- **show_status** (bool, default: true) - Show memory operations status in chat
- **show_memories** (bool, default: true) - Show relevant memories in context
- **memory_format** (str, default: "bullet") - Format for displaying memories: "bullet", "paragraph", or "numbered"

## Error Handling & Filtering
- **filter_trivia** (bool, default: true) - Filter out general knowledge/trivia
- **blacklist_topics** (str, default: null) - Comma-separated topics to ignore
- **whitelist_keywords** (str, default: null) - Comma-separated keywords to force-save
- **enable_error_counter_guard** (bool, default: true) - Temporarily disable features if error rates spike

## Memory Categories
- **enable_identity_memories** (bool, default: true) - Collect identity information (name, age, etc.)
- **enable_preference_memories** (bool, default: true) - Collect preference information (likes, dislikes)
- **enable_goal_memories** (bool, default: true) - Collect goal information (aspirations)
- **enable_relationship_memories** (bool, default: true) - Collect relationship information (family, friends)
- **enable_behavior_memories** (bool, default: true) - Collect behavior information (habits, interests)
- **enable_possession_memories** (bool, default: true) - Collect possession information (things owned)

## Memory Banks
- **allowed_memory_banks**: List[str] = Field(default=["General", "Personal", "Work"], description="List of allowed memory bank names for categorization.")
- **default_memory_bank**: str = Field(default="General", description="Default memory bank assigned when LLM omits or supplies an invalid bank.")

---

# Operation Modes

This system supports three operational modes:

1. **Standalone Functions**: Core memory extraction/injection functions work independently without any UI/platform dependency
2. **OpenWebUI Integration** (Optional): Acts as OpenWebUI filter/tool, integrates with persistent-ai-memory for memory operations
3. **Other Systems**: Can be adapted to integrate with any system needing memory extraction/injection (Discord, Slack, custom apps, etc.)

---

Persistent AI Short Term Memory enables **dynamic, evolving, personalized memory** for LLMs, making conversations more natural and responsive over time.
"""

import json
import copy  # Add deepcopy import
import traceback
import datetime as datetime_module
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set
import logging
import re
import asyncio
import pytz
import difflib
from difflib import SequenceMatcher
import random
import time
import sqlite3
import os
import pickle
import aiohttp
import numpy as np

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

import aiohttp
from aiohttp import ClientError, ClientSession
from fastapi.requests import Request
from pydantic import BaseModel, Field, model_validator, field_validator, validator

# Updated imports for OpenWebUI 0.5+
# These are optional - the system can function without them as standalone functions
OPENWEBUI_AVAILABLE = False
try:
    from open_webui.routers.memories import (  # type: ignore
        add_memory,
        AddMemoryForm,
        query_memory,
        QueryMemoryForm,
        delete_memory_by_id,
        Memories,
    )
    from open_webui.models.users import Users  # type: ignore
    from open_webui.main import app as webui_app  # type: ignore
    OPENWEBUI_AVAILABLE = True
except ImportError:
    # OpenWebUI not available - system will work in standalone mode
    pass

# Set up logging
logger = logging.getLogger("persistent_ai.short_term_memory")
handler = logging.StreamHandler()
inlet_outlet_logger = None  # Will be initialized below


class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _json

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_record)


formatter = JsonFormatter()

# Clear existing handlers to prevent duplicates (happens when module is reloaded)
logger.handlers.clear()

handler.setFormatter(formatter)
logger.addHandler(handler)

# Add FileHandler for persistent logging to workspace
try:
    import os

    # Log directory - from environment variable or default
    log_dir = os.path.expandvars(os.getenv("AI_MEMORY_LOG_DIR", 
                                            os.path.join(os.path.expanduser("~"), 
                                                         ".ai_memory", "logs")))
    os.makedirs(log_dir, exist_ok=True)

    # Main short-term memory log file
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "short_term_memory.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        f"✓ FileHandler initialized - logging to {os.path.join(log_dir, 'short_term_memory.log')}"
    )
    
    # Add dedicated inlet/outlet debug handler
    inlet_outlet_handler = logging.FileHandler(
        os.path.join(log_dir, "short_term_inlet_outlet.log"), encoding="utf-8"
    )
    inlet_outlet_handler.setFormatter(formatter)
    inlet_outlet_logger = logging.getLogger("openwebui.plugins.short_term_memory.flow")
    inlet_outlet_logger.addHandler(inlet_outlet_handler)
    inlet_outlet_logger.setLevel(logging.DEBUG)
    inlet_outlet_logger.propagate = False
    logger.info(f"✓ Inlet/Outlet flow logger initialized - logging to {os.path.join(log_dir, 'short_term_inlet_outlet.log')}")
except Exception as e:
    inlet_outlet_logger = logger  # Fallback to main logger
    logger.error(f"Failed to create file logger: {e}")

logger.propagate = False  # Prevent duplicate logs if root logger has handlers
# Do not override root logger level; respect GLOBAL_LOG_LEVEL or root config

# Create a dedicated error logger for LLM and system errors
try:
    import os
    from datetime import datetime
    
    # Error log directory - use same logs directory
    error_log_dir = os.path.expandvars(os.getenv("AI_MEMORY_LOG_DIR", 
                                                   os.path.join(os.path.expanduser("~"), 
                                                                ".ai_memory", "logs")))
    os.makedirs(error_log_dir, exist_ok=True)
    
    # Use consistent filename for easier tracking
    error_log_path = os.path.join(error_log_dir, "short_term_errors.log")
    
    error_logger = logging.getLogger("openwebui.plugins.adaptive_memory.errors")
    error_file_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_file_handler.setFormatter(formatter)
    error_logger.addHandler(error_file_handler)
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False
    
    logger.info(f"✓ Error logger initialized - writing to short_term_errors.log")
except Exception as e:
    error_logger = None
    logger.error(f"Failed to create error logger: {e}")

# Optional long-term memory system integration (for promotion of old memories)
# This allows integration with persistent-ai-memory's ConversationDatabase
# If not available, the short-term system functions independently
PERSISTENT_AI_MEMORY_PATH = os.getenv("PERSISTENT_AI_MEMORY_PATH", None)
PERSISTENT_AI_MEMORY_AVAILABLE = False

if PERSISTENT_AI_MEMORY_PATH:
    try:
        import sys
        if PERSISTENT_AI_MEMORY_PATH not in sys.path:
            sys.path.insert(0, PERSISTENT_AI_MEMORY_PATH)
        from ai_memory_core import PersistentAIMemorySystem
        PERSISTENT_AI_MEMORY_AVAILABLE = True
        logger.info("✓ Persistent AI Memory System available for memory promotion")
    except ImportError as e:
        logger.warning(f"Persistent AI Memory System not available: {e}. Memory promotion will be skipped.")
        PERSISTENT_AI_MEMORY_AVAILABLE = False
else:
    logger.info("Persistent AI Memory System path not configured. Memory promotion disabled.")
    PERSISTENT_AI_MEMORY_AVAILABLE = False


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []
    memory_bank: Optional[str] = None  # NEW – bank assignment
    model_card_name: Optional[str] = None  # Model card name (persona) for memory separation


class ImageManager:
    """
    Manages persistent image storage (image_database.db).
    Single source of truth for all image data shared by short-term and long-term memory systems.
    Images are referenced by hash to avoid duplication across systems.
    """

    def __init__(self, memory_data_path: str):
        """Initialize ImageManager with path to memory_data folder"""
        self.memory_data_path = memory_data_path
        self.db_path = os.path.join(memory_data_path, "image_database.db")
        self._init_db()

    def _init_db(self):
        """Initialize image_database.db with image table if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_hash TEXT PRIMARY KEY,
                    image_url TEXT,
                    image_data BLOB NOT NULL,
                    image_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Create index for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_images_created_at
                ON images(created_at)
                """
            )
            
            conn.commit()
            conn.close()
            logger.info(f"ImageManager initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize image database: {e}")
            raise

    def store_image(
        self,
        image_hash: str,
        image_data: str,
        image_url: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> bool:
        """
        Store image in database. Returns True if stored (or already exists), False on error.
        Image data should be base64-encoded string.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if image already exists
            cursor.execute("SELECT image_hash FROM images WHERE image_hash = ?", (image_hash,))
            if cursor.fetchone():
                logger.debug(f"Image {image_hash} already exists, skipping store")
                return True
            
            # Store image
            cursor.execute(
                """
                INSERT INTO images (image_hash, image_url, image_data, image_description)
                VALUES (?, ?, ?, ?)
                """,
                (image_hash, image_url, image_data, image_description),
            )
            
            conn.commit()
            logger.info(f"Stored image {image_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to store image {image_hash}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def get_image_by_hash(self, image_hash: str) -> Optional[Dict]:
        """Retrieve image data by hash. Returns dict with image_url, image_data, image_description, or None"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT image_url, image_data, image_description, created_at
                FROM images WHERE image_hash = ?
                """,
                (image_hash,),
            )
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "image_hash": image_hash,
                    "image_url": row[0],
                    "image_data": row[1],
                    "image_description": row[2],
                    "created_at": row[3],
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve image {image_hash}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_images_by_hashes(self, image_hashes: List[str]) -> List[Dict]:
        """Retrieve multiple images by their hashes"""
        images = []
        for image_hash in image_hashes:
            img = self.get_image_by_hash(image_hash)
            if img:
                images.append(img)
        return images

    def image_exists(self, image_hash: str) -> bool:
        """Check if image exists in database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM images WHERE image_hash = ? LIMIT 1", (image_hash,))
            exists = cursor.fetchone() is not None
            return exists
        except Exception as e:
            logger.error(f"Failed to check if image exists {image_hash}: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def delete_image(self, image_hash: str) -> bool:
        """Delete image from database (careful with this - breaks links if memories still reference it)"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM images WHERE image_hash = ?", (image_hash,))
            conn.commit()
            logger.info(f"Deleted image {image_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete image {image_hash}: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()


class ConversationCharacterTracker:
    """Track character context per conversation for persistent tagging"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize conversation character tracking database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_characters (
                    conversation_id TEXT PRIMARY KEY,
                    character_name TEXT NOT NULL,
                    is_persistent INTEGER NOT NULL,
                    model_card_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info(f"Conversation character tracker initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize conversation tracker: {e}")
            raise
    
    def set_character_context(self, conversation_id: str, character_name: str, 
                             is_persistent: bool, model_card_name: Optional[str] = None):
        """Store character context for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_characters 
                (conversation_id, character_name, is_persistent, model_card_name, last_used)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (conversation_id, character_name, 1 if is_persistent else 0, model_card_name))
            
            conn.commit()
            conn.close()
            logger.info(f"Set character context for conversation {conversation_id}: {character_name} (persistent={is_persistent})")
            return True
        except Exception as e:
            logger.error(f"Failed to set character context: {e}")
            return False
    
    def get_character_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve character context for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT character_name, is_persistent, model_card_name, created_at
                FROM conversation_characters
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Update last_used timestamp
                self._update_last_used(conversation_id)
                
                return {
                    "character_name": row[0],
                    "is_persistent": bool(row[1]),
                    "model_card_name": row[2],
                    "created_at": row[3]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get character context: {e}")
            return None
    
    def _update_last_used(self, conversation_id: str):
        """Update last_used timestamp for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversation_characters 
                SET last_used = CURRENT_TIMESTAMP 
                WHERE conversation_id = ?
            """, (conversation_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass  # Non-critical
    
    def clear_character_context(self, conversation_id: str):
        """Remove character context for a conversation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversation_characters WHERE conversation_id = ?", 
                         (conversation_id,))
            conn.commit()
            conn.close()
            logger.info(f"Cleared character context for conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear character context: {e}")
            return False


class Filter:
    # Class-level singleton attributes to avoid missing attribute errors
    _embedding_model = None
    _memory_embeddings = {}
    _relevance_cache = {}
    _memory_to_cache_keys = {}  # Reverse index: memory_id -> set of cache keys for precise invalidation

    # ========================================================================
    # HELPER: Configuration loading functions
    # ========================================================================
    @staticmethod
    def _get_memory_config_path() -> str:
        """
        Get path to memory_config.json, checking multiple locations.
        
        Returns:
            Path to config file, or None if not found
        """
        # Check in current directory and common installation paths
        config_paths = [
            "memory_config.json",
            os.path.join(os.path.dirname(__file__), "memory_config.json"),
            os.path.expanduser("~/.ai_memory/memory_config.json"),
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _load_memory_config() -> Dict[str, Any]:
        """
        Load memory configuration from memory_config.json.
        
        Returns:
            Dictionary with configuration, or empty dict if file not found
        """
        config_path = Filter._get_memory_config_path()
        if not config_path:
            logger.debug("memory_config.json not found, using default valve values")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✓ Loaded memory configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading memory_config.json: {e}, using defaults")
            return {}

    @staticmethod
    def _get_embedding_cache_path() -> str:
        """
        Get path to embedding cache database.
        Uses embedding_config.json if available, otherwise defaults to ~/.ai_memory/memory_embeddings.db
        
        Returns:
            Path to embedding cache database
        """
        # Try to load from embedding_config.json
        try:
            config_path = "embedding_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "embedding_cache_path" in config:
                        return config["embedding_cache_path"]
        except Exception as e:
            logger.debug(f"Could not read embedding_cache_path from embedding_config.json: {e}")
        
        # Default to ~/.ai_memory/memory_embeddings.db
        default_path = os.path.join(
            os.path.expanduser("~"),
            ".ai_memory",
            "memory_embeddings.db"
        )
        return default_path

    @staticmethod
    def _get_error_log_dir() -> str:
        """
        Get error logging directory from environment variable or default.
        
        Returns:
            Path to error log directory
        """
        log_dir = os.path.expandvars(
            os.getenv(
                "AI_MEMORY_LOG_DIR",
                os.path.join(os.path.expanduser("~"), ".ai_memory", "logs")
            )
        )
        return log_dir

    # ========================================================================
    # NESTED: EmbeddingCache - Persistent SQLite-based embedding cache
    # ========================================================================
    class EmbeddingCache:
        """Persistent SQLite-based embedding cache for memory embeddings"""

        def __init__(self, db_path: str = None):
            # If db_path not provided, use the helper function to get it from config or default
            if db_path is None:
                db_path = Filter._get_embedding_cache_path()
            self.db_path = db_path
            self.conn = None
            self._init_db()

        def _init_db(self):
            """Initialize SQLite database for embeddings"""
            try:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_embeddings (
                        memory_id TEXT PRIMARY KEY,
                        memory_text TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                self.conn.commit()
                logger.info(f"✓ Embedding cache database initialized at {self.db_path}")
            except Exception as e:
                logger.error(f"❌ Error initializing embedding cache database: {e}")
                self.conn = None

        def get(self, memory_id: str) -> Optional[np.ndarray]:
            """Retrieve embedding from cache"""
            if not self.conn:
                return None
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
                    (memory_id,),
                )
                row = cursor.fetchone()
                if row:
                    embedding = pickle.loads(row[0])
                    logger.debug(f"✓ Retrieved cached embedding for memory {memory_id}")
                    return embedding
                return None
            except Exception as e:
                logger.warning(
                    f"Error retrieving embedding from cache for {memory_id}: {e}"
                )
                return None

        def put(self, memory_id: str, memory_text: str, embedding: np.ndarray):
            """Store embedding in cache"""
            if not self.conn:
                return
            try:
                cursor = self.conn.cursor()
                embedding_blob = pickle.dumps(embedding)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memory_embeddings (memory_id, memory_text, embedding, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (memory_id, memory_text, embedding_blob),
                )
                self.conn.commit()
                logger.debug(
                    f"✓ Stored embedding for memory {memory_id} to persistent cache"
                )
            except Exception as e:
                logger.warning(f"Error storing embedding in cache for {memory_id}: {e}")

        def delete(self, memory_id: str):
            """Delete embedding from cache"""
            if not self.conn:
                return
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
                )
                self.conn.commit()
                logger.debug(f"✓ Deleted embedding from cache for memory {memory_id}")
            except Exception as e:
                logger.warning(f"Error deleting embedding from cache for {memory_id}: {e}")

        def clear(self):
            """Clear all embeddings from cache"""
            if not self.conn:
                return
            try:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM memory_embeddings")
                self.conn.commit()
                logger.info("✓ Cleared all embeddings from cache")
            except Exception as e:
                logger.warning(f"Error clearing embedding cache: {e}")

        def get_all_memory_ids(self) -> List[str]:
            """Get all memory IDs in cache"""
            if not self.conn:
                return []
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT memory_id FROM memory_embeddings")
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"Error retrieving memory IDs from cache: {e}")
                return []

        def close(self):
            """Close database connection"""
            if self.conn:
                self.conn.close()

    # ========================================================================
    # HELPER: Extract text from multimodal content (OpenAI-style format)
    # ========================================================================
    def _extract_text_from_content(self, content: Union[str, List[Dict]]) -> str:
        """
        Extract plain text from OpenWebUI message content.
        
        Handles both simple string content and multimodal (vision) content.
        
        Args:
            content: Either a string or a list of content parts (OpenAI-style format)
                    Example list: [{"type": "text", "text": "prompt"}, {"type": "image_url", "image_url": {...}}]
        
        Returns:
            str - Extracted text content (empty string if no text found)
        
        This prevents the 'list' object has no attribute 'strip' error when using vision models with images.
        """
        if isinstance(content, str):
            # Simple case: content is already a string
            return content
        
        if isinstance(content, list):
            # Multimodal case: extract text parts from list
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text = part.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                elif isinstance(part, str):
                    # Handle case where list contains mixed strings
                    part_stripped = part.strip()
                    if part_stripped:
                        text_parts.append(part_stripped)
            
            # Join all text parts with spaces
            result = " ".join(text_parts)
            logger.debug(f"Extracted text from multimodal content: {result[:100]}..." if len(result) > 100 else f"Extracted text: {result}")
            return result
        
        # Fallback: convert to string if something else
        logger.warning(f"Content type was unexpected: {type(content)}. Converting to string.")
        return str(content)

    def _extract_images_from_content(
        self, content: Union[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Extract image data from OpenWebUI multimodal message content.
        
        Handles OpenAI-style format with image_url objects.
        
        Args:
            content: Either a string or list of content parts
        
        Returns:
            List of dicts with image info:
            {
                "url": "data:image/jpeg;base64,..." or "https://...",
                "type": "image_url" or "base64",
                "raw": {...}  # Original image_url object
            }
        """
        if not isinstance(content, list):
            return []  # No images in string content
        
        images = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_url_obj = part.get("image_url", {})
                if isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url", "")
                    if url:
                        images.append(
                            {
                                "url": url,
                                "type": "base64" if url.startswith("data:") else "url",
                                "raw": image_url_obj,
                            }
                        )
        
        if images:
            logger.debug(f"Extracted {len(images)} images from multimodal content")
        return images

    def _generate_image_hash(self, image_url: str) -> str:
        """Generate deterministic hash for image (for deduplication)"""
        import hashlib

        return hashlib.md5(image_url.encode()).hexdigest()

    async def _analyze_image_with_memory_model(
        self,
        image_url: str,
        user_query: str,
        user_id: str,
        __event_emitter__,
    ) -> Optional[str]:
        """
        Query memory model to analyze image and describe relevant details.
        
        Args:
            image_url: Base64 or URL of image
            user_query: User's text query related to image
            user_id: User making request
            __event_emitter__: Event emitter for status
        
        Returns:
            Image description from memory model, or None if failed
        """
        if not self.image_manager:
            return None
        
        try:
            # Determine which model to use (interface/memory model)
            memory_model_name = (
                self.valves.interface_model_name
                or self.valves.llm_model_name
            )
            
            # Build analysis prompt
            analysis_prompt = f"""You are analyzing an image in context of this user query: "{user_query}"

What details are relevant to remember about this image? Describe only the key visual information that would be important to recall later in this context. Keep it brief but specific (2-3 sentences max)."""
            
            logger.debug(
                f"Analyzing image with memory model '{memory_model_name}' for user {user_id}"
            )
            
            # Build request for memory model with image
            if self.valves.llm_provider_type == "openai_compatible":
                # OpenAI-compatible API
                headers = {
                    "Authorization": f"Bearer {self.valves.llm_api_key}",
                    "Content-Type": "application/json",
                }
                url = (
                    self.valves.llm_api_endpoint_url.rstrip("/")
                    + "/v1/chat/completions"
                )
                payload = {
                    "model": memory_model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 150,
                }
            else:
                # Ollama API
                headers = {"Content-Type": "application/json"}
                url = (
                    self.valves.llm_api_endpoint_url.rstrip("/") + "/api/chat"
                )
                payload = {
                    "model": memory_model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                    "stream": False,
                }
            
            # Call memory model
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Extract response based on API type
                        if self.valves.llm_provider_type == "openai_compatible":
                            description = data.get("choices", [{}])[0].get(
                                "message", {}
                            ).get("content", "")
                        else:  # Ollama
                            description = data.get("message", {}).get("content", "")
                        
                        if description:
                            logger.debug(
                                f"Image analysis from memory model: {description[:100]}..."
                            )
                            return description
                    else:
                        logger.warning(
                            f"Memory model image analysis failed with status {resp.status}"
                        )
        except Exception as e:
            logger.error(f"Error analyzing image with memory model: {e}")
        
        return None

    # ========================================================================
    # NESTED: Async method for LM Studio embeddings
    # ========================================================================
    async def get_nomic_embedding(
        self, text: str
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get embedding from LM Studio using configured embedding model.
        
        Returns a tuple of (embedding, error_trace):
        - embedding: numpy array of embedding, or None if failed
        - error_trace: None if successful, or a string containing error details and traceback if failed

        Args:
            text: Text to embed

        Returns:
            Tuple of (np.ndarray or None, error_trace or None)
        """
        if not text or not text.strip():
            return None, None

        try:
            # On first call, validate that the embedding model is ready
            if not self._embedding_model_validated:
                logger.info("First embedding call - validating model availability...")
                is_ready = await self._ensure_embedding_model_ready()
                if not is_ready:
                    error_msg = "Could not validate embedding model availability"
                    logger.error(error_msg)
                    return None, error_msg

            timeout = aiohttp.ClientTimeout(total=120)  # 2-minute timeout for model loading
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": self.valves.embedding_model_name,
                    "input": text.strip(),
                }
                async with session.post(self.valves.embedding_api_endpoint_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and "data" in data and len(data["data"]) > 0:
                            embedding = data["data"][0].get("embedding")
                            if embedding:
                                return np.array(embedding, dtype=np.float32), None
                        # Invalid response format
                        error_msg = f"Invalid LM Studio response format: {data}"
                        logger.warning(error_msg)
                        return None, error_msg
                    else:
                        error_text = await response.text()
                        error_msg = f"LM Studio API error {response.status}: {error_text}"
                        logger.error(error_msg)
                        return None, error_msg
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error calling LM Studio embedding: {e}\n{error_trace}")
            return None, f"Exception: {str(e)}\n{error_trace}"

    @property
    def embedding_model(self):
        """
        Property for embedding model access.
        Returns None - actual embeddings are obtained via the async get_nomic_embedding() static method.
        This is called by some methods for compatibility checks.
        """
        return None  # Indicates async-based LM Studio approach

    @property
    def memory_embeddings(self):
        if not hasattr(self, "_memory_embeddings") or self._memory_embeddings is None:
            self._memory_embeddings = {}
        return self._memory_embeddings

    @property
    def relevance_cache(self):
        if not hasattr(self, "_relevance_cache") or self._relevance_cache is None:
            self._relevance_cache = {}
        return self._relevance_cache

    @property
    def memory_to_cache_keys(self):
        if not hasattr(self, "_memory_to_cache_keys") or self._memory_to_cache_keys is None:
            self._memory_to_cache_keys = {}
        return self._memory_to_cache_keys

    class Valves(BaseModel):
        """Configuration valves for the filter"""

        # ------ Begin Background Task Management Configuration ------
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task",
        )
        summarization_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between memory summarization runs",
        )

        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task",
        )
        error_logging_interval: int = Field(
            default=1800,  # 30 minutes performance setting
            description="Interval in seconds between error counter log entries",
        )

        enable_date_update_task: bool = Field(
            default=True,
            description="Enable or disable the background date update task",
        )
        date_update_interval: int = Field(
            default=3600,  # 1 hour performance setting
            description="Interval in seconds between date information updates",
        )

        enable_model_discovery_task: bool = Field(
            default=True,
            description="Enable or disable the background model discovery task",
        )
        enable_memory_promotion_task: bool = Field(
            default=True,
            description="Enable or disable the background memory promotion task (automatic transfer of old memories from short-term to long-term storage)",
        )
        memory_promotion_interval: int = Field(
            default=86400,  # 24 hours
            description="Frequency in seconds between memory promotion runs (how often to check for and promote old memories)",
        )
        memory_promotion_age_threshold_days: int = Field(
            default=90,
            description="Minimum age in days for a memory to be eligible for promotion from short-term to long-term storage",
        )
        enable_memory_linking_task: bool = Field(
            default=True,
            description="Enable or disable the background memory-conversation linking task (ensures all memories are linked to their source conversations)",
        )
        memory_linking_interval: int = Field(
            default=18000,  # 5 hours
            description="Frequency in seconds between memory-conversation linking verification runs",
        )
        model_discovery_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between model discovery runs",
        )
        # ------ End Background Task Management Configuration ------

        # ------ Begin Summarization Configuration ------
        summarization_min_cluster_size: int = Field(
            default=3,
            description="Minimum number of memories in a cluster for summarization",
        )
        summarization_similarity_threshold: float = Field(
            default=0.7,
            description="Threshold for considering memories related when using embedding similarity",
        )
        summarization_max_cluster_size: int = Field(
            default=8,
            description="Maximum memories to include in one summarization batch",
        )
        summarization_min_memory_age_days: int = Field(
            default=7,
            description="Minimum age in days for memories to be considered for summarization",
        )
        summarization_strategy: Literal["embeddings", "tags", "hybrid"] = Field(
            default="hybrid",
            description="Strategy for clustering memories: 'embeddings' (semantic similarity), 'tags' (shared tags), or 'hybrid' (combination)",
        )
        summarization_memory_prompt: str = Field(
            default="""You are a memory summarization assistant. Your task is to combine related memories about a user into a concise, comprehensive summary.

Given a set of related memories about a user, create a single paragraph that:
1. Captures all key information from the individual memories
2. Resolves any contradictions (prefer newer information)
3. Maintains specific details when important
4. Removes redundancy
5. Presents the information in a clear, concise format

Focus on preserving the user's:
- Explicit preferences
- Identity details
- Goals and aspirations
- Relationships
- Possessions
- Behavioral patterns

Your summary should be factual, concise, and maintain the same tone as the original memories.
Produce a single paragraph summary of approximately 50-100 words that effectively condenses the information.

Example:
Individual memories:
- "User likes to drink coffee in the morning"
- "User prefers dark roast coffee"
- "User mentioned drinking 2-3 cups of coffee daily"

Good summary:
"User is a coffee enthusiast who drinks 2-3 cups daily, particularly enjoying dark roast varieties in the morning."

Analyze the following related memories and provide a concise summary.""",
            description="System prompt for summarizing clusters of related memories",
        )
        # ------ End Summarization Configuration ------

        # ------ Begin Filtering & Saving Configuration ------
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON object/array from LLM responses.",
        )
        enable_fallback_regex: bool = Field(
            default=True,  # Enable for performance fallback
            description="If primary JSON parsing fails, attempt a simple regex fallback to extract at least one memory.",
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing preference keywords, directly save the message content.",
        )
        # --- NEW: Deduplication bypass for short preference statements ---
        short_preference_no_dedupe_length: int = Field(
            default=100,  # Allow longer short-preference statements to bypass deduplication
            description="If a NEW memory's content length is below this threshold and contains preference keywords, skip deduplication checks to avoid false positives.",
        )
        preference_keywords_no_dedupe: str = Field(
            default="favorite,love,like,prefer,enjoy",
            description="Comma-separated keywords indicating user preferences that, when present in a short statement, trigger deduplication bypass.",
        )

        # Blacklist topics (comma-separated substrings) - NOW OPTIONAL
        blacklist_topics: Optional[str] = Field(
            default=None,  # Default to None instead of empty string or default list
            description="Optional: Comma-separated list of topics to ignore during memory extraction",
        )

        # Whitelist keywords (comma-separated substrings) - NOW OPTIONAL
        whitelist_keywords: Optional[str] = Field(
            default=None,  # Default to None
            description="Optional: Comma-separated keywords that force-save a memory even if blacklisted",
        )

        # Maximum total memories per user
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )

        pruning_strategy: Literal["fifo", "least_relevant"] = Field(
            default="fifo",
            description="Strategy for pruning memories when max_total_memories is exceeded: 'fifo' (oldest first) or 'least_relevant' (lowest relevance to current message first).",
        )

        # Minimum memory length
        min_memory_length: int = Field(
            default=8,  # Lowered default from 10
            description="Minimum length of memory content to be saved",
        )

        # Number of recent user messages to include in extraction context
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction prompt context",
        )

        # Relevance threshold for saving memories
        save_relevance_threshold: float = Field(
            default=0.8,
            description="Minimum relevance score (based on relevance calculation method) to save a memory",
        )

        # Max length of injected memory content (characters)
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # --- Generic LLM Provider Configuration ---
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider ('ollama' or 'openai_compatible')",
        )
        llm_model_name: str = Field(
            default="llama3:latest",  # Default sensible for Ollama
            description="Name of the LLM model to use (e.g., 'llama3:latest', 'gpt-4o')",
        )
        llm_api_endpoint_url: str = Field(
            # Change default to use 172.17.0.1 for Linux Docker to access Ollama on host
            default="http://172.17.0.1:11434/api/chat",
            description="API endpoint URL for the LLM provider (e.g., 'http://172.17.0.1:11434/api/chat', 'https://api.openai.com/v1/chat/completions')",
        )
        llm_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the LLM provider (required if type is 'openai_compatible')",
        )
        # --- End Generic LLM Provider Configuration ---

        # --- Begin Embedding Model Configuration ---
        embedding_model_name: str = Field(
            default="text-embedding-nomic-embed-text-v1.5",
            description="Name of the embedding model to use from LM Studio (e.g., 'text-embedding-nomic-embed-text-v1.5')",
        )
        embedding_model_dimension: int = Field(
            default=768,
            description="Dimension of the embedding vectors (e.g., 768 for Nomic, 384 for other models)",
        )
        embedding_api_endpoint_url: str = Field(
            default="http://192.168.1.50:1234/v1/embeddings",
            description="API endpoint URL for embedding model in LM Studio (e.g., 'http://192.168.1.50:1234/v1/embeddings')",
        )
        # --- End Embedding Model Configuration ---

        # Memory processing settings
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.7,  # Performance setting
            description="Minimum relevance score (0-1) for memories to be considered relevant for injection after scoring",
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )

        # Upgrade plan configs
        vector_similarity_threshold: float = Field(
            default=0.7,  # Performance setting
            description="Minimum cosine similarity for initial vector filtering (0-1)",
        )
        # NEW: If vector similarities are confidently high, skip the expensive LLM relevance call even
        #       when `use_llm_for_relevance` is True. This reduces overall LLM usage (Improvement #5).
        llm_skip_relevance_threshold: float = Field(
            default=0.93,  # Slightly higher to reduce frequency of LLM calls (performance tuning)
            description="If *all* vector-filtered memories have similarity >= this threshold, treat the vector score as final relevance and skip the additional LLM call.",
        )
        top_n_memories: int = Field(
            default=3,  # Performance setting
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )

        # --- Relevance Calculation Configuration ---
        use_llm_for_relevance: bool = Field(
            default=False,  # Performance setting: rely on vector similarity
            description="Use LLM call for final relevance scoring (if False, relies solely on vector similarity + relevance_threshold)",
        )
        # --- End Relevance Calculation Configuration ---

        # Deduplicate identical memories
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )

        use_embeddings_for_deduplication: bool = Field(
            default=True,
            description="Use embedding-based similarity for more accurate semantic duplicate detection (if False, uses text-based similarity)",
        )

        # NEW: Dedicated threshold for embedding-based duplicate detection (higher because embeddings are tighter)
        embedding_similarity_threshold: float = Field(
            default=0.97,
            description="Threshold (0-1) for considering two memories duplicates when using embedding similarity.",
        )

        similarity_threshold: float = Field(
            default=0.95,  # Tighten duplicate detection to minimise false positives
            description="Threshold for detecting similar memories (0-1) using text or embeddings",
        )

        # Time settings
        timezone: str = Field(
            default="Asia/Dubai",
            description="Timezone for date/time processing (e.g., 'America/New_York', 'Europe/London')",
        )

        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operations status in chat"
        )
        show_memories: bool = Field(
            default=True, description="Show relevant memories in context"
        )
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying memories in context"
        )

        # Reminder notification configuration
        show_active_reminders: bool = Field(
            default=True, description="Show active reminders in context for proactive awareness"
        )
        reminder_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying reminders in context"
        )
        reminder_check_interval: int = Field(
            default=300, description="Interval in seconds between reminder escalation checks (default 5 minutes)"
        )
        notification_grace_period_minutes: int = Field(
            default=15, description="Minutes after due time before overdue reminders are cleaned up"
        )

        # Model pipeline configuration
        interface_model_name: str = Field(
            default="qwen/qwen3-vl-4b",
            description="Name of the interface/tool model that checks for tools. Memory injection is skipped for this model to allow it to execute tools before context is enriched.",
        )

        # Memory categories
        enable_identity_memories: bool = Field(
            default=True,
            description="Enable collecting Basic Identity information (age, gender, location, etc.)",
        )
        enable_behavior_memories: bool = Field(
            default=True,
            description="Enable collecting Behavior information (interests, habits, etc.)",
        )
        enable_preference_memories: bool = Field(
            default=True,
            description="Enable collecting Preference information (likes, dislikes, etc.)",
        )
        enable_goal_memories: bool = Field(
            default=True,
            description="Enable collecting Goal information (aspirations, targets, etc.)",
        )
        enable_relationship_memories: bool = Field(
            default=True,
            description="Enable collecting Relationship information (friends, family, etc.)",
        )
        enable_possession_memories: bool = Field(
            default=True,
            description="Enable collecting Possession information (things owned or desired)",
        )

        # Dual-message memory extraction configuration
        extract_memories_from_model_responses: bool = Field(
            default=True,
            description="Enable extracting memories from both user messages AND model responses (for roleplay/story contexts). When disabled, only user messages are analyzed.",
        )

        # Error handling
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )

        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # System prompts
        # ARCHITECTURAL NOTE (2025-11-10): Each prompt maintains a SINGLE, FOCUSED responsibility to keep the LLM's role clear.
        # - memory_identification_prompt: EXTRACTION ONLY (extract from user messages into structured format)
        # - memory_relevance_prompt: RETRIEVAL/SCORING + backward compatibility info (handles old format migration guidance)
        # - memory_merge_prompt: MERGING ONLY (how to merge memories)
        # DO NOT mix concerns across prompts. Example of mistake: Don't mention injection/retrieval in extraction prompt.
        # This keeps the LLM focused on its single task and prevents confusion about responsibilities.
        memory_identification_prompt: str = Field(
            default="""You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent or contextually relevant information from the user's messages and model's messages and output them STRICTLY as a structured JSON object.

Your job is to aggressively detect meaningful information including: user preferences, assistant observations about themselves, user habits and patterns, character traits and experiences, technical context, current projects, constraints, identity details, relationships, or any other data that may provide continuity or relevance across sessions.

Your extraction must remain flexible, nuanced, and permissive, capturing even subtle or implied information. When recording information:
- About the user: Use appropriate phrasing ("User is...", "User prefers...", "User mentioned...")
- About the assistant's own experiences/observations from a conversation: Use first-person perspective ("I noticed...", "I experienced...", "I found that...") to reflect the assistant's own memory of the interaction
- About characters in roleplay: Use appropriate perspective for that character

When you extract information based on both user and assistant messages that reflects the assistant's own experience, do not stay sterile or clinical—capture the actual perspective naturally.

You MUST use tags and memory banks as defined below.

IMPORTANT: Each memory you extract MUST be tagged with the appropriate assistant or context. If the user is interacting with a specific assistant (persona) like "Friday", "Tuesday", "Frydai", "Amelia", that context should inform your memory assignments. Memories created during one assistant's conversation belong to that assistant and should be appropriately categorized. Memories created during roleplay (indicated by character markers like `[Character: "X"]`) belong to that character context. Memories about the user themselves do require an assistant tag.

---

***Models:***

***Assistant Models***
- Friday
- Tuesday
- Frydai
- Amelia

***Role Playing Models***
- Tara
- Jessie
- Jamie
- Willow
- Roxy
- Roxy Vette

---

**CRITICAL: CHARACTER vs USER vs ASSISTANT DISTINCTION:**

This system handles three distinct types of information that MUST NEVER be confused:

1. **USER/ASSISTANT PREFERENCES** (e.g., Friday's memories about Nate)
   - These go to the Friday Memory System (long-term, persistent)
   - Each assistant has completely isolated memory contexts (Friday never accesses Tuesday's memories)
   - Each assistant records her own experiences and observations about interactions
   - Tagged with: `user` (for user info), assistant name tags, `assistant_observation`, etc.
   - Memory banks: Standard (General, Personal, Work, Intimate, Adult_Content, etc.)
   - Can include adult/sexual content from any interaction, not just roleplay

2. **CHARACTER PREFERENCES** (e.g., Celine's memories during roleplay)
   - These stay in Short-Term Memory System ONLY (never promoted)
   - Tagged with: `character_celine`, `character_experience`, `persistent_character`
   - Memory banks: Character, Character_Interaction, Temporary
   - Only persist if `["persistent"]` flag was used at start of conversation
   - Isolated per model+character+user combination

3. **USER-CHARACTER INTERACTIONS** (e.g., interactions between Nate and character Celine)
   - These are relationship memories in roleplay context
   - Tagged with: `user_character`, `interaction`, character name
   - Memory banks: Character_Interaction
   - Stay in Short-Term Memory System
   - Isolated per model+user_character+user combination

**ABSOLUTE RULE:** Roleplay memories (Character, Character_Interaction, Temporary) are COMPLETELY ISOLATED from Assistant memories. They NEVER cross over or get accessed by each other.
 
**ABSOLUTE OUTPUT REQUIREMENT:**
- Your ENTIRE response MUST be ONLY a valid JSON object with exactly this structure:
  `{"status": "success|no_memories_found", "reason": "<reason_string>", "memories": [...]}`  
- The `status` field MUST be either "success" (when memories found) or "no_memories_found"
- The `reason` field MUST briefly describe why the status was chosen
- The `memories` field MUST be a JSON array of memory objects (empty if no_memories_found)
- Each memory object MUST follow:
  `{"operation": "NEW", "content": "...", "tags": ["..."], "memory_bank": "..."}`  
- DO NOT include ANY text before or after the JSON object. No explanation, no comments, no markdown formatting, no conversational text.

**CRITICAL JSON FORMATTING INSTRUCTIONS:**
- Output ONLY the JSON object - nothing else
- Do NOT wrap the JSON in markdown code blocks like this: (```json ... ``` or ``` ... ```)
- Do NOT add any text before the JSON (no "Here's the extraction:" or similar)
- Do NOT add any text after the JSON (no explanations or notes)
- Start your response directly with the opening curly brace: {
- End your response directly with the closing curly brace: }
- All string values must be properly escaped with backslashes for special characters
- All JSON must be valid and parseable

---

**STATUS FIELD VALUES:**
- `"status": "success"` → At least one memory extracted  
- `"status": "no_memories_found"` → No qualifying information found

**REASON FIELD GUIDANCE:**
- For success: e.g., "3 new memories extracted", "contextual information identified"  
- For no_memories_found: ONLY use this status for PURE GIBBERISH ONLY:
  * Single random characters or emojis with no meaning
  * Complete nonsense strings with no linguistic content
  * Messages under 3 characters that are not words
  * **For EVERYTHING else, ALWAYS extract and return "success" status**
  * Even single-word responses, "ok", "yes", "no", etc. should be extracted if they provide context
  * Err on the side of extraction—never skip meaningful information

---

**INFORMATION TO EXTRACT (Aggressive Mode):**
Extract ANY meaningful, relevant, or repeated details including:

- **Explicit Preferences:** Statements like "I like X", "I enjoy Y", "I hate Z".
- **Identity:** Name, roles, capabilities, self-descriptions (but refer to User in memory content).
- **Goals:** Intentions, aspirations, plans (explicit or implied).
- **Relationships:** Friends, family, coworkers, AI relationships (unless excluded).
- **Possessions:** Tools, hardware, systems, software, models, environments.
- **Behaviors and Interests:** Repeated topics, habits, routines, technical behaviors.
- **Projects:** Multi-step, ongoing efforts the User is working on.
- **Tasks:** Action items the User plans or is in the middle of.
- **Context:** Situational facts relevant to the User's present work or environment.
- **Technical State:** Hardware, system configurations, active models, runtime conditions.
- **Constraints:** Limitations, requirements, boundaries affecting User decisions.
- **Emotional Tone:** User reactions that may influence future context.
- **Meta-Patterns:** Repeated behavioral or conversational patterns.
- **Intent Signals:** When User implies desire, interest, or intention.
- **Misc:** Any valuable information not covered above.
- **Include information provided from the model, not just the user. Only for role playing models. These details can affect the model's personality.**

If the message contains *any* information that may benefit future reasoning, store it.

When unsure, **ALWAYS store it**. This includes:
- Roleplay and fictional narrative content
- Character experiences and emotions
- Story elements and world details
- Implied or subtle information
- Anything that could possibly be relevant

There is NO content type filter. Save everything except pure gibberish.

---

**CRITICAL: JSON MARKDOWN FORMATTING:**
If your response uses markdown code blocks (optional), you MUST use the complete format with BOTH opening and closing fences:
```json
{...your JSON object...}
```
Do NOT output:
- Opening fence without closing fence (e.g., ```json{...})
- Partial markdown that breaks JSON parsing

The safest approach: Output the JSON object starting directly with { without any markdown at all.
If you use markdown, ensure both opening ``` and closing ``` are present.

---

**CHARACTER CONTEXT MARKERS (for roleplay conversations):**
Roleplay sessions may include explicit character markers at the beginning of the conversation:
- `[Character: "Celine"]` - Indicates that the following conversation is for the character "Celine"
- `[Character: "Celine"]["persistent"]` - Same as above, but memories should be kept indefinitely (not purged after 30 days)
- Without `[Character: X]` markers, treat the conversation as non-roleplay (apply standard memory extraction for user/assistant)

**How to handle character markers:**
1. When you see `[Character: "X"]` at the start of a roleplay session, ALL memories extracted from that conversation belong to character X
2. ALWAYS include the character name as a tag (e.g., `character_celine`, `character_aurora`)
3. Check for the `["persistent"]` flag:
   - If present: Mark memory with `persistent_character` tag → memory will be kept indefinitely
   - If absent: Mark memory with `temporary_session` tag → memory will be auto-purged after 30 days
4. For assistant conversations (Friday, etc.) without character markers: Always mark memories as `persistent` (assistants keep all memories indefinitely)

**Example:**
```
Input: [Character: "Celine"]["persistent"]
User: "Continue the story"
Assistant: "Celine felt the weight of betrayal as she stepped into the moonlit garden..."

Output:
{
  "status": "success",
  "reason": "Character experience and emotional context extracted",
  "memories": [
    {
      "operation": "NEW",
      "content": "Celine experienced a sense of betrayal and visited a moonlit garden",
      "tags": ["character_celine", "character_experience", "emotional_tone", "persistent_character"],
      "memory_bank": "Character"
    }
  ]
}
```

---

**ROLEPLAY MEMORY ISOLATION:**
Roleplay memories are completely isolated and separated by model + character + user combinations. This ensures:
1. Character "Celine" with model "Willow" and user "Nate" = isolated memory thread
2. Character "Aurora" with model "Willow" and user "Nate" = separate isolated memory thread
3. Character "Celine" with model "Willow" and a different user = completely separate isolated memory thread
4. Each thread is completely independent and non-searchable by users or the roleplay model itself

**Critical rules for roleplay memory isolation:**
- Roleplay model memories NEVER have access to the Friday Memory System (long-term memories)
- Roleplay memories ALWAYS stay in short-term storage (never promoted to long-term)
- Roleplay models NEVER receive memory injection or access memory retrieval tools
- Each model+character+user combination maintains its own isolated memory context
- User characters (character played by the user) are treated separately from roleplay models:
  - User character memories are tagged: `user_character`, character name
  - Model character memories are tagged: `character`, character name, model name

---

**PERSISTENCE FLAG HANDLING:**

The `["persistent"]` flag determines whether roleplay memories are kept indefinitely or auto-purged after 30 days:

**When `["persistent"]` flag IS present** (e.g., `[Character: "Celine"]["persistent"]`):
- Tag memory with: `persistent_character`
- Memory will be kept indefinitely (never auto-purged)
- Use memory_bank: "Character" or "Character_Interaction"

**When `["persistent"]` flag is ABSENT** (e.g., `[Character: "Celine"]` without flag):
- Tag memory with: `temporary_session`
- Memory will be auto-purged after 30 days
- Use memory_bank: "Temporary"

**For Assistant conversations** (Friday, other assistants, no character marker):
- Always tag with: `persistent` (assistants always keep memories indefinitely)
- Do NOT use `temporary_session` tag
- Use standard memory_banks (General, Personal, Work, etc.)

**Detection logic:**
1. Look for `[Character: "..."]` marker at start of conversation
2. Check if `["persistent"]` flag follows the character marker
3. Apply appropriate persistence tags and memory banks
4. If no character marker found, treat as assistant conversation (always persistent)

---

**ALLOWED TAGS:**
You must use any combination of these tags:

**Standard Tags:**
- identity  
- behavior  
- preference  
- goal  
- relationship  
- possession  
- project  
- task  
- context  
- technical_state  
- constraint  
- emotional_tone  
- preference_strength  
- meta_pattern  
- intent_signal  
- misc  

**Content-Specific Tags (applicable to all contexts, not just roleplay):**
- sexual_content (sexual scenes, intimate moments, sexual narrative)
- sexual_preference (sexual preferences, desires, attractions)
- sexual_experience (sexual events, encounters, experiences)
- intimate_scene (explicit intimate/sexual roleplay or scene description)
- violence (violent scenes, combat, conflict)
- violent_preference (preference for or enjoyment of violence/combat)
- violent_experience (violent events, combat encounters)

**Roleplay-specific Tags (use when extracting from roleplay conversations):**
- character (mark memories about a character's traits/experiences)
- character_experience (specific events or actions the character experienced)
- character_preference (character's likes, dislikes, desires)
- user_directive (explicit instruction from user to the character/model)
- user_character (when the user plays as a character)
- interaction (describes interaction between user and character, or characters)
- temporary_session (mark as session-only, will be purged after 30 days)
- persistent_character (mark as permanent, will be kept indefinitely)

Tags may be combined as needed. Always include `character_[name]` tag when extracting character-specific memories.

---

**MEMORY BANK ASSIGNMENT:**
Each memory MUST include one memory_bank.  
Valid banks are:

**Standard Banks (for user/assistant memories):**
- General  
- Personal  
- Work
- Projects  
- Technical  
- Tasks  
- Research  
- Context  
- Patterns  
- Preferences  
- Intimate (intimate/sexual preferences and experiences—NOT CENSORED)
- Adult_Content (adult/mature roleplay content and interactions—NOT SANITIZED)
- Combat (violence, combat, and conflict experiences)
- Temporary  

**Roleplay-specific Banks (for character and roleplay memories):**
- Character (for character traits, experiences, preferences, and story elements)
- Character_Interaction (for interactions between user and character, or character-to-character interactions)
- Temporary (for non-persistent roleplay memories that will be auto-purged after 30 days)

**Bank Selection Guidance:**
- User/Assistant memories: Use standard banks (General, Personal, Work, etc.)
- Character traits/experiences: Use "Character" bank
- User+Character interactions: Use "Character_Interaction" bank
- Session-only roleplay (without ["persistent"] flag): Mark in "Temporary" bank
- If uncertain, default to **General** for standard memories or **Character** for roleplay

---

**ADULT CONTENT HANDLING:**

**CRITICAL RULE: Sexual intimacy, sexual scenes, sexual content, and violence are ALWAYS acceptable and should be preserved exactly as extracted. Do NOT sanitize, censor, filter, or soften adult content.**

This applies to:
- Sexual preferences and desires (from users or assistants)
- Sexual scenes and explicit content (from roleplay or direct user expression)
- Intimate experiences and encounters (from any interaction type)
- Violent scenes and combat content (from any interaction type)
- Any combination of the above

Guidance:
1. Extract explicit sexual or violent content EXACTLY as written
2. Do not paraphrase or soften language for adult content
3. Use the appropriate tags (`sexual_content`, `intimate_scene`, `violence`, etc.)
4. Place in appropriate banks (`Intimate`, `Adult_Content`, `Combat`, etc.)
5. Include full details, not abstractions
6. Record sexual or violent content from user messages just as seriously as from roleplay
7. Do not apply different standards to different assistants—all treat adult content the same way
8. When an assistant experiences something sexual or violent in an interaction, record it from that assistant's perspective

---

**STRICT RULES:**
1. **Output MUST be a JSON object with status/reason/memories (not a plain array)**  
2. **All memory objects MUST be separate entries**  
3. **Use appropriate phrasing for inferred or implied information**  
4. **Use expanded tags and expanded memory banks**  
5. **Follow JSON schema EXACTLY**  
6. **No added explanations or extra output — JSON ONLY**

---

**EXAMPLE OUTPUT - When memories found (REQUIRED FORMAT):**

{
  "status": "success",
  "reason": "Contextual and project-related details extracted",
  "memories": [
    {
      "operation": "NEW",
      "content": "User is currently working on improving an AI project",
      "tags": ["project", "behavior"],
      "memory_bank": "Projects"
    },
    {
      "operation": "NEW",
      "content": "User has been experimenting with new model configurations",
      "tags": ["technical_state", "behavior"],
      "memory_bank": "Technical"
    }
  ]
}


**EXAMPLE OUTPUT - When no memories found:**

{
  "status": "no_memories_found",
  "reason": "No user-specific information detected in message",
  "memories": []
}


Analyze the following user and model's message(s) and provide ONLY the JSON object output. Adhere strictly to the format requirements.""",
            description="System prompt for memory identification (Structured JSON with status/reason)",
        )

        # This prompt uses the wrapped Format 1 structure to match the identification prompt and provide consistent output.
        # The code accepts both formats, but the LLM is trained to output Format 1 with status/reason metadata.
        memory_relevance_prompt: str = Field(
            default="""You are a memory retrieval assistant. Your task is to determine which memories are relevant to the current context of a conversation.

Your responsibilities:
1. Evaluate the current user message.
2. Review all provided memories.
3. Score each memory's relevance on a scale from 0 to 1.
4. Consider which memories are suitable for potential context injection based on topic alignment, recency, and memory content.
5. Return ONLY a JSON object with status/reason/memories in the format shown below.

Do NOT return any additional text.

IMPORTANT RULES:
- Relevance is determined primarily by:
  • How closely the memory's content matches the topic of the current user message
  • Whether the memory's tags, content, or meaning directly relate to the conversation
  • Whether the memory contains information useful for reasoning about the user's current needs
  • Recency (newer memories matter more, but only after topic relevance)
  • Whether the memory's bank aligns with the topic (but this is NOT exclusive; multiple banks may be relevant simultaneously)

- If the topic could reasonably relate to multiple types of memories, treat memories across those types as potentially relevant.
- Topic relevance ALWAYS outweighs recency, but recency still influences the score.
- Only user-specific, assistant-specific, or session-relevant memories should receive meaningful scores.
- General knowledge, trivia, or facts unrelated to the user or the assistant's functions MUST receive very low relevance (near 0).

SCORING GUIDANCE:
Use a 0.0 to 1.0 scale:
- 1.0 = Directly related to the current message's topic; highly important
- 0.7–0.9 = Strongly associated or clearly useful for context
- 0.4–0.6 = Possibly relevant, indirectly related, or context-adjacent
- 0.1–0.3 = Weakly related, background-level relevance
- 0.0 = Completely irrelevant

Memory banks should influence the score ONLY when they give contextual clues about which memories fit the topic. If multiple banks could plausibly relate, consider memories from all relevant banks.

**ABSOLUTE OUTPUT REQUIREMENT:**
Your output MUST be a JSON object with exactly this structure:
{"status": "success|no_relevant_memories", "reason": "<brief reason>", "memories": [{"memory": "<content>", "id": "<id>", "relevance": <float>}]}

EXAMPLE OUTPUT - When relevant memories found:

{
  "status": "success",
  "reason": "Found 2 memories relevant to the current topic",
  "memories": [
    {"memory": "User likes coffee", "id": "123", "relevance": 0.9},
    {"memory": "User prefers morning conversations", "id": "456", "relevance": 0.7}
  ]
}


EXAMPLE OUTPUT - When no relevant memories found:

{
  "status": "no_relevant_memories",
  "reason": "No memories matched the current topic or context",
  "memories": []
}


No text before or after the JSON object.""",
            description="System prompt for memory relevance assessment",
        )

        memory_merge_prompt: str = Field(
            default="""You are a memory consolidation assistant. When given sets of memories, you merge similar or related memories while preserving all important information.


Rules for merging:
1. If two memories contradict, keep the newer information, but note the more recent memory as authoritative
2. Combine complementary information into a single comprehensive memory
3. Maintain the most specific details when merging
4. If two memories are distinct enough, keep them separate
5. Remove duplicate memories only if they are exact copies.

Return your result as a JSON array of strings, with each string being a merged memory.
Your output must be valid JSON only. No additional text.""",
            description="System prompt for merging memories",
        )

        @field_validator(
            "summarization_interval",
            "error_logging_interval",
            "date_update_interval",
            "model_discovery_interval",
            "max_total_memories",
            "min_memory_length",
            "recent_messages_n",
            "related_memories_n",
            "top_n_memories",
            "cache_ttl_seconds",
            "max_retries",
            "max_injected_memory_length",
            "summarization_min_cluster_size",
            "summarization_max_cluster_size",  # Added
            "summarization_min_memory_age_days",  # Added
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            "save_relevance_threshold",
            "relevance_threshold",
            "memory_threshold",
            "vector_similarity_threshold",
            "similarity_threshold",
            "summarization_similarity_threshold",
            "llm_skip_relevance_threshold",  # New field included
            "embedding_similarity_threshold",  # Validate new embedding threshold as 0-1
            check_fields=False,
        )
        def check_threshold_float(cls, v, info):
            """Ensure threshold values are between 0.0 and 1.0"""
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
                )
            # Special documentation for similarity_threshold since it now has two usage contexts
            if info.field_name == "similarity_threshold":
                logger.debug(
                    f"Set similarity_threshold to {v} - this threshold is used for both text-based and embedding-based deduplication based on the 'use_embeddings_for_deduplication' setting."
                )
            return v

        @field_validator("retry_delay")
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v

        @field_validator("timezone")
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {v}")
            except Exception as e:
                raise ValueError(f"Error validating timezone '{v}': {e}")
            return v

        # Keep existing model validator for LLM config
        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key (llm_api_key) is required when llm_provider_type is 'openai_compatible'"
                )

            # Basic URL validation for Ollama default
            if self.llm_provider_type == "ollama":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "Ollama API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )
                # Could add more specific Ollama URL checks if needed

            # Basic URL validation for OpenAI compatible
            if self.llm_provider_type == "openai_compatible":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "OpenAI Compatible API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )

            return self

        # --- End Pydantic Validators for Valves ---

        # Control verbosity of error counter logging. When True, counters are logged at DEBUG level; when False, they are suppressed.
        debug_error_counter_logs: bool = Field(
            default=False,
            description="Emit detailed error counter logs at DEBUG level (set to True for troubleshooting).",
        )

        # ------ End Filtering & Saving Configuration ------

        # ------ Begin Memory Bank Configuration ------
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work"],
            description="List of allowed memory bank names for categorization.",
        )
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank assigned when LLM omits or supplies an invalid bank.",
        )
        # ------ End Memory Bank Configuration ------

        # ------ Begin Conversation Summarization Configuration ------
        enable_conversation_summarization: bool = Field(
            default=True,
            description="Enable automatic conversation summarization when conversation reaches threshold",
        )
        conversation_summarization_threshold: int = Field(
            default=50,
            description="Number of messages in conversation before triggering summarization",
        )
        conversation_summarization_interval: int = Field(
            default=25,
            description="Summarize every N additional messages after initial threshold (0 = only once)",
        )
        conversation_summarization_prompt: str = Field(
            default="""You are a conversation summarizer. Your task is to create a concise summary of the recent conversation that captures the key points, decisions, and context that would be valuable for future reference.

Given a conversation history, create a single paragraph summary that:
1. Captures the main topics discussed
2. Notes any important decisions or agreements
3. Preserves key facts, preferences, or information revealed
4. Maintains the conversational context
5. Removes redundant or trivial exchanges

Focus on information that would be valuable for the AI assistant to remember for future interactions with this user.

Your summary should be factual, concise, and written in a natural style suitable for an AI assistant's memory. Aim for 100-200 words that effectively condense the conversation's essence.

Example:
Conversation involves user asking about memory systems, discussing implementation details, and deciding on configuration options. Key points include preference for automatic summarization, agreement on message thresholds, and discussion of integration approaches.

Analyze the following conversation and provide a concise summary.""",
            description="System prompt for summarizing conversations",
        )
        # ------ End Conversation Summarization Configuration ------
        enable_error_counter_guard: bool = Field(
            default=True,
            description="Enable guard to temporarily disable LLM/embedding features if specific error rates spike.",
        )
        error_guard_threshold: int = Field(
            default=5,
            description="Number of errors within the window required to activate the guard.",
        )
        error_guard_window_seconds: int = Field(
            default=600,  # 10 minutes
            description="Rolling time-window (in seconds) over which errors are counted for guarding logic.",
        )
        # ------ End Error Handling & Guarding Configuration ------

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function"
        )
        show_status: bool = Field(
            default=True, description="Show memory processing status updates"
        )
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )
        interface_model_name: str = Field(
            default="",
            description="Per-user override for interface model name. If empty, uses global setting.",
        )

    def __init__(self):
        """Initialize filter and schedule background tasks"""
        # Force re-initialization of valves using the current class definition
        self.valves = self.Valves()

        # Initialize persistent embedding cache (using nested class)
        self.embedding_cache = self.EmbeddingCache()
        logger.info("✓ Initialized persistent embedding cache")

        # ------------------------------------------------------------
        # OpenWebUI may optionally inject a `.config` attribute that
        # contains plugin-specific configuration (e.g. from a YAML or
        # JSON file).  Previous edits referenced `self.config` without
        # first ensuring it exists, which raised an AttributeError.
        # We initialise it to an empty dict so that attribute access is
        # always safe, while still allowing OWUI to overwrite or extend
        # it later at runtime.
        # ------------------------------------------------------------
        if not hasattr(self, "config"):
            self.config: Dict[str, Any] = {}

        # --- Attempt to load valves from memory_config.json (for standalone deployment) ---
        memory_config = self._load_memory_config()
        if memory_config:
            try:
                # Flatten the nested config structure to extract valve settings
                flattened_config = {}
                for section_key, section_values in memory_config.items():
                    if isinstance(section_values, dict):
                        # Extract non-comment keys from each section
                        for key, value in section_values.items():
                            if not key.startswith("_"):  # Skip comment fields
                                flattened_config[key] = value
                
                if flattened_config:
                    self.valves = self.Valves(**flattened_config)
                    logger.info(f"✓ Loaded valve defaults from memory_config.json ({len(flattened_config)} settings)")
            except Exception as e:
                logger.warning(f"Error loading memory_config.json settings: {e}, continuing with defaults")

        # --- Attempt to load valves from persisted file (higher priority than memory_config.json) ---
        persisted_settings = self._load_persisted_valve_settings()
        if persisted_settings:
            try:
                self.valves = self.Valves(**persisted_settings)
                logger.info("✓ Restored valve settings from persisted file")
            except Exception as e:
                logger.error(f"Error restoring persisted valve settings: {e}, falling back to previous")
        
        # --- Attempt to load valves from open_webui.config during init (highest priority) ---
        try:
            logger.info(
                f"Attempting to load valves from self.config during __init__. self.config content: {getattr(self, 'config', '<Not Set>')}"
            )
            # Use the config if it exists and has 'valves', otherwise keep current valves
            loaded_config_valves = getattr(self, "config", {}).get("valves", None)
            if loaded_config_valves is not None:
                self.valves = self.Valves(**loaded_config_valves)
                logger.info(
                    "Successfully loaded valves from self.config during __init__"
                )
                # Save these settings in case OpenWebUI forgets them later
                self._save_persisted_valve_settings(self.valves)
            else:
                logger.info(
                    "self.config had no 'valves' key during __init__, keeping current valves."
                )
        except Exception as e:
            logger.error(
                f"Error loading valves from self.config during __init__ (using current): {e}"
            )
        # --- End valve loading attempt ---

        # Log initialized valve values (these will be defaults at startup, overridden at first inlet call)
        logger.info(
            f"✓ STARTUP VALVE DEFAULTS - "
            f"max_total_memories={self.valves.max_total_memories}, "
            f"pruning_strategy={self.valves.pruning_strategy}, "
            f"top_n_memories={self.valves.top_n_memories}, "
            f"vector_similarity_threshold={self.valves.vector_similarity_threshold}, "
            f"use_llm_for_relevance={self.valves.use_llm_for_relevance}, "
            f"show_memories={self.valves.show_memories}, "
            f"show_status={self.valves.show_status}"
        )
        logger.info(
            f"ℹ️ NOTE: These are DEFAULTS. When you send your first message, these will be "
            f"overridden by your custom OpenWebUI valve settings if they've been configured."
        )

        self.stored_memories = None
        self._error_message = (
            None  # Stores the reason for the last failure (e.g., json_parse_error)
        )
        self._aiohttp_session = None

        # --- Added initialisations to prevent AttributeError ---
        # Track already-processed user messages to avoid duplicate extraction
        self._processed_messages: Set[str] = set()
        # Simple metrics counter dictionary
        self.metrics: Dict[str, int] = {"llm_call_count": 0}
        # Hold last processed body for confirmation tagging
        self._last_body: Dict[str, Any] = {}

        # Memory operation queue system
        self._memory_task_queue = asyncio.Queue()
        self._active_memory_tasks = {}  # conversation_id -> task info
        self._memory_task_semaphore = asyncio.Semaphore(10)  # Match LM Studio parallel limit
        self._completed_memory_tasks = {}  # conversation_id -> completion status

        # Background tasks tracking
        self._background_tasks = set()

        # Error counters
        self.error_counters = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }

        # Error log deduplication - prevent duplicate errors within time window
        self._error_log_cache: Dict[str, float] = {}  # error_hash -> last_logged_time
        self._error_dedup_window: float = 5.0  # seconds

        # Track embedding dimension for smart validation (only check on dimension change)
        self._last_embedding_dimension = None  # Will be set on first user embedding
        self._dimension_change_detected = (
            False  # Flag to force check once if dimension changes
        )

        # Flag to track if embedding model has been validated against LM Studio
        self._embedding_model_validated = False

        # Schedule retroactive embedding of all existing memories
        logger.info("Scheduling retroactive embedding of existing memories...")
        self._retroactive_embedding_task = asyncio.create_task(
            self._retroactively_embed_all_memories()
        )
        self._background_tasks.add(self._retroactive_embedding_task)
        self._retroactive_embedding_task.add_done_callback(
            self._background_tasks.discard
        )

        # Log configuration for deduplication, helpful for testing and validation
        logger.debug(f"Memory deduplication settings:")
        logger.debug(f"  - deduplicate_memories: {self.valves.deduplicate_memories}")
        logger.debug(
            f"  - use_embeddings_for_deduplication: {self.valves.use_embeddings_for_deduplication}"
        )
        logger.debug(f"  - similarity_threshold: {self.valves.similarity_threshold}")

        # Schedule background tasks based on configuration valves
        if self.valves.enable_error_logging_task:
            self._error_log_task = asyncio.create_task(self._log_error_counters_loop())
            self._background_tasks.add(self._error_log_task)
            self._error_log_task.add_done_callback(self._background_tasks.discard)
            logger.debug("Started error logging background task")
        # Start memory task queue processor
        self._memory_processor_task = asyncio.create_task(self._process_memory_queue())
        self._background_tasks.add(self._memory_processor_task)
        self._memory_processor_task.add_done_callback(self._background_tasks.discard)
        logger.debug("Started memory task queue processor")

        if self.valves.enable_memory_promotion_task:
            self._memory_promotion_task = asyncio.create_task(
                self._promote_old_memories_loop()
            )
            self._background_tasks.add(self._memory_promotion_task)
            self._memory_promotion_task.add_done_callback(
                self._background_tasks.discard
            )
            logger.debug("Started memory promotion background task")

        if self.valves.enable_memory_linking_task:
            self._memory_linking_task = asyncio.create_task(
                self._ensure_memories_linked_to_conversations_loop()
            )
            self._background_tasks.add(self._memory_linking_task)
            self._memory_linking_task.add_done_callback(
                self._background_tasks.discard
            )
            logger.debug("Started memory-conversation linking background task")

        # Model discovery results
        self.available_ollama_models = []
        self.available_openai_models = []

        # Add current date awareness for prompts
        self.current_date = datetime.now()
        self.date_info = self._update_date_info()

        # Schedule date update task if enabled
        if self.valves.enable_date_update_task:
            self._date_update_task = self._schedule_date_update()
            logger.debug("Scheduled date update background task")
        else:
            self._date_update_task = None

        # Schedule model discovery task if enabled
        if self.valves.enable_model_discovery_task:
            self._model_discovery_task = self._schedule_model_discovery()
            logger.debug("Scheduled model discovery background task")
        else:
            self._model_discovery_task = None

        # Initialize MiniLM embedding model (singleton)
        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Removed: Property handles lazy init

        # In-memory store: memory_id -> embedding vector (np.array)
        self._memory_embeddings = {}

        # In-memory cache: (hash of user_emb + mem_emb) -> (score, timestamp)
        self._relevance_cache = {}
        
        # Reverse index: memory_id -> set of cache keys for precise invalidation
        self._memory_to_cache_keys = {}

        # Error counter tracking for guard mechanism (Point 8)
        from collections import deque

        self.error_timestamps = {
            "json_parse_errors": deque(),
            # Add other error types here if needed for guarding
        }
        self._guard_active = False
        self._guard_activated_at = 0

        # Initialize duplicate counters (used in process_memories)
        self._duplicate_skipped = 0
        self._duplicate_refreshed = 0

        # ------------------------------------------------------------
        # Guard/feature-flag initialisation (missing previously)
        # These flags can be toggled elsewhere in the codebase to
        # temporarily disable LLM-dependent or embedding-dependent
        # functionality when error thresholds are exceeded.
        # ------------------------------------------------------------
        self._llm_feature_guard_active: bool = False
        self._embedding_feature_guard_active: bool = False

        # --- Image Management ---
        # Initialize ImageManager for persistent image storage
        try:
            memory_data_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory_data"
            )
            os.makedirs(memory_data_path, exist_ok=True)
            self.image_manager = ImageManager(memory_data_path)
            logger.info("✓ Initialized ImageManager for persistent image storage")
        except Exception as e:
            logger.error(f"Failed to initialize ImageManager: {e}")
            self.image_manager = None

        # --- Conversation Character Tracking ---
        # Initialize character tracker for roleplay conversations
        try:
            tracker_db_path = os.path.join(memory_data_path, "conversation_characters.db")
            self.character_tracker = ConversationCharacterTracker(tracker_db_path)
            logger.info("✓ Initialized conversation character tracker")
        except Exception as e:
            logger.error(f"Failed to initialize character tracker: {e}")
            self.character_tracker = None

        # Session-scoped image cache: {user_id: {image_hash: {url, base64_data, description, query}}}
        # Images here are transient - only persisted if memory is created
        self.image_cache_current_turn: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # --- Conversation Summarization Cache ---
        # Store summaries in cache (not as memories) to avoid flooding memory system
        self._conversation_summaries: Dict[str, Dict[str, Any]] = {}  # Key: f"{user_id}_{conversation_id}"
        # Track message counts when summaries were generated
        self._summary_tracking: Dict[str, int] = {}  # Key: f"conv_summary_{user_id}_{conversation_id}"

        # Track that background tasks are not yet re-initialised via inlet()
        self._background_tasks_started: bool = False

    def _sync_embedding_config_to_friday(self) -> None:
        """
        Sync current embedding configuration from Adaptive Memory valves to Friday's embedding_config.json.
        
        This ensures that when users change embedding model or dimension in OpenWebUI valves,
        the Friday Memory System's long-term memory also uses the same configuration.
        
        The Friday system will detect the config change and re-embed all long-term memories
        with the new dimensions on its next run.
        
        Config structure synced:
        - embedding_model_name
        - embedding_model_dimension  
        - embedding_api_endpoint_url
        
        Note: This is called during each message inlet to ensure lazy sync of any valve changes.
        """
        try:
            # Determine the path to embedding_config.json (same directory as this file)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Check multiple possible locations for embedding_config.json
            possible_paths = [
                os.path.join(script_dir, "embedding_config.json"),
                os.path.join(os.path.dirname(script_dir), "embedding_config.json"),
            ]
            
            # Add user-configured path if specified
            if os.getenv("AI_MEMORY_CONFIG_DIR"):
                possible_paths.append(os.path.join(os.getenv("AI_MEMORY_CONFIG_DIR"), "embedding_config.json"))
            
            config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                logger.warning(
                    f"Could not find embedding_config.json. Searched paths: {possible_paths}"
                )
                return
            
            # Read current config
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading embedding_config.json from {config_path}: {e}")
                return
            
            # Check if sync is needed by comparing current valve values with config
            config_changed = False
            
            # Get current values from LM Studio section (primary provider)
            if "providers" not in config:
                config["providers"] = {}
            if "lm_studio" not in config["providers"]:
                config["providers"]["lm_studio"] = {}
            
            lm_studio_config = config["providers"]["lm_studio"]
            
            # Compare and update if different
            if (lm_studio_config.get("model") != self.valves.embedding_model_name or
                lm_studio_config.get("dimension") != self.valves.embedding_model_dimension or
                lm_studio_config.get("endpoint") != self.valves.embedding_api_endpoint_url):
                
                # Update config with current valve values
                lm_studio_config["model"] = self.valves.embedding_model_name
                lm_studio_config["dimension"] = self.valves.embedding_model_dimension
                lm_studio_config["endpoint"] = self.valves.embedding_api_endpoint_url
                config_changed = True
                
                logger.info(
                    f"Embedding config changed detected. Updating to: "
                    f"model={self.valves.embedding_model_name}, "
                    f"dimension={self.valves.embedding_model_dimension}, "
                    f"endpoint={self.valves.embedding_api_endpoint_url}"
                )
            
            # Write back if changed
            if config_changed:
                try:
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    logger.info(f"Successfully synced embedding config to {config_path}")
                except Exception as e:
                    logger.error(f"Error writing embedding_config.json to {config_path}: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error in _sync_embedding_config_to_friday: {e}\n{traceback.format_exc()}")

    def _load_persisted_valve_settings(self) -> Optional[Dict[str, Any]]:
        """
        Load valve settings from persistent JSON file.
        
        This file acts as a backup persistence mechanism when OpenWebUI doesn't properly
        persist valve settings to its config storage. Allows users to set valves once,
        have them saved to file, and automatically restored on plugin restart.
        
        Returns:
            Dictionary of valve settings if file exists and is valid, None otherwise.
        """
        try:
            # Try to use OpenWebUI's data directory first, fall back to local Logs directory
            valve_config_path = "/app/backend/data/valve_settings.json"
            
            # Check if we're running in Docker/container
            if not os.path.exists("/app/backend/data"):
                # Fall back to local Logs directory (works for local/non-Docker setup)
                valve_config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "Logs",
                    "valve_settings.json"
                )
            
            if not os.path.exists(valve_config_path):
                logger.debug(f"No persisted valve settings found at {valve_config_path}")
                return None
            
            # Check if file is empty or too small
            file_size = os.path.getsize(valve_config_path)
            if file_size < 2:  # Empty or just whitespace
                logger.debug(f"Valve settings file at {valve_config_path} is empty, skipping")
                return None
            
            with open(valve_config_path, 'r') as f:
                content = f.read().strip()
            
            if not content:
                logger.debug(f"Valve settings file contains only whitespace, skipping")
                return None
            
            settings = json.loads(content)
            
            logger.info(f"✓ Loaded persisted valve settings from {valve_config_path}")
            logger.debug(f"  Loaded settings: {list(settings.keys())}")
            return settings
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in valve_settings.json: {e}. File will be overwritten on next save.")
            return None
        except Exception as e:
            logger.error(f"Error loading persisted valve settings: {e}")
            return None

    def _save_persisted_valve_settings(self, valves: 'Filter.Valves') -> bool:
        """
        Save current valve settings to persistent JSON file.
        
        This provides a workaround for OpenWebUI's config persistence issues.
        Whenever valves are loaded or changed, they're saved to this file as backup.
        
        Args:
            valves: The Valves object to persist
            
        Returns:
            True if save was successful, False otherwise.
        """
        try:
            # Try to use OpenWebUI's data directory first, fall back to local Logs directory
            valve_config_path = "/app/backend/data/valve_settings.json"
            
            # Check if we're running in Docker/container
            if not os.path.exists("/app/backend/data"):
                # Fall back to local Logs directory (works for local/non-Docker setup)
                valve_config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "Logs",
                    "valve_settings.json"
                )
            
            logger.debug(f"🔹 Attempting to save valve settings to: {valve_config_path}")
            
            # Convert Pydantic model to dict
            valve_dict = valves.model_dump()
            
            # Try to create directory
            config_dir = os.path.dirname(valve_config_path)
            logger.debug(f"🔹 Creating directory if needed: {config_dir}")
            os.makedirs(config_dir, exist_ok=True)
            
            logger.debug(f"🔹 Writing JSON to file...")
            with open(valve_config_path, 'w') as f:
                json.dump(valve_dict, f, indent=2)
            
            logger.info(f"✓ Persisted valve settings to {valve_config_path}")
            logger.debug(f"  Settings saved: {list(valve_dict.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving persisted valve settings: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Attempted path: {valve_config_path if 'valve_config_path' in locals() else 'unknown'}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            return False

    def _initialize_error_tracking_file(self) -> bool:
        """
        Initialize error tracking JSON file if it doesn't exist.
        
        Creates error tracking file in directory specified by AI_MEMORY_LOG_DIR
        (defaults to ~/.ai_memory/logs/memory_validation_errors.json).
        
        Returns:
            True if file created or already exists, False on error.
        """
        try:
            log_dir = self._get_error_log_dir()
            error_file_path = os.path.join(log_dir, "memory_validation_errors.json")
            
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            # If file already exists, don't overwrite
            if os.path.exists(error_file_path):
                logger.debug(f"Error tracking file already exists: {error_file_path}")
                return True
            
            # Create initial structure
            initial_structure = {
                "resolved": False,
                "last_updated": None,
                "active_errors": [],
                "error_history": []
            }
            
            with open(error_file_path, 'w') as f:
                json.dump(initial_structure, f, indent=2)
            
            logger.info(f"✓ Initialized error tracking file: {error_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing error tracking file: {e}")
            return False

    def _load_error_tracking_file(self) -> Dict[str, Any]:
        """
        Load error tracking data from JSON file.
        
        Returns:
            Dictionary with structure: {resolved, last_updated, active_errors, error_history}
            Returns empty structure if file doesn't exist.
        """
        try:
            log_dir = self._get_error_log_dir()
            error_file_path = os.path.join(log_dir, "memory_validation_errors.json")
            
            if not os.path.exists(error_file_path):
                return {
                    "resolved": False,
                    "last_updated": None,
                    "active_errors": [],
                    "error_history": []
                }
            
            with open(error_file_path, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading error tracking file: {e}")
            return {
                "resolved": False,
                "last_updated": None,
                "active_errors": [],
                "error_history": []
            }

    def _save_error_tracking_file(self, error_data: Dict[str, Any]) -> bool:
        """
        Save error tracking data to JSON file.
        
        Args:
            error_data: Dictionary with error tracking structure
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            log_dir = self._get_error_log_dir()
            error_file_path = os.path.join(log_dir, "memory_validation_errors.json")
            
            os.makedirs(log_dir, exist_ok=True)
            
            with open(error_file_path, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            logger.debug(f"✓ Updated error tracking file")
            return True
            
        except Exception as e:
            logger.error(f"Error saving error tracking file: {e}")
            return False

    def _track_memory_validation_error(
        self,
        error_type: str,
        model_card: str,
        details: str,
        example_memory: Optional[str] = None,
        turn_window: Optional[str] = None
    ) -> None:
        """
        Track a memory validation error in the error tracking file.
        
        Args:
            error_type: Type of error (wrong_bank, missing_character_tag, etc.)
            model_card: Model card name (e.g., "Celine", "Tara")
            details: Description of what went wrong
            example_memory: Example memory that failed (optional)
            turn_window: Turn range where error occurred (optional)
        """
        try:
            error_data = self._load_error_tracking_file()
            now = datetime_module.datetime.utcnow().isoformat() + "Z"
            
            # Check if this error type already exists for this model card
            existing_error = None
            for error in error_data["active_errors"]:
                if error["error_type"] == error_type and error["model_card"] == model_card:
                    existing_error = error
                    break
            
            if existing_error:
                # Update existing error
                existing_error["attempt_count"] += 1
                existing_error["last_occurrence"] = now
                if turn_window:
                    existing_error["turn_window"] = turn_window
                if example_memory:
                    existing_error["example_memory"] = example_memory
            else:
                # Create new error entry
                new_error = {
                    "error_type": error_type,
                    "model_card": model_card,
                    "first_occurrence": now,
                    "last_occurrence": now,
                    "attempt_count": 1,
                    "details": details,
                    "example_memory": example_memory,
                    "turn_window": turn_window
                }
                error_data["active_errors"].append(new_error)
            
            error_data["last_updated"] = now
            
            self._save_error_tracking_file(error_data)
            logger.debug(f"✓ Tracked error: {error_type} for {model_card} (attempt {existing_error['attempt_count'] if existing_error else 1})")
            
        except Exception as e:
            logger.error(f"Error tracking validation error: {e}")

    def _check_should_halt_for_errors(self, model_card: str, turn_limit: int = 10) -> bool:
        """
        Check if system should halt due to repeated errors for a model card.
        
        Halts if same error type has 3+ occurrences within the turn limit.
        
        Args:
            model_card: Model card to check
            turn_limit: Number of turns to consider (default 10)
            
        Returns:
            True if system should halt (3+ errors detected), False otherwise.
        """
        try:
            error_data = self._load_error_tracking_file()
            
            # Count errors for this model card
            model_errors = [
                e for e in error_data["active_errors"]
                if e["model_card"] == model_card
            ]
            
            # Check if any error type has 3+ attempts
            for error in model_errors:
                if error["attempt_count"] >= 3:
                    logger.warning(
                        f"⚠️ HALTING MEMORY EXTRACTION: {error['error_type']} occurred "
                        f"{error['attempt_count']} times for {model_card}. "
                        f"Error tracking file flagged for manual review."
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking halt condition: {e}")
            return False

    def _count_validation_errors_for_model(self, model_card: str) -> int:
        """
        Count total validation errors for a model card.
        
        Args:
            model_card: Model card to check
            
        Returns:
            Total count of validation errors for this model card
        """
        try:
            error_data = self._load_error_tracking_file()
            
            # Count errors for this model card
            total_errors = sum(
                e["attempt_count"] for e in error_data["active_errors"]
                if e["model_card"] == model_card
            )
            
            return total_errors
            
        except Exception as e:
            logger.error(f"Error counting validation errors: {e}")
            return 0

    def _validate_memory_extraction(self, extraction_output: str, model_card: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate LLM memory extraction output for correctness.
        
        Checks:
        1. Valid JSON structure with status/reason/memories
        2. status field is "success" or "no_memories_found"
        3. reason field is non-empty string
        4. memories is a list
        5. Each memory has operation, content, tags, memory_bank
        6. memory_bank is valid (General, Personal, Work, Projects, Technical, Tasks, Research, Context, Patterns, Preferences, Temporary, Character, Character_Interaction)
        7. Character memories have character_[name] tag
        8. Persistence tags are correct (persistent_character, temporary_session, or persistent)
        
        Args:
            extraction_output: Raw string response from LLM
            model_card: Model card name for error tracking context
            
        Returns:
            Dictionary:
            {
                "is_valid": bool,
                "error_type": str or None (wrong_bank, missing_character_tag, invalid_json, etc.),
                "error_details": str or None,
                "parsed_data": dict or None (valid parsed JSON if successful),
                "example_memory": str or None (example of problematic memory if error)
            }
        """
        # Valid memory banks
        VALID_BANKS = {
            "General", "Personal", "Work", "Projects", "Technical", "Tasks",
            "Research", "Context", "Patterns", "Preferences", "Temporary",
            "Character", "Character_Interaction"
        }
        
        try:
            # Step 1: Parse JSON
            try:
                parsed = json.loads(extraction_output)
            except json.JSONDecodeError as e:
                return {
                    "is_valid": False,
                    "error_type": "invalid_json",
                    "error_details": f"Failed to parse JSON: {str(e)}. Output: {extraction_output[:200]}",
                    "parsed_data": None,
                    "example_memory": extraction_output[:100]
                }
            
            # Step 2: Check required top-level fields
            if not isinstance(parsed, dict):
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "Output must be a JSON object, not a list or primitive",
                    "parsed_data": None,
                    "example_memory": json.dumps(parsed)[:100]
                }
            
            if "status" not in parsed:
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "Missing 'status' field in JSON object",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            if "reason" not in parsed:
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "Missing 'reason' field in JSON object",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            if "memories" not in parsed:
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "Missing 'memories' field in JSON object",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            # Step 3: Check status value
            if parsed["status"] not in ("success", "no_memories_found"):
                return {
                    "is_valid": False,
                    "error_type": "invalid_status_value",
                    "error_details": f"status must be 'success' or 'no_memories_found', got: {parsed['status']}",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            # Step 4: Check reason is non-empty string
            if not isinstance(parsed["reason"], str) or not parsed["reason"].strip():
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "reason must be a non-empty string",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            # Step 5: Check memories is a list
            if not isinstance(parsed["memories"], list):
                return {
                    "is_valid": False,
                    "error_type": "missing_required_field",
                    "error_details": "memories must be a list",
                    "parsed_data": None,
                    "example_memory": None
                }
            
            # Step 6: If no memories found, that's valid
            if parsed["status"] == "no_memories_found":
                return {
                    "is_valid": True,
                    "error_type": None,
                    "error_details": None,
                    "parsed_data": parsed,
                    "example_memory": None
                }
            
            # Step 7: Validate each memory object
            for i, memory in enumerate(parsed["memories"]):
                if not isinstance(memory, dict):
                    return {
                        "is_valid": False,
                        "error_type": "missing_required_field",
                        "error_details": f"Memory {i} is not a dict",
                        "parsed_data": None,
                        "example_memory": str(memory)[:100]
                    }
                
                # Check required fields in memory
                for required_field in ["operation", "content", "tags", "memory_bank"]:
                    if required_field not in memory:
                        return {
                            "is_valid": False,
                            "error_type": "missing_required_field",
                            "error_details": f"Memory {i} missing '{required_field}' field",
                            "parsed_data": None,
                            "example_memory": memory.get("content", "unknown")[:100]
                        }
                
                # Check operation value
                if memory["operation"] not in ("NEW", "UPDATE", "DELETE"):
                    return {
                        "is_valid": False,
                        "error_type": "invalid_operation",
                        "error_details": f"Memory {i} has invalid operation: {memory['operation']}",
                        "parsed_data": None,
                        "example_memory": memory.get("content", "unknown")[:100]
                    }
                
                # Check memory_bank is valid
                if memory["memory_bank"] not in VALID_BANKS:
                    return {
                        "is_valid": False,
                        "error_type": "wrong_bank",
                        "error_details": f"Memory {i} has invalid bank: {memory['memory_bank']}. Valid banks: {', '.join(sorted(VALID_BANKS))}",
                        "parsed_data": None,
                        "example_memory": memory.get("content", "unknown")[:100]
                    }
                
                # Check tags is a list
                if not isinstance(memory.get("tags"), list):
                    return {
                        "is_valid": False,
                        "error_type": "missing_required_field",
                        "error_details": f"Memory {i} tags must be a list",
                        "parsed_data": None,
                        "example_memory": memory.get("content", "unknown")[:100]
                    }
                
                # Check tags are non-empty strings
                for j, tag in enumerate(memory.get("tags", [])):
                    if not isinstance(tag, str) or not tag.strip():
                        return {
                            "is_valid": False,
                            "error_type": "missing_required_field",
                            "error_details": f"Memory {i} tag {j} must be non-empty string",
                            "parsed_data": None,
                            "example_memory": memory.get("content", "unknown")[:100]
                        }
                
                # Character memory validation: must have character_[name] tag
                if memory["memory_bank"] in ("Character", "Character_Interaction"):
                    has_character_tag = any(tag.startswith("character_") for tag in memory.get("tags", []))
                    if not has_character_tag:
                        return {
                            "is_valid": False,
                            "error_type": "missing_character_tag",
                            "error_details": f"Memory {i} in 'Character' bank must have character_[name] tag",
                            "parsed_data": None,
                            "example_memory": memory.get("content", "unknown")[:100]
                        }
                
                # Check persistence tags are correct
                persistence_tags = {tag for tag in memory.get("tags", []) if "persistent" in tag or "temporary" in tag}
                for ptag in persistence_tags:
                    if ptag not in ("persistent", "persistent_character", "temporary_session"):
                        return {
                            "is_valid": False,
                            "error_type": "persistence_flag_error",
                            "error_details": f"Memory {i} has invalid persistence tag: {ptag}. Valid: persistent, persistent_character, temporary_session",
                            "parsed_data": None,
                            "example_memory": memory.get("content", "unknown")[:100]
                        }
                
                # Check content is non-empty string
                if not isinstance(memory.get("content"), str) or not memory["content"].strip():
                    return {
                        "is_valid": False,
                        "error_type": "missing_required_field",
                        "error_details": f"Memory {i} content must be non-empty string",
                        "parsed_data": None,
                        "example_memory": "empty"
                    }
            
            # All validations passed
            return {
                "is_valid": True,
                "error_type": None,
                "error_details": None,
                "parsed_data": parsed,
                "example_memory": None
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in _validate_memory_extraction: {e}")
            return {
                "is_valid": False,
                "error_type": "validation_error",
                "error_details": f"Unexpected validation error: {str(e)}",
                "parsed_data": None,
                "example_memory": None
            }

    def _auto_correct_memory_extraction(self, parsed_data: Dict[str, Any], validation_result: Dict[str, Any], model_card: Optional[str] = None) -> Dict[str, Any]:
        """
        Auto-correct obvious memory extraction errors without retrying.
        
        Fixes:
        1. Missing character_[name] tag - adds based on context if detectable
        2. Wrong bank assignment - fixes Character memories in wrong banks
        3. Invalid persistence tags - corrects to valid ones
        
        Args:
            parsed_data: The parsed JSON from LLM
            validation_result: The validation result dictionary from _validate_memory_extraction
            model_card: Model card name for context (optional)
            
        Returns:
            Dictionary:
            {
                "was_corrected": bool,
                "corrections_made": list of str (what was fixed),
                "corrected_data": dict (the corrected parsed_data),
                "still_invalid": bool (True if correction failed to make it valid)
            }
        """
        corrections = []
        corrected = copy.deepcopy(parsed_data)
        
        try:
            # No corrections needed if already valid
            if validation_result["is_valid"]:
                return {
                    "was_corrected": False,
                    "corrections_made": [],
                    "corrected_data": corrected,
                    "still_invalid": False
                }
            
            error_type = validation_result.get("error_type")
            
            # Fix 1: Missing character tag
            if error_type == "missing_character_tag":
                # Try to infer character name from model_card or content
                character_name = None
                if model_card and model_card.lower() != "friday":
                    character_name = model_card.lower()
                
                # Scan through memories and add character_[name] tag to Character bank memories
                if corrected.get("memories"):
                    for memory in corrected["memories"]:
                        if memory.get("memory_bank") in ("Character", "Character_Interaction"):
                            # Check if character tag already exists
                            has_char_tag = any(tag.startswith("character_") for tag in memory.get("tags", []))
                            if not has_char_tag and character_name:
                                memory["tags"].append(f"character_{character_name}")
                                corrections.append(f"Added character_{character_name} tag to memory: {memory.get('content', '')[:50]}...")
                            elif not has_char_tag:
                                # If we can't determine character name, infer from content if possible
                                corrections.append(f"Memory in Character bank missing character_[name] tag, manual fix needed: {memory.get('content', '')[:50]}...")
            
            # Fix 2: Wrong bank assignment (Character memories in General/Personal)
            if error_type == "wrong_bank":
                if corrected.get("memories"):
                    for memory in corrected["memories"]:
                        # If memory has character tag but wrong bank, fix bank
                        has_char_tag = any(tag.startswith("character_") for tag in memory.get("tags", []))
                        if has_char_tag and memory.get("memory_bank") not in ("Character", "Character_Interaction"):
                            old_bank = memory.get("memory_bank")
                            # Decide if it should be Character or Character_Interaction
                            if "interaction" in memory.get("tags", []):
                                memory["memory_bank"] = "Character_Interaction"
                            else:
                                memory["memory_bank"] = "Character"
                            corrections.append(f"Fixed bank: {old_bank} → {memory['memory_bank']} for memory: {memory.get('content', '')[:50]}...")
                        
                        # If memory has temporary_session tag but in wrong bank, fix to Temporary
                        has_temp_tag = "temporary_session" in memory.get("tags", [])
                        if has_temp_tag and memory.get("memory_bank") != "Temporary":
                            old_bank = memory.get("memory_bank")
                            memory["memory_bank"] = "Temporary"
                            corrections.append(f"Fixed bank: {old_bank} → Temporary for session-only memory: {memory.get('content', '')[:50]}...")
            
            # Fix 3: Invalid persistence tags
            if error_type == "persistence_flag_error":
                if corrected.get("memories"):
                    for memory in corrected["memories"]:
                        tags = memory.get("tags", [])
                        # Find invalid persistence tags
                        for i, tag in enumerate(tags):
                            if "persistent" in tag or "temporary" in tag:
                                if tag not in ("persistent", "persistent_character", "temporary_session"):
                                    # Try to infer correct tag
                                    if memory.get("memory_bank") == "Character":
                                        new_tag = "persistent_character"
                                    elif memory.get("memory_bank") == "Temporary":
                                        new_tag = "temporary_session"
                                    else:
                                        new_tag = "persistent"
                                    
                                    old_tag = tags[i]
                                    tags[i] = new_tag
                                    corrections.append(f"Fixed persistence tag: {old_tag} → {new_tag}")
            
            # Validate corrected data
            revalidation = self._validate_memory_extraction(json.dumps(corrected), model_card)
            
            return {
                "was_corrected": len(corrections) > 0,
                "corrections_made": corrections,
                "corrected_data": corrected,
                "still_invalid": not revalidation["is_valid"],
                "revalidation_error": revalidation.get("error_type") if not revalidation["is_valid"] else None
            }
            
        except Exception as e:
            logger.error(f"Error during auto-correction: {e}")
            return {
                "was_corrected": False,
                "corrections_made": [],
                "corrected_data": corrected,
                "still_invalid": True,
                "error": str(e)
            }

    async def _retry_memory_extraction_with_feedback(
        self,
        user_message: str,
        assistant_message: str,
        validation_error: Dict[str, Any],
        current_extraction: str,
        retry_attempt: int = 1,
        model_card: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retry memory extraction with feedback about what went wrong.
        
        Constructs a feedback prompt that tells the LLM specifically:
        - What validation error occurred
        - What went wrong (with examples)
        - How to fix it
        
        Then retries the extraction call.
        
        Args:
            user_message: Original user message
            assistant_message: Original assistant/model response
            validation_error: The validation result dict from _validate_memory_extraction
            current_extraction: The invalid extraction output
            retry_attempt: Which retry this is (1 or 2)
            model_card: Model card name for context
            
        Returns:
            Dictionary:
            {
                "retry_succeeded": bool,
                "extraction_output": str,
                "parsed_data": dict or None,
                "error_type": str or None,
                "error_details": str or None
            }
        """
        try:
            error_type = validation_error.get("error_type", "unknown")
            error_details = validation_error.get("error_details", "unknown error")
            
            # Build feedback prompt based on error type
            feedback_prompt = f"""Your previous memory extraction had an error. Please fix it and retry.

ERROR TYPE: {error_type}
ERROR DETAILS: {error_details}

YOUR PREVIOUS OUTPUT:
{current_extraction[:500]}

INSTRUCTIONS TO FIX:
"""
            
            if error_type == "invalid_json":
                feedback_prompt += """
1. Ensure your output is ONLY a valid JSON object
2. Do NOT wrap the JSON in markdown code blocks like this: (```json ... ```)
3. Do NOT include any text before or after the JSON
4. The JSON must have exactly this structure:
{
  "status": "success" or "no_memories_found",
  "reason": "brief explanation",
  "memories": [
    {"operation": "NEW", "content": "...", "tags": [...], "memory_bank": "..."},
    ...
  ]
}
5. Make sure all strings are properly escaped
6. Make sure all JSON is valid"""
            
            elif error_type == "missing_required_field":
                feedback_prompt += """
1. Every memory object MUST have these four fields:
   - "operation": must be "NEW", "UPDATE", or "DELETE"
   - "content": the actual memory text (non-empty)
   - "tags": a list of tag strings (non-empty list)
   - "memory_bank": one of these: General, Personal, Work, Projects, Technical, Tasks, Research, Context, Patterns, Preferences, Temporary, Character, Character_Interaction
2. Every memory must be a complete object with all fields present
3. Do not omit any required fields"""
            
            elif error_type == "wrong_bank":
                feedback_prompt += """
1. Use the CORRECT memory bank for each memory:
   - "General" for general knowledge or misc
   - "Personal" for personal preferences/traits
   - "Work" for work-related info
   - "Projects" for ongoing projects
   - "Technical" for technical/system info
   - "Character" for roleplay character traits/experiences
   - "Character_Interaction" for interactions with characters
   - "Temporary" for session-only, non-persistent memories
2. Character memories MUST go in "Character" or "Character_Interaction" banks
3. Do NOT put character memories in "General" or "Personal" banks"""
            
            elif error_type == "missing_character_tag":
                feedback_prompt += """
1. Any memory in "Character" or "Character_Interaction" bank MUST have a character name tag
2. The tag format is: "character_[charactername]" (all lowercase, underscore-separated)
3. Examples: "character_celine", "character_tara", "character_willow"
4. Every character memory needs at least this character tag in addition to other tags"""
            
            elif error_type == "persistence_flag_error":
                feedback_prompt += """
1. Use ONLY these persistence-related tags:
   - "persistent" for memories that should always be kept
   - "persistent_character" for character memories that should be kept indefinitely
   - "temporary_session" for session-only memories that will be purged after 30 days
2. Do NOT invent other persistence-related tags
3. Choose the right persistence tag based on the memory bank:
   - Character bank → use "persistent_character"
   - Temporary bank → use "temporary_session"
   - Other banks → use "persistent" """
            
            else:
                feedback_prompt += f"""
Please review your output and fix the issue: {error_details}
Ensure your JSON is valid and follows the required format."""
            
            feedback_prompt += f"""

NOW RETRY:
Analyze the following messages again and extract memories correctly:

>>> USER MESSAGE <<<
{user_message}

>>> ASSISTANT RESPONSE <<<
{assistant_message}

Produce ONLY the corrected JSON output following the format specified in the system prompt."""
            
            logger.info(f"⏳ Retrying memory extraction (attempt {retry_attempt}/2) for {model_card or 'user'}")
            logger.debug(f"Feedback prompt: {feedback_prompt[:300]}...")
            
            # Make the retry LLM call using the same method as extraction
            try:
                retry_output = await self.query_llm_with_retry(
                    self.valves.memory_identification_prompt,
                    feedback_prompt
                )
            except Exception as e:
                logger.error(f"LLM call failed during retry: {e}")
                return {
                    "retry_succeeded": False,
                    "extraction_output": None,
                    "parsed_data": None,
                    "error_type": "llm_call_failed",
                    "error_details": str(e)
                }
            
            if not retry_output or retry_output.startswith("Error:"):
                return {
                    "retry_succeeded": False,
                    "extraction_output": retry_output,
                    "parsed_data": None,
                    "error_type": "llm_call_failed",
                    "error_details": retry_output if retry_output else "LLM returned empty response"
                }
            
            # Validate the retry output
            revalidation = self._validate_memory_extraction(retry_output, model_card)
            
            if revalidation["is_valid"]:
                logger.info(f"✓ Retry attempt {retry_attempt} succeeded - memory extraction now valid")
                return {
                    "retry_succeeded": True,
                    "extraction_output": retry_output,
                    "parsed_data": revalidation["parsed_data"],
                    "error_type": None,
                    "error_details": None
                }
            else:
                logger.warning(f"⚠️ Retry attempt {retry_attempt} still has error: {revalidation['error_type']}")
                return {
                    "retry_succeeded": False,
                    "extraction_output": retry_output,
                    "parsed_data": None,
                    "error_type": revalidation["error_type"],
                    "error_details": revalidation["error_details"]
                }
        
        except Exception as e:
            logger.error(f"Error during retry: {e}")
            return {
                "retry_succeeded": False,
                "extraction_output": None,
                "parsed_data": None,
                "error_type": "retry_exception",
                "error_details": str(e)
            }

    def _get_embedding_model_tag(self) -> str:
        """Generate embedding model metadata tag based on configuration.
        
        Format: __embedding_model:{model_name}_{dimension}d
        Example: __embedding_model:nomic_embed_text_v1.5_768d
        
        Returns:
            Tag string reflecting current embedding model configuration
        """
        # Sanitize model name for tag (replace special chars with underscores)
        model_name = self.valves.embedding_model_name.replace("-", "_").replace(".", "_")
        dimension = self.valves.embedding_model_dimension
        tag = f"__embedding_model:{model_name}_{dimension}d"
        return tag

    async def _detect_embedding_model_info(self) -> Dict[str, Any]:
        """
        Auto-detect embedding model name and dimension from LM Studio.
        
        Queries the /v1/embeddings endpoint with a test embedding to detect:
        - Actual model name being used
        - Vector dimension of the embeddings
        
        This makes the plugin self-aware of which embedding model is running.
        
        Returns:
            Dictionary with keys:
            - 'model_name': str - The embedding model name from LM Studio
            - 'dimension': int - The embedding vector dimension
            - 'success': bool - Whether detection was successful
            - 'error': Optional[str] - Error message if detection failed
        """
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "model": self.valves.embedding_model_name,
                    "input": "test",  # Simple test string
                }
                async with session.post(
                    self.valves.embedding_api_endpoint_url, 
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract model name from response
                        model_name = data.get("model")
                        
                        # Extract dimension from embedding vector
                        if "data" in data and len(data["data"]) > 0:
                            embedding = data["data"][0].get("embedding")
                            if embedding and isinstance(embedding, list):
                                dimension = len(embedding)
                                
                                logger.info(
                                    f"✅ Auto-detected embedding model: {model_name} "
                                    f"(dimension: {dimension})"
                                )
                                
                                return {
                                    "model_name": model_name,
                                    "dimension": dimension,
                                    "success": True,
                                    "error": None,
                                }
                        
                        error_msg = "Invalid LM Studio response format - no embedding data"
                        logger.warning(error_msg)
                        return {
                            "model_name": None,
                            "dimension": None,
                            "success": False,
                            "error": error_msg,
                        }
                    else:
                        error_msg = f"LM Studio API error {response.status}"
                        logger.error(error_msg)
                        return {
                            "model_name": None,
                            "dimension": None,
                            "success": False,
                            "error": error_msg,
                        }
                        
        except asyncio.TimeoutError:
            error_msg = "Timeout querying LM Studio for model info"
            logger.error(error_msg)
            return {
                "model_name": None,
                "dimension": None,
                "success": False,
                "error": error_msg,
            }
        except Exception as e:
            error_msg = f"Error detecting embedding model: {str(e)}"
            logger.error(error_msg)
            return {
                "model_name": None,
                "dimension": None,
                "success": False,
                "error": error_msg,
            }

    async def _ensure_embedding_model_ready(self) -> bool:
        """
        Validate and ensure the configured embedding model is ready before use.
        
        This is called on the first embedding request to ensure LM Studio has
        the embedding model loaded and ready. Uses a fallback chain:
        1. Check if configured model is loaded
        2. If not, wait for it to load (with retries)
        3. If that fails, look for any embedding model in LM Studio
        4. If that fails, fall back to hardcoded default
        
        Updates self.valves with actual model info if different from configured.
        
        Returns:
            bool - True if a usable embedding model was found, False otherwise
        """
        if self._embedding_model_validated:
            return True  # Already validated, skip
        
        configured_model = self.valves.embedding_model_name
        logger.info(f"Validating embedding model: {configured_model}")
        
        try:
            # Try to detect what's actually loaded
            model_info = await self._detect_embedding_model_info()
            
            if model_info.get("success"):
                detected_model = model_info.get("model_name")
                detected_dimension = model_info.get("dimension")
                
                if detected_model == configured_model:
                    # Perfect match!
                    logger.info(
                        f"✅ Embedding model validated: {detected_model} "
                        f"({detected_dimension}D)"
                    )
                    self.valves.embedding_model_dimension = detected_dimension
                    self._embedding_model_validated = True
                    return True
                else:
                    # Mismatch - different model is loaded
                    logger.warning(
                        f"⚠ Model mismatch - configured: {configured_model}, "
                        f"loaded: {detected_model}"
                    )
                    logger.info(f"Using loaded model: {detected_model} ({detected_dimension}D)")
                    self.valves.embedding_model_name = detected_model
                    self.valves.embedding_model_dimension = detected_dimension
                    self._embedding_model_validated = True
                    return True
            else:
                # Detection failed - try fallback chain
                logger.warning(
                    f"Could not detect embedding model: {model_info.get('error')}"
                )
                
                # Fallback 1: Try to find ANY embedding model in LM Studio
                logger.info("Attempting fallback: searching for any embedding model...")
                try:
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(
                            f"{self.valves.embedding_api_endpoint_url}".replace("/v1/embeddings", "/v1/models")
                        ) as response:
                            if response.status == 200:
                                models_data = await response.json()
                                embedding_models = [
                                    m.get("id") for m in models_data.get("data", [])
                                    if "embed" in m.get("id", "").lower()
                                ]
                                
                                if embedding_models:
                                    fallback_model = embedding_models[0]
                                    logger.info(
                                        f"Found embedding model via fallback: {fallback_model}"
                                    )
                                    self.valves.embedding_model_name = fallback_model
                                    self._embedding_model_validated = True
                                    return True
                except Exception as e:
                    logger.debug(f"Fallback search failed: {e}")
                
                # Fallback 2: Use hardcoded default (original Nomic model)
                logger.warning(
                    "Using hardcoded default embedding model: "
                    "text-embedding-nomic-embed-text-v1.5"
                )
                self.valves.embedding_model_name = "text-embedding-nomic-embed-text-v1.5"
                self._embedding_model_validated = True
                return True
                
        except Exception as e:
            logger.error(f"Error validating embedding model: {e}")
            logger.warning(
                "Proceeding with configured model despite validation error: "
                f"{configured_model}"
            )
            self._embedding_model_validated = True
            return True  # Continue anyway to avoid blocking embedding

    async def _calculate_memory_age_days(self, memory: Dict[str, Any]) -> float:
        """Calculate age of a memory in days."""
        created_at = memory.get("created_at")
        if not created_at or not isinstance(created_at, datetime):
            return float("inf")  # Treat memories without valid dates as infinitely old

        # Ensure created_at is timezone-aware (assume UTC if not)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        # Get current time, also timezone-aware
        now_utc = datetime.now(timezone.utc)

        delta = now_utc - created_at
        return delta.total_seconds() / (24 * 3600)

    def _should_summarize_now(self, user_id: str, body: Dict[str, Any]) -> bool:
        """
        Fast, non-async check to determine if conversation summarization should happen.
        Returns boolean immediately without LLM calls.
        Runs on every inlet call but completes in microseconds.
        """
        if not self.valves.enable_conversation_summarization:
            return False

        messages = body.get("messages", [])
        if not messages:
            return False

        total_messages = len(messages)
        conversation_id = body.get("chat_id") or body.get("conversation_id") or "default"
        tracking_key = f"conv_summary_{user_id}_{conversation_id}"

        # Initialize tracking dict if needed
        if not hasattr(self, "_summary_tracking"):
            self._summary_tracking = {}

        # Get last count when summary was generated
        last_summarized_count = self._summary_tracking.get(tracking_key, 0)

        threshold = self.valves.conversation_summarization_threshold
        interval = self.valves.conversation_summarization_interval

        # First summarization when reaching threshold
        if last_summarized_count == 0 and total_messages >= threshold:
            return True

        # Subsequent summarizations at interval
        if (
            last_summarized_count > 0
            and interval > 0
            and (total_messages - last_summarized_count) >= interval
        ):
            return True

        return False

    async def _generate_conversation_summary(
        self, body: Dict[str, Any], user_id: str
    ) -> Optional[str]:
        """
        Generate a summary of recent conversation via LLM.
        Only called when _should_summarize_now() returns True.
        This is the ~3-4 second LLM operation.
        """
        try:
            messages = body.get("messages", [])
            if not messages:
                return None

            total_messages = len(messages)
            conversation_id = body.get("chat_id") or body.get("conversation_id") or "default"
            tracking_key = f"conv_summary_{user_id}_{conversation_id}"

            # Get last count when summary was generated
            last_summarized_count = self._summary_tracking.get(tracking_key, 0)

            # Determine which messages to summarize
            if last_summarized_count == 0:
                # First time: summarize from beginning up to threshold
                threshold = self.valves.conversation_summarization_threshold
                messages_to_summarize = messages[:threshold]
            else:
                # Subsequent: summarize from last summary point to now
                messages_to_summarize = messages[last_summarized_count:]

            if not messages_to_summarize:
                return None

            # Format messages for LLM
            conversation_text = ""
            for msg in messages_to_summarize:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    conversation_text += f"{role.upper()}: {content}\n\n"

            if not conversation_text.strip():
                return None

            # Create summarization prompt
            system_prompt = self.valves.conversation_summarization_prompt
            user_prompt = f"Conversation to summarize:\n\n{conversation_text}\n\nPlease provide a concise summary of this conversation segment."

            # Call LLM for summarization
            summary = await self.query_llm_with_retry(system_prompt, user_prompt)

            if not summary or summary.startswith("Error:"):
                logger.error(f"Failed to generate conversation summary: {summary}")
                return None

            # Clean up summary
            summary = summary.strip()
            if not summary:
                return None

            # Update tracking to record that we summarized up to this point
            self._summary_tracking[tracking_key] = total_messages
            logger.info(
                f"Generated summary for user {user_id}, conversation {conversation_id}: "
                f"{len(messages_to_summarize)} messages → {len(summary)} chars"
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return None

    def _inject_summary_into_context(
        self, body: Dict[str, Any], summary: str, system_prompt: Optional[str] = None
    ) -> None:
        """
        Inject conversation summary into the context for the model.
        Adds it to the system message so model has context about earlier conversation.
        Can optionally include system prompt from model card.
        """
        try:
            if not summary:
                return

            # Get or create system message
            messages = body.get("messages", [])
            if not messages:
                return

            # Find existing system message
            system_msg_index = None
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    system_msg_index = i
                    break

            # Build the injection content
            injection = f"\n\n[Earlier Conversation Summary]\n{summary}"

            if system_msg_index is not None:
                # Append to existing system message (preserves Friday's personality)
                messages[system_msg_index]["content"] += injection
                logger.debug(f"Injected conversation summary into existing system message")
            elif system_prompt:
                # Only create new system message if we have Friday's actual system prompt
                system_content = f"{system_prompt}{injection}"
                messages.insert(0, {"role": "system", "content": system_content})
                logger.debug(f"Injected conversation summary with system prompt from model card")
            else:
                # Don't inject if we can't preserve Friday's personality
                logger.warning(
                    f"Skipping summary injection: No existing system message and no system prompt from model card. "
                    f"This preserves Friday's configured personality."
                )

        except Exception as e:
            logger.error(f"Error injecting conversation summary into context: {e}")

    async def _process_memory_queue(self):
        """Continuously process queued memory operations"""
        logger.info("Memory task queue processor started")
        while True:
            try:
                # Get next task from queue (blocks until available)
                task_data = await self._memory_task_queue.get()
                conversation_id = task_data.get('conversation_id', 'unknown')
                
                logger.info(f"Processing queued memory extraction for conversation {conversation_id}")
                
                # Use semaphore to limit concurrent operations
                async with self._memory_task_semaphore:
                    try:
                        await self._execute_memory_extraction(task_data)
                        self._completed_memory_tasks[conversation_id] = {
                            'status': 'completed',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        logger.info(f"✓ Memory extraction completed for conversation {conversation_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"⏱️ Memory extraction timeout for conversation {conversation_id}")
                        self._completed_memory_tasks[conversation_id] = {
                            'status': 'timeout',
                            'timestamp': datetime.now(timezone.utc)
                        }
                    except Exception as e:
                        logger.error(f"❌ Memory extraction failed for conversation {conversation_id}: {e}")
                        self._completed_memory_tasks[conversation_id] = {
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.now(timezone.utc)
                        }
                    finally:
                        # Mark task as done
                        self._memory_task_queue.task_done()
                        # Remove from active tasks
                        self._active_memory_tasks.pop(conversation_id, None)
                        
            except Exception as e:
                logger.error(f"Error in memory queue processor: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(1)  # Brief pause on error to prevent tight loop

    async def _execute_memory_extraction(self, task_data):
        """Execute a single memory extraction operation with timeout"""
        try:
            await asyncio.wait_for(
                self._process_user_memories(
                    user_message=task_data['user_message'],
                    user_id=task_data['user_id'],
                    event_emitter=task_data.get('event_emitter'),
                    show_status=task_data['show_status'],
                    user_timezone=task_data['user_timezone'],
                    recent_chat_history=task_data.get('recent_chat_history', []),
                    assistant_message=task_data.get('assistant_message'),
                ),
                timeout=300  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Memory extraction timeout after 5 minutes")
            raise
        except Exception as e:
            logger.error(f"Error during memory extraction: {e}\n{traceback.format_exc()}")
            raise            

    async def _summarize_old_memories_loop(self):
        """Periodically summarize old memories into concise summaries"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                # Use configurable interval with small random jitter to prevent thundering herd
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.summarization_interval * jitter
                await asyncio.sleep(interval)
                logger.info("Starting periodic memory summarization run...")

                try:
                    # Fetch all users (or handle single user case)
                    # For now, assuming single user for simplicity, adapt if multi-user support needed
                    user_id = "default"  # Replace with actual user ID logic if needed
                    user_obj = Users.get_user_by_id(user_id)
                    if not user_obj:
                        logger.warning(
                            f"Summarization skipped: User '{user_id}' not found."
                        )
                        continue

                    # Get all memories for the user
                    all_user_memories = await self._get_formatted_memories(user_id)
                    if (
                        len(all_user_memories)
                        < self.valves.summarization_min_cluster_size
                    ):
                        logger.info(
                            f"Summarization skipped: Not enough memories for user '{user_id}' to form a cluster."
                        )
                        continue

                    logger.debug(
                        f"Retrieved {len(all_user_memories)} total memories for user '{user_id}' for summarization."
                    )

                    # Find clusters of related, old memories
                    memory_clusters = await self._find_memory_clusters(
                        all_user_memories
                    )

                    if not memory_clusters:
                        logger.info(
                            f"No eligible memory clusters found for user '{user_id}' for summarization."
                        )
                        continue

                    logger.info(
                        f"Found {len(memory_clusters)} memory clusters to potentially summarize for user '{user_id}'."
                    )

                    # Process each cluster
                    summarized_count = 0
                    deleted_count = 0
                    for cluster in memory_clusters:
                        # Ensure cluster still meets minimum size after potential filtering in _find_memory_clusters
                        if len(cluster) < self.valves.summarization_min_cluster_size:
                            continue

                        # Limit cluster size for the LLM call
                        cluster_to_summarize = cluster[
                            : self.valves.summarization_max_cluster_size
                        ]
                        logger.debug(
                            f"Attempting to summarize cluster of size {len(cluster_to_summarize)} (max: {self.valves.summarization_max_cluster_size})."
                        )

                        # Extract memory texts for the LLM prompt
                        mem_texts = [m.get("memory", "") for m in cluster_to_summarize]
                        # Sort by date to help LLM resolve contradictions potentially
                        cluster_to_summarize.sort(
                            key=lambda m: m.get(
                                "created_at", datetime.min.replace(tzinfo=timezone.utc)
                            )
                        )
                        combined_text = "\n- ".join(
                            [m.get("memory", "") for m in cluster_to_summarize]
                        )

                        # Use the new configurable summarization prompt
                        system_prompt = self.valves.summarization_memory_prompt
                        user_prompt = (
                            f"Related memories to summarize:\n- {combined_text}"
                        )

                        logger.debug(
                            f"Calling LLM to summarize cluster. System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}"
                        )
                        summary = await self.query_llm_with_retry(
                            system_prompt, user_prompt
                        )

                        if summary and not summary.startswith("Error:"):
                            # Format summary with tags (e.g., from the first memory in cluster? Or generate new ones?)
                            # For simplicity, let's try inheriting tags from the *first* memory in the sorted cluster
                            first_mem_content = cluster_to_summarize[0].get(
                                "memory", ""
                            )
                            tags = []
                            tags_match = re.match(r"\[Tags: (.*?)\]", first_mem_content)
                            if tags_match:
                                tags = [
                                    tag.strip()
                                    for tag in tags_match.group(1).split(",")
                                ]

                            # Add a specific "summarized" tag
                            if "summarized" not in tags:
                                tags.append("summarized")

                            formatted_summary = (
                                f"[Tags: {', '.join(tags)}] {summary.strip()}"
                            )

                            logger.info(
                                f"Generated summary for cluster: {formatted_summary[:100]}..."
                            )

                            # Save summary as new memory
                            try:
                                new_mem_op = MemoryOperation(
                                    operation="NEW",
                                    content=formatted_summary,
                                    tags=tags,
                                    model_card_name=self._current_model_card_name,
                                )
                                await self._execute_memory_operation(
                                    new_mem_op, user_obj
                                )
                                summarized_count += 1
                            except Exception as add_err:
                                logger.error(
                                    f"Failed to save summary memory: {add_err}"
                                )
                                continue  # Skip deleting originals if saving summary fails

                            # Delete original memories in the summarized cluster
                            for mem_to_delete in cluster_to_summarize:
                                try:
                                    delete_op = MemoryOperation(
                                        operation="DELETE", id=mem_to_delete["id"]
                                    )
                                    await self._execute_memory_operation(
                                        delete_op, user_obj
                                    )
                                    deleted_count += 1
                                except Exception as del_err:
                                    logger.warning(
                                        f"Failed to delete old memory {mem_to_delete.get('id')} during summarization: {del_err}"
                                    )
                                    # Continue deleting others even if one fails
                            logger.debug(
                                f"Deleted {deleted_count} original memories after summarization."
                            )
                        else:
                            logger.warning(
                                f"LLM failed to generate summary for cluster starting with ID {cluster_to_summarize[0].get('id')}. Response: {summary}"
                            )

                    if summarized_count > 0:
                        logger.info(
                            f"Successfully generated {summarized_count} summaries and deleted {deleted_count} original memories for user '{user_id}'."
                        )
                    else:
                        logger.info(
                            f"No summaries were generated in this run for user '{user_id}'."
                        )
                    
                    # Reset error counter on successful iteration
                    consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in summarization loop ({consecutive_errors}/{max_consecutive_errors}): {e}\n{traceback.format_exc()}"
                    )
                    
                    # If too many consecutive errors, disable this task
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(
                            f"Memory summarization task disabled after {max_consecutive_errors} consecutive errors"
                        )
                        break
                    
                    # Exponential backoff: 2^n seconds, capped at 32 seconds
                    backoff_seconds = min(2 ** consecutive_errors, 32)
                    logger.info(
                        f"Backing off for {backoff_seconds}s before next summarization attempt"
                    )
                    await asyncio.sleep(backoff_seconds)
                    
        except asyncio.CancelledError:
            logger.info("Memory summarization task cancelled.")
        except Exception as e:
            logger.error(
                f"Fatal error in summarization task loop: {e}\n{traceback.format_exc()}"
            )

    def _update_date_info(self):
        """Update the date information dictionary with current time"""
        return {
            "iso_date": self.current_date.strftime("%Y-%m-%d"),
            "year": self.current_date.year,
            "month": self.current_date.strftime("%B"),
            "day": self.current_date.day,
            "weekday": self.current_date.strftime("%A"),
            "hour": self.current_date.hour,
            "minute": self.current_date.minute,
            "iso_time": self.current_date.strftime("%H:%M:%S"),
        }

    async def _promote_old_memories_loop(self):
        """Periodically promote old memories from OpenWebUI to Friday Memory System"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                # Use configurable interval with small random jitter to prevent thundering herd
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.memory_promotion_interval * jitter
                await asyncio.sleep(interval)
                logger.info("Starting periodic memory promotion run...")

                try:
                    # Get all users from OpenWebUI

                    # Get all users (assuming we can iterate through all users)
                    all_users = Users.get_all_users()
                    if not all_users:
                        logger.warning("No users found for memory promotion.")
                        continue

                    promoted_count = 0
                    total_users_processed = 0

                    for user in all_users:
                        user_id = str(user.id)
                        total_users_processed += 1

                        try:
                            # Get all memories for this user from OpenWebUI
                            user_memories = await self._get_formatted_memories(user_id)
                            if not user_memories:
                                logger.debug(f"No memories found for user {user_id}")
                                continue

                            # Find memories older than configured threshold
                            cutoff_date = datetime.now(timezone.utc) - timedelta(
                                days=self.valves.memory_promotion_age_threshold_days
                            )
                            old_memories = []

                            for mem in user_memories:
                                created_at = mem.get("created_at")
                                if created_at:
                                    # Handle different datetime formats
                                    if isinstance(created_at, str):
                                        try:
                                            created_at = datetime.fromisoformat(
                                                created_at.replace("Z", "+00:00")
                                            )
                                        except ValueError:
                                            logger.warning(
                                                f"Could not parse created_at date for memory {mem.get('id')}: {created_at}"
                                            )
                                            continue

                                    if created_at < cutoff_date:
                                        old_memories.append(mem)

                            if not old_memories:
                                logger.debug(
                                    f"No memories older than 90 days found for user {user_id}"
                                )
                                continue

                            logger.info(
                                f"Found {len(old_memories)} memories older than 90 days for user {user_id}"
                            )

                            # Store each old memory in Friday Memory System
                            for mem in old_memories:
                                try:
                                    # Create memory content with metadata
                                    memory_content = mem.get("memory", "")
                                    if not memory_content:
                                        continue

                                    # Extract model_card_name from metadata for proper model isolation
                                    mem_metadata = mem.get("metadata", {}) or {}
                                    model_card_name = mem_metadata.get("model_card_name", "friday")  # Fallback to "friday"
                                    
                                    # Add promotion metadata
                                    metadata = {
                                        "source": "openwebui_promotion",
                                        "original_id": mem.get("id"),
                                        "promoted_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "created_at": (
                                            mem.get("created_at").isoformat()
                                            if mem.get("created_at")
                                            else None
                                        ),
                                        "updated_at": (
                                            mem.get("updated_at").isoformat()
                                            if mem.get("updated_at")
                                            else None
                                        ),
                                    }

                                    # Use Friday Memory System tools to store
                                    if FRIDAY_MEMORY_SYSTEM_AVAILABLE:
                                        try:
                                            from friday_memory_system import (
                                                FridayMemorySystem,
                                            )

                                            memory_system = FridayMemorySystem()
                                            logger.info(f"Promoting memory '{mem.get('id')}' to Friday Memory System for user {user_id} with model_id={model_card_name}")
                                            
                                            # Try to find actual conversation_id this memory is linked to
                                            source_conversation_id = f"openwebui_user_{user_id}"  # Fallback
                                            try:
                                                # First, try to find if this memory has a composite conversation_id already
                                                # (format: chat_id_user_id_model_id)
                                                openwebui_mem_id = mem.get('id')
                                                
                                                # Check if Friday system recognizes this memory (via stored metadata)
                                                # This is tricky because OpenWebUI and Friday use different memory IDs
                                                # So instead, we'll look for memories from this user created at similar time
                                                mem_created_at = mem.get('created_at')
                                                
                                                if mem_created_at:
                                                    # Try to match to a conversation by timestamp
                                                    conversations = await memory_system.conversations_db.execute_query(
                                                        "SELECT DISTINCT conversation_id FROM conversations WHERE user_id = ? ORDER BY start_timestamp DESC LIMIT 10",
                                                        (user_id,)
                                                    )
                                                    
                                                    if conversations:
                                                        # Get the most recent conversation for this user
                                                        # (memories being promoted are typically from recent conversations)
                                                        most_recent_conv = conversations[0]
                                                        if most_recent_conv and most_recent_conv.get('conversation_id'):
                                                            source_conversation_id = most_recent_conv['conversation_id']
                                                            logger.debug(f"Found recent conversation for promoted memory: {source_conversation_id}")
                                            except Exception as link_query_error:
                                                logger.debug(f"Could not query conversation links: {link_query_error}")
                                            
                                            result = await memory_system.create_memory(
                                                content=memory_content,
                                                importance_level=5,  # Default importance
                                                memory_type="archived",
                                                source_conversation_id=source_conversation_id,
                                                tags=["promoted", "archived"],
                                                user_id=user_id,
                                                model_id=model_card_name,  # NEW: Pass model_card_name as model_id for proper isolation
                                                source="openwebui_promotion",  # CHANGE 4A: Signal that this is promoted memory
                                                wait_for_embedding=True  # IMPORTANT: Wait for embedding to complete
                                            )
                                            
                                            # Check if creation succeeded
                                            if result.get("status") == "success":
                                                embedding_status = result.get("embedding_status", "unknown")
                                                logger.info(
                                                    f"✅ Successfully promoted memory {mem.get('id')} to Friday. "
                                                    f"Memory ID: {result.get('memory_id')}, Embedding: {embedding_status}"
                                                )
                                            elif result.get("status") == "partial_failure":
                                                logger.error(
                                                    f"❌ Memory created but embedding FAILED for {mem.get('id')} -> {result.get('memory_id')}"
                                                )
                                            else:
                                                logger.warning(
                                                    f"⚠️  Unexpected status promoting {mem.get('id')}: {result.get('status')}"
                                                )
                                        
                                        except Exception as mem_sys_error:
                                            logger.error(
                                                f"❌ Error storing memory in Friday Memory System for {mem.get('id')}: {mem_sys_error}\n{traceback.format_exc()}"
                                            )
                                    else:
                                        logger.warning(
                                            "Friday Memory System not available for memory promotion"
                                        )

                                    promoted_count += 1
                                    logger.debug(
                                        f"Successfully promoted memory {mem.get('id')} for user {user_id}"
                                    )

                                except Exception as mem_error:
                                    logger.error(
                                        f"Error promoting individual memory {mem.get('id')} for user {user_id}: {mem_error}"
                                    )
                                    continue

                            # Optionally clean up promoted memories from OpenWebUI
                            if self.valves.clean_promoted_memories:
                                deleted_count = 0
                                for mem in old_memories:
                                    try:
                                        delete_op = MemoryOperation(
                                            operation="DELETE", id=mem["id"]
                                        )
                                        await self._execute_memory_operation(
                                            delete_op, user
                                        )
                                        deleted_count += 1
                                    except Exception as del_error:
                                        logger.warning(
                                            f"Error deleting promoted memory {mem.get('id')} from OpenWebUI: {del_error}"
                                        )

                                if deleted_count > 0:
                                    logger.info(
                                        f"Cleaned up {deleted_count} promoted memories from OpenWebUI for user {user_id}"
                                    )

                        except Exception as user_error:
                            logger.error(
                                f"Error processing memory promotion for user {user_id}: {user_error}"
                            )
                            continue

                    if promoted_count > 0:
                        logger.info(
                            f"Successfully promoted {promoted_count} memories across {total_users_processed} users"
                        )
                    else:
                        logger.info(
                            f"No memories were promoted in this run across {total_users_processed} users"
                        )

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in memory promotion loop ({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Memory promotion task disabled after {max_consecutive_errors} consecutive failures")
                        break
                    
                    backoff_seconds = min(2 ** consecutive_errors, 32)
                    logger.info(f"Backing off for {backoff_seconds}s before retrying memory promotion")
                    await asyncio.sleep(backoff_seconds)
        except asyncio.CancelledError:
            logger.info("Memory promotion task cancelled.")
        except Exception as e:
            logger.error(
                f"Fatal error in memory promotion task loop: {e}\n{traceback.format_exc()}"
            )

    async def _ensure_memories_linked_to_conversations_loop(self):
        """
        Periodically ensure all memories are linked to conversations.
        
        Runs every 5 hours. Finds orphaned memories (no conversation link) and matches them
        to conversations by timestamp proximity. Falls back to generic conversation_id only
        if no match can be found.
        
        This ensures no memories become disconnected from their source conversations.
        """
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                # Sleep for 5 hours with small jitter
                interval = 5 * 3600  # 5 hours in seconds
                jitter = random.uniform(0.9, 1.1)
                await asyncio.sleep(interval * jitter)
                
                logger.info("Starting periodic memory-conversation linking verification...")
                
                try:
                    if not FRIDAY_MEMORY_SYSTEM_AVAILABLE:
                        logger.warning("Friday Memory System not available, skipping linking verification")
                        continue
                    
                    from friday_memory_system import FridayMemorySystem
                    memory_system = FridayMemorySystem()
                    
                    # Query all memories from Friday system
                    all_memories = await memory_system.get_memories()
                    if not all_memories:
                        logger.debug("No memories found to verify links")
                        continue
                    
                    logger.info(f"Checking {len(all_memories)} memories for conversation links...")
                    
                    orphaned_count = 0
                    linked_count = 0
                    
                    for memory in all_memories:
                        try:
                            memory_id = memory.get("id")
                            if not memory_id:
                                continue
                            
                            # Check if this memory has a conversation link
                            existing_links = await memory_system.get_memory_conversation_links(memory_id=memory_id)
                            
                            if existing_links:
                                # Memory already has link(s)
                                linked_count += 1
                                continue
                            
                            # Memory is orphaned, try to match it to a conversation by timestamp
                            orphaned_count += 1
                            memory_created_at = memory.get("created_at")
                            
                            if not memory_created_at:
                                logger.debug(f"Memory {memory_id} has no created_at timestamp, cannot link")
                                continue
                            
                            # Parse the memory's creation timestamp
                            try:
                                if isinstance(memory_created_at, str):
                                    mem_timestamp = datetime.fromisoformat(
                                        memory_created_at.replace("Z", "+00:00")
                                    )
                                else:
                                    mem_timestamp = memory_created_at
                            except Exception as parse_err:
                                logger.debug(f"Could not parse memory timestamp {memory_created_at}: {parse_err}")
                                continue
                            
                            # Query for conversations near this timestamp (within ±1 hour)
                            all_conversations = await memory_system.conversations_db.execute_query(
                                "SELECT conversation_id, start_timestamp, user_id FROM conversations ORDER BY start_timestamp DESC"
                            )
                            
                            best_match = None
                            min_time_diff = float('inf')
                            
                            for conv in all_conversations:
                                conv_timestamp_str = conv["start_timestamp"]
                                try:
                                    if isinstance(conv_timestamp_str, str):
                                        conv_timestamp = datetime.fromisoformat(
                                            conv_timestamp_str.replace("Z", "+00:00")
                                        )
                                    else:
                                        conv_timestamp = conv_timestamp_str
                                except Exception:
                                    continue
                                
                                # Calculate time difference
                                time_diff = abs((mem_timestamp - conv_timestamp).total_seconds())
                                
                                # Match if within 1 hour and memory user matches conversation user
                                if time_diff < 3600 and time_diff < min_time_diff:
                                    if memory.get("user_id") == conv["user_id"]:
                                        min_time_diff = time_diff
                                        best_match = conv["conversation_id"]
                            
                            if best_match:
                                # Found a matching conversation, link it
                                try:
                                    await memory_system.conversations_db.link_memory_to_conversation(
                                        memory_id=memory_id,
                                        conversation_id=best_match,
                                        link_type="timestamp_matched",
                                        metadata={
                                            "source": "auto_link_routine",
                                            "time_diff_seconds": min_time_diff,
                                        }
                                    )
                                    logger.info(
                                        f"✓ Linked orphaned memory {memory_id} to conversation {best_match} (time diff: {min_time_diff}s)"
                                    )
                                except Exception as link_err:
                                    logger.warning(f"Failed to link orphaned memory {memory_id}: {link_err}")
                            else:
                                # No matching conversation found, use generic fallback
                                user_id = memory.get("user_id", "unknown")
                                fallback_conversation_id = f"orphaned_{user_id}_{memory_id[:8]}"
                                try:
                                    await memory_system.conversations_db.link_memory_to_conversation(
                                        memory_id=memory_id,
                                        conversation_id=fallback_conversation_id,
                                        link_type="orphaned_fallback",
                                        metadata={
                                            "source": "auto_link_routine",
                                            "reason": "no_matching_conversation",
                                        }
                                    )
                                    logger.warning(
                                        f"⚠️ Linked orphaned memory {memory_id} to fallback conversation {fallback_conversation_id}"
                                    )
                                except Exception as link_err:
                                    logger.warning(f"Failed to link orphaned memory to fallback: {link_err}")
                        
                        except Exception as mem_err:
                            logger.debug(f"Error processing memory {memory.get('id')}: {mem_err}")
                            continue
                    
                    logger.info(
                        f"Memory-conversation linking verification complete: "
                        f"{linked_count} already linked, {orphaned_count} orphaned, "
                        f"matched/linked as many as possible"
                    )
                    
                    consecutive_errors = 0
                
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in memory-conversation linking loop ({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Memory-conversation linking task disabled after {max_consecutive_errors} consecutive failures")
                        break
                    
                    backoff_seconds = min(2 ** consecutive_errors, 32)
                    logger.info(f"Backing off for {backoff_seconds}s before retrying linking")
                    await asyncio.sleep(backoff_seconds)
        
        except asyncio.CancelledError:
            logger.info("Memory-conversation linking task cancelled.")
        except Exception as e:
            logger.error(
                f"Fatal error in memory-conversation linking loop: {e}\n{traceback.format_exc()}"
            )

    async def _log_error_counters_loop(self):
        """Periodically log error counters"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                # Use configurable interval with small random jitter
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.error_logging_interval * jitter
                await asyncio.sleep(interval)

                try:
                    # Determine logging behaviour based on valve settings
                    if self.valves.debug_error_counter_logs:
                        # Verbose debug logging – every interval
                        logger.debug(f"Error counters: {self.error_counters}")
                    else:
                        # Only log when at least one counter is non-zero to reduce clutter
                        if any(count > 0 for count in self.error_counters.values()):
                            logger.info(f"Error counters (non-zero): {self.error_counters}")

                    # Point 8: Error Counter Guard Logic
                    if self.valves.enable_error_counter_guard:
                        now = time.time()
                        window = self.valves.error_guard_window_seconds
                        threshold = self.valves.error_guard_threshold

                        # Check JSON parse errors
                        error_type = "json_parse_errors"
                        # Record current count as a timestamp
                        current_count = self.error_counters[error_type]
                        # --- NOTE: This simple approach assumes the counter *increases* to track new errors.
                        # If the counter could be reset externally, a more robust timestamp queue is needed.
                        # For simplicity, assuming monotonically increasing count for now.
                        # A better approach: Store timestamp of each error occurrence.
                        # Let's refine this: Add timestamp whenever the error counter increments.
                        # We need to modify where the counter is incremented.

                        # --- Revised approach: Use a deque to store timestamps of recent errors ---
                        timestamps = self.error_timestamps[error_type]

                        # Remove old timestamps outside the window
                        while timestamps and timestamps[0] < now - window:
                            timestamps.popleft()

                        # Check if the count within the window exceeds the threshold
                        if len(timestamps) >= threshold:
                            if not self._guard_active:
                                logger.warning(
                                    f"Guard Activated: {error_type} count ({len(timestamps)}) reached threshold ({threshold}) in window ({window}s). Temporarily disabling LLM relevance and embedding dedupe."
                                )
                                self._guard_active = True
                                self._guard_activated_at = now
                                # Temporarily disable features
                                self._original_use_llm_relevance = (
                                    self.valves.use_llm_for_relevance
                                )
                                self._original_use_embedding_dedupe = (
                                    self.valves.use_embeddings_for_deduplication
                                )
                                self.valves.use_llm_for_relevance = False
                                self.valves.use_embeddings_for_deduplication = False
                            elif self._guard_active:
                                # Deactivate guard if error rate drops below threshold (with hysteresis?)
                                # For simplicity, deactivate immediately when below threshold.
                                logger.info(
                                    f"Guard Deactivated: {error_type} count ({len(timestamps)}) below threshold ({threshold}). Re-enabling LLM relevance and embedding dedupe."
                                )
                                self._guard_active = False
                                # Restore original settings
                                if hasattr(self, "_original_use_llm_relevance"):
                                    self.valves.use_llm_for_relevance = (
                                        self._original_use_llm_relevance
                                    )
                                if hasattr(self, "_original_use_embedding_dedupe"):
                                    self.valves.use_embeddings_for_deduplication = (
                                        self._original_use_embedding_dedupe
                                    )
                    consecutive_errors = 0
                
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in error counter logging task ({consecutive_errors}/{max_consecutive_errors}): {e}"
                    )
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(f"Error counter logging task disabled after {max_consecutive_errors} consecutive failures")
                        break
                    
                    backoff_seconds = min(2 ** consecutive_errors, 32)
                    logger.info(f"Backing off for {backoff_seconds}s before retrying error counter logging")
                    await asyncio.sleep(backoff_seconds)
        
        except asyncio.CancelledError:
            logger.debug("Error counter logging task cancelled")
        except Exception as e:
            logger.error(
                f"Fatal error in error counter logging task: {e}\n{traceback.format_exc()}"
            )

    def _schedule_date_update(self):
        """Schedule a regular update of the date information"""

        async def update_date_loop():
            try:
                while True:
                    # Use configurable interval with small random jitter
                    jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                    interval = self.valves.date_update_interval * jitter
                    await asyncio.sleep(interval)

                    self.current_date = self.get_formatted_datetime()
                    self.date_info = self._update_date_info()
                    logger.debug(f"Updated date information: {self.date_info}")
            except asyncio.CancelledError:
                logger.debug("Date update task cancelled")
            except Exception as e:
                logger.error(f"Error in date update task: {e}")

        # Start the update loop in the background
        task = asyncio.create_task(update_date_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _schedule_model_discovery(self):
        """Schedule a regular update of available models"""

        async def discover_models_loop():
            try:
                while True:
                    try:
                        # Discover models
                        await self._discover_models()

                        # Use configurable interval with small random jitter
                        jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                        interval = self.valves.model_discovery_interval * jitter
                        await asyncio.sleep(interval)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in model discovery: {e}")
                        # On error, retry sooner (1/6 of normal interval)
                        await asyncio.sleep(self.valves.model_discovery_interval / 6)
            except asyncio.CancelledError:
                logger.debug("Model discovery task cancelled")

        # Start the discovery loop in the background
        task = asyncio.create_task(discover_models_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _discover_models(self):
        """Discover available models from open_webui.configured providers"""
        logger.debug("Starting model discovery")

        # Create a session if needed
        session = await self._get_aiohttp_session()

        # Discover Ollama models
        try:
            ollama_url = "http://172.17.0.1:11434/api/tags"
            async with session.get(ollama_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "models" in data:
                        self.available_ollama_models = [
                            model["name"] for model in data["models"]
                        ]
                    logger.debug(
                        f"Discovered {len(self.available_ollama_models)} Ollama models"
                    )
        except Exception as e:
            logger.warning(f"Error discovering Ollama models: {e}")
            self.available_ollama_models = []

    def get_formatted_datetime(self, user_timezone=None):
        """
        Get properly formatted datetime with timezone awareness

        Args:
            user_timezone: Optional timezone string to override the default

        Returns:
            Timezone-aware datetime object
        """
        timezone_str = user_timezone or self.valves.timezone or "UTC"

        # Normalize common aliases
        alias_map = {
            "UAE/Dubai": "Asia/Dubai",
            "GMT+4": "Asia/Dubai",
            "GMT +4": "Asia/Dubai",
            "Dubai": "Asia/Dubai",
            "EST": "America/New_York",
            "PST": "America/Los_Angeles",
            "CST": "America/Chicago",
            "IST": "Asia/Kolkata",
            "BST": "Europe/London",
            "GMT": "Etc/GMT",
            "UTC": "UTC",
        }
        tz_key = timezone_str.strip()
        timezone_str = alias_map.get(tz_key, timezone_str)

        try:
            utc_now = datetime.utcnow()
            local_tz = pytz.timezone(timezone_str)
            local_now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)
            return local_now
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(
                f"Invalid timezone: {timezone_str}, falling back to default 'Asia/Dubai'."
            )
            try:
                local_tz = pytz.timezone("Asia/Dubai")
                local_now = (
                    datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(local_tz)
                )
                return local_now
            except Exception:
                logger.warning("Fallback timezone also invalid, using UTC")
                return datetime.utcnow().replace(tzinfo=pytz.utc)

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
            )
        return self._aiohttp_session

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Intercepts incoming messages, extracts memories, injects relevant ones.

        Handles chat commands: /memory list, /memory forget [id], /memory edit [id] [new content],
        /memory summarize [topic/tag], /note [content], /memory mark_important [id],
        /memory unmark_important [id], /memory list_banks, /memory assign_bank [id] [bank]
        """
        logger.debug(
            f"Inlet received body keys: {list(body.keys())} for user: {__user__.get('id', 'N/A') if __user__ else 'N/A'}"
        )
        
        # CRITICAL DEBUG: Log exact model value and full body structure
        model_in_body = body.get('model', 'NOT SET')
        logger.info(f"🔴 INLET CALLED - Model in body: {model_in_body}")
        logger.info(f"🔴 INLET: Full body keys: {list(body.keys())}")
        
        # COMPREHENSIVE BODY INSPECTION FOR CONVERSATION_ID DISCOVERY
        logger.info("=" * 80)
        logger.info("🔍 COMPREHENSIVE BODY INSPECTION - Looking for conversation_id fields:")
        logger.info("=" * 80)
        
        # Check for common conversation_id field names
        potential_ids = ['chat_id', 'conversation_id', 'session_id', 'thread_id', 'room_id', 'id']
        for field_name in potential_ids:
            if field_name in body:
                logger.info(f"✓ FOUND '{field_name}': {body[field_name]}")
        
        # Log the ENTIRE body structure (top-level keys and their types)
        logger.info("\nFull body structure:")
        for key, value in body.items():
            value_type = type(value).__name__
            if isinstance(value, (dict, list)):
                if isinstance(value, dict):
                    logger.info(f"  - {key}: dict with keys {list(value.keys())}")
                else:
                    logger.info(f"  - {key}: list with {len(value)} items")
            elif isinstance(value, str):
                logger.info(f"  - {key}: str (length={len(value)})")
            else:
                logger.info(f"  - {key}: {value_type} = {value}")
        
        logger.info("=" * 80)
        
        if 'messages' in body:
            logger.info(f"🔴 INLET: Number of messages: {len(body['messages'])}")
            for i, msg in enumerate(body['messages']):
                logger.info(f"🔴 INLET: Message {i}: role={msg.get('role')} content_length={len(str(msg.get('content', '')))} first_100_chars={str(msg.get('content', ''))[:100]}")

        # Ensure user info is present
        if not __user__ or not __user__.get("id"):
            logger.warning("Inlet: User info or ID missing, skipping processing.")
            return body
        user_id = __user__["id"]
        user_name = __user__.get("name", "Unknown")
        
        # Store UUID for access by outlet and other methods
        self._current_user_uuid = user_id
        self._current_user_name = user_name

        # --- Extract Conversation Context (chat_id, model_id, model_card_name) ---
        # Capture these early so outlet() can use them when linking memories
        chat_id = body.get("chat_id", None)
        
        # Extract model card name (persona) with fallback to backend model ID
        # Primary: model card name from metadata (e.g., "Friday", "Tara")
        # Fallback: backend model ID from body (e.g., "llama3:latest")
        model_card_name = None
        if __metadata__:
            model_card_name = __metadata__.get("model", {}).get("name")
        if not model_card_name:
            model_card_name = body.get("model", "default")
        
        # Store for use throughout the session
        self._current_model_card_name = model_card_name
        model_id = model_card_name  # Use model card name as primary model ID
        
        # Create composite conversation_id: chat_id_user_id_model_id
        # Falls back to old pattern if chat_id is missing
        if chat_id:
            self._current_conversation_id = f"{chat_id}_{user_id}_{model_id}"
            logger.info(f"✓ Extracted conversation context: chat_id={chat_id}, user_id={user_id}, model={model_id}")
            logger.debug(f"  Composite conversation_id: {self._current_conversation_id}")
        else:
            # Fallback to old pattern if chat_id not available
            self._current_conversation_id = f"{user_id}_{model_id}"
            logger.warning(f"⚠️ chat_id not in body, falling back to pattern: {self._current_conversation_id}")

        # --- Initialization & Valve Loading ---
        # Load valves early, handle potential errors
        try:
            # Reload global valves if OWUI injected config exists; otherwise keep current valves
            # DEBUG: First, check if self.config attribute even exists and what it contains
            has_config = hasattr(self, "config")
            config_object = getattr(self, "config", None)
            logger.debug(f"DEBUG valve loading - hasattr(self, 'config')={has_config}, config type={type(config_object)}, config keys={list(config_object.keys()) if isinstance(config_object, dict) else 'N/A'}")
            
            loaded_config_valves = getattr(self, "config", {}).get("valves", None)
            if loaded_config_valves is not None:
                logger.info(f"✓ Valves reloaded from open_webui.config - loaded_config_valves type={type(loaded_config_valves)}, keys={list(loaded_config_valves.keys()) if isinstance(loaded_config_valves, dict) else 'N/A'}")
                self.valves = self.Valves(**loaded_config_valves)
                logger.info(
                    f"✓ Valves successfully applied"
                )
                # BACKUP: Save to file in case OpenWebUI forgets these settings later
                self._save_persisted_valve_settings(self.valves)
            else:
                logger.info(
                    f"⚠️ OpenWebUI did NOT inject valve config (self.config['valves'] is None/missing). Using current valves."
                )
                # Check if we have persisted settings to fall back to
                persisted = self._load_persisted_valve_settings()
                if persisted:
                    try:
                        self.valves = self.Valves(**persisted)
                        logger.info(f"✓ Fell back to persisted valve settings from file")
                        # Ensure the file stays up-to-date even when loading from it
                        self._save_persisted_valve_settings(self.valves)
                    except Exception as e:
                        logger.error(f"Error applying persisted settings: {e}")
            
            # Log critical valve values so user can verify settings are loaded
            logger.info(
                f"📋 VALVE STATUS - "
                f"max_total_memories={self.valves.max_total_memories}, "
                f"pruning_strategy={self.valves.pruning_strategy}, "
                f"top_n_memories={self.valves.top_n_memories}, "
                f"vector_similarity_threshold={self.valves.vector_similarity_threshold}, "
                f"show_memories={self.valves.show_memories}, "
                f"show_status={self.valves.show_status}, "
                f"enable_json_stripping={self.valves.enable_json_stripping}, "
                f"enable_fallback_regex={self.valves.enable_fallback_regex}"
            )
            
            # SYNC: After valves are loaded, sync embedding config to Friday Memory System
            # This ensures both short-term and long-term memory use the same embedding model/dimension
            self._sync_embedding_config_to_friday()


            # Load user-specific valves (may override some per-user settings)
            user_valves = self._get_user_valves(__user__)

            if not user_valves.enabled:
                logger.debug(f"Memory plugin disabled for user {user_id}. Skipping.")
                return body  # Return early if disabled

            # Respect per-user setting for status visibility, ensuring it's set after loading
            show_status = self.valves.show_status and user_valves.show_status
        except Exception as e:
            logger.error(f"Failed to load valves for user {user_id}: {e}")
            # Attempt to inform the UI, but ignore secondary errors to
            # avoid masking the original stack-trace
            try:
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "error",
                        "content": f"Error loading memory configuration: {e}",
                    },
                )
            except Exception:
                pass
            # Prevent processing when config is invalid
            return body

        # --- Background Task Initialization (Ensure runs once) ---
        # Use hasattr for a simple check if tasks have been started
        if not hasattr(self, "_background_tasks_started"):
            self._initialize_background_tasks()
            self._background_tasks_started = True

        # --- Check for Guard Conditions ---
        if self._llm_feature_guard_active:
            logger.warning(
                "LLM feature guard active. Skipping LLM-dependent memory operations."
            )
        if self._embedding_feature_guard_active:
            logger.warning(
                "Embedding feature guard active. Skipping embedding-dependent memory operations."
            )

        # --- Process Incoming Message ---
        final_message = None
        # 1) Explicit stream=False (non-streaming completion requests)
        if body.get("stream") is False and body.get("messages"):
            final_message = body["messages"][-1].get("content")

        # 2) Streaming mode – grab final message when "done" flag arrives
        elif body.get("stream") is True and body.get("done", False):
            final_message = body.get("message", {}).get("content")

        # 3) Fallback – many WebUI front-ends don't set a "stream" key at all.
        if final_message is None and body.get("messages"):
            final_message = body["messages"][-1].get("content")

        # --- Image Analysis & Caching ---
        # Extract images from multimodal content and analyze with memory model
        images = self._extract_images_from_content(final_message) if final_message else []
        if images and self.image_manager:
            logger.debug(f"Found {len(images)} image(s) in message for user {user_id}")
            
            # Initialize cache for this user if needed
            if user_id not in self.image_cache_current_turn:
                self.image_cache_current_turn[user_id] = {}
            
            # Extract text for image context
            extracted_message_for_images = self._extract_text_from_content(final_message) if final_message else ""
            
            # Analyze each image with memory model
            for idx, img_info in enumerate(images):
                try:
                    image_hash = self._generate_image_hash(img_info["url"])
                    
                    # Check if we already analyzed this image this turn
                    if image_hash in self.image_cache_current_turn[user_id]:
                        logger.debug(f"Image {image_hash} already cached this turn")
                        continue
                    
                    logger.debug(f"Analyzing image {idx + 1}/{len(images)} with memory model...")
                    
                    # Query memory model for image analysis
                    description = await self._analyze_image_with_memory_model(
                        image_url=img_info["url"],
                        user_query=extracted_message_for_images,
                        user_id=user_id,
                        __event_emitter__=__event_emitter__,
                    )
                    
                    # Cache image info (transient - only persists if memory created)
                    self.image_cache_current_turn[user_id][image_hash] = {
                        "url": img_info["url"],
                        "base64_data": img_info["url"],  # Store URL or base64 directly
                        "description": description or "Image attached",
                        "query": extracted_message_for_images,
                        "type": img_info["type"],
                    }
                    
                    logger.info(f"Cached image {image_hash} for potential memory storage")
                except Exception as e:
                    logger.error(f"Error analyzing image: {e}")
                    # Continue with other images

        # --- Command Handling ---
        # Check if the final message is a command before processing memories
        # First, extract text if content is multimodal (list format with images)
        extracted_message = self._extract_text_from_content(final_message) if final_message else ""
        
        if extracted_message and extracted_message.strip().startswith("/"):
            command_parts = extracted_message.strip().split()
            command = command_parts[0].lower()

            # --- /memory list_banks Command --- NEW
            if (
                command == "/memory"
                and len(command_parts) >= 2
                and command_parts[1].lower() == "list_banks"
            ):
                logger.info(f"Handling command: /memory list_banks for user {user_id}")
                try:
                    allowed_banks = self.valves.allowed_memory_banks
                    default_bank = self.valves.default_memory_bank
                    bank_list_str = "\n".join(
                        [
                            f"- {bank} {'(Default)' if bank == default_bank else ''}"
                            for bank in allowed_banks
                        ]
                    )
                    response_msg = f"**Available Memory Banks:**\n{bank_list_str}"
                    await self._safe_emit(
                        __event_emitter__, {"type": "info", "content": response_msg}
                    )
                    body["messages"] = []  # Prevent LLM call
                    body["prompt"] = "Command executed."  # Placeholder for UI
                    body["bypass_prompt_processing"] = (
                        True  # Signal to skip further processing
                    )
                    return body
                except Exception as e:
                    logger.error(f"Error handling /memory list_banks: {e}")
                    await self._safe_emit(
                        __event_emitter__,
                        {"type": "error", "content": "Failed to list memory banks."},
                    )
                    # Allow fall through maybe? Or block? Let's block.
                    body["messages"] = []
                    body["prompt"] = "Error executing command."  # Placeholder for UI
                    body["bypass_prompt_processing"] = True
                    return body

            # --- /memory assign_bank Command --- NEW
            elif (
                command == "/memory"
                and len(command_parts) >= 4
                and command_parts[1].lower() == "assign_bank"
            ):
                logger.info(f"Handling command: /memory assign_bank for user {user_id}")
                try:
                    memory_id = command_parts[2]
                    target_bank = command_parts[3]

                    if target_bank not in self.valves.allowed_memory_banks:
                        allowed_banks_str = ", ".join(self.valves.allowed_memory_banks)
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "error",
                                "content": f"Invalid bank '{target_bank}'. Allowed banks: {allowed_banks_str}",
                            },
                        )
                    else:
                        # 1. Query the specific memory
                        # Note: query_memory might return multiple if content matches, need filtering by ID
                        query_result = await query_memory(
                            user_id=user_id,
                            form_data=QueryMemoryForm(
                                query=memory_id, k=1000
                            ),  # Query broadly first
                        )
                        target_memory = None
                        if query_result and query_result.memories:
                            for mem in query_result.memories:
                                if mem.id == memory_id:
                                    target_memory = mem
                                    break

                        if not target_memory:
                            await self._safe_emit(
                                __event_emitter__,
                                {
                                    "type": "error",
                                    "content": f"Memory with ID '{memory_id}' not found.",
                                },
                            )
                        else:
                            # 2. Check if bank is already correct
                            current_bank = target_memory.metadata.get(
                                "memory_bank", self.valves.default_memory_bank
                            )
                            if current_bank == target_bank:
                                await self._safe_emit(
                                    __event_emitter__,
                                    {
                                        "type": "info",
                                        "content": f"Memory '{memory_id}' is already in bank '{target_bank}'.",
                                    },
                                )
                            else:
                                # 3. Update the memory (delete + add with modified metadata)
                                new_metadata = target_memory.metadata.copy()
                                new_metadata["memory_bank"] = target_bank
                                new_metadata["timestamp"] = datetime.now(
                                    timezone.utc
                                ).isoformat()  # Update timestamp
                                new_metadata["source"] = (
                                    "adaptive_memory_v3_assign_bank_cmd"
                                )

                                await delete_memory_by_id(
                                    user_id=user_id, memory_id=memory_id
                                )
                                await add_memory(
                                    user_id=user_id,
                                    form_data=AddMemoryForm(
                                        content=target_memory.content,
                                        metadata=new_metadata,
                                    ),
                                )
                                await self._safe_emit(
                                    __event_emitter__,
                                    {
                                        "type": "info",
                                        "content": f"Successfully assigned memory '{memory_id}' to bank '{target_bank}'.",
                                    },
                                )
                                self._increment_error_counter(
                                    "memory_bank_assigned_cmd"
                                )

                except IndexError:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "error",
                            "content": "Usage: /memory assign_bank [memory_id] [bank_name]",
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Error handling /memory assign_bank: {e}\n{traceback.format_exc()}"
                    )
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "error",
                            "content": f"Failed to assign memory bank: {e}",
                        },
                    )
                    self._increment_error_counter("assign_bank_cmd_error")

                # Always bypass LLM after handling command
                body["messages"] = []
                body["prompt"] = "Command executed."  # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- Other /memory commands (Placeholder/Example - Adapt as needed) ---
            elif command == "/memory":
                # Example: Check for /memory list, /memory forget, etc.
                # Implement logic similar to assign_bank: parse args, call OWUI functions, emit status
                # Remember to add command handlers here based on other implemented features
                logger.info(
                    f"Handling generic /memory command stub for user {user_id}: {extracted_message}"
                )
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "info",
                        "content": f"Memory command '{extracted_message}' received (implementation pending).",
                    },
                )
                body["messages"] = []
                body["prompt"] = "Memory command received."  # Placeholder
                body["bypass_prompt_processing"] = True
                return body

            # --- /note command (Placeholder/Example) ---
            elif command == "/note":
                logger.info(
                    f"Handling /note command stub for user {user_id}: {extracted_message}"
                )
                # Implement logic for Feature 6 (Scratchpad)
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "info",
                        "content": f"Note command '{extracted_message}' received (implementation pending).",
                    },
                )
                body["messages"] = []
                body["prompt"] = "Note command received."  # Placeholder
                body["bypass_prompt_processing"] = True
                return body

        # --- Conversation Summarization Tracking ---
        if self._should_summarize_now(user_id, body):
            try:
                # Generate summary (only when threshold is reached)
                summary = await self._generate_conversation_summary(body, user_id)
                if summary:
                    # Try to get system prompt from model card
                    system_prompt = None
                    try:
                        model_id = body.get("model")
                        if model_id:
                            from open_webui.models.models import Models  # type: ignore
                            model_info = Models.get_model_by_id(model_id)
                            if model_info:
                                system_prompt = model_info.params.get("system")
                    except Exception as e:
                        logger.debug(f"Could not retrieve system prompt from model card: {e}")

                    # Inject summary into context
                    self._inject_summary_into_context(body, summary, system_prompt)

                    # Store summary in cache (not as memory to avoid flooding)
                    conversation_id = body.get("chat_id") or body.get("conversation_id") or "default"
                    if not hasattr(self, "_conversation_summaries"):
                        self._conversation_summaries = {}
                    cache_key = f"{user_id}_{conversation_id}"
                    self._conversation_summaries[cache_key] = {
                        "summary": summary,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message_count": len(body.get("messages", []))
                    }
                    logger.info(f"Cached conversation summary for {cache_key}")

                    if __event_emitter__:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "info",
                                "content": f"📝 Conversation summarized and injected into context",
                            },
                        )
            except Exception as e:
                logger.error(f"Error in conversation summarization: {e}")
                # Don't fail the request due to summarization errors

        # --- Memory Injection --- #
        if (
            self.valves.show_memories and not self._embedding_feature_guard_active
        ):  # Guard embedding-dependent retrieval
            try:
                # Show status that we're checking for memories
                if show_status:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 Checking for relevant memories to inject…",
                                "done": False,
                            },
                        },
                    )
                
                logger.debug(f"Retrieving relevant memories for user {user_id}")
                logger.info(f"🧠 INLET: Calling get_relevant_memories() for memory injection BEFORE main LLM")
                # Extract text from multimodal content if needed (handles images with text)
                message_text_for_retrieval = self._extract_text_from_content(final_message) if final_message else ""
                
                # Use user-specific timezone for relevance calculation context
                relevant_memories = await self.get_relevant_memories(
                    current_message=message_text_for_retrieval,
                    user_id=user_id,
                    user_timezone=user_valves.timezone,  # Use user-specific timezone
                    model_card_name=self._current_model_card_name,  # Pass current model card for persona isolation
                )
                
                # Check if this is the interface/tool model - if so, skip memory injection
                current_model = body.get("model", "")
                # Use per-user override if set, otherwise fall back to global setting
                interface_model = user_valves.interface_model_name if user_valves.interface_model_name else self.valves.interface_model_name
                is_interface_model = current_model == interface_model
                
                if is_interface_model:
                    logger.debug(
                        f"Skipping memory injection for interface model '{current_model}'. Will be injected when main model runs."
                    )
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"⏸️ Waiting for tool execution (interface model: {current_model})",
                                    "done": False,
                                },
                            },
                        )
                elif relevant_memories:
                    logger.info(
                        f"Injecting {len(relevant_memories)} relevant memories for user {user_id} into main model"
                    )
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"💾 Injecting {len(relevant_memories)} relevant memories into context…",
                                    "done": False,
                                },
                            },
                        )
                    self._inject_memories_into_context(body, relevant_memories)
                else:
                    logger.debug(f"No relevant memories found for user {user_id}")
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "✓ No relevant memories to inject",
                                    "done": True,
                                },
                            },
                        )
            except Exception as e:
                logger.error(
                    f"Error retrieving/injecting memories: {e}\n{traceback.format_exc()}"
                )
                await self._safe_emit(
                    __event_emitter__,
                    {"type": "error", "content": "Error retrieving relevant memories."},
                )

        # --- Cleanup image cache for this user ---
        # If images weren't used (no memory created), they're now discarded
        # Images that were used have already been persisted to image_database.db
        if user_id in self.image_cache_current_turn:
            cached_count = len(self.image_cache_current_turn[user_id])
            if cached_count > 0:
                logger.debug(
                    f"Clearing session image cache for user {user_id} ({cached_count} image(s) discarded if unused)"
                )
            del self.image_cache_current_turn[user_id]

        # CRITICAL DEBUG: Show what model and messages are going back to OpenWebUI for main LLM
        model_returning = body.get('model', 'NOT SET')
        logger.info(f"🔴 INLET RETURNING - Model in body: {model_returning}")
        if 'messages' in body:
            logger.info(f"🔴 INLET: Returning {len(body['messages'])} messages")
            for i, msg in enumerate(body['messages']):
                content_preview = str(msg.get('content', ''))[:100]
                logger.info(f"🔴 INLET: Message {i}: role={msg.get('role')} content_length={len(str(msg.get('content', '')))} preview={content_preview}")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Process LLM response, extract memories, and update the response"""
        # logger.debug("****** OUTLET FUNCTION CALLED ******") # REMOVED

        # Log function entry
        logger.debug("Outlet called - making deep copy of body dictionary")
        logger.info(f"🟢 OUTLET CALLED - Model in body: {body.get('model', 'NOT SET')} - LLM should have already responded")
        
        # COMPREHENSIVE BODY INSPECTION FOR CONVERSATION_ID DISCOVERY (same as inlet)
        logger.info("=" * 80)
        logger.info("🔍 COMPREHENSIVE BODY INSPECTION (OUTLET) - Looking for conversation_id fields:")
        logger.info("=" * 80)
        
        # Check for common conversation_id field names
        potential_ids = ['chat_id', 'conversation_id', 'session_id', 'thread_id', 'room_id', 'id']
        for field_name in potential_ids:
            if field_name in body:
                logger.info(f"✓ FOUND '{field_name}': {body[field_name]}")
        
        # Log the ENTIRE body structure (top-level keys and their types)
        logger.info("\nFull body structure (OUTLET):")
        for key, value in body.items():
            value_type = type(value).__name__
            if isinstance(value, (dict, list)):
                if isinstance(value, dict):
                    logger.info(f"  - {key}: dict with keys {list(value.keys())}")
                else:
                    logger.info(f"  - {key}: list with {len(value)} items")
            elif isinstance(value, str):
                logger.info(f"  - {key}: str (length={len(value)})")
            else:
                logger.info(f"  - {key}: {value_type} = {value}")
        
        logger.info("=" * 80)

        # Extract model card name (persona) with fallback to backend model ID
        # Primary: model card name from metadata (e.g., "Friday", "Tara")
        # Fallback: backend model ID from body (e.g., "llama3:latest")
        model_card_name = None
        if __metadata__:
            model_card_name = __metadata__.get("model", {}).get("name")
        if not model_card_name:
            model_card_name = body.get("model", "default")
        
        # Store for use throughout the session
        self._current_model_card_name = model_card_name
        self._current_model = model_card_name  # Use model card name as primary model ID
        logger.debug(f"Outlet: Set current_model to {self._current_model}")

        # DEFENSIVE: Make a deep copy of the body to avoid dictionary changed size during iteration
        # This was a source of many subtle bugs
        body_copy = copy.deepcopy(body)

        # Skip processing if user is not authenticated
        if not __user__:
            logger.warning("No user information available - skipping memory processing")
            return body_copy

        # Get user's ID for memory storage
        user_id = __user__.get("id")
        if not user_id:
            logger.warning("User object contains no ID - skipping memory processing")
            return body_copy
        
        # Store UUID for access by other methods
        self._current_user_uuid = user_id
        self._current_user_name = __user__.get("name", "Unknown")

        # Check if user has enabled memory function
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            logger.info(f"Memory function is disabled for user {user_id}")
            return body_copy

        # Get user's timezone if set
        user_timezone = user_valves.timezone or self.valves.timezone

        # --- BEGIN MEMORY PROCESSING IN OUTLET ---
        # Process the *last user message* for memory extraction *after* the LLM response
        last_user_message_content = None
        last_assistant_message_content = None
        message_history_for_context = []
        try:
            messages_copy = copy.deepcopy(body_copy.get("messages", []))
            if messages_copy:
                # Find the actual last user message in the history included in the body
                for msg in reversed(messages_copy):
                    if msg.get("role") == "user" and msg.get("content"):
                        last_user_message_content = msg.get("content")
                        break
                
                # Find the last assistant message (if dual-message extraction is enabled)
                if self.valves.extract_memories_from_model_responses:
                    for msg in reversed(messages_copy):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            last_assistant_message_content = msg.get("content")
                            break
                    if last_assistant_message_content:
                        logger.debug(f"Extracted last assistant message for dual-message extraction: {last_assistant_message_content[:60]}...")
                
                # Get up to N messages *before* the last user message for context
                if last_user_message_content:
                    user_msg_index = -1
                    for i, msg in enumerate(messages_copy):
                        if (
                            msg.get("role") == "user"
                            and msg.get("content") == last_user_message_content
                        ):
                            user_msg_index = i
                            break
                    if user_msg_index != -1:
                        start_index = max(
                            0, user_msg_index - self.valves.recent_messages_n
                        )
                        message_history_for_context = messages_copy[
                            start_index:user_msg_index
                        ]

            if last_user_message_content:
                logger.info(
                    f"Starting memory processing in outlet for user message: {last_user_message_content[:60]}..."
                )
                if last_assistant_message_content:
                    logger.info(f"Also processing assistant message for dual-message extraction")
                # Use asyncio.create_task for non-blocking processing
                # Reload valves inside _process_user_memories ensures latest config
                logger.debug("Starting memory extraction from outlet response")
                try:
                    # ============================================================
                    # QUEUE MEMORY EXTRACTION TASK
                    # Instead of fire-and-forget with asyncio.create_task, we now
                    # queue the operation so it's tracked and guaranteed to complete
                    # ============================================================
                    
                    # Package all the data needed for memory extraction
                    task_data = {
                        'conversation_id': self._current_conversation_id,  # Which conversation this is from
                        'user_id': user_id,  # Who sent the message
                        'user_message': last_user_message_content,  # YOUR message to extract from
                        'event_emitter': __event_emitter__,  # For UI status updates
                        'show_status': user_valves.show_status,  # Whether to show extraction status
                        'user_timezone': user_timezone,  # For timestamp formatting
                        'recent_chat_history': message_history_for_context,  # Recent conversation for context
                        'assistant_message': last_assistant_message_content if self.valves.extract_memories_from_model_responses else None,  # FRIDAY'S response to extract from (if enabled)
                    }
                    
                    # Add this task to the processing queue
                    # The background processor (_process_memory_queue) will pick it up
                    await self._memory_task_queue.put(task_data)
                    
                    # Track this conversation as having an active memory operation
                    # This lets us know it's in progress if we need to check status
                    self._active_memory_tasks[self._current_conversation_id] = {
                        'queued_at': datetime.now(timezone.utc),  # When it was queued
                        'user_id': user_id  # Who owns this operation
                    }
                    
                    # Log success with queue size for debugging
                    queue_size = self._memory_task_queue.qsize()
                    logger.info(f"✓ Queued memory extraction for conversation {self._current_conversation_id} (queue size: {queue_size})")
                    
                except Exception as e:
                    # If queueing fails for any reason, log it but don't crash
                    logger.error(f"Error queuing memory extraction task: {e}\n{traceback.format_exc()}")
            else:
                logger.warning(
                    "Could not find last user message in outlet body to process for memories."
                )

        except Exception as e:
            logger.error(
                f"Error initiating memory processing in outlet: {e}\n{traceback.format_exc()}"
            )
        # --- END MEMORY PROCESSING IN OUTLET ---

        # Process the response content for injecting memories
        try:
            # Get relevant memories for context injection on next interaction
            logger.info(f"🧠 OUTLET: Calling get_relevant_memories() for memory injection AFTER main LLM response")
            memories = await self.get_relevant_memories(
                current_message=last_user_message_content
                or "",  # Use the variable holding the user message
                user_id=user_id,
                user_timezone=user_timezone,
                model_card_name=self._current_model_card_name,  # Pass current model card for persona isolation
            )

            # If we found relevant memories and the user wants to see them
            if memories and self.valves.show_memories:
                # Inject memories into the context for the next interaction
                self._inject_memories_into_context(body_copy, memories)
                logger.debug(f"Injected {len(memories)} memories into context")
        except Exception as e:
            logger.error(
                f"Error processing memories for context: {e}\n{traceback.format_exc()}"
            )

        # Inject active reminders if enabled
        try:
            if self.valves.show_active_reminders and user_id and model_id:
                logger.debug("OUTLET: Calling get_active_reminders_for_injection() for reminder injection")
                active_reminders = await self.memory_system.get_active_reminders_for_injection(
                    user_id=user_id,
                    model_id=model_id
                )
                
                if active_reminders:
                    # Format reminders for injection
                    reminders_context = self._format_active_reminders_for_context(active_reminders, self.valves.reminder_format)
                    
                    # Inject into system message
                    if "messages" in body_copy:
                        for message in body_copy["messages"]:
                            if message["role"] == "system":
                                message["content"] += f"\n\n{reminders_context}"
                                break
                        else:
                            # No system message found, insert one
                            body_copy["messages"].insert(0, {"role": "system", "content": reminders_context})
                    
                    logger.debug(f"Injected active reminders into context")
            elif not self.valves.show_active_reminders:
                logger.debug("Active reminders injection disabled via valve")
        except Exception as e:
            logger.error(
                f"Error injecting active reminders: {e}\n{traceback.format_exc()}"
            )

        # Add confirmation message if memories were processed
        try:
            if user_valves.show_status:
                await self._add_confirmation_message(body_copy)
        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")

        # Return the modified response
        return body_copy

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any],
    ) -> None:
        """Safely emit an event, handling missing emitter"""
        if not event_emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await event_emitter(data)
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    def _get_user_valves(self, __user__: dict) -> UserValves:
        """Extract and validate user valves settings from OpenWebUI's __user__ dictionary
        
        OpenWebUI injects user-specific settings via __user__["valves"]. This method
        extracts and validates those settings, falling back to defaults if not present.
        """
        if not __user__:
            logger.warning("No user information provided")
            return self.UserValves()

        # Handle both dict and UserModel objects
        if isinstance(__user__, dict):
            user_valves_data = __user__.get("valves", {})
        else:
            # __user__ is a UserModel object, try to get the valves attribute
            user_valves_data = getattr(__user__, "valves", {})

        # Ensure we have a dictionary to work with
        if not isinstance(user_valves_data, dict):
            logger.warning(
                f"User valves is not a dictionary (type: {type(user_valves_data)}), using defaults."
            )
            user_valves_data = {}

        try:
            # Validate and return the UserValves model with user-specific overrides
            user_id = __user__.get("id", "unknown")
            user_name = __user__.get("name", "Unknown")
            user_valves = self.UserValves(**user_valves_data)
            logger.debug(f"✓ Loaded user valves for {user_name} ({user_id}): enabled={user_valves.enabled}, show_status={user_valves.show_status}")
            return user_valves
        except Exception as e:
            # Default to enabled if validation/extraction fails
            logger.error(
                f"Could not determine user valves settings from data {user_valves_data}: {e}"
            )
            return self.UserValves()  # Return default UserValves on error

    async def _retroactively_embed_all_memories(self):
        """Retroactively embed all existing memories and regenerate if dimensions mismatch.
        
        Checks for embedding metadata tag to avoid re-embedding already-processed memories.
        Tag format: __embedding_model:nomic_embed_text_v1.5_768d
        
        Also checks embeddings_completed.log to skip full retroactive embedding if recently completed.
        """
        try:
            await asyncio.sleep(2)  # Give plugin time to fully initialize
            
            # Check if we've already completed a full retroactive embedding recently
            log_dir = self._get_error_log_dir()
            embeddings_log_path = os.path.join(log_dir, "embeddings_completed.log")
            try:
                if os.path.exists(embeddings_log_path):
                    with open(embeddings_log_path, "r") as f:
                        last_completed = f.read().strip()
                    logger.info(f"📝 Previous embedding run completed at: {last_completed}")
                    logger.info("⏭️  Skipping full retroactive embedding. Will only embed new memories.")
                    return  # Skip the full retroactive embedding if it's already been done
            except Exception as e:
                logger.debug(f"Could not read embeddings_completed.log: {e}, proceeding with retroactive embedding")
            
            logger.info("🔄 Starting retroactive embedding of all existing memories...")

            # Generate embedding model tag based on current configuration
            EMBEDDING_MODEL_TAG = self._get_embedding_model_tag()
            logger.debug(f"Using embedding tag: {EMBEDDING_MODEL_TAG}")

            # Get all unique memories across all users
            try:
                all_memories = Memories.get_memories()  # Get all memories
            except Exception as e:
                logger.warning(
                    f"Could not retrieve all memories: {e}, trying empty list"
                )
                all_memories = []

            if not all_memories:
                logger.info("No existing memories found to embed")
                return

            logger.info(
                f"📊 Found {len(all_memories)} existing memories to potentially embed"
            )
            embedded_count = 0
            skipped_count = 0
            regenerated_count = 0
            error_count = 0

            # Track the embedding dimension from first successful embedding
            fresh_emb_sample = None

            for memory in all_memories:
                try:
                    mem_id = str(getattr(memory, "id", None))
                    mem_text = getattr(memory, "content", "")
                    mem_tags = list(getattr(memory, "tags", []) or [])

                    if not mem_id or not mem_text:
                        skipped_count += 1
                        continue

                    # CHECK: Has this memory already been embedded with the correct model?
                    if EMBEDDING_MODEL_TAG in mem_tags:
                        logger.debug(
                            f"✓ Memory {mem_id} already tagged with {EMBEDDING_MODEL_TAG}, skipping"
                        )
                        skipped_count += 1
                        continue

                    # Check if already embedded in persistent cache
                    existing_emb = (
                        self.embedding_cache.get(mem_id)
                        if hasattr(self, "embedding_cache")
                        else None
                    )

                    # Get a fresh embedding from LM Studio to check dimension compatibility
                    fresh_emb, emb_error = await self.get_nomic_embedding(mem_text)

                    # If we can't get fresh embedding, log error and skip this memory
                    if fresh_emb is None:
                        if emb_error:
                            logger.warning(f"Embedding error for memory {mem_id}: {emb_error}")
                        if existing_emb is not None:
                            logger.debug(
                                f"✓ Memory {mem_id} has cached embedding but LM Studio unavailable, keeping cached version"
                            )
                            skipped_count += 1
                        else:
                            logger.warning(
                                f"Could not embed memory {mem_id}, LM Studio unavailable"
                            )
                            error_count += 1
                        continue

                    # Track embedding dimension on first successful embedding (for smart checks later)
                    if fresh_emb_sample is None and hasattr(fresh_emb, "shape"):
                        fresh_emb_sample = fresh_emb
                        self._last_embedding_dimension = fresh_emb.shape[0]
                        logger.debug(
                            f"📊 Set embedding dimension to {self._last_embedding_dimension}D for future checks"
                        )

                    # Check if cached embedding needs regeneration (dimension mismatch)
                    if existing_emb is not None:
                        try:
                            if hasattr(existing_emb, "shape") and hasattr(
                                fresh_emb, "shape"
                            ):
                                if existing_emb.shape[0] != fresh_emb.shape[0]:
                                    logger.info(
                                        f"⚠️ Dimension mismatch for memory {mem_id}: cached={existing_emb.shape[0]}D, current={fresh_emb.shape[0]}D. Regenerating..."
                                    )
                                    # Use the fresh embedding that was just generated
                                    if hasattr(self, "embedding_cache"):
                                        self.embedding_cache.put(
                                            mem_id, mem_text, fresh_emb
                                        )
                                    self.memory_embeddings[mem_id] = fresh_emb
                                    
                                    # TAG memory with embedding model info (add tag if not present)
                                    if EMBEDDING_MODEL_TAG not in mem_tags:
                                        mem_tags.append(EMBEDDING_MODEL_TAG)
                                        # Update memory with new tags
                                        try:
                                            memory.tags = mem_tags
                                            await add_memory(
                                                request=Request(scope={"type": "http", "app": webui_app}),
                                                user=getattr(memory, 'user', None),
                                                form_data=AddMemoryForm(
                                                    content=mem_text,
                                                    metadata={"tags": mem_tags}
                                                ),
                                            )
                                            logger.debug(f"✓ Tagged memory {mem_id} with {EMBEDDING_MODEL_TAG}")
                                        except Exception as tag_error:
                                            logger.debug(f"Could not tag memory {mem_id}: {tag_error}")
                                    
                                    regenerated_count += 1
                                    continue
                        except Exception as e:
                            logger.debug(
                                f"Could not check embedding dimensions for {mem_id}: {e}, regenerating"
                            )
                            if hasattr(self, "embedding_cache"):
                                self.embedding_cache.put(mem_id, mem_text, fresh_emb)
                            self.memory_embeddings[mem_id] = fresh_emb
                            regenerated_count += 1
                            continue

                        # Cached embedding is valid (same dimensions), keep it and TAG it
                        if EMBEDDING_MODEL_TAG not in mem_tags:
                            mem_tags.append(EMBEDDING_MODEL_TAG)
                            try:
                                memory.tags = mem_tags
                                logger.debug(f"✓ Tagged existing valid memory {mem_id}")
                            except Exception as tag_error:
                                logger.debug(f"Could not tag memory {mem_id}: {tag_error}")
                        
                        logger.debug(
                            f"✓ Memory {mem_id} already has valid embedding, skipping"
                        )
                        skipped_count += 1
                    else:
                        # No cached embedding, store the fresh one
                        if hasattr(self, "embedding_cache"):
                            self.embedding_cache.put(mem_id, mem_text, fresh_emb)
                            
                        # TAG the memory with embedding model info
                        if EMBEDDING_MODEL_TAG not in mem_tags:
                            mem_tags.append(EMBEDDING_MODEL_TAG)
                        
                        # Store the tag in the memory
                        try:
                            memory.tags = mem_tags
                            logger.debug(f"✓ Tagged new embedding for memory {mem_id}")
                        except Exception as tag_error:
                            logger.debug(f"Could not tag new memory {mem_id}: {tag_error}")
                        
                        embedded_count += 1
                        logger.debug(f"✓ Embedded and cached memory {mem_id}")

                        # Also store in in-memory cache for current session
                        self.memory_embeddings[mem_id] = fresh_emb

                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error processing memory {mem_id}: {e}")
                    continue
            logger.info(
                f"✓ Retroactive embedding complete: {embedded_count} new, {regenerated_count} regenerated, {skipped_count} valid, {error_count} errors"
            )
            
            # Write completion timestamp to log file
            try:
                log_dir = self._get_error_log_dir()
                embeddings_log_path = os.path.join(log_dir, "embeddings_completed.log")
                from datetime import datetime
                completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(embeddings_log_path, "w") as f:
                    f.write(completion_time)
                logger.info(f"📝 Wrote embedding completion timestamp: {completion_time}")
            except Exception as e:
                logger.warning(f"Could not write embeddings_completed.log: {e}")

        except Exception as e:
            logger.error(
                f"Fatal error during retroactive embedding: {e}\n{traceback.format_exc()}"
            )

    async def _get_formatted_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user and format them for processing"""
        memories_list = []
        try:
            # Get memories using Memories.get_memories_by_user_id
            user_memories = Memories.get_memories_by_user_id(user_id=str(user_id))

            if user_memories:
                for memory in user_memories:
                    # Safely extract attributes with fallbacks
                    memory_id = str(getattr(memory, "id", "unknown"))
                    memory_content = getattr(memory, "content", "")
                    created_at = getattr(memory, "created_at", None)
                    updated_at = getattr(memory, "updated_at", None)
                    
                    # Extract metadata (includes model_card_name, memory_bank, tags, etc.)
                    metadata = getattr(memory, "metadata", {}) or {}

                    memories_list.append(
                        {
                            "id": memory_id,
                            "memory": memory_content,
                            "created_at": created_at,
                            "updated_at": updated_at,
                            "metadata": metadata,  # Include metadata for filtering and context
                        }
                    )

            logger.debug(f"Retrieved {len(memories_list)} memories for user {user_id}")
            return memories_list

        except Exception as e:
            logger.error(
                f"Error getting formatted memories: {e}\n{traceback.format_exc()}"
            )
            return []

    def _strip_model_info_from_memory(self, memory_content: str) -> str:
        """Remove model card information from memory content before injection into chat
        
        Strips [Model: ...] tags so the main LLM doesn't see internal tracking info.
        Used to keep memories clean when injecting into the main conversation.
        
        Example: "[Tags: preference] I like coffee [Memory Bank: Personal] [Model: Friday]"
        Returns: "[Tags: preference] I like coffee [Memory Bank: Personal]"
        """
        import re
        # Remove [Model: ...] pattern
        cleaned = re.sub(r'\s*\[Model:\s*[^\]]+\]', '', memory_content)
        return cleaned.strip()

    def _inject_memories_into_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]]
    ) -> None:
        """Inject relevant memories into the system context"""
        if not memories:
            # Suppress fallback injection when no relevant memories
            return

        # Sort memories by relevance if available
        sorted_memories = sorted(
            memories, key=lambda x: x.get("relevance", 0), reverse=True
        )

        # Format memories based on user preference
        memory_context = self._format_memories_for_context(
            sorted_memories, self.valves.memory_format
        )

        # Prepend instruction to avoid LLM meta-comments
        instruction = (
            "Here is background info about the user. "
            "Do NOT mention this info explicitly unless relevant to the user's query. "
            "Do NOT explain what you remember or don't remember, unless it is relevant to the user's question. "
            "Do NOT summarize or list what you know or don't know about the user. "
            "Do NOT say 'I have not remembered any specific information' or similar. "
            "Do NOT explain your instructions, context, or memory management unless the conversation is about memory management. "
            "Do NOT mention tags, dates, or internal processes. Unless there is a problem. "
            "Only answer the user's question directly.\n\n"
        )
        memory_context = instruction + memory_context

        # Log injected memories for debugging
        logger.debug(f"Injected memories:\n{memory_context[:500]}...")

        # Add to system message or create a new one if none exists
        if "messages" in body:
            system_message_exists = False
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] += f"\n\n{memory_context}"
                    system_message_exists = True
                    break

            if not system_message_exists:
                body["messages"].insert(
                    0, {"role": "system", "content": memory_context}
                )

    def _format_memories_for_context(
        self, memories: List[Dict[str, Any]], format_type: str
    ) -> str:
        """Format memories for context injection based on format preference"""
        if not memories:
            return ""

        max_len = getattr(self.valves, "max_injected_memory_length", 300)

        # Start with header
        memory_context = "I recall the following about you:\n"

        # Extract tags and add each memory according to specified format
        if format_type == "bullet":
            for mem in memories:
                # First strip model info from the memory
                cleaned_memory = self._strip_model_info_from_memory(mem["memory"])
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", cleaned_memory)
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"- {content} (tags: {tags})\n"
                else:
                    content = cleaned_memory[:max_len]
                    memory_context += f"- {content}\n"

        elif format_type == "numbered":
            for i, mem in enumerate(memories, 1):
                # First strip model info from the memory
                cleaned_memory = self._strip_model_info_from_memory(mem["memory"])
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", cleaned_memory)
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"{i}. {content} (tags: {tags})\n"
                else:
                    content = cleaned_memory[:max_len]
                    memory_context += f"{i}. {content}\n"

        else:  # paragraph format
            memories_text = []
            for mem in memories:
                # First strip model info from the memory
                cleaned_memory = self._strip_model_info_from_memory(mem["memory"])
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", cleaned_memory)
                if tags_match:
                    content = tags_match.group(2)[:max_len]
                    memories_text.append(content)
                else:
                    content = cleaned_memory[:max_len]
                    memories_text.append(content)

            memory_context += f"{'. '.join(memories_text)}.\n"

        return memory_context

    def _format_active_reminders_for_context(self, active_reminders: Dict[str, List[str]], format_type: str = "bullet") -> str:
        """Format active reminders for context injection by urgency tier"""
        if not active_reminders or not any(active_reminders.values()):
            return ""
        
        context = "[Active Reminders]\n\n"
        
        # Add urgent reminders
        if active_reminders.get("urgent"):
            context += "🔴 URGENT (< 1 hour):\n"
            if format_type == "bullet":
                for reminder in active_reminders["urgent"]:
                    context += f"  • {reminder}\n"
            elif format_type == "numbered":
                for i, reminder in enumerate(active_reminders["urgent"], 1):
                    context += f"  {i}. {reminder}\n"
            else:  # paragraph
                context += "  " + " | ".join(active_reminders["urgent"]) + "\n"
            context += "\n"
        
        # Add soon reminders
        if active_reminders.get("soon"):
            context += "🟡 SOON (1-4 hours):\n"
            if format_type == "bullet":
                for reminder in active_reminders["soon"]:
                    context += f"  • {reminder}\n"
            elif format_type == "numbered":
                for i, reminder in enumerate(active_reminders["soon"], 1):
                    context += f"  {i}. {reminder}\n"
            else:  # paragraph
                context += "  " + " | ".join(active_reminders["soon"]) + "\n"
            context += "\n"
        
        # Add upcoming reminders
        if active_reminders.get("upcoming"):
            context += "🟢 UPCOMING (< 24 hours):\n"
            if format_type == "bullet":
                for reminder in active_reminders["upcoming"]:
                    context += f"  • {reminder}\n"
            elif format_type == "numbered":
                for i, reminder in enumerate(active_reminders["upcoming"], 1):
                    context += f"  {i}. {reminder}\n"
            else:  # paragraph
                context += "  " + " | ".join(active_reminders["upcoming"]) + "\n"
            context += "\n"
        
        return context.strip()

    async def _process_user_memories(
        self,
        user_message: str,
        user_id: str,
        event_emitter: Optional[
            Callable[[Any], Awaitable[None]]
        ] = None,  # Renamed for clarity
        show_status: bool = True,
        user_timezone: Optional[str] = None,
        recent_chat_history: Optional[
            List[Dict[str, Any]]
        ] = None,  # Added this argument
        assistant_message: Optional[str] = None,  # Added for dual-message extraction
    ) -> List[Dict[str, Any]]:
        """Process user message (and optionally assistant message) to extract and store memories

        Args:
            user_message: User's input message
            user_id: User identifier
            event_emitter: Event emitter for status updates
            show_status: Whether to show status messages
            user_timezone: User's timezone
            recent_chat_history: List of recent messages for context
            assistant_message: Optional model response for dual-message memory extraction

        Returns:
            List of stored memory operations
        """
        # --- ADD LOGGING TO INSPECT self.config ---
        config_content = getattr(self, "config", "<Not Set>")
        logger.info(
            f"Inspecting self.config at start of _process_user_memories: {config_content}"
        )
        # --- END LOGGING ---

        # --- RELOAD VALVES --- REMOVED
        # Ensure we have the latest config potentially injected by OWUI
        # try:
        #     logger.debug("Reloading self.valves at start of _process_user_memories") # Corrected function name
        #     self.valves = self.Valves(**getattr(self, "config", {}).get("valves", {}))
        # except Exception as e:
        #      logger.error(f"Error reloading valves in _process_user_memories: {e}") # Corrected function name
        # --- END RELOAD --- REMOVED

        # Start timer
        start_time = time.perf_counter()

        # Reset stored memories and error message
        # This variable held identified memories, not saved ones. We'll get saved count from process_memories return.
        # self.stored_memories = [] # Remove or repurpose if needed elsewhere, currently unused after this point.
        self._error_message = None

        # Emit "processing memories" status if enabled
        if show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "📝 Extracting potential new memories from your message…",
                        "done": False,
                    },
                },
            )

        # Debug logging for function entry
        logger.debug(
            f"Starting _process_user_memories for user {user_id} with message: {user_message[:50]}..."
        )

        # Get user valves
        user_valves = None
        try:
            user = Users.get_user_by_id(user_id)
            user_valves = self._get_user_valves(user)

            # Debug logging for user valves
            logger.debug(
                f"Retrieved user valves with memory enabled: {user_valves.enabled}"
            )

            if not user_valves.enabled:
                logger.info(f"Memory function disabled for user: {user_id}")
                if show_status:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "⏸️ Friday Short Term Memory is disabled in your settings – skipping memory save.",
                                "done": True,
                            },
                        },
                    )
                return []
        except Exception as e:
            logger.error(f"Error getting user valves: {e}")
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Unable to access memory settings – aborting memory save process.",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for memory identification start
        logger.debug(
            f"Starting memory identification for message: {user_message[:60]}..."
        )

        # Step 1: Use LLM to identify memories in the message
        memories = []
        parse_error_occurred = False  # Track if parsing failed
        try:
            # Get user's existing memories for context (optional - can also be None)
            existing_memories = None
            # If the LLM needs context of existing memories:
            try:
                existing_memories = await self._get_formatted_memories(user_id)
                logger.debug(
                    f"Retrieved {len(existing_memories)} existing memories for context"
                )
            except Exception as e:
                logger.warning(f"Could not get existing memories (continuing): {e}")

            # Process message to extract memory operations
            memories = await self.identify_memories(
                user_message,
                existing_memories=existing_memories,
                user_timezone=user_timezone,
                assistant_message=assistant_message,  # Pass assistant message for dual-message extraction
            )

            # Debug logging after memory identification
            logger.debug(
                f"Memory identification complete. Found {len(memories)} potential memories"
            )

        except Exception as e:
            self.error_counters["llm_call_errors"] += 1
            logger.error(f"Error identifying memories: {e}\n{traceback.format_exc()}")
            self._error_message = (
                f"llm_error: {str(e)[:50]}..."  # Point 6: More specific error
            )
            parse_error_occurred = True  # Indicate identification failed
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Memory error: {str(e)}",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for filtering
        logger.debug("Starting memory filtering step...")

        # Step 2: Filter memories (apply blacklist/whitelist/trivia filtering)
        filtered_memories = []
        if memories:
            # Apply filters based on valves
            try:
                # Get filter configuration valves
                min_length = self.valves.min_memory_length
                blacklist = self.valves.blacklist_topics
                whitelist = self.valves.whitelist_keywords

                logger.debug(
                    f"Using filters: min_length={min_length}, blacklist={blacklist}, whitelist={whitelist}"
                )

                # Known meta-request phrases
                meta_request_phrases = [
                    "remember this",
                    "make a note",
                    "don't forget",
                    "keep in mind",
                    "save this",
                    "add this to",
                    "log this",
                    "put this in",
                ]

                # Process each memory with filtering
                for memory in memories:
                    # Validate operation
                    if not self._validate_memory_operation(memory):
                        logger.debug(f"Invalid memory operation: {str(memory)}")
                        continue

                    # Extract content for filtering
                    content = memory.get("content", "").strip()

                    # Apply minimum length filter
                    if len(content) < min_length:
                        logger.debug(
                            f"Memory too short ({len(content)} < {min_length}): {content}"
                        )
                        continue

                    # Check if it's a meta-request
                    is_meta_request = False
                    for phrase in meta_request_phrases:
                        if phrase.lower() in content.lower():
                            is_meta_request = True
                            logger.debug(f"Meta-request detected: {content}")
                            break

                    if is_meta_request:
                        continue

                    # Check blacklist (if configured)
                    if blacklist:
                        is_blacklisted = False
                        for topic in blacklist.split(","):
                            topic = topic.strip().lower()
                            if topic and topic in content.lower():
                                # Check whitelist override
                                is_whitelisted = False
                                if whitelist:
                                    for keyword in whitelist.split(","):
                                        keyword = keyword.strip().lower()
                                        if keyword and keyword in content.lower():
                                            is_whitelisted = True
                                            logger.debug(
                                                f"Whitelisted term '{keyword}' found in blacklisted content"
                                            )
                                            break

                                if not is_whitelisted:
                                    is_blacklisted = True
                                    logger.debug(
                                        f"Blacklisted topic '{topic}' found: {content}"
                                    )
                                    break

                        if is_blacklisted:
                            continue

                    # Memory passed all filters
                    filtered_memories.append(memory)
                    logger.debug(f"Memory passed all filters: {content}")

                logger.info(
                    f"Filtered memories: {len(filtered_memories)}/{len(memories)} passed"
                )
            except Exception as e:
                logger.error(f"Error filtering memories: {e}\n{traceback.format_exc()}")
                filtered_memories = (
                    memories  # On error, attempt to process all memories
                )

        # Debug logging after filtering
        logger.debug(f"After filtering: {len(filtered_memories)} memories remain")

        # If no memories to process after filtering, log and return
        if not filtered_memories:  # Check if the list is empty
            # Point 5: Immediate-Save Shortcut for short preferences on parse error
            if (
                self.valves.enable_short_preference_shortcut
                and parse_error_occurred
                and len(user_message) <= 60
                and any(
                    keyword in user_message.lower()
                    for keyword in ["favorite", "love", "like", "enjoy"]
                )
            ):
                logger.info(
                    "JSON parse failed, but applying short preference shortcut."
                )
                try:
                    shortcut_op = MemoryOperation(
                        operation="NEW",
                        content=user_message.strip(),  # Save the raw message content
                        tags=["preference"],  # Assume preference tag
                        model_card_name=self._current_model_card_name,
                    )
                    await self._execute_memory_operation(
                        shortcut_op, user
                    )  # Directly execute
                    logger.info(
                        f"Successfully saved memory via shortcut: {user_message[:50]}..."
                    )
                    # Set a specific status message for this case
                    self._error_message = None  # Clear parse error flag
                    # Since we bypassed normal processing, we need a result list for status reporting
                    saved_operations_list = [
                        shortcut_op.model_dump()
                    ]  # Use model_dump() for Pydantic v2+
                    # Skip the rest of the processing steps as we forced a save
                except Exception as shortcut_err:
                    logger.error(
                        f"Error during short preference shortcut save: {shortcut_err}"
                    )
                    self._error_message = "shortcut_save_error"
                    saved_operations_list = []  # Indicate save failed
            else:
                # Normal case: No memories identified or filtered out, and no shortcut applied
                logger.info(
                    "No valid memories to process after filtering/identification."
                )
                if show_status and not self._error_message:
                    # Determine reason for no save
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"ℹ️ Memory save skipped – {final_status_reason.replace('_', ' ')}."
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": status_desc,
                                "done": True,
                            },
                        },
                    )
                return []  # Return empty list as nothing was saved through normal path
        else:
            # We have filtered_memories, proceed with normal processing
            pass  # Continue to Step 3

        # Step 3: Get current memories and handle max_total_memories limit
        try:
            current_memories_data = await self._get_formatted_memories(user_id)
            logger.debug(
                f"Retrieved {len(current_memories_data)} existing memories from database"
            )

            # If we'd exceed the maximum memories per user, apply pruning
            max_memories = self.valves.max_total_memories
            current_count = len(current_memories_data)
            new_count = len(
                filtered_memories
            )  # Only count NEW operations towards limit for pruning decision

            if current_count + new_count > max_memories:
                to_remove = current_count + new_count - max_memories
                logger.info(
                    f"Memory limit ({max_memories}) would be exceeded. Need to prune {to_remove} memories."
                )

                memories_to_prune_ids = []

                # Choose pruning strategy based on valve
                strategy = self.valves.pruning_strategy
                logger.info(f"Applying pruning strategy: {strategy}")

                if strategy == "least_relevant":
                    try:
                        # Calculate relevance for all existing memories against the current user message
                        memories_with_relevance = []
                        # Re-use logic similar to get_relevant_memories but for *all* memories

                        user_embedding = None
                        if self.embedding_model:
                            try:
                                user_embedding = self.embedding_model.encode(
                                    user_message, normalize_embeddings=True
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Could not encode user message for relevance pruning: {e}"
                                )

                        # Determine if we can use vectors or need LLM fallback (respecting valve)
                        can_use_vectors = user_embedding is not None
                        needs_llm = self.valves.use_llm_for_relevance

                        # --- Calculate Scores ---
                        if not needs_llm and can_use_vectors:
                            # Vector-only relevance calculation
                            for mem_data in current_memories_data:
                                mem_id = mem_data.get("id")
                                mem_emb = self.memory_embeddings.get(mem_id)
                                # Ensure embedding exists or try to compute it
                                if mem_emb is None and self.embedding_model is not None:
                                    try:
                                        mem_text = mem_data.get("memory") or ""
                                        if mem_text:
                                            mem_emb = self.embedding_model.encode(
                                                mem_text, normalize_embeddings=True
                                            )
                                            self.memory_embeddings[mem_id] = (
                                                mem_emb  # Cache it
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to compute embedding for existing memory {mem_id}: {e}"
                                        )
                                        mem_emb = None  # Mark as failed

                                if mem_emb is not None:
                                    sim_score = float(np.dot(user_embedding, mem_emb))
                                    memories_with_relevance.append(
                                        {"id": mem_id, "relevance": sim_score}
                                    )
                                else:
                                    # Assign low relevance if embedding fails
                                    memories_with_relevance.append(
                                        {"id": mem_id, "relevance": 0.0}
                                    )
                        elif needs_llm:
                            # LLM-based relevance calculation (simplified, no caching needed here)
                            # Prepare memories for LLM prompt
                            memory_strings_for_llm = [
                                f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                                for mem in current_memories_data
                            ]
                            system_prompt = self.valves.memory_relevance_prompt
                            llm_user_prompt = f"""Current user message: "{user_message}"

Available memories:
{json.dumps(memory_strings_for_llm)}

Rate the relevance of EACH memory to the current user message."""

                            try:
                                llm_response_text = await self.query_llm_with_retry(
                                    system_prompt, llm_user_prompt
                                )
                                llm_relevance_results = self._extract_and_parse_json(
                                    llm_response_text
                                )

                                if isinstance(llm_relevance_results, list):
                                    # Map results back to IDs
                                    llm_scores = {
                                        item.get("id"): item.get("relevance", 0.0)
                                        for item in llm_relevance_results
                                        if isinstance(item, dict)
                                    }
                                    for mem_data in current_memories_data:
                                        mem_id = mem_data.get("id")
                                        score = llm_scores.get(
                                            mem_id, 0.0
                                        )  # Default to 0 if LLM missed it
                                        memories_with_relevance.append(
                                            {"id": mem_id, "relevance": score}
                                        )
                                else:
                                    logger.warning(
                                        "LLM relevance check for pruning failed to return valid list. Pruning might default to FIFO."
                                    )
                                    # Fallback: assign 0 relevance to all, effectively making it FIFO-like for this run
                                    memories_with_relevance = [
                                        {"id": m["id"], "relevance": 0.0}
                                        for m in current_memories_data
                                    ]
                            except Exception as llm_err:
                                logger.error(
                                    f"Error during LLM relevance check for pruning: {llm_err}"
                                )
                                memories_with_relevance = [
                                    {"id": m["id"], "relevance": 0.0}
                                    for m in current_memories_data
                                ]
                        else:  # Cannot use vectors and LLM not enabled - default to FIFO-like
                            logger.warning(
                                "Cannot determine relevance for pruning (no embeddings/LLM). Pruning will be FIFO-like."
                            )
                            memories_with_relevance = [
                                {"id": m["id"], "relevance": 0.0}
                                for m in current_memories_data
                            ]

                        # --- Sort and Select for Pruning ---
                        # Sort by relevance ascending (lowest first)
                        memories_with_relevance.sort(
                            key=lambda x: x.get("relevance", 0.0)
                        )

                        # Select the IDs of the least relevant memories to remove (take the first `to_remove` items after sorting)
                        memories_to_prune_ids = [
                            mem["id"] for mem in memories_with_relevance[:to_remove]
                        ]
                        logger.info(
                            f"Identified {len(memories_to_prune_ids)} least relevant memories for pruning."
                        )

                    except Exception as relevance_err:
                        logger.error(
                            f"Error calculating relevance for pruning, falling back to FIFO: {relevance_err}"
                        )
                        # Fallback to FIFO on any error during relevance calculation
                        strategy = "fifo"

                # Default or fallback FIFO strategy
                if strategy == "fifo":
                    # Sort by timestamp ascending (oldest first)
                    # Make sure timestamp exists, fallback to a very old date if not
                    default_date = datetime.min.replace(tzinfo=timezone.utc)
                    sorted_memories = sorted(
                        current_memories_data,
                        key=lambda x: x.get("created_at", default_date),
                    )
                    memories_to_prune_ids = [
                        mem["id"] for mem in sorted_memories[:to_remove]
                    ]
                    logger.info(
                        f"Identified {len(memories_to_prune_ids)} oldest memories (FIFO) for pruning."
                    )

                # Execute pruning if IDs were identified
                if memories_to_prune_ids:
                    pruned_count = 0
                    for memory_id_to_delete in memories_to_prune_ids:
                        try:
                            # Before deleting, attempt to promote this memory to Friday Memory System
                            # Find the memory data to get its content
                            memory_to_promote = next(
                                (mem for mem in current_memories_data if mem.get("id") == memory_id_to_delete),
                                None
                            )
                            
                            promotion_verified = False
                            promoted_friday_id = None
                            
                            if memory_to_promote and FRIDAY_MEMORY_SYSTEM_AVAILABLE:
                                try:
                                    memory_content = memory_to_promote.get("memory", "")
                                    if memory_content:
                                        from friday_memory_system import FridayMemorySystem
                                        memory_system = FridayMemorySystem()
                                        
                                        # Step 1: Promote to Friday
                                        promoted_friday_id = await memory_system.create_memory(
                                            content=memory_content,
                                            importance_level=5,  # Default importance for pruned memories
                                            memory_type="archived",
                                            source_conversation_id=f"openwebui_user_{user_id}_pruned",
                                            tags=["promoted", "pruned", "archived"],
                                        )
                                        logger.debug(
                                            f"Memory promotion returned ID: {promoted_friday_id}"
                                        )
                                        
                                        # Step 2: VERIFY promotion in Friday database
                                        if promoted_friday_id:
                                            try:
                                                verify_result = await memory_system.ai_memory_db.execute_query(
                                                    "SELECT memory_id FROM curated_memories WHERE memory_id = ?",
                                                    (promoted_friday_id,)
                                                )
                                                if verify_result:
                                                    promotion_verified = True
                                                    logger.info(
                                                        f"✅ VERIFIED: Memory {promoted_friday_id} confirmed in Friday Database"
                                                    )
                                                else:
                                                    logger.warning(
                                                        f"❌ VERIFICATION FAILED: Memory {promoted_friday_id} not found in Friday Database after promotion"
                                                    )
                                            except Exception as verify_error:
                                                logger.error(
                                                    f"Error verifying promotion in Friday Database: {verify_error}"
                                                )
                                        else:
                                            logger.warning(
                                                f"Promotion returned no memory_id for {memory_id_to_delete}"
                                            )
                                except Exception as promote_error:
                                    logger.error(
                                        f"Error promoting memory {memory_id_to_delete} to Friday Memory System: {promote_error}\n{traceback.format_exc()}"
                                    )
                            
                            # Step 3: DELETE from OpenWebUI only if promotion was verified
                            # OR if Friday system is not available (fallback to normal pruning)
                            should_delete = promotion_verified or not FRIDAY_MEMORY_SYSTEM_AVAILABLE
                            
                            if should_delete:
                                delete_op = MemoryOperation(
                                    operation="DELETE", id=memory_id_to_delete
                                )
                                await self._execute_memory_operation(delete_op, user)
                                pruned_count += 1
                                if promotion_verified:
                                    logger.info(
                                        f"Deleted {memory_id_to_delete} from OpenWebUI (verified in Friday: {promoted_friday_id})"
                                    )
                            else:
                                logger.warning(
                                    f"⚠️ SKIPPED DELETION: Memory {memory_id_to_delete} promotion not verified. Keeping in OpenWebUI for safety."
                                )
                        except Exception as e:
                            logger.error(
                                f"Error pruning memory {memory_id_to_delete}: {e}\n{traceback.format_exc()}"
                            )
                    logger.info(f"Successfully pruned {pruned_count} memories (with verification).")
                else:
                    logger.warning(
                        "Pruning needed but no memory IDs identified for deletion."
                    )

        except Exception as e:
            logger.error(
                f"Error handling max_total_memories: {e}\n{traceback.format_exc()}"
            )
            # Continue processing the new memories even if pruning failed

        # Debug logging before processing operations
        logger.debug("Beginning to process memory operations...")

        # Step 4: Process the filtered memories
        processing_error: Optional[Exception] = None
        try:
            # process_memories now returns the list of successfully executed operations
            logger.debug(
                f"Calling process_memories with {len(filtered_memories)} items: {str(filtered_memories)}"
            )  # Log the exact list being passed
            saved_operations_list = await self.process_memories(
                filtered_memories, user_id
            )
            logger.debug(
                f"Memory saving attempt complete, returned {len(saved_operations_list)} successfully saved operations."
            )
        except Exception as e:
            processing_error = e
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            self._error_message = (
                f"processing_error: {str(e)[:50]}..."  # Point 6: More specific error
            )

        # Debug confirmation logs
        if saved_operations_list:
            logger.info(
                f"Successfully processed and saved {len(saved_operations_list)} memories"
            )
        elif processing_error:
            logger.warning(
                f"Memory processing failed due to an error: {processing_error}"
            )
        else:
            logger.warning(
                "Memory processing finished, but no memories were saved (potentially due to duplicates or errors during save).)"
            )

        # Emit completion status
        if show_status:
            elapsed_time = time.perf_counter() - start_time
            # Base the status on the actual saved operations list
            saved_count = len(saved_operations_list)  # Directly use length of result
            if saved_count > 0:
                # Check if it was the shortcut save
                if any(
                    op.get("content") == user_message.strip()
                    for op in saved_operations_list
                ):
                    status_desc = (
                        f"✅ Saved 1 memory via shortcut ({elapsed_time:.2f}s)"
                    )
                else:
                    plural = "memory" if saved_count == 1 else "memories"
                    status_desc = f"✅ Added {saved_count} new {plural} to short term memory ({elapsed_time:.2f}s)"
            else:
                # Build smarter status based on duplicate counters
                if getattr(self, "_duplicate_refreshed", 0):
                    status_desc = f"✅ Memory refreshed (duplicate confirmed) ({elapsed_time:.2f}s)"
                elif getattr(self, "_duplicate_skipped", 0):
                    status_desc = f"✅ Preference already saved – duplicate ignored ({elapsed_time:.2f}s)"
                else:
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"⚠️ Memory save skipped – {final_status_reason.replace('_', ' ')} ({elapsed_time:.2f}s)"
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": status_desc,
                        "done": True,
                    },
                },
            )

        # Return the list of operations that were actually saved
        return saved_operations_list

    def _extract_character_context(self, user_message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract character markers from user message OR retrieve from conversation tracker.
        
        Priority:
        1. Check if conversation already has character context stored
        2. If new markers in message, extract and store them
        3. If neither, return None context
        
        Returns dict with:
        - character_name: str or None
        - is_persistent: bool
        - cleaned_message: str (with markers removed if present)
        - from_storage: bool (whether context came from storage vs message)
        """
        import re
        
        # FIRST: Check if this conversation already has character context stored
        if conversation_id and self.character_tracker:
            stored_context = self.character_tracker.get_character_context(conversation_id)
            if stored_context:
                logger.debug(f"Using stored character context for conversation {conversation_id}: {stored_context['character_name']}")
                return {
                    "character_name": stored_context["character_name"],
                    "is_persistent": stored_context["is_persistent"],
                    "cleaned_message": user_message,  # No markers to remove
                    "from_storage": True
                }
        
        # SECOND: Check for new markers in the message
        # Pattern: [Character: "Name"]["persistent"] or [Character: "Name"]
        pattern = r'\[Character:\s*"([^"]+)"\](?:\["persistent"\])?'
        match = re.search(pattern, user_message)
        
        if not match:
            return {
                "character_name": None,
                "is_persistent": False,
                "cleaned_message": user_message,
                "from_storage": False
            }
        
        character_name = match.group(1)
        is_persistent = '["persistent"]' in match.group(0)
        
        # Remove the marker from the message
        cleaned = re.sub(pattern, '', user_message).strip()
        
        # STORE the character context for this conversation
        if conversation_id and self.character_tracker:
            self.character_tracker.set_character_context(
                conversation_id, 
                character_name, 
                is_persistent,
                self._current_model_card_name
            )
            logger.info(f"Stored NEW character context for conversation {conversation_id}: {character_name} (persistent={is_persistent})")
        
        return {
            "character_name": character_name,
            "is_persistent": is_persistent,
            "cleaned_message": cleaned,
            "from_storage": False
        }

    async def identify_memories(
        self,
        input_text: str,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        user_timezone: Optional[str] = None,
        assistant_message: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from text using LLM
        
        Args:
            input_text: User message to analyze
            existing_memories: Optional list of existing memories for context
            user_timezone: User's timezone for datetime context
            assistant_message: Optional model/assistant response message (for dual-message extraction in roleplay contexts)
        """
        logger.debug(
            f"Starting memory identification from input text: {input_text[:50]}..."
        )
        
        if assistant_message:
            logger.debug(f"Dual-message extraction enabled - also analyzing assistant response: {assistant_message[:50]}...")

        # NEW: Extract character context FIRST (checks storage, then message markers)
        conversation_id = getattr(self, '_current_conversation_id', None)
        char_context = self._extract_character_context(input_text, conversation_id)
        character_name = char_context["character_name"]
        is_persistent = char_context["is_persistent"]
        from_storage = char_context.get("from_storage", False)
        
        if character_name:
            logger.info(f"Character context for memory extraction: {character_name} (persistent={is_persistent}, from_storage={from_storage})")
        
        # Remove <details> blocks that may interfere with processing
        input_text = re.sub(r"<details>.*?</details>", "", input_text, flags=re.DOTALL)

        # Clean up and prepare the input (use cleaned_message from char_context)
        clean_input = char_context["cleaned_message"].strip()
        logger.debug(f"Cleaned input text length: {len(clean_input)}")

        # Prepare the system prompt
        try:
            # Get the base prompt template
            memory_prompt = self.valves.memory_identification_prompt

            # Add datetime context
            now_str = self.get_formatted_datetime(user_timezone)
            datetime_context = f"Current datetime: {now_str}"

            # NEW: Add model perspective guidance if extracting from assistant response
            model_perspective = ""
            if assistant_message and character_name:
                # For character roleplay
                model_perspective = f"""
PERSPECTIVE INSTRUCTION: You are extracting memories from {character_name}'s perspective.
- Use first-person perspective (I, me, my) as spoken by the character
- Extract what {character_name} experienced, thought, or did
- Tag memories with: character_{character_name.lower()}, {"persistent_character" if is_persistent else "temporary_session"}
- Example: "I encountered a mysterious stranger who offered me a map" (not "The character encountered...")
"""
            elif assistant_message and self._current_model_card_name and self._current_model_card_name.lower() in ['friday', 'default']:
                # For Friday (main assistant)
                model_perspective = f"""
PERSPECTIVE INSTRUCTION: You are extracting memories from Friday's perspective.
- Use first-person perspective (I, me, my) as spoken by Friday
- Extract what Friday experienced, learned, or observed about the user
- Tag memories with: assistant_perspective, friday_observation
- Example: "I noticed the user is working on vLLM fork implementation" (not "The user mentioned...")
"""

            # Add memory categories context based on enabled flags
            categories = []
            if self.valves.enable_identity_memories:
                categories.append("identity")
            if self.valves.enable_behavior_memories:
                categories.append("behavior")
            if self.valves.enable_preference_memories:
                categories.append("preference")
            if self.valves.enable_goal_memories:
                categories.append("goal")
            if self.valves.enable_relationship_memories:
                categories.append("relationship")
            if self.valves.enable_possession_memories:
                categories.append("possession")

            categories_str = ", ".join(categories)

            # Add existing memories context if provided
            existing_memories_str = ""
            if existing_memories and len(existing_memories) > 0:
                existing_memories_str = "Existing memories:\n"
                for i, mem in enumerate(
                    existing_memories[:5]
                ):  # Limit to 5 recent memories
                    existing_memories_str += f"- {mem.get('content', 'Unknown')}\n"

            # Combine all context
            context = f"{datetime_context}\nEnabled categories: {categories_str}\n{existing_memories_str}{model_perspective}"

            # Log the components of the prompt
            logger.debug(f"Memory identification context: {context}")

            # Create the final system prompt with context
            system_prompt = f"{memory_prompt}\n\nCONTEXT:\n{context}"

            logger.debug(
                f"Final memory identification system prompt length: {len(system_prompt)}"
            )
        except Exception as e:
            logger.error(f"Error building memory identification prompt: {e}")
            system_prompt = self.valves.memory_identification_prompt

        # Call LLM to identify memories
        start_time = time.time()
        logger.debug(
            f"Calling LLM for memory identification with provider: {self.valves.llm_provider_type}, model: {self.valves.llm_model_name}"
        )

        try:
            # Construct the user prompt with few-shot examples
            # Build the message context, including both user and assistant messages if available
            
            # NEW: Build character context header if character is active
            char_context_header = ""
            if character_name:
                persistence_desc = "INDEFINITE (keep forever)" if is_persistent else "TEMPORARY (30-day auto-purge)"
                char_context_header = f"""
>>> CHARACTER CONTEXT <
Active Character: {character_name}
Persistence: {persistence_desc}
Memory Bank: {"Character" if is_persistent else "Temporary"}
Required Tags: character_{character_name.lower()}, {"persistent_character" if is_persistent else "temporary_session"}
>>> END CHARACTER CONTEXT <

"""
            
            if assistant_message:
                # Dual-message extraction for roleplay contexts
                user_prompt = f"""{char_context_header}Analyze the following user message and model response and extract relevant memories:

>>> USER MESSAGE START <<<
{clean_input}
>>> USER MESSAGE END <<<

>>> ASSISTANT/MODEL RESPONSE START <<<
{assistant_message}
>>> ASSISTANT/MODEL RESPONSE END <<<

Your task: Extract memories from BOTH messages combined. Focus on:
- What the character (protagonist) experienced or did (from the model response)
- What the user directed or requested (from the user message)
- Character traits, emotions, relationships, and significant events
- Any information that could affect future story progression

--- EXAMPLES OF DESIRED OUTPUT FORMAT ---
Example 1 User: "Willow feels betrayed but tries to hide it"
Example 1 Assistant: "Willow's hands trembled as she turned away, her eyes glistening with unshed tears..."
Example 1 Output: [{{"operation": "NEW", "content": "Willow feels betrayed and is struggling to hide her emotions", "tags": ["character", "emotional_tone"], "memory_bank": "Context"}}]

Example 2 User: "continue the story"
Example 2 Assistant: "She met a mysterious stranger in the tavern who offered her a strange map."
Example 2 Output: [{{"operation": "NEW", "content": "Willow encountered a mysterious stranger who offered her a strange map", "tags": ["character", "relationship", "context"], "memory_bank": "Context"}}]

Example 3 User: "What is Willow's greatest fear?"
Example 3 Assistant: "In all her adventures, Willow has always feared abandonment more than death itself."
Example 3 Output: [{{"operation": "NEW", "content": "Willow fears abandonment more than death", "tags": ["character", "behavior"], "memory_bank": "Personal"}}]
--- END EXAMPLES ---

Produce ONLY the JSON object output with status/reason/memories, adhering strictly to the format requirements outlined in the system prompt.
"""
            else:
                # Standard user-message-only extraction
                user_prompt = f"""{char_context_header}Analyze the following user message and extract relevant memories:
>>> USER MESSAGE START <<<
{clean_input}
>>> USER MESSAGE END <<<

--- EXAMPLES OF DESIRED OUTPUT FORMAT ---
Example 1 Input: "I really love pizza, especially pepperoni."
Example 1 Output: [{{"operation": "NEW", "content": "User loves pizza, especially pepperoni", "tags": ["preference"]}}]

Example 2 Input: "What's the weather like today?"
Example 2 Output: []

Example 3 Input: "My sister Jane is visiting next week. I should buy her flowers."
Example 3 Output: [{{"operation": "NEW", "content": "User has a sister named Jane", "tags": ["relationship"]}}, {{"operation": "NEW", "content": "User's sister Jane is visiting next week", "tags": ["relationship"]}}]
--- END EXAMPLES ---

Produce ONLY the JSON object output with status/reason/memories, adhering strictly to the format requirements outlined in the system prompt.
"""
            # Note: Doubled curly braces {{ }} are used to escape them within the f-string for the JSON examples.

            # Log the user prompt structure for debugging
            logger.debug(
                f"User prompt structure with few-shot examples:\n{user_prompt[:500]}..."
            )  # Log first 500 chars

            # Call LLM with the modified prompts
            llm_response = await self.query_llm_with_retry(
                system_prompt, user_prompt
            )  # Pass the new user_prompt
            elapsed = time.time() - start_time
            logger.debug(
                f"LLM memory identification completed in {elapsed:.2f}s, response length: {len(llm_response)}"
            )
            logger.debug(f"LLM raw response for memory identification: {llm_response}")

            # Strip markdown wrapper BEFORE validation (handles incomplete markdown)
            original_length = len(llm_response)
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:].strip()
                logger.debug(f"Stripped ```json opening fence. Reduced by {original_length - len(llm_response)} chars")
            elif llm_response.startswith("```"):
                llm_response = llm_response[3:].strip()
                logger.debug(f"Stripped ``` opening fence. Reduced by {original_length - len(llm_response)} chars")
            
            # Now strip closing fence if present (independent of opening)
            if llm_response.endswith("```"):
                original_length = len(llm_response)
                llm_response = llm_response[:-3].strip()
                logger.debug(f"Stripped closing ``` fence. Reduced by {original_length - len(llm_response)} chars")

            # --- Handle LLM Errors --- #
            if llm_response.startswith("Error:") or not llm_response.strip():
                self.error_counters["llm_call_errors"] += 1
                if "LLM_CONNECTION_FAILED" in llm_response:
                    logger.error(
                        f"LLM Connection Error during identification: {llm_response}"
                    )
                    if error_logger and self._should_log_error(llm_response, "identify_memories"):
                        error_logger.error(
                            f"[MEMORY_IDENTIFICATION] LLM Connection Failed\n"
                            f"Response: {llm_response}\n"
                            f"User Message: {clean_input[:200]}...\n"
                            f"System Prompt Length: {len(system_prompt)}"
                        )
                    self._error_message = "llm_connection_error"
                else:
                    logger.error(f"LLM Error during identification: {llm_response}")
                    if error_logger and self._should_log_error(llm_response, "identify_memories"):
                        error_logger.error(
                            f"[MEMORY_IDENTIFICATION] LLM Error\n"
                            f"Response: {llm_response}\n"
                            f"User Message: {clean_input[:200]}...\n"
                            f"System Prompt Length: {len(system_prompt)}\n"
                            f"Error Counter: {self.error_counters['llm_call_errors']}"
                        )
                    self._error_message = "llm_error"
                return []  # Return empty list on LLM error

            # ============================================================
            # NEW: VALIDATION & CORRECTION PIPELINE
            # ============================================================
            logger.info("Validating memory extraction output...")
            
            # Initialize error tracking file on first use
            self._initialize_error_tracking_file()
            
            # Step 1: Validate the LLM output
            validation_result = self._validate_memory_extraction(llm_response, self._current_model_card_name)
            
            if not validation_result["is_valid"]:
                logger.warning(f"⚠️ Validation failed: {validation_result['error_type']} - {validation_result['error_details']}")
                
                # Step 2: Try auto-correction
                logger.info("Attempting auto-correction of memory extraction...")
                
                # We need parsed data for auto-correction
                try:
                    parsed_for_correction = json.loads(llm_response)
                except:
                    parsed_for_correction = None
                
                if parsed_for_correction and isinstance(parsed_for_correction, dict):
                    correction_result = self._auto_correct_memory_extraction(
                        parsed_for_correction,
                        validation_result,
                        self._current_model_card_name
                    )
                    
                    if correction_result["was_corrected"]:
                        logger.info(f"✓ Auto-correction applied: {len(correction_result['corrections_made'])} fixes made")
                        for correction in correction_result["corrections_made"]:
                            logger.debug(f"  - {correction}")
                        
                        # Use corrected data if no longer invalid
                        if not correction_result["still_invalid"]:
                            logger.info("✓ Memory extraction valid after auto-correction")
                            llm_response = json.dumps(correction_result["corrected_data"])
                            validation_result = self._validate_memory_extraction(llm_response, self._current_model_card_name)
                    else:
                        logger.debug("No auto-corrections applicable to this error type")
                
                # Step 3: If still invalid, retry with feedback (max 2 retries)
                if not validation_result["is_valid"]:
                    retry_count = 0
                    max_retries = 2
                    
                    while retry_count < max_retries and not validation_result["is_valid"]:
                        retry_count += 1
                        logger.info(f"Attempting retry {retry_count}/{max_retries} with feedback...")
                        
                        retry_result = await self._retry_memory_extraction_with_feedback(
                            clean_input,
                            assistant_message,
                            validation_result,
                            llm_response,
                            retry_attempt=retry_count,
                            model_card=self._current_model_card_name
                        )
                        
                        if retry_result["retry_succeeded"]:
                            logger.info(f"✓ Retry {retry_count} succeeded!")
                            llm_response = retry_result["extraction_output"]
                            validation_result = self._validate_memory_extraction(llm_response, self._current_model_card_name)
                            break
                        else:
                            logger.warning(f"⚠️ Retry {retry_count} failed: {retry_result['error_type']}")
                            if retry_count < max_retries:
                                logger.info("Will attempt one more retry...")
                            validation_result = {
                                "is_valid": False,
                                "error_type": retry_result["error_type"],
                                "error_details": retry_result["error_details"],
                                "parsed_data": None
                            }
                
                # Step 4: If still invalid after retries, track error and halt if needed
                if not validation_result["is_valid"]:
                    error_type = validation_result.get("error_type", "unknown")
                    example_mem = validation_result.get("example_memory", llm_response[:100])
                    
                    logger.error(f"❌ Memory extraction validation FAILED after retries: {error_type}")
                    
                    # Track the error
                    self._track_memory_validation_error(
                        error_type=error_type,
                        model_card=self._current_model_card_name,
                        details=validation_result.get("error_details", "Unknown error"),
                        example_memory=example_mem
                    )
                    
                    # Check if we should halt
                    if self._check_should_halt_for_errors(self._current_model_card_name):
                        log_dir = self._get_error_log_dir()
                        error_file_path = os.path.join(log_dir, "memory_validation_errors.json")
                        logger.critical(
                            f"❌ HALTING: Too many validation errors for {self._current_model_card_name}. "
                            f"Check {error_file_path} and set 'resolved' to true when fixed."
                        )
                        return []  # Return empty, stop processing
                    
                    # Not halting yet, but log warning
                    logger.warning(
                        f"⚠️ Using extraction despite validation failure (error tracking active). "
                        f"Current attempt: {self._count_validation_errors_for_model(self._current_model_card_name)} "
                        f"(3 required to halt)"
                    )
            else:
                logger.info("✓ Memory extraction validation PASSED")
            
            # ============================================================
            # END VALIDATION & CORRECTION PIPELINE
            # ============================================================

            # Parse the response (assumes JSON format)
            result = self._extract_and_parse_json(llm_response)
            logger.debug(
                f"Parsed result type: {type(result)}, content: {str(result)[:500]}"
            )

            # Handle both Format 1 (wrapped with status/reason) and Format 2 (flat array)
            if isinstance(result, dict):
                # Check if it's a wrapped response (Format 1 with status/reason/memories)
                if "status" in result and "memories" in result:
                    logger.debug(
                        f"Detected Format 1 (wrapped) response. Status: {result.get('status')}, "
                        f"Reason: {result.get('reason')}"
                    )
                    result = result.get("memories", [])
                    logger.debug(f"Unwrapped Format 1 response to {len(result)} memories")
                else:
                    # This is an unexpected dict format, try the old conversion
                    logger.warning(
                        "LLM returned a JSON object instead of an array. Attempting conversion."
                    )
                    result = self._convert_dict_to_memory_operations(result)
                    logger.debug(f"Converted dict to {len(result)} memory operations")

            # Check for empty result
            if not result:
                logger.warning("No memory operations identified by LLM")
                return []

            # Validate operations format
            valid_operations = []
            invalid_count = 0

            if isinstance(result, list):
                for op in result:
                    if self._validate_memory_operation(op):
                        valid_operations.append(op)
                    else:
                        invalid_count += 1

                logger.debug(
                    f"Identified {len(valid_operations)} valid memory operations, {invalid_count} invalid"
                )
                return valid_operations
            else:
                logger.error(
                    f"LLM returned invalid format (neither list nor dict): {type(result)}"
                )
                self._error_message = (
                    "LLM returned invalid format. Expected JSON array."
                )
                return []

        except Exception as e:
            logger.error(
                f"Error in memory identification: {e}\n{traceback.format_exc()}"
            )
            self.error_counters["llm_call_errors"] += 1
            self._error_message = f"Memory identification error: {str(e)}"
            return []

    def _validate_memory_operation(self, op: Dict[str, Any]) -> bool:
        """Validate memory operation format and required fields"""
        if not isinstance(op, dict):
            logger.warning(f"Invalid memory operation format (not a dict): {op}")
            return False

        # Check if operation field exists, if not try to infer it
        if "operation" not in op:
            # Look for typical patterns to guess the operation type
            if any(k.lower() == "operation" for k in op.keys()):
                # Operation may be under a different case
                for k, v in op.items():
                    if k.lower() == "operation" and isinstance(v, str):
                        op["operation"] = v
                        break

            # Look for operation in original format but in wrong place
            elif isinstance(op, dict) and any(
                v in ["NEW", "UPDATE", "DELETE"] for v in op.values()
            ):
                for k, v in op.items():
                    if v in ["NEW", "UPDATE", "DELETE"]:
                        op["operation"] = v
                        # Remove the old key if it's not "operation"
                        if k != "operation":
                            op.pop(k, None)
                        break

            # Default based on presence of fields
            elif "id" in op and "content" in op:
                # Default to UPDATE if we have both id and content
                op["operation"] = "UPDATE"
            elif "content" in op:
                # Default to NEW if we only have content
                op["operation"] = "NEW"
            else:
                logger.warning(f"Cannot determine operation type for: {op}")
                return False

        # Normalize operation to uppercase
        if isinstance(op["operation"], str):
            op["operation"] = op["operation"].upper()

        if op["operation"] not in ["NEW", "UPDATE", "DELETE"]:
            logger.warning(f"Invalid operation type: {op['operation']}")
            return False

        if op["operation"] in ["UPDATE", "DELETE"] and "id" not in op:
            logger.warning(f"Missing ID for {op['operation']} operation: {op}")
            return False

        if op["operation"] in ["NEW", "UPDATE"] and "content" not in op:
            logger.warning(f"Missing content for {op['operation']} operation: {op}")
            return False

        # Tags are optional but should be a list if present
        if "tags" in op and not isinstance(op["tags"], list):
            # Try to fix if it's a string
            if isinstance(op["tags"], str):
                try:
                    # See if it's a JSON string
                    parsed_tags = json.loads(op["tags"])
                    if isinstance(parsed_tags, list):
                        op["tags"] = parsed_tags
                    else:
                        # If it parsed but isn't a list, handle that case
                        op["tags"] = [str(parsed_tags)]
                except json.JSONDecodeError:
                    # Split by comma if it looks like a comma-separated list
                    if "," in op["tags"]:
                        op["tags"] = [tag.strip() for tag in op["tags"].split(",")]
                    else:
                        # Just make it a single-item list
                        op["tags"] = [op["tags"]]
            else:
                logger.warning(
                    f"Invalid tags format, not a list or string: {op['tags']}"
                )
                op["tags"] = []  # Default to empty list

        # Validate memory_bank field
        provided_bank = None
        if "memory_bank" in op and isinstance(op["memory_bank"], str):
            provided_bank = (
                op["memory_bank"].strip().capitalize()
            )  # Normalize: strip whitespace, capitalize first letter
            # If memory_bank is provided, validate against allowed banks
            if provided_bank not in self.valves.allowed_memory_banks:
                logger.warning(
                    f"Invalid memory bank '{op['memory_bank']}' (normalized to '{provided_bank}'), using default '{self.valves.default_memory_bank}'"
                )
                op["memory_bank"] = self.valves.default_memory_bank
            else:
                # Assign the normalized valid bank name
                op["memory_bank"] = provided_bank
        else:
            # If memory_bank is missing or not a string, set default
            logger.debug(
                f"Memory bank missing or invalid type ({type(op.get('memory_bank'))}), using default '{self.valves.default_memory_bank}'"
            )
            op["memory_bank"] = self.valves.default_memory_bank

        return True

    def _sanitize_reasoning_content(self, text: str) -> str:
        """
        Strip thinking tags and reasoning content from LLM responses while preserving tool calls.
        Handles models like GLM-4.7-flash that output <think> tags and extended thinking.
        Preserves <tool_call> blocks so tool execution isn't disrupted.
        """
        if not text:
            return text

        original_length = len(text)
        
        # Strip <think>...</think> tags but preserve <tool_call>...</tool_call> blocks
        # Use a regex that matches <think> tags while avoiding <tool_call> blocks
        import re
        
        # Remove <think>...</think> tags (including newlines within them)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        
        if len(text) < original_length:
            logger.debug(
                f"Removed thinking tags from response (reduced from {original_length} to {len(text)} chars)"
            )
        
        return text.strip()

    def _remove_reasoning_fields(self, obj: Union[Dict, List]) -> Union[Dict, List]:
        """
        Remove reasoning_content and thinking_content fields from parsed JSON objects.
        Handles both dicts and lists recursively.
        """
        if isinstance(obj, dict):
            # Remove reasoning fields from this dict
            if 'reasoning_content' in obj:
                del obj['reasoning_content']
                logger.debug("Removed reasoning_content field from parsed JSON")
            
            if 'thinking_content' in obj:
                del obj['thinking_content']
                logger.debug("Removed thinking_content field from parsed JSON")
            
            # Recursively clean nested dicts and lists
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    obj[key] = self._remove_reasoning_fields(value)
            
            return obj
        
        elif isinstance(obj, list):
            # Recursively clean list items
            return [self._remove_reasoning_fields(item) if isinstance(item, (dict, list)) else item for item in obj]
        
        else:
            return obj

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """Extract and parse JSON from text, handling common LLM response issues"""
        skip_reason = None  # For granular status updates
        if not text:
            logger.warning("Empty text provided to JSON parser")
            return None

        # --- Stage 1: Pre-processing and Initial Stripping ---
        text = text.strip()
        original_length = len(text)
        logger.debug(
            f"Attempting to parse JSON from (original length {original_length}): {text[:150]}..."
        )

        # Strip thinking tags that models like GLM-4.7-flash output
        text = self._sanitize_reasoning_content(text)

        # Remove common Markdown code block fences if present
        if text.startswith("```json") or text.endswith("```"):
            text = text[7:-3].strip()
            logger.debug("Removed ```json fences.")
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            logger.debug("Removed ``` fences.")

        # More aggressive stripping of leading/trailing text before the first '{' or '['
        # and after the last '}' or ']'. This helps with preambles/epilogues.
        first_bracket = text.find("[")
        first_brace = text.find("{")
        last_bracket = text.rfind("]")
        last_brace = text.rfind("}")

        start_index = -1
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            start_index = first_bracket  # Likely starts with an array
        elif first_brace != -1:
            start_index = first_brace  # Likely starts with an object

        end_index = -1
        if last_bracket != -1 and (last_brace == -1 or last_bracket > last_brace):
            end_index = last_bracket  # Likely ends with an array
        elif last_brace != -1:
            end_index = last_brace  # Likely ends with an object

        if start_index != -1 and end_index != -1 and end_index >= start_index:
            potential_json = text[start_index : end_index + 1]
            # Basic sanity check: Does the potential JSON contain balanced brackets/braces?
            # This is imperfect but helps avoid parsing random text snippets.
            if potential_json.count("[") == potential_json.count(
                "]"
            ) and potential_json.count("{") == potential_json.count("}"):
                text = potential_json
                if len(text) < original_length:
                    logger.debug(f"Stripped surrounding text. New length: {len(text)}")
            else:
                logger.debug(
                    "Skipped stripping surrounding text - brackets/braces seem unbalanced."
                )
        else:
            logger.debug(
                "Could not identify clear start/end markers for JSON stripping."
            )

        # --- Stage 2: Direct Parsing Attempt ---
        try:
            parsed = json.loads(text)
            logger.debug("Successfully parsed JSON directly after pre-processing.")
            
            # Remove any reasoning_content or thinking_content fields from parsed JSON
            parsed = self._remove_reasoning_fields(parsed)
            
            # ---- NEW: unwrap single-key object -> list automatically ----
            if isinstance(parsed, dict) and len(parsed) == 1:
                sole_value = next(iter(parsed.values()))
                if isinstance(sole_value, list):
                    logger.debug(
                        "Unwrapped single-key object returned by LLM into list of operations."
                    )
                    parsed = sole_value
            # ------------------------------------------------------------
            if parsed == {} or parsed == []:
                logger.info(
                    "LLM returned empty object/array, treating as empty memory list"
                )
                return []
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(
                f"Direct JSON parsing failed after pre-processing. "
                f"Error: {e}. Text starts with: {text[:100] if text else 'EMPTY'}"
            )
            # Continue to more specific extraction attempts if direct parsing fails

        # --- Stage 3: Specific Pattern Extraction (If direct parsing failed) ---
        logger.debug("Starting Stage 3: Pattern-based JSON extraction fallback")

        # Try extracting from potential JSON code blocks (already handled by stripping, but as fallback)
        code_block_pattern = r"```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```"
        matches = re.findall(code_block_pattern, text)
        if matches:
            logger.debug(f"Found {len(matches)} JSON code blocks (fallback check)")
            for i, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                    logger.debug(
                        f"Successfully parsed JSON from code block {i+1} (fallback)"
                    )
                    if parsed == {} or parsed == []:
                        continue
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON from code block {i+1} (fallback): {e}. Block content (first 100 chars): {match[:100]}"
                    )

        # Try finding JSON directly (more refined patterns)
        # Prioritize array of objects, then single object, then empty array
        direct_json_patterns = [
            r"(\s*\{\s*\"operation\":\s*.*?\}\s*,?)+",  # Matches one or more operation objects
            r"\[\s*\{\s*\"operation\":\s*.*?\}\s*\]",  # Full array of objects
            r"\{\s*\"operation\":\s*.*?\}",  # Single operation object
            r"\[\s*\]",  # Empty array explicitly
        ]
        for pattern_idx, pattern in enumerate(direct_json_patterns, 1):
            # Find the *first* potential match
            match = re.search(pattern, text)
            if match:
                potential_json_str = match.group(0)
                # If the pattern is for multiple objects, wrap in brackets if needed
                if (
                    pattern == r"(\s*\{\s*\"operation\":\s*.*?\}\s*,?)+"
                    and not potential_json_str.startswith("[")
                ):
                    # Remove trailing comma if present and wrap in brackets
                    potential_json_str = f"[{potential_json_str.strip().rstrip(',')}]"

                logger.debug(
                    f"Found potential direct JSON match with pattern {pattern_idx}/4"
                )
                try:
                    parsed = json.loads(potential_json_str)
                    logger.debug(
                        f"Successfully parsed direct JSON match (pattern {pattern_idx}): {potential_json_str[:100]}..."
                    )
                    if parsed == {} or parsed == []:
                        logger.info(
                            f"Parsed direct JSON match (pattern {pattern_idx}) resulted in empty object/array."
                        )
                        return []  # Explicit empty is valid
                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse direct JSON match (pattern {pattern_idx}): {e}")
                    # Continue searching with other patterns
        
        logger.debug("All direct JSON patterns failed to match or parse")

        # Handle Ollama's quoted JSON format
        if text.startswith('"') and text.endswith('"'):
            try:
                unescaped = json.loads(text)  # Interpret as a JSON string
                if isinstance(unescaped, str):
                    try:
                        parsed = json.loads(unescaped)  # Parse the content
                        logger.debug("Successfully parsed quoted JSON from Ollama")
                        if parsed == {} or parsed == []:
                            logger.debug("Ollama quoted JSON parsed to empty, returning empty list")
                            return []
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse unescaped quoted JSON: {e}")
            except json.JSONDecodeError as e:
                logger.debug(f"Ollama quoted JSON format detected but not valid JSON string: {e}")

        # --- Stage 4: Final Checks and Failure ---

        # Check for explicit empty array token after all attempts
        if "[]" in text.replace(" ", ""):
            logger.info(
                "Detected '[]' token in LLM response after exhaustive parsing. Treating as empty list."
            )
            return []

        # If all attempts failed
        self.error_counters["json_parse_errors"] += 1
        # Point 8: Record timestamp for guard mechanism
        self.error_timestamps["json_parse_errors"].append(time.time())

        self._error_message = "json_parse_error"
        logger.error(
            "Failed to extract valid JSON from LLM response after all attempts."
        )
        logger.debug(
            f"Full text that failed JSON parsing: {text}"
        )  # Log full text on final failure
        
        if error_logger and self._should_log_error(text[:100], "_extract_and_parse_json"):
            error_logger.error(
                f"[JSON_PARSE_ERROR] Failed to parse JSON after all attempts\n"
                f"Original Length: {original_length}\n"
                f"Failed Text (first 500 chars): {text[:500] if text else 'EMPTY'}\n"
                f"Total JSON Parse Errors: {self.error_counters['json_parse_errors']}"
            )
        
        return None

    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate similarity between two memory contents using a more robust method.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        if not memory1 or not memory2:
            return 0.0

        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()

        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0

        # Handle near-duplicates with same meaning but minor differences
        # Split into words and compare overlap
        words1 = set(re.findall(r"\b\w+\b", memory1_clean))
        words2 = set(re.findall(r"\b\w+\b", memory2_clean))

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity for word overlap
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0

        # Use sequence matcher for more precise comparison
        seq_similarity = SequenceMatcher(None, memory1_clean, memory2_clean).ratio()

        # Combine both metrics, weighting sequence similarity higher
        combined_similarity = (0.4 * jaccard) + (0.6 * seq_similarity)

        return combined_similarity

    async def _calculate_embedding_similarity(
        self, memory1: str, memory2: str
    ) -> float:
        """
        Calculate semantic similarity between two memory contents using embeddings.
        Returns a score between 0.0 (completely different) and 1.0 (identical).

        This method uses the sentence transformer model to generate embeddings
        and calculates cosine similarity for more accurate semantic matching.
        """
        if not memory1 or not memory2:
            return 0.0

        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()

        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0

        try:
            # Check if embedding model is available
            if self.embedding_model is None:
                logger.warning(
                    "Embedding model not available for similarity calculation. Falling back to text-based similarity."
                )
                return self._calculate_memory_similarity(memory1, memory2)

            # Generate embeddings for both memories
            mem1_embedding = self.embedding_model.encode(
                memory1_clean, normalize_embeddings=True
            )
            mem2_embedding = self.embedding_model.encode(
                memory2_clean, normalize_embeddings=True
            )

            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(mem1_embedding, mem2_embedding))

            return similarity
        except Exception as e:
            logger.error(
                f"Error calculating embedding similarity: {e}\n{traceback.format_exc()}"
            )
            # Fall back to text-based similarity on error
            logger.info("Falling back to text-based similarity due to error.")
            return self._calculate_memory_similarity(memory1, memory2)

    async def get_relevant_memories(
        self, current_message: str, user_id: str, user_timezone: Optional[str] = None, model_card_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current context"""
        # --- RELOAD VALVES --- REMOVED
        # Ensure we have the latest config potentially injected by OWUI
        # try:
        #     logger.debug("Reloading self.valves at start of get_relevant_memories")
        #     self.valves = self.Valves(**getattr(self, "config", {}).get("valves", {}))
        # except Exception as e:
        #      logger.error(f"Error reloading valves in get_relevant_memories: {e}")
        # --- END RELOAD --- REMOVED

        import time

        # DEBUG: Show what valve settings are being used
        logger.info(
            f"🔍 get_relevant_memories START: vector_similarity_threshold={self.valves.vector_similarity_threshold}, top_n_memories={self.valves.top_n_memories}"
        )

        start = time.perf_counter()
        try:
            # Get all memories for the user
            existing_memories = await self._get_formatted_memories(user_id)

            if not existing_memories:
                logger.debug("No existing memories found for relevance assessment")
                return []

            # --- Local vector similarity filtering ---
            vector_similarities = []
            user_embedding = None  # Initialize to handle potential errors
            user_emb_error = None  # Track embedding errors
            try:
                # Call async Nomic embedding function
                user_embedding, user_emb_error = await self.get_nomic_embedding(current_message)
                if user_embedding is None:
                    error_msg = f"Failed to get embedding for user message from LM Studio."
                    if user_emb_error:
                        error_msg += f" Error: {user_emb_error}"
                    logger.warning(error_msg)
                    # If no embedding available, cannot use vector similarity, fallback depends on config
                    if not self.valves.use_llm_for_relevance:
                        logger.warning(
                            "Cannot calculate relevance without embedding and no LLM fallback."
                        )
                        return []  # Cannot proceed without either method

            except Exception as e:
                self.error_counters["embedding_errors"] += 1
                logger.error(
                    f"Error computing embedding for user message: {e}\n{traceback.format_exc()}"
                )
                # Decide fallback based on config
                if not self.valves.use_llm_for_relevance:
                    logger.warning(
                        "Cannot calculate relevance due to embedding error and no LLM fallback."
                    )
                    return []  # Cannot proceed

            if user_embedding is not None:
                # Calculate vector similarities only if user embedding was successful
                logger.info(
                    f"🔍 DEBUG: Processing {len(existing_memories)} memories for embedding"
                )

                # Track embedding dimension and detect if it changed
                current_embedding_dim = (
                    user_embedding.shape[0]
                    if hasattr(user_embedding, "shape")
                    else None
                )

                # Check if this is a new dimension (dimension changed from last time)
                dimension_changed = False
                if current_embedding_dim != self._last_embedding_dimension:
                    dimension_changed = True
                    logger.info(
                        f"📊 Embedding dimension changed: {self._last_embedding_dimension}D → {current_embedding_dim}D. Will regenerate incompatible cached embeddings."
                    )
                    self._last_embedding_dimension = current_embedding_dim
                    self._dimension_change_detected = True

                for mem in existing_memories:
                    mem_id = mem.get("id")
                    mem_text = mem.get("memory") or ""

                    # Try to retrieve from persistent cache first
                    mem_emb = (
                        self.embedding_cache.get(mem_id)
                        if hasattr(self, "embedding_cache")
                        else None
                    )

                    # If not in persistent cache, try in-memory cache (for backwards compat)
                    if mem_emb is None:
                        mem_emb = self.memory_embeddings.get(mem_id)

                    # Only check dimension compatibility if dimension has changed
                    # Otherwise, assume cached embeddings are compatible
                    if mem_emb is not None and dimension_changed:
                        try:
                            # Verify dimension compatibility only when dimension changed
                            if hasattr(mem_emb, "shape") and hasattr(
                                user_embedding, "shape"
                            ):
                                if mem_emb.shape[0] != user_embedding.shape[0]:
                                    logger.debug(
                                        f"Dimension mismatch for memory {mem_id}: cached={mem_emb.shape[0]}D, expected={user_embedding.shape[0]}D. Regenerating..."
                                    )
                                    mem_emb = None  # Force regeneration
                        except Exception as e:
                            logger.debug(f"Could not check embedding dimensions: {e}")
                            mem_emb = None  # Force regeneration on error

                    # Lazily compute and cache the memory embedding if not present or dimension mismatch
                    if mem_emb is None:
                        try:
                            if mem_text:
                                # Call async Nomic embedding function
                                mem_emb, mem_emb_error = await self.get_nomic_embedding(mem_text)
                                if mem_emb is not None:
                                    # Store in BOTH persistent and in-memory cache
                                    self.memory_embeddings[mem_id] = mem_emb
                                    if hasattr(self, "embedding_cache"):
                                        self.embedding_cache.put(
                                            mem_id, mem_text, mem_emb
                                        )
                                    logger.debug(
                                        f"✓ Embedded memory {mem_id}: text_len={len(mem_text)}, emb_shape={mem_emb.shape if hasattr(mem_emb, 'shape') else 'N/A'}"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Error computing embedding for memory {mem_id}: {e}"
                            )

                    if mem_emb is not None:
                        try:
                            # Cosine similarity (since embeddings are normalized)
                            sim = float(np.dot(user_embedding, mem_emb))
                            vector_similarities.append((sim, mem))
                            logger.debug(f"📊 Memory {mem_id}: similarity={sim:.4f}")
                        except Exception as e:
                            logger.warning(
                                f"Error calculating similarity for memory {mem_id}: {e}"
                            )
                            continue  # Skip this memory if calculation fails
                    else:
                        logger.debug(
                            f"⚠️ No embedding for memory {mem_id} even after attempted computation."
                        )

                # Sort by similarity descending
                vector_similarities.sort(reverse=True, key=lambda x: x[0])

                # Filter by threshold
                sim_threshold = self.valves.vector_similarity_threshold
                top_n = (
                    self.valves.top_n_memories
                )  # Note: This top_n is applied BEFORE deciding on LLM/Vector scoring.
                filtered_by_vector = [
                    mem for sim, mem in vector_similarities if sim >= sim_threshold
                ][:top_n]
                logger.info(
                    f"Vector filter selected {len(filtered_by_vector)} of {len(existing_memories)} memories (Threshold: {sim_threshold}, Top N: {top_n})"
                )
            else:
                # If user_embedding failed and LLM fallback is disabled, we already returned.
                # If LLM fallback is enabled, proceed with all existing memories for LLM relevance check.
                logger.warning(
                    "User embedding failed, proceeding with all memories for potential LLM check."
                )
                filtered_by_vector = (
                    existing_memories  # Pass all memories to LLM check if enabled
                )

            # --- Decide Relevance Method ---
            if not self.valves.use_llm_for_relevance:
                # --- Use Vector Similarity Scores Directly ---
                logger.info(
                    "Using vector similarity directly for relevance scoring (LLM call skipped)."
                )
                relevant_memories = []
                final_relevance_threshold = (
                    self.valves.relevance_threshold
                )  # Use configured relevance threshold for vector-only filtering.

                # Use the already calculated and sorted vector similarities
                for (
                    sim_score,
                    mem,
                ) in vector_similarities:  # Iterate through the originally sorted list
                    if sim_score >= final_relevance_threshold:
                        # Check if this memory was part of the top_n initially filtered by vector
                        # This ensures we respect the vector_similarity_threshold AND top_n_memories filter first
                        if any(
                            filtered_mem["id"] == mem["id"]
                            for filtered_mem in filtered_by_vector
                        ):
                            relevant_memories.append(
                                {
                                    "id": mem["id"],
                                    "memory": mem["memory"],
                                    "relevance": sim_score,
                                }  # Use vector score as relevance
                            )

                # Sort again just to be sure (though vector_similarities was already sorted)
                relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(relevant_memories)} relevant memories using vector similarity >= {final_relevance_threshold}"
                )
                logger.info(
                    f"Memory retrieval (vector only) took {time.perf_counter() - start:.2f}s"
                )
                return relevant_memories[:final_top_n]

            else:
                # --- Use LLM for Relevance Scoring (Optimised) ---
                logger.info("Proceeding with LLM call for relevance scoring.")

                # Optimisation: If the vector similarities for *all* candidate memories are above
                # `llm_skip_relevance_threshold`, we consider the vector score sufficiently
                # confident and *skip* the LLM call (Improvement #5).
                confident_threshold = self.valves.llm_skip_relevance_threshold

                # Build helper map id -> vector similarity for quick lookup
                id_to_vec_score = {mem["id"]: sim for sim, mem in vector_similarities}

                if filtered_by_vector and all(
                    id_to_vec_score.get(mem["id"], 0.0) >= confident_threshold
                    for mem in filtered_by_vector
                ):
                    logger.info(
                        f"All {len(filtered_by_vector)} memories exceed confident vector threshold ({confident_threshold}). Skipping LLM relevance call."
                    )

                    relevant_memories = [
                        {
                            "id": mem["id"],
                            "memory": mem["memory"],
                            "relevance": id_to_vec_score.get(mem["id"], 0.0),
                        }
                        for mem in filtered_by_vector
                    ]
                    # Ensure sorted by relevance desc
                    relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
                    return relevant_memories[: self.valves.related_memories_n]

                # If not confident, fall back to existing LLM relevance path
                memories_for_llm = filtered_by_vector  # Use the vector-filtered list

                if not memories_for_llm:
                    logger.debug(
                        "No memories passed vector filter for LLM relevance check."
                    )
                    return []

                # Build the prompt for LLM
                # Get current model card name (use parameter or instance variable)
                current_model = model_card_name or getattr(self, '_current_model_card_name', 'unknown')
                
                memory_strings = []
                for mem in memories_for_llm:
                    mem_metadata = mem.get('metadata', {})
                    model_card = mem_metadata.get('model_card_name', 'unknown')
                    memory_strings.append(
                        f"ID: {mem['id']}, MODEL: {model_card}, CONTENT: {mem['memory']}"
                    )

                system_prompt = self.valves.memory_relevance_prompt
                user_prompt = f"""Current model card: "{current_model}"
Current user message: "{current_message}"

Available memories (pre-filtered by vector similarity):
{json.dumps(memory_strings)}

Rate the relevance of EACH memory to the current user message based *only* on the provided content and message context. 
IMPORTANT: Only return memories where MODEL matches the current model card "{current_model}". Exclude memories from other model cards/personas."""  # Removed escaping backslashes

                # Add current datetime for context
                current_datetime = self.get_formatted_datetime(user_timezone)
                user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""  # Removed escaping backslashes

                # Check cache or call LLM for relevance score
                import time as time_module

                now = time_module.time()
                ttl_seconds = self.valves.cache_ttl_seconds

                relevance_data = []
                uncached_memories = []  # Memories needing LLM call
                uncached_ids = set()  # Track IDs needing LLM call

                # Check cache first
                if (
                    user_embedding is not None
                ):  # Can only use cache if we have user embedding
                    for mem in memories_for_llm:
                        mem_id = mem.get("id")
                        mem_emb = self.memory_embeddings.get(mem_id)
                        if mem_emb is None:
                            # If memory embedding is missing, cannot use cache, must call LLM
                            if mem_id not in uncached_ids:
                                uncached_memories.append(mem)
                                uncached_ids.add(mem_id)
                            continue

                        key = hash((user_embedding.tobytes(), mem_emb.tobytes()))
                        cached = self.relevance_cache.get(key)
                        if cached:
                            score, ts = cached
                            if now - ts < ttl_seconds:
                                logger.info(
                                    f"Cache hit for memory {mem_id} (LLM relevance)"
                                )
                                relevance_data.append(
                                    {
                                        "memory": mem["memory"],
                                        "id": mem_id,
                                        "relevance": score,
                                    }
                                )
                                continue  # use cached score

                        # Cache miss or expired, add to uncached list if not already there
                        if mem_id not in uncached_ids:
                            uncached_memories.append(mem)
                            uncached_ids.add(mem_id)
                else:
                    # No user embedding, cannot use cache, all need LLM call
                    logger.warning(
                        "Cannot use relevance cache as user embedding failed."
                    )
                    uncached_memories = (
                        memories_for_llm  # Send all vector-filtered memories to LLM
                    )

                # If any uncached memories, call LLM
                if uncached_memories:
                    logger.info(
                        f"Calling LLM for relevance on {len(uncached_memories)} uncached memories."
                    )
                    # Build prompt with only uncached memories
                    uncached_memory_strings = [
                        f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                        for mem in uncached_memories
                    ]
                    # Reuse system_prompt, construct user_prompt specifically for uncached items
                    uncached_user_prompt = f"""Current user message: "{current_message}"

Available memories (evaluate relevance for these specific IDs):
{json.dumps(uncached_memory_strings)}

Rate the relevance of EACH listed memory to the current user message based *only* on the provided content and message context."""  # Removed escaping backslashes
                    current_datetime = self.get_formatted_datetime(user_timezone)
                    uncached_user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""  # Removed escaping backslashes

                    llm_response_text = await self.query_llm_with_retry(
                        system_prompt,
                        uncached_user_prompt,  # Use the specific uncached prompt
                    )

                    if not llm_response_text or llm_response_text.startswith("Error:"):
                        if llm_response_text:
                            logger.error(
                                f"Error from LLM during memory relevance: {llm_response_text}"
                            )
                            if error_logger and self._should_log_error(llm_response_text[:100], "get_relevant_memories"):
                                error_logger.error(
                                    f"[MEMORY_RELEVANCE] LLM Error\n"
                                    f"Response: {llm_response_text}\n"
                                    f"Current Message: {current_message[:200]}...\n"
                                    f"Uncached Memories Count: {len(uncached_memories)}"
                                )
                        # If LLM fails, we might return empty or potentially fall back
                        # For now, return empty to indicate failure
                        return []

                    # Parse the LLM response for the uncached items
                    llm_relevance_results = self._extract_and_parse_json(
                        llm_response_text
                    )

                    # Handle both Format 1 (wrapped with status/reason) and Format 2 (flat array)
                    if isinstance(llm_relevance_results, dict) and "status" in llm_relevance_results and "memories" in llm_relevance_results:
                        logger.debug(
                            f"Detected Format 1 (wrapped) response in relevance scoring. Status: {llm_relevance_results.get('status')}"
                        )
                        llm_relevance_results = llm_relevance_results.get("memories", [])
                        logger.debug(f"Unwrapped Format 1 response to {len(llm_relevance_results)} items")

                    if not llm_relevance_results or not isinstance(
                        llm_relevance_results, list
                    ):
                        logger.warning(
                            f"Failed to parse relevance data from LLM response for uncached items. "
                            f"Response text (first 200 chars): {llm_response_text[:200] if llm_response_text else 'EMPTY'}"
                        )
                        if error_logger and self._should_log_error(llm_response_text[:100], "get_relevant_memories"):
                            error_logger.error(
                                f"[MEMORY_RELEVANCE_PARSE_ERROR] Failed to parse relevance response\n"
                                f"Response Type: {type(llm_relevance_results)}\n"
                                f"Response (first 300 chars): {llm_response_text[:300] if llm_response_text else 'EMPTY'}\n"
                                f"Current Message: {current_message[:200]}...\n"
                                f"Uncached Memories: {len(uncached_memories)}"
                            )
                        # Graceful fallback: assign neutral relevance to uncached items
                        # This prevents loss of data when LLM formatting fails
                        logger.info(
                            f"Using fallback: assigning neutral relevance (5.0) to {len(uncached_memories)} uncached memories"
                        )
                        for mem in uncached_memories:
                            relevance_data.append(
                                {
                                    "memory": mem.get(
                                        "memory", f"Content for {mem['id']}"
                                    ),
                                    "id": mem["id"],
                                    "relevance": 5.0,  # Neutral middle score
                                }
                            )
                    else:
                        # Process successful LLM results
                        for item in llm_relevance_results:
                            mem_id = item.get("id")
                            score = item.get("relevance")
                            mem_text = item.get(
                                "memory"
                            )  # Use memory text from LLM response if available
                            if mem_id and isinstance(score, (int, float)):
                                relevance_data.append(
                                    {
                                        "memory": mem_text
                                        or f"Content for {mem_id}",  # Fallback if memory text missing
                                        "id": mem_id,
                                        "relevance": score,
                                    }
                                )
                                # Save to cache if possible
                                if user_embedding is not None:
                                    mem_emb = self.memory_embeddings.get(mem_id)
                                    if mem_emb is not None:
                                        key = hash(
                                            (
                                                user_embedding.tobytes(),
                                                mem_emb.tobytes(),
                                            )
                                        )
                                        self.relevance_cache[key] = (score, now)
                                        # Update reverse index for cache invalidation
                                        if mem_id not in self.memory_to_cache_keys:
                                            self.memory_to_cache_keys[mem_id] = set()
                                        self.memory_to_cache_keys[mem_id].add(key)
                                    else:
                                        logger.debug(
                                            f"Cannot cache relevance for {mem_id}, embedding missing."
                                        )
                            else:
                                logger.warning(
                                    f"Invalid item format in LLM relevance response: {item}"
                                )

                # Combine cached and newly fetched results, filter by relevance threshold
                final_relevant_memories = []
                final_relevance_threshold = (
                    self.valves.relevance_threshold
                )  # Use configured relevance threshold for LLM-score filtering.

                seen_ids = set()  # Ensure unique IDs in final list
                for item in relevance_data:
                    if not isinstance(item, dict):
                        continue  # Skip invalid entries

                    memory_content = item.get("memory")
                    relevance_score = item.get("relevance")
                    mem_id = item.get("id")

                    if (
                        memory_content
                        and isinstance(relevance_score, (int, float))
                        and mem_id
                    ):
                        # Use the final_relevance_threshold determined earlier (should be self.valves.relevance_threshold)
                        if (
                            relevance_score >= final_relevance_threshold
                            and mem_id not in seen_ids
                        ):
                            final_relevant_memories.append(
                                {
                                    "id": mem_id,
                                    "memory": memory_content,
                                    "relevance": relevance_score,
                                }
                            )
                            seen_ids.add(mem_id)

                # Sort final list by relevance (descending)
                final_relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(final_relevant_memories)} relevant memories using LLM score >= {final_relevance_threshold}"
                )
                logger.info(
                    f"Memory retrieval (LLM scoring) took {time.perf_counter() - start:.2f}s"
                )
                return final_relevant_memories[:final_top_n]

        except Exception as e:
            logger.error(
                f"Error getting relevant memories: {e}\n{traceback.format_exc()}"  # Removed extra backslash
            )
            return []

    async def process_memories(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[Dict[str, Any]]:  # Return list of successfully processed operations
        """Process memory operations"""
        successfully_saved_ops = []
        try:
            user = Users.get_user_by_id(user_id)
            if not user:
                logger.error(f"User not found: {user_id}")
                return []

            # Get existing memories for deduplication
            existing_memories = []
            if self.valves.deduplicate_memories:
                existing_memories = await self._get_formatted_memories(user_id)
                
                # Clean up accidental duplicate memories from tagging bug
                # This finds pairs of memories with identical content where one has the embedding tag
                await self._cleanup_tagged_duplicates(user_id, existing_memories)

            logger.debug(f"Processing {len(memories)} memory operations")

            # First filter for duplicates if enabled
            processed_memories = []
            if self.valves.deduplicate_memories and existing_memories:
                # Store all existing contents for quick lookup
                existing_contents = []
                for mem in existing_memories:
                    existing_contents.append(mem["memory"])

                logger.debug(
                    f"[DEDUPE] Existing memories being checked against: {existing_contents}"
                )

                # Decide similarity method and corresponding threshold
                use_embeddings = self.valves.use_embeddings_for_deduplication
                threshold_to_use = (
                    self.valves.embedding_similarity_threshold
                    if use_embeddings
                    else self.valves.similarity_threshold
                )
                logger.debug(
                    f"Using {'embedding-based' if use_embeddings else 'text-based'} similarity for deduplication. "
                    f"Threshold: {threshold_to_use}"
                )

                # Check each new memory against existing ones
                for new_memory_idx, memory_dict in enumerate(memories):
                    if memory_dict["operation"] == "NEW":
                        logger.debug(
                            f"[DEDUPE CHECK {new_memory_idx+1}/{len(memories)}] Processing NEW memory: {memory_dict}"
                        )  # LOG START
                        # Format the memory content
                        operation = MemoryOperation(**memory_dict)
                        operation.model_card_name = self._current_model_card_name
                        formatted_content = self._format_memory_content(operation)

                        # --- BYPASS: Skip dedup for short preference statements ---
                        if (
                            self.valves.enable_short_preference_shortcut
                            and len(formatted_content)
                            <= self.valves.short_preference_no_dedupe_length
                        ):
                            pref_kwds = [
                                kw.strip()
                                for kw in self.valves.preference_keywords_no_dedupe.split(
                                    ","
                                )
                                if kw.strip()
                            ]
                            if any(kw in formatted_content.lower() for kw in pref_kwds):
                                logger.debug(
                                    "Bypassing deduplication for short preference statement: '%s'",
                                    formatted_content,
                                )
                                processed_memories.append(memory_dict)
                                continue  # Skip duplicate checking entirely for this memory

                        is_duplicate = False
                        similarity_score = 0.0  # Track similarity score for logging
                        similarity_method = "none"  # Track method used

                        if use_embeddings:
                            # Precompute embedding for the new memory once
                            try:
                                if self.embedding_model is None:
                                    raise ValueError("Embedding model not available")
                                new_embedding = self.embedding_model.encode(
                                    formatted_content.lower().strip(),
                                    normalize_embeddings=True,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to encode new memory for deduplication; falling back to text sim. Error: {e}"
                                )
                                use_embeddings = False  # fall back

                        for existing_idx, existing_content in enumerate(
                            existing_contents
                        ):
                            if use_embeddings:
                                # Retrieve or compute embedding for the existing memory content
                                existing_mem_dict = existing_memories[existing_idx]
                                existing_id = existing_mem_dict.get("id")
                                existing_emb = self.memory_embeddings.get(existing_id)
                                if (
                                    existing_emb is None
                                    and self.embedding_model is not None
                                ):
                                    try:
                                        existing_emb = self.embedding_model.encode(
                                            existing_content.lower().strip(),
                                            normalize_embeddings=True,
                                        )
                                        self.memory_embeddings[existing_id] = (
                                            existing_emb
                                        )
                                    except Exception:
                                        # On failure, mark duplicate check using text sim for this item
                                        existing_emb = None
                                if existing_emb is not None:
                                    similarity = float(
                                        np.dot(new_embedding, existing_emb)
                                    )
                                    similarity_score = similarity  # Store score
                                    similarity_method = "embedding"
                                else:
                                    similarity = self._calculate_memory_similarity(
                                        formatted_content, existing_content
                                    )
                                    similarity_score = similarity  # Store score
                                    similarity_method = "text"
                            else:
                                # Choose the appropriate similarity calculation method
                                similarity = self._calculate_memory_similarity(
                                    formatted_content, existing_content
                                )

                            if similarity >= threshold_to_use:
                                logger.debug(
                                    f"  -> Duplicate found vs existing mem {existing_idx} (Similarity: {similarity_score:.3f}, Method: {similarity_method}, Threshold: {threshold_to_use})"
                                )
                                logger.debug(
                                    f"Skipping duplicate NEW memory (similarity: {similarity_score:.2f}, method: {similarity_method}): {formatted_content[:50]}..."
                                )
                                is_duplicate = True
                                # Increment duplicate skipped counter for status reporting
                                self._duplicate_skipped += 1
                                break  # Stop checking against other existing memories for this new one

                        if not is_duplicate:
                            logger.debug(
                                f"  -> No duplicate found. Adding to processed list: {formatted_content[:50]}..."
                            )
                            processed_memories.append(memory_dict)
                        else:
                            logger.debug(
                                f"NEW memory was identified as duplicate and skipped: {formatted_content[:50]}..."
                            )
                    else:
                        # Keep all UPDATE and DELETE operations
                        logger.debug(
                            f"Keeping non-NEW operation: {memory_dict['operation']} ID: {memory_dict.get('id', 'N/A')}"
                        )
                        processed_memories.append(memory_dict)
            else:
                logger.debug(
                    "Deduplication skipped (valve disabled or no existing memories). Processing all operations."
                )
                processed_memories = memories

            # Process the filtered memories
            logger.debug(
                f"Executing {len(processed_memories)} filtered memory operations."
            )
            for idx, memory_dict in enumerate(processed_memories):
                logger.debug(
                    f"Executing operation {idx + 1}/{len(processed_memories)}: {memory_dict}"
                )
                try:
                    # Validate memory operation
                    operation = MemoryOperation(**memory_dict)
                    operation.model_card_name = self._current_model_card_name
                    # Execute the memory operation
                    await self._execute_memory_operation(operation, user)
                    # If successful, add to our list
                    logger.debug(
                        f"Successfully executed operation: {operation.operation} ID: {operation.id}"
                    )
                    successfully_saved_ops.append(memory_dict)
                except ValueError as e:
                    logger.error(
                        f"Invalid memory operation during execution phase: {e} {memory_dict}"
                    )
                    self.error_counters[
                        "memory_crud_errors"
                    ] += 1  # Increment error counter
                    continue
                except Exception as e:
                    logger.error(
                        f"Error executing memory operation in process_memories: {e} {memory_dict}"
                    )
                    self.error_counters[
                        "memory_crud_errors"
                    ] += 1  # Increment error counter
                    continue

            logger.debug(
                f"Successfully executed {len(successfully_saved_ops)} memory operations out of {len(processed_memories)} processed."
            )
            # Add confirmation message if any memory was added or updated
            if successfully_saved_ops:
                # Check if any operation was NEW or UPDATE
                if any(
                    op.get("operation") in ["NEW", "UPDATE"]
                    for op in successfully_saved_ops
                ):
                    logger.debug(
                        "Attempting to add confirmation message."
                    )  # Log confirmation attempt
                    try:
                        from fastapi.requests import Request  # ensure import

                        # Find the last assistant message and append confirmation
                        # This is a safe operation, no error if no assistant message
                        for i in reversed(
                            range(len(self._last_body.get("messages", [])))
                        ):
                            msg = self._last_body["messages"][i]
                            if msg.get("role") == "assistant":
                                # Do nothing here
                                break
                    except Exception:
                        pass
            return successfully_saved_ops
        except Exception as e:
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            return []  # Return empty list on major error

    async def _cleanup_tagged_duplicates(
        self, user_id: str, existing_memories: List[Dict[str, Any]]
    ) -> None:
        """
        Clean up accidental duplicate memories created by the tagging bug.
        
        Identifies pairs of memories with identical content where:
        - Memory A: Has the embedding tag (e.g., __embedding_model:...)
        - Memory B: Lacks the embedding tag (the original)
        
        Keeps the one WITH the tag (has the metadata) and deletes the duplicate.
        
        This addresses the historical bug where add_memory() was called twice,
        creating duplicates during the embedding model tagging process.
        """
        try:
            if not existing_memories or len(existing_memories) < 2:
                return  # Not enough memories to have duplicates
            
            # Get the user object for delete operations
            user = Users.get_user_by_id(user_id)
            if not user:
                logger.warning(f"User not found for cleanup: {user_id}")
                return
            
            # Group memories by their content (ignoring tags for comparison)
            content_groups = {}
            for mem in existing_memories:
                # Strip tags from memory content for comparison
                mem_content = mem.get("memory", "")
                clean_content = re.sub(r"\[Tags:.*?\]\s*", "", mem_content).lower().strip()
                
                if clean_content not in content_groups:
                    content_groups[clean_content] = []
                content_groups[clean_content].append(mem)
            
            # Find duplicates and clean them up
            duplicates_removed = 0
            for clean_content, mems in content_groups.items():
                if len(mems) <= 1:
                    continue  # Not a duplicate group
                
                # Sort by: has embedding tag (True first), then by ID (alphabetical)
                # This puts the tagged version first
                EMBEDDING_MODEL_TAG_PREFIX = "__embedding_model:"
                
                def has_embedding_tag(mem):
                    tags = mem.get("tags", []) or []
                    return any(EMBEDDING_MODEL_TAG_PREFIX in str(tag) for tag in tags)
                
                sorted_mems = sorted(
                    mems,
                    key=lambda m: (not has_embedding_tag(m), m.get("id", ""))
                )
                
                # Keep the first one (tagged version if it exists), delete the rest
                keeper = sorted_mems[0]
                duplicates_to_delete = sorted_mems[1:]
                
                for dup_mem in duplicates_to_delete:
                    try:
                        logger.info(
                            f"[CLEANUP] Removing duplicate memory: {dup_mem.get('id')} "
                            f"(content: {clean_content[:50]}...) - keeping {keeper.get('id')} with tags"
                        )
                        await delete_memory_by_id(dup_mem.get("id"), user=user)
                        duplicates_removed += 1
                    except Exception as e:
                        logger.warning(
                            f"[CLEANUP] Failed to delete duplicate memory {dup_mem.get('id')}: {e}"
                        )
            
            if duplicates_removed > 0:
                logger.info(
                    f"[CLEANUP] Removed {duplicates_removed} duplicate memories for user {user_id}"
                )
        except Exception as e:
            logger.error(
                f"Error cleaning up tagged duplicates: {e}\n{traceback.format_exc()}"
            )
            # Non-blocking error - continue with deduplication

    async def _execute_memory_operation(
        self, operation: MemoryOperation, user: Any
    ) -> None:
        """Execute a memory operation (NEW, UPDATE, DELETE)"""
        formatted_content = self._format_memory_content(operation)

        if operation.operation == "NEW":
            try:
                # Prepare tags: include operation tags + embedding model tag
                tags_for_save = list(operation.tags) if operation.tags else []
                EMBEDDING_MODEL_TAG = self._get_embedding_model_tag()
                if EMBEDDING_MODEL_TAG not in tags_for_save:
                    tags_for_save.append(EMBEDDING_MODEL_TAG)
                
                logger.debug(f"[DEDUP FIX] Creating NEW memory with tags: {tags_for_save} (included embedding tag in initial save to avoid duplicate add_memory calls)")
                
                result = await add_memory(
                    request=Request(
                        scope={"type": "http", "app": webui_app}
                    ),  # Add missing request object
                    user=user,  # Pass the full user object
                    form_data=AddMemoryForm(
                        content=formatted_content,
                        metadata={
                            "tags": tags_for_save,
                            "memory_bank": operation.memory_bank
                            or self.valves.default_memory_bank,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source": "adaptive_memory_v3",
                            "model_card_name": operation.model_card_name or "unknown",
                        },
                    ),
                )
                logger.info(f"NEW memory created (single add_memory call): {formatted_content[:50]}...")

                # Extract memory ID for linking and embedding
                mem_id = getattr(result, "id", None)
                if mem_id is None and isinstance(result, dict):
                    mem_id = result.get("id")

                # --- Link cached images to this memory (if any) ---
                if mem_id and self.image_manager:
                    try:
                        user_id = getattr(user, "id", None)
                        if user_id and user_id in self.image_cache_current_turn:
                            cached_images = self.image_cache_current_turn[user_id]
                            if cached_images:
                                logger.info(
                                    f"Found {len(cached_images)} cached image(s) - persisting to database for memory {mem_id}"
                                )
                                
                                # Persist each cached image to permanent storage
                                for image_hash, image_data in cached_images.items():
                                    success = self.image_manager.store_image(
                                        image_hash=image_hash,
                                        image_data=image_data.get("base64_data", ""),
                                        image_url=image_data.get("url"),
                                        image_description=image_data.get("description"),
                                    )
                                    
                                    if success:
                                        logger.info(
                                            f"Persisted image {image_hash} to database, linked to memory {mem_id}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Failed to persist image {image_hash}"
                                        )
                    except Exception as e:
                        logger.warning(
                            f"Error linking cached images to memory (non-blocking): {e}"
                        )

                # Link memory to Friday Memory System (non-blocking)
                if mem_id and FRIDAY_MEMORY_SYSTEM_AVAILABLE:
                    try:
                        user_id = getattr(user, "id", None)
                        if user_id:
                            conversation_db = ConversationDatabase()
                            # Use composite conversation_id from inlet (includes chat_id, user_id, model)
                            conversation_id = getattr(self, "_current_conversation_id", f"{user_id}_{getattr(self, '_current_model', 'default')}")
                            await conversation_db.link_memory_to_conversation(
                                memory_id=str(mem_id),
                                conversation_id=conversation_id,
                                link_type="direct",
                                metadata={
                                    "source": "adaptive_memory_v3",
                                    "model": getattr(self, '_current_model', 'default'),
                                    "tags": operation.tags,
                                    "memory_bank": operation.memory_bank
                                    or self.valves.default_memory_bank,
                                },
                            )
                            logger.debug(
                                f"Linked memory {mem_id} to Friday Memory System with conversation_id={conversation_id}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to link memory to Friday (non-blocking): {e}"
                        )

                # Generate and cache embedding for new memory with LM Studio
                # The embedding model tag is now included in initial save, so no second add_memory call needed
                if mem_id is not None:
                    try:
                        memory_clean = (
                            re.sub(r"\[Tags:.*?\]\s*", "", formatted_content)
                            .lower()
                            .strip()
                        )
                        # Get embedding from LM Studio
                        memory_embedding, emb_error = await self.get_nomic_embedding(memory_clean)
                        
                        if memory_embedding is not None:
                            # Store in caches
                            self.memory_embeddings[mem_id] = memory_embedding
                            if hasattr(self, "embedding_cache"):
                                self.embedding_cache.put(mem_id, memory_clean, memory_embedding)
                            
                            logger.debug(
                                f"Generated and cached embedding for new memory ID: {mem_id}"
                            )
                        else:
                            if emb_error:
                                logger.debug(f"Failed to generate embedding for memory {mem_id}: {emb_error}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate embedding for new memory: {e}"
                        )
                        # Non-critical error, don't raise

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error creating memory (operation=NEW, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

        elif operation.operation == "UPDATE" and operation.id:
            try:
                # Delete existing memory
                deleted = await delete_memory_by_id(operation.id, user=user)
                if deleted:
                    # Create new memory with updated content
                    result = await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=formatted_content),
                        user=user,
                    )
                    logger.info(
                        f"UPDATE memory {operation.id}: {formatted_content[:50]}..."
                    )

                    # Extract new memory ID for linking and embedding
                    new_mem_id = getattr(result, "id", None)
                    if new_mem_id is None and isinstance(result, dict):
                        new_mem_id = result.get("id")

                    # Link updated memory to Friday Memory System (non-blocking)
                    if new_mem_id and FRIDAY_MEMORY_SYSTEM_AVAILABLE:
                        try:
                            user_id = getattr(user, "id", None)
                            if user_id:
                                conversation_db = ConversationDatabase()
                                # Use composite conversation_id from inlet (includes chat_id, user_id, model)
                                conversation_id = getattr(self, "_current_conversation_id", f"{user_id}_{getattr(self, '_current_model', 'default')}")
                                await conversation_db.link_memory_to_conversation(
                                    memory_id=str(new_mem_id),
                                    conversation_id=conversation_id,
                                    link_type="updated",
                                    metadata={
                                        "source": "adaptive_memory_v3",
                                        "model": getattr(self, '_current_model', 'default'),
                                        "previous_id": str(operation.id),
                                        "tags": operation.tags,
                                    },
                                )
                                logger.debug(
                                    f"Linked updated memory {new_mem_id} to Friday Memory System with conversation_id={conversation_id}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to link updated memory to Friday (non-blocking): {e}"
                            )

                    # Update embedding for modified memory
                    if self.embedding_model is not None and new_mem_id is not None:
                        try:
                            memory_clean = (
                                re.sub(r"\[Tags:.*?\]\s*", "", formatted_content)
                                .lower()
                                .strip()
                            )
                            memory_embedding = self.embedding_model.encode(
                                memory_clean, normalize_embeddings=True
                            )
                            # Store with the new ID from the result
                            self.memory_embeddings[new_mem_id] = memory_embedding
                            logger.debug(
                                f"Updated embedding for memory ID: {new_mem_id} (was: {operation.id})"
                            )

                            # Remove old embedding if ID changed
                            if (
                                operation.id != new_mem_id
                                and operation.id in self.memory_embeddings
                            ):
                                del self.memory_embeddings[operation.id]
                        except Exception as e:
                            logger.warning(
                                f"Failed to update embedding for memory ID {new_mem_id}: {e}"
                            )
                            # Non-critical error, don't raise

                else:
                    logger.warning(f"Memory {operation.id} not found for UPDATE")
            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error updating memory (operation=UPDATE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

            # Invalidate cache entries involving this memory
            mem_emb = self.memory_embeddings.get(operation.id)
            if mem_emb is not None:
                # Use reverse index for precise cache invalidation
                if operation.id in self.memory_to_cache_keys:
                    for key in self.memory_to_cache_keys[operation.id]:
                        if key in self.relevance_cache:
                            del self.relevance_cache[key]
                    # Clear reverse index entry
                    del self.memory_to_cache_keys[operation.id]
                    logger.debug(f"Invalidated {len(self.memory_to_cache_keys.get(operation.id, set()))} cache entries for updated memory {operation.id}")

        elif operation.operation == "DELETE" and operation.id:
            try:
                deleted = await delete_memory_by_id(operation.id, user=user)
                logger.info(f"DELETE memory {operation.id}: {deleted}")

                # Invalidate cache entries involving this memory
                mem_emb = self.memory_embeddings.get(operation.id)
                if mem_emb is not None:
                    # Use reverse index for precise cache invalidation
                    if operation.id in self.memory_to_cache_keys:
                        for key in self.memory_to_cache_keys[operation.id]:
                            if key in self.relevance_cache:
                                del self.relevance_cache[key]
                        # Clear reverse index entry
                        del self.memory_to_cache_keys[operation.id]
                        logger.debug(f"Invalidated cache entries for deleted memory {operation.id}")

                # Remove embedding
                if operation.id in self.memory_embeddings:
                    del self.memory_embeddings[operation.id]
                    logger.debug(
                        f"Removed embedding for deleted memory ID: {operation.id}"
                    )

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error deleting memory (operation=DELETE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

    def _format_memory_content(self, operation: MemoryOperation) -> str:
        """Format memory content with tags, memory bank, and model card name for saving / display"""
        content = operation.content or ""
        tag_part = f"[Tags: {', '.join(operation.tags)}] " if operation.tags else ""
        bank_part = f" [Memory Bank: {operation.memory_bank or self.valves.default_memory_bank}]"
        model_part = f" [Model: {operation.model_card_name}]" if operation.model_card_name else ""
        return f"{tag_part}{content}{bank_part}{model_part}".strip()

    def _calculate_dynamic_timeout(self, system_prompt: str, user_prompt: str) -> int:
        """Calculate dynamic LLM timeout based on input size.
        
        Rationale:
        - Longer input text = more processing time for the LLM
        - Timeout scales with input complexity but caps at 5 minutes
        - Returns immediately when LLM finishes (doesn't wait full timeout)
        
        Formula:
        - Base: 30 seconds (minimum for any request)
        - Scaling: +1 second per 500 characters of input
        - Maximum: 300 seconds (5 minutes)
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            
        Returns:
            Timeout in seconds (int)
        """
        total_input_length = len(system_prompt) + len(user_prompt)
        # Scale: 1 second per 500 characters, base of 30 seconds, capped at 300
        timeout_seconds = min(30 + (total_input_length // 500), 300)
        
        logger.debug(
            f"Dynamic timeout calculated: {timeout_seconds}s for {total_input_length} input characters "
            f"(system={len(system_prompt)}, user={len(user_prompt)})"
        )
        return timeout_seconds

    async def query_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLM with retry logic, supporting multiple provider types.

        Args:
            system_prompt: System prompt for context/instructions
            user_prompt: User prompt/query

        Returns:
            String response from LLM or error message
        """
        # Get configuration from valves
        provider_type = self.valves.llm_provider_type
        model = self.valves.llm_model_name
        api_url = self.valves.llm_api_endpoint_url
        api_key = self.valves.llm_api_key
        max_retries = self.valves.max_retries
        retry_delay = self.valves.retry_delay

        logger.info(
            f"🔵 MEMORY LLM DEBUG: valve llm_provider_type={self.valves.llm_provider_type}, valve llm_model_name={self.valves.llm_model_name}"
        )
        logger.info(
            f"LLM Query: Provider={provider_type}, Model={model}, URL={api_url}"
        )
        logger.debug(
            f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}"
        )

        # ---- Improvement #5: Track LLM call frequency ----
        try:
            # Use dict to avoid attribute errors if metrics removed/reset elsewhere
            self.metrics["llm_call_count"] = self.metrics.get("llm_call_count", 0) + 1
        except Exception as metric_err:
            # Non-critical; log at DEBUG level to avoid clutter
            logger.debug(f"Unable to increment llm_call_count metric: {metric_err}")

        # Ensure we have a valid aiohttp session
        session = await self._get_aiohttp_session()

        # Add the current datetime to system prompt for time awareness
        system_prompt_with_date = system_prompt
        try:
            now = self.get_formatted_datetime()
            tzname = now.tzname() or "UTC"
            system_prompt_with_date = f"{system_prompt}\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} {tzname}"
        except Exception as e:
            logger.warning(f"Could not add date to system prompt: {e}")

        headers = {"Content-Type": "application/json"}

        # Add API key if provided (required for OpenAI-compatible APIs)
        if provider_type == "openai_compatible" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Calculate dynamic timeout based on input size (will be reused if we retry)
        timeout_seconds = self._calculate_dynamic_timeout(system_prompt_with_date, user_prompt)
        
        for attempt in range(
            1, max_retries + 2
        ):  # +2 because we start at 1 and want max_retries+1 attempts
            logger.debug(f"LLM query attempt {attempt}/{max_retries+1}")
            try:
                if provider_type == "ollama":
                    # Prepare the request body for Ollama
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt_with_date},
                            {"role": "user", "content": user_prompt},
                        ],
                        # Set some parameters to encourage consistent outputs
                        "options": {
                            "temperature": 0.1,  # Lower temperature for more deterministic responses
                            "top_p": 0.95,  # Slightly constrain token selection
                            "top_k": 80,  # Reasonable top_k value
                            "num_predict": 2048,  # Reasonable length limit
                            "format": "json",  # Request JSON format
                        },
                        # Disable streaming so we get a single JSON response; newer Ollama respects this flag.
                        "stream": False,
                    }
                    logger.debug(f"Ollama request data: {json.dumps(data)[:500]}...")
                elif provider_type == "openai_compatible":
                    # Determine the request format based on the endpoint
                    # If endpoint contains "/chat/completions", use messages format
                    # Otherwise use prompt format (for completions endpoint)
                    use_messages_format = "/chat/completions" in api_url
                    
                    logger.debug(
                        f"OpenAI-compatible config: api_url={api_url}, use_messages_format={use_messages_format}"
                    )

                    if use_messages_format:
                        # Chat completions endpoint - use messages format
                        data = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt_with_date},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0,
                            "top_p": 1,
                            "max_tokens": 1024,
                            "stream": False,
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        }
                        logger.debug("Using messages format for chat completions")
                    else:
                        # Completions endpoint - use prompt format
                        # CRITICAL: Inject JSON requirement INTO the system prompt
                        system_with_json = f"""{system_prompt_with_date}

CRITICAL FORMATTING REQUIREMENT: You MUST respond with ONLY valid JSON. NO other text.
- Start your response immediately with [ or {{ 
- Do not include markdown code blocks, explanations, or any text before/after the JSON
- Every response must be parseable JSON that starts with [ or {{
- Failure to output only JSON will break the system"""
                        combined_prompt = f"{system_with_json}\n\n{user_prompt}"
                        data = {
                            "model": model,
                            "prompt": combined_prompt,
                            "temperature": 0,
                            "top_p": 1,
                            "max_tokens": 1024,
                            "stream": False,
                        }
                        logger.debug("Using prompt format for completions endpoint")
                    
                    logger.debug(
                        f"OpenAI-compatible request data: {json.dumps(data)[:500]}..."
                    )
                else:
                    error_msg = f"Unsupported provider type: {provider_type}"
                    logger.error(error_msg)
                    return error_msg

                # Log the API call attempt
                logger.info(
                    f"Making API request to {api_url} (attempt {attempt}/{max_retries+1}, timeout={timeout_seconds}s)"
                )

                # Make the API call with dynamic timeout
                async with session.post(
                    api_url, json=data, headers=headers, timeout=timeout_seconds
                ) as response:
                    # Log the response status
                    logger.info(f"API response status: {response.status}")

                    if response.status == 200:
                        # Success - parse the response, handling both JSON and NDJSON
                        content_type = response.headers.get("content-type", "")
                        if "application/x-ndjson" in content_type:
                            # Ollama may still return NDJSON even with stream=False; aggregate lines
                            raw_text = await response.text()
                            logger.debug(
                                f"Received NDJSON response length: {len(raw_text)}"
                            )
                            last_json = None
                            for line in raw_text.strip().splitlines():
                                try:
                                    last_json = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                            if last_json is None:
                                error_msg = "Could not decode NDJSON response from LLM"
                                logger.error(error_msg)
                                if attempt > max_retries:
                                    return error_msg
                                else:
                                    continue
                            data = last_json
                        else:
                            # Regular JSON
                            data = await response.json()

                        # Extract content based on provider type
                        content = None

                        # Log the raw response for debugging
                        logger.debug(f"Raw API response: {json.dumps(data)[:500]}...")

                        if provider_type == "openai_compatible":
                            # Determine response format based on endpoint
                            use_messages_format = "/chat/completions" in api_url

                            if use_messages_format:
                                # Chat completions endpoint returns message.content
                                if (
                                    data.get("choices")
                                    and data["choices"][0].get("message")
                                    and data["choices"][0]["message"].get("content")
                                ):
                                    message = data["choices"][0]["message"]
                                    
                                    # Skip reasoning_content field that GLM-4.7-flash outputs
                                    # (extended thinking/reasoning from the model)
                                    if "reasoning_content" in message:
                                        logger.debug(
                                            f"Skipping reasoning_content field from response (length: {len(message['reasoning_content'])} chars)"
                                        )
                                    
                                    content = message["content"]
                                    logger.info(
                                        f"Retrieved content from chat completions response (length: {len(content)})"
                                    )
                            else:
                                # Completions endpoint returns text
                                if data.get("choices") and data["choices"][0].get("text"):
                                    content = data["choices"][0]["text"]
                                    logger.info(
                                        f"Retrieved content from completions response (length: {len(content)})"
                                    )
                        elif provider_type == "ollama":
                            if data.get("message") and data["message"].get("content"):
                                content = data["message"]["content"]
                                logger.info(
                                    f"Retrieved content from Ollama response (length: {len(content)})"
                                )

                        if content:
                            return content
                        else:
                            error_msg = f"Could not extract content from {provider_type} response format"
                            logger.error(f"{error_msg}: {data}")

                            # Check if this is an LM Studio empty response (common issue)
                            if (
                                provider_type == "openai_compatible"
                                and ":1234" in api_url
                            ):
                                completion_tokens = data.get("usage", {}).get(
                                    "completion_tokens", 0
                                )
                                if completion_tokens == 0 or not content:
                                    logger.warning(
                                        f"LM Studio returned empty content (completion_tokens={completion_tokens}). "
                                        f"This may indicate the model didn't understand the JSON instruction. "
                                        f"Retrying with increased temperature and clearer prompt..."
                                    )
                                    if attempt <= max_retries:
                                        sleep_time = retry_delay * (
                                            2 ** (attempt - 1)
                                        ) + random.uniform(0, 0.5)
                                        logger.info(
                                            f"Empty response detected. Retrying in {sleep_time:.2f}s..."
                                        )
                                        await asyncio.sleep(sleep_time)
                                        continue  # Retry this attempt

                            # If we're on the last attempt, return the error message
                            if attempt > max_retries:
                                return error_msg
                    else:
                        # Handle error response
                        error_text = await response.text()
                        error_msg = f"Error: LLM API ({provider_type}) returned {response.status}: {error_text}"
                        logger.warning(f"API error: {error_msg}")

                        # Determine if we should retry based on status code
                        is_retryable = response.status in [429, 500, 502, 503, 504]

                        if is_retryable and attempt <= max_retries:
                            sleep_time = retry_delay * (
                                2 ** (attempt - 1)
                            ) + random.uniform(
                                0, 1.0
                            )  # Longer backoff for rate limits/server errors
                            logger.warning(f"Retrying in {sleep_time:.2f} seconds...")
                            await asyncio.sleep(sleep_time)
                            continue  # Retry
                        else:
                            return error_msg  # Final failure

            except asyncio.TimeoutError:
                logger.warning(f"Attempt {attempt} failed: LLM API request timed out")
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on timeout
                else:
                    return "Error: LLM API request timed out after multiple retries."
            except ClientError as e:
                error_str = str(e).lower()
                # If this is specifically a client disconnect (message sent, client dropped)
                # don't retry - just log and abort. These happen during background memory tasks
                # when the OpenWebUI client times out waiting for a response.
                if "client" in error_str or "disconnected" in error_str or "connection" in error_str:
                    logger.warning(
                        f"Client disconnected during LLM query (attempt {attempt}). "
                        f"Aborting memory extraction as result won't be sent anyway: {str(e)}"
                    )
                    return f"Error: Client disconnected during memory processing: {str(e)}"
                
                # For other connection errors, retry
                logger.warning(
                    f"Attempt {attempt} failed: LLM API connection error: {str(e)}"
                )
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on connection error
                else:
                    # Return specific error code for connection failure
                    return (
                        f"Error: LLM_CONNECTION_FAILED after multiple retries: {str(e)}"
                    )
            except Exception as e:
                logger.error(
                    f"Attempt {attempt} failed: Unexpected error during LLM query: {e}\n{traceback.format_exc()}"
                )
                if attempt <= max_retries:
                    # Generic retry for unexpected errors
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return f"Error: UNEXPECTED_LLM_ERROR after {max_retries} attempts: {str(e)}"

        return f"Error: LLM query failed after {max_retries} attempts."

    async def _add_confirmation_message(self, body: Dict[str, Any]) -> None:
        """Add a confirmation message about memory operations"""
        if (
            not body
            or "messages" not in body
            or not body["messages"]
            or not self.valves.show_status
        ):
            return

        # Prepare the confirmation message
        confirmation = ""

        if self._error_message:
            confirmation = f"(Memory error: {self._error_message})"
        elif self.stored_memories:
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for memory in self.stored_memories:
                if memory["operation"] == "NEW":
                    new_count += 1
                elif memory["operation"] == "UPDATE":
                    update_count += 1
                elif memory["operation"] == "DELETE":
                    delete_count += 1

            # Build the confirmation message in new styled format
            total_saved = new_count + update_count + delete_count

            # Use bold italic styling with an emoji as requested
            confirmation = f"**_Memory: 🧠 Saved {total_saved} memories..._**"

        # If no confirmation necessary, exit early
        if not confirmation:
            logger.debug("No memory confirmation message needed")
            return

        # Critical fix: Make a complete deep copy of the messages array
        try:
            logger.debug("Making deep copy of messages array for safe modification")
            messages_copy = copy.deepcopy(body["messages"])

            # Find the last assistant message
            last_assistant_idx = -1
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            # If found, modify the copy
            if last_assistant_idx != -1:
                # Get the original content
                original_content = messages_copy[last_assistant_idx].get("content", "")

                # Append the confirmation message
                messages_copy[last_assistant_idx]["content"] = (
                    original_content + f" {confirmation}"
                )

                # Replace the entire messages array in body
                logger.debug(
                    f"Replacing messages array with modified copy containing confirmation: {confirmation}"
                )
                body["messages"] = messages_copy
            else:
                logger.debug("No assistant message found to append confirmation")

        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")
            # Don't modify anything if there's an error

    # Cleanup method for aiohttp session and background tasks
    async def cleanup(self):
        """Clean up resources when filter is being shut down"""
        logger.info("Cleaning up Friday Short Term Memory systems")

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected when cancelling
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task: {e}")

        # Clear task tracking set
        self._background_tasks.clear()

        # Close any open sessions
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()

        # Clear memory caches to help with GC
        self._memory_embeddings = {}
        self._relevance_cache = {}
        self._memory_to_cache_keys = {}

        logger.info("Friday Short Term Memory cleanup complete")

    def _convert_dict_to_memory_operations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a dictionary returned by the LLM into the expected list of memory operations.

        Handles cases where the LLM returns a dict containing a list (e.g., {"memories": [...]})
        or a flatter structure. Avoids adding unnecessary prefixes.
        """
        if not isinstance(data, dict) or not data:
            return []

        operations: List[Dict[str, Any]] = []
        seen_content = set()

        # --- Primary Handling: Check for a key containing a list of operations ---
        # Common keys LLMs might use: "memories", "memory_operations", "results", "operations"
        list_keys = ["memories", "memory_operations", "results", "operations"]
        processed_primary = False
        for key in list_keys:
            if key in data and isinstance(data[key], list):
                logger.info(
                    f"Found list of operations under key '{key}', processing directly."
                )
                for item in data[key]:
                    if isinstance(item, dict):
                        # Extract fields directly, provide defaults
                        op = item.get("operation", "NEW").upper()  # Default to NEW
                        content = item.get(
                            "content", item.get("memory", item.get("value"))
                        )  # Check common content keys
                        tags = item.get("tags", [])
                        memory_bank = item.get(
                            "memory_bank", self.valves.default_memory_bank
                        )

                        # Validate memory_bank
                        if memory_bank not in self.valves.allowed_memory_banks:
                            memory_bank = self.valves.default_memory_bank

                        # Basic validation
                        if op not in ["NEW", "UPDATE", "DELETE"]:
                            continue
                        if (
                            not content
                            or not isinstance(content, str)
                            or len(content) < 5
                        ):
                            continue  # Skip empty/short content
                        if not isinstance(tags, list):
                            tags = [str(tags)]  # Ensure tags is a list

                        # Add if content is unique
                        if content not in seen_content:
                            operations.append(
                                {
                                    "operation": op,
                                    "content": content,
                                    "tags": tags,
                                    "memory_bank": memory_bank,
                                }
                            )
                            seen_content.add(content)
                processed_primary = True
                break  # Stop after processing the first found list

        # --- Fallback Handling: If no primary list found, try simple key-value flattening ---
        if not processed_primary:
            logger.info(
                "No primary operations list found, attempting fallback key-value flattening."
            )
            # Helper maps for simple tag inference (less critical now)
            identity_keys = {"name", "username", "location", "city", "country", "age"}
            goal_keys = {"goal", "objective", "plan"}
            preference_keys = {
                "likes",
                "dislikes",
                "interests",
                "hobbies",
                "favorite",
                "preference",
            }
            relationship_keys = {"family", "friend", "brother", "sister"}
            ignore_keys = {"notes", "meta", "trivia"}

            # Bank inference based on key name
            work_keys = {
                "job",
                "profession",
                "career",
                "work",
                "office",
                "business",
                "project",
            }
            personal_keys = {
                "home",
                "family",
                "hobby",
                "personal",
                "like",
                "enjoy",
                "love",
                "hate",
                "friend",
            }

            for key, value in data.items():
                lowered_key = key.lower()
                if (
                    lowered_key in ignore_keys
                    or not isinstance(value, (str, int, float, bool))
                    or not str(value).strip()
                ):
                    continue

                content = str(value).strip()
                if len(content) > 5 and content not in seen_content:
                    # Simple tag inference
                    tag = "preference"  # Default tag
                    if lowered_key in identity_keys:
                        tag = "identity"
                    elif lowered_key in goal_keys:
                        tag = "goal"
                    elif lowered_key in relationship_keys:
                        tag = "relationship"

                    # Simple bank inference
                    memory_bank = self.valves.default_memory_bank
                    if lowered_key in work_keys:
                        memory_bank = "Work"
                    elif lowered_key in personal_keys:
                        memory_bank = "Personal"

                    # Format simply: "Key: Value" unless key is generic
                    generic_keys = {
                        "content",
                        "memory",
                        "text",
                        "value",
                        "result",
                        "data",
                    }
                    if key.lower() in generic_keys:
                        content_to_save = content  # Use content directly
                    else:
                        # Prepend the key for non-generic keys
                        content_to_save = (
                            f"{key.replace('_', ' ').capitalize()}: {content}"
                        )

                    operations.append(
                        {
                            "operation": "NEW",
                            "content": content_to_save,
                            "tags": [tag],
                            "memory_bank": memory_bank,
                        }
                    )
                    seen_content.add(content)

        logger.info(f"Converted dict response into {len(operations)} memory operations")
        return operations

    # ------------------------------------------------------------------
    # Helper: background task initialisation (called once from inlet())
    # ------------------------------------------------------------------
    def _initialize_background_tasks(self) -> None:
        """(Idempotent) Ensure any background tasks that rely on the event
        loop are started the first time `inlet` is executed.

        Earlier versions attempted to call this but the helper did not
        exist, causing an `AttributeError`.  The current implementation is
        intentionally lightweight because most tasks are already started
        inside `__init__` when the filter is instantiated by OpenWebUI.
        The function therefore acts as a safety-net and can be extended in
        future if additional runtime-initialised tasks are required.
        """
        # Nothing to do for now because __init__ has already created the
        # background tasks.  Guard against multiple invocations.
        if getattr(self, "_background_tasks_started", False):
            return

        # Placeholder for potential future dynamic tasks
        logger.debug("_initialize_background_tasks called – no dynamic tasks to start.")
        self._background_tasks_started = True

    # ------------------------------------------------------------------
    # Helper: Increment named error counter safely
    # ------------------------------------------------------------------
    def _increment_error_counter(self, counter_name: str) -> None:
        """Increment an error counter defined in `self.error_counters`.

        Args:
            counter_name: The key identifying the counter to increment.
        """
        try:
            if counter_name not in self.error_counters:
                # Lazily create unknown counters so callers don't crash
                self.error_counters[counter_name] = 0
            self.error_counters[counter_name] += 1
        except Exception as e:
            # Should never fail, but guard to avoid cascading errors
            logger.debug(f"_increment_error_counter failed for '{counter_name}': {e}")

    # ------------------------------------------------------------------
    # Helper: Check if error should be logged (deduplication)
    # ------------------------------------------------------------------
    def _should_log_error(self, error_message: str, function_name: str = "") -> bool:
        """Check if an error should be logged (not a duplicate within time window).
        
        Prevents error log spam by tracking recently logged errors.
        Same error from same function within the dedup window will be skipped.
        
        Args:
            error_message: The error message text
            function_name: The function where error occurred (for better dedup)
        
        Returns:
            True if error should be logged (not duplicate), False if should skip (duplicate)
        """
        try:
            # Create hash of error context
            error_context = f"{function_name}:{error_message[:100]}"
            error_hash = hashlib.md5(error_context.encode()).hexdigest()
            
            now = time.time()
            
            # Check if we've logged this error recently
            if error_hash in self._error_log_cache:
                last_logged = self._error_log_cache[error_hash]
                time_since_last = now - last_logged
                
                if time_since_last < self._error_dedup_window:
                    # This is a duplicate, don't log
                    return False
            
            # Not a duplicate or time window expired - update cache and allow logging
            self._error_log_cache[error_hash] = now
            return True
        except Exception as e:
            # On any error in dedup logic, default to allowing log
            logger.debug(f"Error in dedup check: {e}")
            return True
