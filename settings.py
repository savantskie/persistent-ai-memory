#!/usr/bin/env python3
"""
Settings configuration for Persistent AI Memory System

This module provides centralized configuration management using pydantic BaseSettings
with environment variable support. Perfect for the desktop app to configure everything.
"""

from pathlib import Path
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field
import os


class MemorySettings(BaseSettings):
    """
    Centralized configuration for AI Memory System
    
    All settings can be overridden via environment variables prefixed with AI_MEMORY_
    For example: AI_MEMORY_DATA_DIR=/custom/path
    """
    
    # --- Core Paths ---
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".ai_memory",
        env="AI_MEMORY_DATA_DIR",
        description="Base directory for all memory data and databases"
    )
    
    # --- Embedding Configuration ---
    embedding_provider: str = Field(
        default="lm_studio",
        env="AI_MEMORY_EMBED_PROVIDER",
        description="Embedding provider: lm_studio, ollama, or openai"
    )
    
    embedding_model: str = Field(
        default="nomic-embed-text",
        env="AI_MEMORY_EMBED_MODEL",
        description="Model name for embeddings"
    )
    
    embedding_url: str = Field(
        default="http://localhost:1234/v1/embeddings",
        env="AI_MEMORY_EMBED_URL",
        description="URL for embedding service"
    )
    
    openai_api_key: str = Field(
        default="",
        env="OPENAI_API_KEY",
        description="OpenAI API key (if using OpenAI embeddings)"
    )
    
    # --- Weather Configuration ---
    weather_latitude: float = Field(
        default=46.34,
        env="AI_MEMORY_WEATHER_LAT",
        description="Latitude for weather queries"
    )
    
    weather_longitude: float = Field(
        default=-94.63,
        env="AI_MEMORY_WEATHER_LON", 
        description="Longitude for weather queries"
    )
    
    weather_timezone: str = Field(
        default="America/Chicago",
        env="AI_MEMORY_WEATHER_TZ",
        description="Timezone for weather queries"
    )
    
    # --- MCP Server Configuration ---
    mcp_port: int = Field(
        default=1234,
        env="AI_MEMORY_MCP_PORT",
        description="Port for MCP server"
    )
    
    # --- File Monitoring Configuration ---
    enable_file_monitoring: bool = Field(
        default=True,
        env="AI_MEMORY_ENABLE_MONITORING",
        description="Enable automatic file monitoring"
    )
    
    file_monitoring_auto_start: bool = Field(
        default=False,  # GitHub version default - safer for public use
        env="AI_MEMORY_AUTO_START_MONITORING",
        description="Auto-start file monitoring after 3 minutes"
    )
    
    watch_directories: Optional[List[str]] = Field(
        default=None,
        env="AI_MEMORY_WATCH_DIRS",
        description="Custom directories to watch (comma-separated)"
    )
    
    # --- Database Retention Configuration ---
    conversation_retention_days: int = Field(
        default=90,
        env="AI_MEMORY_CONV_RETENTION_DAYS",
        description="Days to keep conversation records"
    )
    
    memory_retention_days: int = Field(
        default=365,
        env="AI_MEMORY_MEMORY_RETENTION_DAYS", 
        description="Days to keep AI memories"
    )
    
    # --- Search Configuration ---
    search_similarity_threshold: float = Field(
        default=0.3,
        env="AI_MEMORY_SIMILARITY_THRESHOLD",
        description="Minimum similarity score for semantic search"
    )
    
    # --- Logging Configuration ---
    log_level: str = Field(
        default="INFO",
        env="AI_MEMORY_LOG_LEVEL",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def conversations_db_path(self) -> Path:
        """Path to conversations database"""
        return self.data_dir / "conversations.db"
    
    @property 
    def ai_memories_db_path(self) -> Path:
        """Path to AI memories database"""
        return self.data_dir / "ai_memories.db"
    
    @property
    def schedule_db_path(self) -> Path:
        """Path to schedule database"""
        return self.data_dir / "schedule.db"
    
    @property
    def vscode_db_path(self) -> Path:
        """Path to VS Code database"""
        return self.data_dir / "vscode_project.db"
    
    @property
    def mcp_db_path(self) -> Path:
        """Path to MCP tool calls database"""
        return self.data_dir / "mcp_tool_calls.db"


# Global settings instance
settings = MemorySettings()


def get_settings() -> MemorySettings:
    """Get the current settings instance"""
    return settings


def reload_settings() -> MemorySettings:
    """Reload settings from environment/config files"""
    global settings
    settings = MemorySettings()
    return settings


def update_settings(**kwargs) -> MemorySettings:
    """Update settings programmatically (for desktop app use)"""
    global settings
    # Create new instance with updated values
    current_dict = settings.dict()
    current_dict.update(kwargs)
    settings = MemorySettings(**current_dict)
    return settings


if __name__ == "__main__":
    # Print current configuration for debugging
    print("🔧 AI Memory System Configuration")
    print("=" * 40)
    print(f"Data Directory: {settings.data_dir}")
    print(f"Embedding Provider: {settings.embedding_provider}")
    print(f"Embedding URL: {settings.embedding_url}")
    print(f"Weather Location: {settings.weather_latitude}, {settings.weather_longitude}")
    print(f"MCP Port: {settings.mcp_port}")
    print(f"File Monitoring: {settings.enable_file_monitoring}")
    print(f"Auto-start Monitoring: {settings.file_monitoring_auto_start}")
    print(f"Conversation Retention: {settings.conversation_retention_days} days")
    print(f"Memory Retention: {settings.memory_retention_days} days")
    print(f"Log Level: {settings.log_level}")