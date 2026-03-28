"""Utility functions for the AI Memory system."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Union
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def get_local_timezone() -> ZoneInfo:
    """Get local timezone based on system settings"""
    try:
        import time
        return ZoneInfo(time.tzname[0])
    except:
        # Fallback to a common timezone if detection fails
        return ZoneInfo("America/Chicago")  # Minnesota is in Central Time

def parse_timestamp(timestamp: Union[str, int, float, None], fallback: Optional[datetime] = None) -> str:
    """
    Parse a timestamp into a consistent ISO 8601 local timezone string.

    Args:
        timestamp (Union[str, int, float, None]): The input timestamp to parse.
            - ISO 8601 string (e.g., "2025-08-04T18:30:29Z")
            - Unix timestamp in seconds or milliseconds (e.g., 1628100000 or 1628100000000)
        fallback (Optional[datetime]): A fallback datetime if parsing fails.

    Returns:
        str: The parsed timestamp as an ISO 8601 string in local timezone.
    """
    if timestamp is None:
        # Use fallback or current local time if no timestamp is provided
        fallback_time = fallback or datetime.now(get_local_timezone())
        return fallback_time.isoformat()

    try:
        # Handle ISO 8601 strings
        if isinstance(timestamp, str):
            # Adjust for common quirks (e.g., "Z" for UTC)
            if "Z" in timestamp:
                timestamp = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(timestamp).astimezone(get_local_timezone()).isoformat()

        # Handle Unix timestamps
        if isinstance(timestamp, (int, float)):
            # Automatically handle milliseconds vs. seconds
            if timestamp > 10**10:  # Likely milliseconds
                timestamp /= 1000
            return datetime.fromtimestamp(timestamp, get_local_timezone()).isoformat()

    except Exception as e:
        # Log the error and use fallback
        logger.warning(f"Failed to parse timestamp '{timestamp}': {e}")
        fallback_time = fallback or datetime.now(get_local_timezone())
        return fallback_time.isoformat()
    
    # If all parsing attempts fail, use fallback
    fallback_time = fallback or datetime.now(get_local_timezone())
    return fallback_time.isoformat()


# ==============================================================================
# Centralized Path Management
# ==============================================================================

def get_memory_data_dir() -> str:
    """
    Get the memory data directory path.
    
    Uses AI_MEMORY_DATA_DIR environment variable if set, otherwise returns ./memory_data/
    relative to the script's location.
    
    Returns:
        Path to memory_data directory
    """
    if env_path := os.getenv("AI_MEMORY_DATA_DIR"):
        return env_path
    
    # Default to ./memory_data/ relative to script location
    script_dir = Path(__file__).parent
    return str(script_dir / "memory_data")


def get_log_dir() -> str:
    """
    Get the logs directory path.
    
    Uses AI_MEMORY_LOG_DIR environment variable if set, otherwise returns ./logs/
    relative to the script's location.
    
    Returns:
        Path to logs directory
    """
    if env_path := os.getenv("AI_MEMORY_LOG_DIR"):
        return env_path
    
    # Default to ./logs/ relative to script location
    script_dir = Path(__file__).parent
    return str(script_dir / "logs")


def get_weather_dir() -> str:
    """
    Get the weather cache directory path.
    
    Uses AI_MEMORY_WEATHER_DIR environment variable if set, otherwise returns ./weather_directory/
    relative to the script's location.
    
    Returns:
        Path to weather directory
    """
    if env_path := os.getenv("AI_MEMORY_WEATHER_DIR"):
        return env_path
    
    # Default to ./weather_directory/ relative to script location
    script_dir = Path(__file__).parent
    return str(script_dir / "weather_directory")


def get_weather_cache_dir() -> str:
    """
    Get the weather cache subdirectory path (weather/ under weather_directory/).
    
    Returns:
        Path to weather cache directory
    """
    weather_dir = get_weather_dir()
    return os.path.join(weather_dir, "weather")


def ensure_directories() -> None:
    """
    Create all essential directories for the AI Memory system if they don't exist.
    
    Creates:
    - memory_data/
    - memory_data/archives/
    - memory_data/backups/
    - logs/
    - weather_directory/weather/
    """
    directories = [
        get_memory_data_dir(),
        os.path.join(get_memory_data_dir(), "archives"),
        os.path.join(get_memory_data_dir(), "backups"),
        get_log_dir(),
        get_weather_cache_dir(),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
