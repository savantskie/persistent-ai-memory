"""Utility functions for the persistent AI memory system."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

def get_local_timezone() -> ZoneInfo:
    """Get local timezone based on system settings"""
    try:
        import time
        return ZoneInfo(time.tzname[0])
    except:
        # Fallback to a common timezone if detection fails
        return ZoneInfo("America/Chicago")  # Central Time fallback

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
