"""
Test utilities for Friday's Memory System
"""

import os
import sys

def setup_test_paths():
    """Add the parent directory to Python path for test imports"""
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add parent directory (main Friday folder) to Python path
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
