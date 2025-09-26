#!/usr/bin/env python3
"""
Setup script for Persistent AI Memory System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="persistent-ai-memory",
    version="1.0.0",
    author="Collaborative AI Development Team",
    author_email="",
    description="A comprehensive, real-time memory system for AI assistants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/savantskie/persistent-ai-memory",
    packages=find_packages(),
    py_modules=["ai_memory_core", "mcp_server"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "watchdog>=2.1.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pams-server=persistent_ai_memory.mcp_server:main",
        ],
    },
)
