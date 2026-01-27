"""
Web Search Initialization Module

This module provides initialization functions for web search resources (online and offline).
The initialization is separated from agent_system.py for better code organization and readability.
"""

from .online_search import initialize_online_resources
from .offline_search import initialize_offline_resources

__all__ = [
    "initialize_online_resources",
    "initialize_offline_resources",
]

