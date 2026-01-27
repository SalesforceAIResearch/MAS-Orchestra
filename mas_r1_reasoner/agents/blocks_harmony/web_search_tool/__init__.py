"""
Web search tool implementations.

This package contains modular implementations for different search backends:
- DuckDuckGo: Free, no API key required
- Tavily: Requires TAVILY_API_KEY, optimized for AI agents
- Serper: Requires SERPER_API_KEY, Google Search quality
- BrightData: Requires BrightData credentials, high-quality web scraping

Each backend provides a consistent interface for fetching and processing search results.
"""

from .duckduckgo import search_duckduckgo
from .tavily import search_tavily
from .serper import search_serper
from .bright import search_brightdata

__all__ = [
    'search_duckduckgo',
    'search_tavily', 
    'search_serper',
    'search_brightdata',
]

