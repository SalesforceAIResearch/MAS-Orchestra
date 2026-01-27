"""
Offline Retrievers for Web Search

This package provides retrieval implementations following BrowseComp-Plus patterns:
- BM25: Lexical search using Pyserini
- Dense: Semantic search using FAISS + embeddings
"""

from .bm25 import initialize_bm25_retriever, search_bm25
from .dense import search_dense

__all__ = [
    'initialize_bm25_retriever',
    'search_bm25',
    'search_dense',
]

