"""
BM25 Retriever for Offline Web Search

Following BrowseComp-Plus BM25Searcher implementation.
Uses Pyserini for lexical search with Lucene indexes.
"""

import os
import json
from typing import Dict, List, Any


def initialize_bm25_retriever(index_path: str) -> Dict[str, Any]:
    """Initialize BM25 retriever using Pyserini.
    
    Args:
        index_path: Path to the directory containing indexes
        
    Returns:
        Dictionary containing retriever metadata and searcher instance
        
    Raises:
        ImportError: If pyserini is not installed
        FileNotFoundError: If BM25 index not found
        RuntimeError: If initialization fails
    """
    try:
        from pyserini.search.lucene import LuceneSearcher
        
        bm25_path = os.path.join(index_path, "bm25")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        
        searcher = LuceneSearcher(bm25_path)
        print(f"   ✓ BM25 loaded: {searcher.num_docs} documents")
        
        return {"type": "bm25", "searcher": searcher}
        
    except ImportError as e:
        raise ImportError(
            f"Pyserini not installed: {e}. "
            "Install with: pip install pyserini"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BM25: {e}")


async def search_bm25(
    retriever: Dict[str, Any],
    query: str,
    corpus: Dict[str, Any],
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """Perform BM25 search following BrowseComp-Plus pattern.
    
    Args:
        retriever: BM25 retriever dictionary from initialize_bm25_retriever
        query: Search query string
        corpus: Dictionary mapping docid to document data
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with docid, title, content, url, and score
    """
    import asyncio
    
    searcher = retriever["searcher"]
    hits = await asyncio.to_thread(searcher.search, query, k=max_results)
    
    # Get full documents from corpus
    results = []
    for hit in hits:
        doc_id = hit.docid
        if doc_id in corpus:
            doc = corpus[doc_id]
            results.append({
                'docid': doc_id,
                'title': doc.get('title', 'Untitled'),
                'content': doc.get('text', doc.get('contents', '')),
                'url': doc.get('url', ''),
                'score': hit.score
            })
    
    return results

