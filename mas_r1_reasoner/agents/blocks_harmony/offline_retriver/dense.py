"""
Dense Retriever for Offline Web Search

Following BrowseComp-Plus FaissSearcher implementation EXACTLY.
Uses Tevatron's DenseModel and FaissFlatSearcher for dense retrieval.

Note: The embedding model initialization has been moved to 
mas_r1_reasoner.agents.blocks_harmony.initialization.offline_search.py
to ensure it only initializes once.
"""

#TODO: Let's do more testing on this...

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def search_dense(
    retriever: Dict[str, Any],
    query: str,
    corpus: Dict[str, Any],
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """Perform dense retrieval using Tevatron (following BrowseComp-Plus FaissSearcher.search EXACTLY).
    
    Data flow:
    1. Encode query using Tevatron's DenseModel
    2. FAISS search returns top-k docids (document identifiers) with scores
    3. Map docids → document text using the 'corpus' parameter (loaded once in AgentSystem)
    4. Return formatted results with title, content, url, and score
    
    Args:
        retriever: Dense retriever dictionary from centralized initialization
        query: Search query string
        corpus: Dictionary mapping docid to document data (loaded in AgentSystem from corpus.jsonl)
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with docid, title, content, url, and score
    """
    import asyncio
    import torch
    
    def _search_dense_sync():
        faiss_searcher = retriever["retriever"]  # Tevatron's FaissFlatSearcher
        lookup = retriever["lookup"]
        model = retriever["model"]  # Tevatron's DenseModel
        tokenizer = retriever["tokenizer"]
        device = retriever["device"]
        task_prefix = retriever["task_prefix"]
        max_length = retriever["max_length"]
        
        if not all([faiss_searcher, model, tokenizer, lookup]):
            raise RuntimeError("Searcher not properly initialized")
        
        # Encode query EXACTLY as BrowseComp-Plus does
        batch_dict = tokenizer(
            task_prefix + query,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        # Use Tevatron's DenseModel.encode_query() method (EXACTLY as BrowseComp-Plus)
        with torch.amp.autocast(device):
            with torch.no_grad():
                q_reps = model.encode_query(batch_dict)
                q_reps = q_reps.cpu().detach().numpy()
        
        # Search using Tevatron's FaissFlatSearcher (EXACTLY as BrowseComp-Plus)
        all_scores, psg_indices = faiss_searcher.search(q_reps, max_results)
        
        # Format results using corpus (passed from AgentSystem)
        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            if index < 0 or index >= len(lookup):
                continue
            
            docid = lookup[index]
            
            # Get document from corpus (already loaded in AgentSystem)
            if docid not in corpus:
                logger.warning(f"Document {docid} not found in corpus")
                continue
            
            doc = corpus[docid]
            text = doc.get('text', doc.get('contents', ''))
            title = doc.get('title', 'Untitled')
            url = doc.get('url', '')
            
            results.append({
                'docid': docid,
                'title': title,
                'content': text,
                'url': url,
                'score': float(score)
            })
        
        return results
    
    return await asyncio.to_thread(_search_dense_sync)
