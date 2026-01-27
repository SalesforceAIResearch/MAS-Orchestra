"""
Tavily search implementation.

Requires TAVILY_API_KEY environment variable.
Optimized for AI agents with built-in full webpage content support.
"""


async def search_tavily(search_query: str, fetch_webpage_content_fn=None) -> list:
    """
    Search using Tavily API.
    
    Args:
        search_query: The search query string
        fetch_webpage_content_fn: Not used (Tavily provides raw content directly)
        
    Returns:
        List of result dictionaries with keys: 'title', 'url', 'content', 'raw_content'
        
    Raises:
        Exception: If search fails or dependencies are missing
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        
        # Fetch results (TavilySearchResults returns list of dicts)
        search = TavilySearchResults(max_results=5, include_raw_content=True)
        tavily_results = await search.ainvoke(search_query)
        
        # Normalize Tavily results to common format
        raw_results = []
        for result in tavily_results:
            raw_results.append({
                'title': result.get('title', 'Untitled'),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'raw_content': result.get('raw_content', '')  # Full webpage content
            })
        
        print(f"📄 GOT {len(raw_results)} RESULTS FROM TAVILY")
        
        return raw_results
        
    except Exception as e:
        raise Exception(f"Search error: {e}. Try installing: pip install tavily-python")

