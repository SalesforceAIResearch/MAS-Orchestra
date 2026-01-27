"""
Google Serper search implementation.

Requires SERPER_API_KEY environment variable.
Provides Google Search quality results with low-cost API access.
"""

import os


async def search_serper(search_query: str, fetch_webpage_content_fn=None) -> list:
    """
    Search using Google Serper API.
    
    Args:
        search_query: The search query string
        fetch_webpage_content_fn: Optional function to fetch full webpage content from URLs
        
    Returns:
        List of result dictionaries with keys: 'title', 'url', 'content', 'raw_content'
        
    Raises:
        Exception: If search fails, API key is missing, or dependencies are missing
    """
    try:
        from langchain_community.tools.google_serper.tool import GoogleSerperResults
        
        # Get API key from environment
        serper_api_key = os.environ.get("SERPER_API_KEY")
        if not serper_api_key:
            raise Exception("SERPER_API_KEY not found in environment. Get a key at: https://serper.dev")
        
        # Use the LangChain tool with ainvoke for async support
        search = GoogleSerperResults(serper_api_key=serper_api_key)
        
        raw_results = []
        
        # Use ainvoke for async execution
        try:
            # Get structured results using ainvoke
            results_list = await search.ainvoke(search_query)
            
            # Handle different response formats from Serper
            if isinstance(results_list, dict):
                # Serper typically returns organic results in a dict
                organic = results_list.get('organic', [])
                for result in organic[:5]:  # Limit to 5 results
                    raw_results.append({
                        'title': result.get('title', 'Untitled'),
                        'url': result.get('link', ''),
                        'content': result.get('snippet', ''),
                        'raw_content': ''  # Will be populated below if needed
                    })
            elif isinstance(results_list, list):
                for result in results_list[:5]:
                    raw_results.append({
                        'title': result.get('title', 'Untitled'),
                        'url': result.get('link', ''),
                        'content': result.get('snippet', ''),
                        'raw_content': ''
                    })
        except AttributeError:
            # Fallback: use ainvoke() which returns formatted text
            # In this case, we can't extract structured results, so raise an error
            # The caller will handle this by returning the formatted string directly
            search_result = await search.ainvoke(search_query)
            raise AttributeError(f"Serper fallback result: {search_result}")
        
        print(f"📄 GOT {len(raw_results)} RESULTS FROM SERPER")
        
        # Fetch raw content from URLs if requested (Serper doesn't provide it directly)
        if raw_results and fetch_webpage_content_fn:
            # Extract URLs from results
            urls = [result['url'] for result in raw_results if result['url']]
            
            if urls:
                # Fetch raw content using provided async function
                url_to_content = await fetch_webpage_content_fn(urls)
                
                # Update raw_results with fetched content
                for result in raw_results:
                    url = result['url']
                    if url in url_to_content:
                        result['raw_content'] = url_to_content[url]
        
        return raw_results
        
    except AttributeError as e:
        # Re-raise AttributeError for fallback handling in caller
        raise e
    except Exception as e:
        raise Exception(f"Search error: {e}. Try installing: pip install langchain-community")

