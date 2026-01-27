"""
DuckDuckGo search implementation.

Free search backend with no API key required.
Fetches full webpage content when summarize_webpages is enabled.
"""



async def search_duckduckgo(search_query: str, fetch_webpage_content_fn=None) -> list:
    """
    Search using DuckDuckGo API with retry logic and exponential backoff.
    
    Args:
        search_query: The search query string
        fetch_webpage_content_fn: Optional function to fetch full webpage content from URLs
        
    Returns:
        List of result dictionaries with keys: 'title', 'url', 'content', 'raw_content'
        
    Raises:
        Exception: If search fails after maximum retries or dependencies are missing
    """
    trial = 0
    while True:
        print(f"\n--- DuckDuckGo Search Trial {trial + 1} ---")
        try:
            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
            import asyncio
            import time
            
            # Use the LangChain tool with async support via asyncio.to_thread
            search = DuckDuckGoSearchAPIWrapper(max_results=5)
            results_list = await asyncio.to_thread(search.results, search_query, max_results=5)
            
            # DuckDuckGo returns list of dicts with 'title', 'link', 'snippet'
            raw_results = []
            for result in results_list:
                raw_results.append({
                    'title': result.get('title', 'Untitled'),
                    'url': result.get('link', ''),
                    'content': result.get('snippet', ''),
                    'raw_content': ''  # Will be populated below if needed
                })
            
            print(f"📄 GOT {len(raw_results)} RESULTS FROM DUCKDUCKGO")
            
            # Fetch raw content from URLs if requested (DuckDuckGo doesn't provide it directly)
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
            
        except Exception as e:
            exception_backoff = 2**trial  # exponential backoff
            print(f"\n✗ DuckDuckGo Search Exception (Trial {trial + 1})")
            print(f"  - Error type: {type(e).__name__}")
            print(f"  - Error message: {e}")
            print(f"  - Error args: {e.args}")
            print(f"  - Backoff time: {exception_backoff} seconds")
            print(f"  - Search query: {search_query}")
            
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower() or "429" in str(e):
                print(f"  - Detected rate limit error")
            elif "timeout" in str(e).lower():
                print(f"  - Detected timeout error")
            elif "connection" in str(e).lower():
                print(f"  - Detected connection error")
            elif "network" in str(e).lower():
                print(f"  - Detected network error")
            else:
                print(f"  - Unknown error type")
            
            print(f"  - Waiting {exception_backoff} seconds before retry...")
            await asyncio.sleep(exception_backoff)
            trial += 1
            
            if trial == 5:  # Max retries reached
                print(f"\n✗ Max trials reached (5)")
                print(f"  - Final error type: {type(e).__name__}")
                print(f"  - Final error message: {e}")
                print(f"  - Returning empty results list")
                return []

