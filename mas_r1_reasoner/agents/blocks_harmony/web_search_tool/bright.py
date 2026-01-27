"""
BrightData search implementation using official LangChain integration.

Requires BrightData API key.
Provides high-quality web scraping and search capabilities via LangChain.
"""

import os
from typing import Optional


# TODO: this one needs to pare with its own browser
# TODO: may need backoff as well


async def search_brightdata(
    search_query: str, 
    fetch_webpage_content_fn=None,
    api_key: Optional[str] = None,
    zone: Optional[str] = None  # BrightData zone (e.g., 'serp', 'sfrrag_agent_phi')
) -> list:
    """
    Search using BrightData SERP API via LangChain.
    
    Args:
        search_query: The search query string
        fetch_webpage_content_fn: Optional function to fetch full webpage content from URLs
        api_key: BrightData API key
        zone: BrightData zone name (e.g., 'sfrrag_agent_phi'). Defaults to 'serp'
        
    Returns:
        List of result dictionaries with keys: 'title', 'url', 'content', 'raw_content'
        
    Raises:
        Exception: If search fails, credentials are missing, or API request fails
    """
    try:
        from langchain_brightdata._utilities import BrightDataSERPAPIWrapper
        
        # Get API key and zone from parameters or environment
        api_key = api_key or os.environ.get("BRIGHTDATA_API_KEY")
        zone = zone or os.environ.get("BRIGHTDATA_ZONE", "sfrrag_agent_phi")  # Default to 'serp'
        
        if not api_key:
            raise Exception(
                "BrightData API key not found. Provide api_key parameter, "
                "or set BRIGHTDATA_API_KEY environment variable."
            )
        
        print(f"🔍 Searching BrightData with query: '{search_query}' (zone: {zone})")
        
        # Use the API wrapper directly to access zone parameter
        api_wrapper = BrightDataSERPAPIWrapper(bright_data_api_key=api_key)
        
        # Execute the search with zone parameter using asyncio.to_thread for async compatibility
        #TODO: according to https://python.langchain.com/docs/integrations/tools/brightdata_serp/, naive async is not supported
        import asyncio
        results_data = await asyncio.to_thread(
            api_wrapper.get_search_results,
            query=search_query,
            zone=zone,  # Pass the zone parameter
            search_engine="google",
            country="us",
            language="en",
            results_count=5,
            parse_results=True,  # Get structured JSON results
        )
        
        # Parse the results into our standard format
        raw_results = []
        
        # BrightData with parse_results=True returns JSON string
        # We need to parse it first
        import json
        
        if isinstance(results_data, str):
            try:
                # Parse the JSON string
                results_data = json.loads(results_data)
                print(f"   ✓ Parsed JSON response from BrightData")
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Failed to parse JSON from BrightData: {e}")
                raise Exception(f"BrightData returned invalid JSON: {results_data[:200]}...")
        
        # Now results_data should be a dict
        if isinstance(results_data, dict):
            # Extract organic search results from the JSON structure
            organic_results = results_data.get('organic', results_data.get('results', []))
            
            if not organic_results:
                print(f"   ⚠️  No 'organic' or 'results' field in response. Keys: {list(results_data.keys())}")
            
            for result in organic_results[:5]:  # Limit to 5 results
                raw_results.append({
                    'title': result.get('title', 'Untitled'),
                    'url': result.get('link', result.get('url', '')),
                    'content': result.get('description', result.get('snippet', '')),
                    'raw_content': result.get('description', result.get('snippet', ''))
                })
        else:
            raise Exception(f"Unexpected type for results_data: {type(results_data)}")
        
        print(f"📄 GOT {len(raw_results)} RESULTS FROM BRIGHTDATA")
        
        # Fetch raw content from URLs if requested (BrightData snippets are limited)
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
        
    except ImportError as e:
        raise Exception(f"BrightData search error: {e}. Try installing: pip install langchain-brightdata")
    except Exception as e:
        raise Exception(f"BrightData search error: {e}")

