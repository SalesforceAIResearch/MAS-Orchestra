"""
Online Web Search Agent using LangChain and Live Web Search APIs

This agent performs web research using live search APIs (DuckDuckGo, Tavily, Serper, BrightData)
to gather and synthesize information. It uses LangChain for tool orchestration and includes
LLM-based summarization of retrieved web pages.

Performance Optimizations:
- Chat model initialization moved to AgentSystem (initialized once per Ray worker)
- Model is reused across all search queries, significantly reducing startup overhead
"""

import inspect

#TODO: Not using LLMAgentBase, but langchain. Is there a problem?
#TODO: "mock" setting is not supported
#TODO: THis is open_deep_research way, but one can also do it in Asearcher way (chunk, memory...)... 
#TODO: Although the MAS-R1's focus on meta-level design, not specific sub-agent design




async def WebSearchAgent(self, agent_input, model: str) -> str:
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool
    from langchain_together import ChatTogether
    from datetime import datetime
    from typing import Any, List, Dict, Optional
    import os

    max_iterations = 5
    # Available search tools:
    # - "duckduckgo": Free, no API key needed. Fetches full content when summarize_webpages=True
    # - "tavily": Requires TAVILY_API_KEY. Built-in full content support (best quality)
    # - "google-serper" or "serper": Requires SERPER_API_KEY. Google Search quality, fetches full content when summarize_webpages=True
    # - "brightdata" or "bright": Requires BRIGHTDATA_API_KEY and optionally BRIGHTDATA_ZONE. High-quality SERP via LangChain BrightDataSERP
    search_tool = "duckduckgo"
    summarize_webpages = True  # If True, fetches and summarizes full webpage content (requires beautifulsoup4, lxml)
    max_chars = 35000  # Character limit to stay within model token limits (matching open_deep_research)

    # Define Summary structure at module level (matching open_deep_research/state.py)
    class Summary(BaseModel):
        """Research summary with key findings."""
        summary: str = Field(description="Main summary of the webpage content")
        key_excerpts: str = Field(description="Important quotes or excerpts from the content")


    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"
    
    # Extract the query from agent_input
    query = agent_input.content

    # Store these in a container so the inner function can access them
    # (will be set after chat_model is initialized)
    context = {
        "chat_model": None,
        "summarize_webpages": summarize_webpages
    }
    

    # Webpage summarization prompt (from open_deep_research)
    summarize_webpage_prompt = """You are tasked with summarizing the raw content of a webpage retrieved from a web search. Your goal is to create a summary that preserves the most important information from the original web page. This summary will be used by a downstream research agent, so it's crucial to maintain the key details without losing essential information.

    Here is the raw content of the webpage:

    <webpage_content>
    {webpage_content}
    </webpage_content>

    Please follow these guidelines to create your summary:

    1. Identify and preserve the main topic or purpose of the webpage.
    2. Retain key facts, statistics, and data points that are central to the content's message.
    3. Keep important quotes from credible sources or experts.
    4. Maintain the chronological order of events if the content is time-sensitive or historical.
    5. Preserve any lists or step-by-step instructions if present.
    6. Include relevant dates, names, and locations that are crucial to understanding the content.
    7. Summarize lengthy explanations while keeping the core message intact.

    When handling different types of content:

    - For news articles: Focus on the who, what, when, where, why, and how.
    - For scientific content: Preserve methodology, results, and conclusions.
    - For opinion pieces: Maintain the main arguments and supporting points.
    - For product pages: Keep key features, specifications, and unique selling points.

    Your summary should be significantly shorter than the original content but comprehensive enough to stand alone as a source of information. Aim for about 25-30 percent of the original length, unless the content is already concise.

    Present your summary in the following XML format:

    <summary> "Your summary here, structured with appropriate paragraphs or bullet points as needed" </summary>
    <key_excerpts> "First important quote or excerpt, Second important quote or excerpt, Third important quote or excerpt, ...Add more excerpts as needed, up to a maximum of 5 </key_excerpts>

    Today's date is {date}."""

    # Research system prompt (adapted from open_deep_research)
    research_system_prompt = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

        <Task>
        Your job is to use tools to gather information about the user's input topic.
        You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
        </Task>

        <Available Tools>
        You have access to two main tools:
        1. **web_search**: For conducting web searches to gather information
        2. **think_tool**: For reflection and strategic planning during research

        **CRITICAL: Use think_tool after each search to reflect on results and plan next steps. Do not call think_tool with the web_search or any other tools. It should be to reflect on the results of the search.**
        </Available Tools>

        <Instructions>
        Think like a human researcher with limited time. Follow these steps:

        1. **Read the question carefully** - What specific information does the user need?
        2. **Start with broader searches** - Use broad, comprehensive queries first
        3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
        4. **Execute narrower searches as you gather information** - Fill in the gaps
        5. **Stop when you can answer confidently** - Don't keep searching for perfection
        </Instructions>

        <Hard Limits>
        **Tool Call Budgets** (Prevent excessive searching):
        - **Simple queries**: Use 2-3 search tool calls maximum
        - **Complex queries**: Use up to 5 search tool calls maximum
        - **Always stop**: After 5 search tool calls if you cannot find the right sources

        **Stop Immediately When**:
        - You can answer the user's question comprehensively
        - You have 3+ relevant examples/sources for the question
        - Your last 2 searches returned similar information
        </Hard Limits>

        <Show Your Thinking>
        After each search tool call, use think_tool to analyze the results:
        - What key information did I find?
        - What's missing?
        - Do I have enough to answer the question comprehensively?
        - Should I search more or provide my answer?
        </Show Your Thinking>
    """


    async def _fetch_webpage_content(urls: list) -> dict:
        """Fetch raw content from URLs using BrightDataUnlocker (generic web scraper).
        
        Args:
            urls: List of URLs to fetch content from
            
        Returns:
            Dictionary mapping URL to raw text content
        """
        try:
            from langchain_brightdata import BrightDataUnlocker
            import os
            import asyncio
            
            if not urls:
                raise ValueError(f"   ⚠️  No URLs provided. Skipping webpage content fetching.")
            
            # Get BrightData API key and zone from environment
            api_key = os.environ.get("BRIGHTDATA_MY_API_KEY")
            zone = os.environ.get("BRIGHTDATA_UNLOCK_ZONE", "sfrrag_agent_phi")  # Use same zone as SERP
            
            if not api_key:
                raise ValueError(f"   ⚠️  BRIGHTDATA_API_KEY not found. Skipping webpage content fetching.")
            
            print(f"   🌐 Fetching raw content from {len(urls)} URLs using BrightData (zone: {zone})...")
            
            # Initialize BrightData Web Unlocker (generic scraper)
            # Use HTML format - will be summarized by LLM later
            unlocker = BrightDataUnlocker(
                bright_data_api_key=api_key,
                zone=zone,  # Use the zone from environment (e.g., "sfrrag_agent_phi")
                data_format="html"  # Returns HTML content (will be summarized by LLM)
            )
            
            # Map content back to results by URL - fetch in parallel
            async def _fetch_single_url(url):
                try:
                    # Scrape the webpage - no dataset_type needed!
                    result = await asyncio.to_thread(unlocker.run, {"url": url})
                    if result:
                        print(f"   ✓ Fetched {result[:50]} {len(result)} chars from {url[:50]}...")
                        return url, result
                except Exception as e:
                    print(f"   ⚠️  Failed to fetch {url[:50]}...: {e}")
                return url, None
            
            # Fetch all URLs in parallel
            tasks = [_fetch_single_url(url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            # Convert results to dictionary
            url_to_content = {url: content for url, content in results if content is not None}
            
            print(f"   ✓ Successfully fetched raw content for {len(url_to_content)} pages")
            return url_to_content
        
        except ImportError as e:
            print(f"   ⚠️  Could not load BrightData Unlocker: {e}")
            return {}
        except Exception as e:
            print(f"   ⚠️  Error fetching raw content: {e}")
            return {}


    async def _summarize_webpage_content(structured_model, base_model, webpage_content: str) -> str:
        """Summarize webpage content using LLM (following open_deep_research pattern).
        
        Args:
            structured_model: LLM model configured with .with_structured_output(Summary) and .with_retry()
            base_model: Base LLM model without structured output (for fallback)
            webpage_content: Raw webpage content (already truncated at call site)
            
        Returns:
            Formatted summary with <summary> and <key_excerpts> tags, or original if fails
        """

        if len(webpage_content) <= 5000: # if small, we don't need to summarize
            print(f"   ✂️  Webpage content is already short enough: {len(webpage_content)} chars")
            return webpage_content

        try:
            import asyncio
            
            # Format prompt with current date (matching open_deep_research)
            today = datetime.now().strftime("%Y-%m-%d")
            prompt_content = summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=today
            )
            try:
                # Invoke the structured model (matching open_deep_research pattern)
                summary = await structured_model.ainvoke(
                    [HumanMessage(content=prompt_content)]
                )
                
                # Format the summary with structured sections (matching open_deep_research)
                formatted_summary = (
                    f"<summary>\n{summary.summary}\n</summary>\n\n"
                    f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
                )
                print(f"   ✂️  Summarized {len(webpage_content)} chars → {len(summary.summary)} chars")
                return formatted_summary

            except Exception as e: # sometimes the structured output fails, so we try regular completion
                # Fallback: If structured output fails, try regular completion with base model
                print(f"   ⚠️  Structured output failed: {e}, trying regular completion")
                response = await base_model.ainvoke([HumanMessage(content=prompt_content)])
                print(f"   ✂️  Summarized {len(webpage_content)} chars → {len(response.content)} chars")
                return response.content           

        except Exception as e:
            # Fallback: return original content on failure (graceful degradation)
            print(f"   ⚠️  Summarization failed: {e}, returning original content")
            return webpage_content



    # Initialize search tool (following open_deep_research's tavily_search pattern)
    SEARCH_DESCRIPTION = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events."
    )
    @tool(description=SEARCH_DESCRIPTION)
    async def web_search(search_query: str) -> str:
        """Search the web for current information. Use this when you need up-to-date facts or news.
        
        This follows the open_deep_research tavily_search pattern for ALL backends:
        - Fetches search results with metadata (title, URL, content)
        - Deduplicates by URL
        - Optionally summarizes long content using LLM
        - Formats output with numbered sources
        """
        print(f"\n🔍 EXECUTING REAL SEARCH: '{search_query}' (ASYNC)")
        
        # Step 1: Fetch raw results based on backend using modular implementations
        from mas_r1_reasoner.agents.blocks_harmony.web_search_tool import (
            search_duckduckgo,
            search_tavily,
            search_serper,
            search_brightdata
        )
        
        raw_results = []
        
        # Prepare webpage content fetcher (only if summarize_webpages is enabled)
        # fetch_fn = _fetch_webpage_content if context["summarize_webpages"] else None
        #TODO: for now, always None. SHould be configurable
        fetch_fn = None

        try:
            if search_tool == "duckduckgo":
                raw_results = await search_duckduckgo(search_query, fetch_webpage_content_fn=fetch_fn)
            
            elif search_tool == "tavily":
                raw_results = await search_tavily(search_query, fetch_webpage_content_fn=fetch_fn)
            
            elif search_tool == "google-serper" or search_tool == "serper":
                try:
                    raw_results = await search_serper(search_query, fetch_webpage_content_fn=fetch_fn)
                except AttributeError as e:
                    # Serper fallback: return formatted text directly
                    return str(e).replace("Serper fallback result: ", "Search results:\n\n")
            
            elif search_tool == "brightdata" or search_tool == "bright":
                raw_results = await search_brightdata(search_query, fetch_webpage_content_fn=fetch_fn)
            
            else:
                return f"Unknown search tool: {search_tool}"
                
        except Exception as e:
            return str(e)
        
        # Step 2: Deduplicate by URL (matching open_deep_research pattern)
        unique_results = {}
        for result in raw_results:
            url = result.get('url', '')
            if url and url not in unique_results:
                unique_results[url] = result
        
        if not unique_results:
            return "No valid search results found. Please try a different search query."
        
        # Step 3: Optionally summarize each result (matching open_deep_research)
        if context["summarize_webpages"]:
            print(f"   📄 Processing {len(unique_results)} unique results...")
            
            # Initialize summarization model ONCE with structured output and retry (matching open_deep_research line 86-93)
            # This is more efficient than creating it inside each function call
            base_model = context["chat_model"]  # Base model without structured output
            structured_model = base_model.with_structured_output(Summary).with_retry(
                stop_after_attempt=3  # Retry up to 3 times if structured output fails
            )
            print(f"   ✓ Initialized summarization models (structured + base for fallback)")
            
            # Process summaries in parallel for better performance
            async def _process_single_result(url, result):
                title = result.get('title', 'Untitled')
                
                # Use raw_content if available (full webpage), otherwise use content (snippet)
                # This matches open_deep_research pattern
                raw_content = result.get('raw_content', '')
                content = result.get('content', '')
                
                # Validate raw_content for this specific result
                if not raw_content and context["summarize_webpages"]:
                    print(f"   ⚠️  No raw_content for {title[:50]}... - using snippet only ({len(content)} chars)")
                
                # Prefer raw_content for summarization (full webpage)
                content_to_process = raw_content if raw_content else content
                
                # Always summarize when summarize_webpages is enabled
                if content_to_process:
                    print(f"   ✂️  Summarizing: {title}: {content_to_process[:100]}... ({len(content_to_process)} chars)")
                    # Truncate at call site and pass both models (matching open_deep_research line 104)
                    content_to_process = await _summarize_webpage_content(
                        structured_model,  # Pre-configured structured model (created once)
                        base_model,  # Base model for fallback
                        content_to_process[:max_chars]  # Truncate at call site
                    )
                
                return url, {
                    'title': title,
                    'content': content_to_process if content_to_process else content
                }
            
            # Process all results in parallel
            tasks = [_process_single_result(url, result) for url, result in unique_results.items()]
            results = await asyncio.gather(*tasks)
            
            # Convert results back to dictionary
            summarized_results = {url: result for url, result in results}
        else:
            # No summarization - use raw_content if available, otherwise content
            summarized_results = {
                url: {
                    'title': result.get('title', 'Untitled'),
                    'content': result.get('raw_content', result.get('content', ''))
                }
                for url, result in unique_results.items()
            }
        
        # Step 4: Format output (matching open_deep_research pattern)
        formatted_output = "Search results:\n\n"
        for i, (url, result) in enumerate(summarized_results.items(), 1):
            formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
            formatted_output += f"URL: {url}\n\n"
            # Use "SUMMARY:" label when summarization is enabled (matching open_deep_research)
            label = "SUMMARY:" if context["summarize_webpages"] else "CONTENT:"
            formatted_output += f"{label}\n{result['content']}\n\n"
            formatted_output += "\n\n" + "-" * 80 + "\n"
        
        return formatted_output
    
    # Initialize think_tool for reflection (from open_deep_research)
    @tool(description="Strategic reflection tool for research planning")
    def think_tool(reflection: str) -> str:
        """Tool for strategic reflection on research progress and decision-making.
        
        Use this tool after each search to analyze results and plan next steps systematically.
        This creates a deliberate pause in the research workflow for quality decision-making.
        
        When to use:
        - After receiving search results: What key information did I find?
        - Before deciding next steps: Do I have enough to answer comprehensively?
        - When assessing research gaps: What specific information am I still missing?
        - Before concluding research: Can I provide a complete answer now?
        
        Reflection should address:
        1. Analysis of current findings - What concrete information have I gathered?
        2. Gap assessment - What crucial information is still missing?
        3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
        4. Strategic decision - Should I continue searching or provide my answer?
        
        Args:
            reflection: Your detailed reflection on research progress, findings, gaps, and next steps
            
        Returns:
            Confirmation that reflection was recorded for decision-making
        """
        print(f"\n💭 THINKING: {reflection[:200]}{'...' if len(reflection) > 200 else ''}")
        return f"Reflection recorded: {reflection}"
    
    # Initialize model directly with ChatTogether (no fallback)
    # Parse model name (GPT-OSS will happen to work)
    #TODO: may need more work
    # Use pre-initialized chat model from AgentSystem (initialized once per worker)
    # If not available (e.g., sync mode or first call), initialize it here
    if hasattr(self, 'online_chat_model') and self.online_chat_model is not None:
        chat_model = self.online_chat_model
        print(f"🧠 Using pre-initialized chat model from AgentSystem (ONLINE MODE)")
    else:
        raise ValueError("Online chat model not initialized. Please initialize it in AgentSystem.")
    
    # Store the model in context so web_search tool can access it for summarization
    context["chat_model"] = chat_model
    
    # Bind tools to model (LangChain handles format automatically - Harmony for GPT-OSS, native for others!)
    # Include both search and think_tool (following open_deep_research pattern)
    model_with_tools = chat_model.bind_tools([web_search, think_tool])
    
    # Prepare system prompt
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = research_system_prompt.format(date=today)
    
    # Initialize conversation
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research Question: {query}\n\nPlease conduct web searches and provide a comprehensive answer with sources.")
    ]
    
    # Tool calling loop
    import asyncio
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        # Call model (LangChain handles everything - including Harmony format for GPT-OSS!)
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        print(f"  - Msg: {messages}")
       
        # Check if model wants to use tools
        if not response.tool_calls:
            # No tool calls - this is the final answer
            print("No tool calls - final answer received")
            return response.content
        
        # Execute tool calls
        print(f"Executing {len(response.tool_calls)} tool call(s)...")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  - {tool_name}({tool_args})")
            
            # Execute the tool
            if tool_name == "web_search":
                search_query = tool_args.get("search_query", "")
                tool_result = await web_search.ainvoke(search_query)
            elif tool_name == "think_tool":
                reflection = tool_args.get("reflection", "")
                tool_result = await think_tool.ainvoke(reflection)
            else:
                tool_result = f"Unknown tool: {tool_name}"
            
            # Add tool result to messages (LangChain handles the format)
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))
    
    # Max iterations reached - return last response
    print(f"Max iterations ({max_iterations}) reached")
    final_response = model_with_tools.invoke(messages)
    return final_response.content




# Export for mas_r1 framework compatibility
func_string = inspect.getsource(WebSearchAgent)

WebSearch = {
    "desciption": "Web search allows models to access up-to-date information from the internet and provide answers with sourced citations.",
    "name": "Web Search Agent (WebSearchAgent)",
    "required_arguments": {
        "agent_input": "The input for the SearchAgent. This is the task question for the WebSearchAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question."
    },
    "implementation": """WebSearchAgent conducts iterative web research using two main tools:
    1. web_search tool: Searches the web, retrieves up to 5 results, optionally summarizes webpage content using LLM, and formats output with citations
    2. think_tool: Enables reflection after each search to analyze findings, assess gaps, and decide next steps

    The agent operates in a tool-calling loop (max 5 iterations by default) following a research workflow:
    - Executes searches based on the query
    - Reflects on results using think_tool
    - Decides whether to continue searching or provide final answer
    - Returns comprehensive answer with sources when sufficient information is gathered"""
}
# Example usage
# if __name__ == "__main__":
#     # Example 1: Using Together AI with GPT-OSS (LangChain handles Harmony format automatically!)
#     result = WebSearchAgent(
#         query="What are the latest AI developments in 2025?",
#         model="together_ai/openai/gpt-oss-120b",  # LangChain knows how to work with GPT-OSS!
#         temperature=0.7,
#         reasoning_effort="medium"  # Options: "low", "medium", "high" - controls thinking depth
#     )
#     print("\n" + "="*80)
#     print("RESULT:")
#     print("="*80)
#     print(result)