"""
Offline Web Search Agent - Based on web_search.py

This is identical to the online web_search.py except it uses offline retrieval (BM25/dense)
instead of live web APIs for reproducible evaluation.

The ONLY difference: web_search tool searches local index instead of internet

Performance Optimizations (Ray-aware):
- Corpus, retriever, and chat model are initialized ONCE per Ray worker in AgentSystem
- Hybrid cache (memory + SQLite): Fast in-memory cache + persistent disk storage
  * Level 1: In-memory dict (fastest, ~0.001ms access)
  * Level 2: SQLite database (persistent, shared across workers/runs, ~1-20ms access)
  * Note: Both are orders of magnitude faster than LLM calls (~10,000ms)
- Cache persists across runs (SQLite) and is shared by all workers
- Thread-safe and process-safe with threading.Lock + SQLite WAL mode
"""

import inspect


async def WebSearchOfflineAgent(self, agent_input, model: str) -> str:
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool
    from langchain_together import ChatTogether
    from datetime import datetime
    from typing import Any, List, Dict, Optional
    import os
    
    # Import retriever functions at the top to avoid repeated imports
    from .offline_retriver import search_bm25, search_dense

    max_iterations = 5
    max_tool_calls_per_iter = 5

    # OFFLINE SEARCH CONFIGURATION (the main difference!)
    # Following BrowseComp-Plus: Supports BM25 (lexical) and Dense (FAISS) retrieval
    retrieval_method = self.retrieval_method # "bm25" or "dense"
    index_path = "./data/browsecomp_plus/indexes"
    
    # Dense retrieval configuration (only used when retrieval_method="dense")
    # Supported models: qwen3-embedding-0.6b, qwen3-embedding-4b, qwen3-embedding-8b
    dense_embedding_model = "qwen3-embedding-8b"
    
    corpus_path = "./data/browsecomp_plus/corpus.jsonl"
    max_search_results = 5
    summarize_webpages = True  # Offline corpus is already clean, no need to summarize
    max_chars = 10000  # Character limit to stay within model token limits (matching open_deep_research)


    # Define Summary structure at module level (matching open_deep_research/state.py)
    class Summary(BaseModel):
        """Research summary with key findings."""
        summary: str = Field(description="Main summary of the webpage content")
        key_excerpts: str = Field(description="Important quotes or excerpts from the content")


    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"
    
    # Extract the query from agent_input
    query = agent_input.content

    # Store configuration in a container
    context = {
        "chat_model": None,
        "summarize_webpages": summarize_webpages,
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
        1. **web_search**: For conducting searches to gather information
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


    async def _summarize_webpage_content(structured_model, base_model, webpage_content: str) -> str:
        """Summarize webpage content using LLM (following open_deep_research pattern).
        
        Uses hybrid cache (memory + SQLite) to avoid re-summarizing duplicates:
        - Level 1: In-memory dict (~1µs access) - per-worker, cleared on restart
        - Level 2: SQLite database (~100µs access) - persistent, shared across workers/runs
        
        Cache is stored in AgentSystem and persists across evaluation runs.
        
        Args:
            structured_model: LLM model configured with .with_structured_output(Summary) and .with_retry()
            base_model: Base LLM model without structured output (for fallback)
            webpage_content: Raw webpage content (already truncated at call site)
            
        Returns:
            Formatted summary with <summary> and <key_excerpts> tags, or original if fails
        """
        import hashlib
        import asyncio

        if len(webpage_content) <= 5000: # if small, we don't need to summarize
            print(f"   ✂️  Webpage content is already short enough: {len(webpage_content)} chars")
            return webpage_content

        try:
            # Create content hash for cache key
            content_hash = hashlib.md5(webpage_content.encode('utf-8')).hexdigest()
            
            # Hybrid cache: Check memory first (fast), then SQLite (persistent), then compute
            if hasattr(self, 'offline_summary_cache') and hasattr(self, 'offline_cache_lock'):
                with self.offline_cache_lock:
                    # Level 1: In-memory cache (fastest)
                    if content_hash in self.offline_summary_cache:
                        print(f"   ✓ Cache hit (memory): {content_hash[:8]}...")
                        return self.offline_summary_cache[content_hash]
                    
                    # Level 2: SQLite persistent cache (fast, shared across workers/runs)
                    if hasattr(self, 'offline_cache_db') and self.offline_cache_db is not None:
                        try:
                            cursor = self.offline_cache_db.execute(
                                "SELECT summary FROM summary_cache WHERE content_hash = ?",
                                (content_hash,)
                            )
                            row = cursor.fetchone()
                            if row:
                                summary = row[0]
                                # Populate memory cache for future access
                                self.offline_summary_cache[content_hash] = summary
                                print(f"   ✓ Cache hit (SQLite): {content_hash[:8]}... → loaded to memory")
                                return summary
                        except Exception as e:
                            print(f"   ⚠️  SQLite read error: {e}")
            
            # Cache miss - summarize the content (outside lock to avoid blocking other tasks)
            print(f"   🔄 Cache miss - summarizing content (hash: {content_hash[:8]}...)...")
            
            # Format prompt with current date (matching open_deep_research)
            today = datetime.now().strftime("%Y-%m-%d")
            prompt_content = summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=today
            )
            # try:
            #     # Invoke the structured model (matching open_deep_research pattern)
            #     summary = structured_model.invoke([HumanMessage(content=prompt_content)])
                
            #     # Format the summary with structured sections (matching open_deep_research)
            #     formatted_summary = (
            #         f"<summary>\n{summary.summary}\n</summary>\n\n"
            #         f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
            #     )
            #     print(f"   ✂️  Summarized {len(webpage_content)} chars → {len(summary.summary)} chars")
            #     return formatted_summary

            # except Exception as e: # sometimes the structured output fails, so we try regular completion
            #     # Fallback: If structured output fails, try regular completion with base model
            #     print(f"   ⚠️  Structured output failed: {e}, trying regular completion")
            # ⚠️  Structured output alsomote 100% not correct
            # Use LangChain for summarization
            response = await base_model.ainvoke(
                [HumanMessage(content=prompt_content)]
            )
            summary = response.content
            print(f"   ✂️  Summarized {len(webpage_content)} chars → {len(summary)} chars")
            
            # Store in hybrid cache (memory + SQLite) - thread-safe write
            if hasattr(self, 'offline_summary_cache') and hasattr(self, 'offline_cache_lock'):
                with self.offline_cache_lock:
                    # Write to memory cache
                    self.offline_summary_cache[content_hash] = summary
                    
                    # Write to SQLite persistent cache (shared across workers/runs)
                    if hasattr(self, 'offline_cache_db') and self.offline_cache_db is not None:
                        try:
                            self.offline_cache_db.execute(
                                "INSERT OR REPLACE INTO summary_cache (content_hash, summary) VALUES (?, ?)",
                                (content_hash, summary)
                            )
                            self.offline_cache_db.commit()
                            print(f"   ✓ Cached to memory + SQLite (total in memory: {len(self.offline_summary_cache)})")
                        except Exception as e:
                            print(f"   ⚠️  SQLite write error (memory cache still updated): {e}")
                    else:
                        print(f"   ✓ Cached to memory only (total: {len(self.offline_summary_cache)})")
            
            return summary           

        except Exception as e:
            # Fallback: return original content on failure (graceful degradation)
            print(f"   ⚠️  Summarization failed: {e}, returning original content")
            return webpage_content


    # ==========================================
    # OFFLINE SEARCH IMPLEMENTATION (THE MAIN DIFFERENCE!)
    # ==========================================
    # Note: Retriever is pre-initialized in AgentSystem.__init__() for speed
    
    def _load_corpus():
        """Load corpus for retrieving full documents."""
        # First check if corpus was pre-loaded in AgentSystem (Ray worker async mode)
        if hasattr(self, 'offline_corpus') and self.offline_corpus is not None:
            return self.offline_corpus
        raise RuntimeError("Corpus not loaded. Please load the corpus first.")



    SEARCH_DESCRIPTION = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events."
    )
    @tool(description=SEARCH_DESCRIPTION)
    async def web_search(search_query: str) -> str:
        """Search for information. Use this when you need up-to-date facts or news.
        
        This follows the open_deep_research tavily_search pattern but uses OFFLINE corpus:
        - Searches local index (BM25/dense) instead of live web
        - Returns documents from fixed corpus for reproducibility
        - Formats output with numbered sources
        """
        print(f"\n🔍 EXECUTING OFFLINE SEARCH: '{search_query}'")
        
        try:
            # Use pre-initialized retriever from AgentSystem (initialized once per worker)
            if hasattr(self, 'offline_retriever') and self.offline_retriever is not None:
                retriever = self.offline_retriever
            else:
                raise RuntimeError("Retriever not initialized. Please initialize the retriever first.")
            
            # Step 1: Search index
            # Load corpus (needed for both methods)
            corpus = _load_corpus()
            
            if retriever["type"] == "bm25":
                raw_results = await search_bm25(retriever, search_query, corpus, max_search_results)
            elif retriever["type"] == "dense":
                raw_results = await search_dense(retriever, search_query, corpus, max_search_results)
            else:
                return f"Retrieval method {retriever['type']} not implemented"
            
            if not raw_results:
                return "No valid search results found. Please try a different search query."
            
            print(f"   ✓ Retrieved {len(raw_results)} documents")
            
        except Exception as e:
            return f"Offline search failed: {str(e)}"
        
        # Step 2: Deduplicate by URL (matching open_deep_research pattern)
        unique_results = {}
        for result in raw_results:
            url = result.get('url', result.get('docid', ''))
            if url and url not in unique_results:
                unique_results[url] = result
        
        if not unique_results:
            return "No valid search results found. Please try a different search query."
        
        # Step 3: Optionally summarize each result (matching open_deep_research)
        if context["summarize_webpages"]:
            print(f"   📄 Processing {len(unique_results)} unique results...")
            
            # Initialize summarization model ONCE with structured output and retry
            base_model = context["chat_model"]
            # structured_model = base_model.with_structured_output(Summary).with_retry(
            #     stop_after_attempt=3
            # )
            
            # Process summaries in parallel for better performance
            async def _process_single_result(url, result):
                title = result.get('title', 'Untitled')
                content = result.get('content', '')
                
                if content:
                    print(f"   ✂️  Summarizing: {title[:50]}... {content[:50]}...({len(content)} chars)")
                    content = await _summarize_webpage_content(
                        structured_model=None,
                        base_model=base_model,
                        webpage_content=content[:max_chars]
                    )
                
                return url, {
                    'title': title,
                    'content': content
                }
            
            # Process all results in parallel
            tasks = [_process_single_result(url, result) for url, result in unique_results.items()]
            results = await asyncio.gather(*tasks)
            
            # Convert results back to dictionary
            summarized_results = {url: result for url, result in results}
        else:
            # No summarization - use content as is
            summarized_results = {
                url: {
                    'title': result.get('title', 'Untitled'),
                    'content': result.get('content', '')[:max_chars]  # Truncate
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
    async def think_tool(reflection: str) -> str:
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
    
    # Use pre-initialized chat model from AgentSystem (initialized once per worker)
    # If not available (e.g., sync mode), initialize it here
    if hasattr(self, 'offline_chat_model') and self.offline_chat_model is not None:
        chat_model = self.offline_chat_model
        print(f"🧠 Using pre-initialized chat model from AgentSystem (OFFLINE MODE)")
    else:
       raise RuntimeError("Chat model not initialized. Please initialize the chat model first.")
    
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
        HumanMessage(content=f"Research Question: {query}\n\nPlease conduct searches and provide a comprehensive answer with sources.")
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
        
        # Limit tool calls to prevent explosion (only if max_tool_calls_per_iter is set)
        if max_tool_calls_per_iter is not None and len(response.tool_calls) > max_tool_calls_per_iter:
            print(f"   ⚠️  Limiting tool calls: {len(response.tool_calls)} → {max_tool_calls_per_iter}")
            response.tool_calls = response.tool_calls[:max_tool_calls_per_iter]
        
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
    final_response = await model_with_tools.ainvoke(messages)
    return final_response.content




# Export for mas_r1 framework compatibility
func_string = inspect.getsource(WebSearchOfflineAgent)

WebSearchOffline = {
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