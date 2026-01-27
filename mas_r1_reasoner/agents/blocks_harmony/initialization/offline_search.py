"""
Offline Search Initialization Module

Handles initialization of offline web search resources including:
- BrowseComp-Plus corpus loading
- Chat model for summarization
- Retriever (BM25 or Dense)
- SQLite persistent cache for summaries
"""

import os
import json
import sqlite3
from typing import Any, Dict
from mas_r1_reasoner.agents.shared_vars import get_global


def initialize_offline_resources(agent_system):
    """
    Initialize all offline search resources for AgentSystem.
    
    This function is called once during AgentSystem initialization if web_search_type == "offline".
    It sets up:
    1. Corpus (100k documents from BrowseComp-Plus)
    2. Chat model for summarization (reused across all searches)
    3. Retriever (BM25 or Dense with FAISS)
    4. Persistent cache (SQLite with WAL mode for thread-safety)
    
    Args:
        agent_system: AgentSystem instance to initialize resources for
    """
    print(f"🔧 AgentSystem: Detected offline search mode, initializing resources...")
    
    # 1. Load corpus
    corpus = _load_corpus()
    if corpus:
        agent_system.offline_corpus = corpus
        print(f"   ✅ Offline corpus loaded: {len(corpus)} documents")
    
    # 2. Initialize chat model for summarization
    chat_model = _initialize_chat_model(agent_system.node_model)
    if chat_model:
        agent_system.offline_chat_model = chat_model
    
    # 3. Initialize retriever (BM25 or dense)
    retriever = _initialize_retriever(agent_system)
    if retriever:
        agent_system.offline_retriever = retriever
    
    # 4. Initialize persistent cache (SQLite)
    cache_db = _initialize_cache()
    if cache_db:
        agent_system.offline_cache_db = cache_db


def _load_corpus():
    """Load BrowseComp-Plus corpus from local JSONL file."""
    corpus_path = os.path.join(os.getcwd(), "data/browsecomp_plus/corpus.jsonl")
    
    if not os.path.exists(corpus_path):
        print(f"   ⚠️  Corpus not found at {corpus_path}")
        return None
    
    print(f"   📚 Loading corpus from {corpus_path}...")
    corpus = {}
    
    try:
        with open(corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get('docid', doc.get('id', ''))
                corpus[doc_id] = doc
        
        return corpus
    except Exception as e:
        print(f"   ⚠️  Error loading corpus: {e}")
        return None


def _initialize_chat_model(node_model):
    """Initialize ChatTogether model for webpage summarization."""
    try:
        from langchain_together import ChatTogether
        
        # Map model names to their full API names (matching grpo_trainer.yaml model_sampler_map)
        # This ensures consistency with BaseDatasetProcessor and model samplers
        model_mapping = {
            "gpt-4o": "gpt-4o",
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-nano": "gpt-4.1-nano",
            "gpt-5-nano": "gpt-5-nano",
            "llama-3.3-70b-instr": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "qwen-2.5-32b-instr": "qwen-2.5-32b-instr",
            "qwen-2.5-7b-instr": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen-2.5-72b-instr": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        }
        
        # Parse model name: if it already has a prefix, use as-is; otherwise map it
        if "/" in node_model:
            # Model already has a prefix (e.g., "openai/gpt-oss-120b")
            model_name = node_model
        else:
            # Try to find in mapping, or add openai/ prefix as fallback
            model_name = model_mapping.get(node_model, f"openai/{node_model}")
        
        # Get reasoning_effort from global config
        reasoning_effort = get_global("global_reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = "low"  # Fallback default
        
        # Configure for offline search (matching web_search_offline.py settings exactly)
        chat_model_kwargs = {
            "model": model_name,
            "temperature": 0.5,
            "together_api_key": os.getenv("TOGETHER_API_KEY"),
            "reasoning_effort": reasoning_effort,
            "timeout": 300,
            "max_tokens": 60000,
            "max_retries": 3,
        }
        
        chat_model = ChatTogether(**chat_model_kwargs)
        print(f"   🧠 Chat model initialized: {model_name}")
        return chat_model
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not initialize offline chat model: {e}")
        return None


def _initialize_retriever(agent_system):
    """Initialize retriever (BM25 or Dense) for offline search."""
    try:
        from mas_r1_reasoner.agents.blocks_harmony.offline_retriver import initialize_bm25_retriever
        
        # Get retrieval method from agent system
        retrieval_method = agent_system.retrieval_method
        index_path = os.path.join(os.getcwd(), "data/browsecomp_plus/indexes")
        
        print(f"   🔧 Initializing {retrieval_method.upper()} retriever...")
        
        if retrieval_method == "bm25":
            retriever = initialize_bm25_retriever(index_path)
            print(f"   ✅ BM25 retriever initialized")
            return retriever
        elif retrieval_method == "dense":
            # Initialize dense retriever with centralized embedding model
            dense_embedding_model = "qwen3-embedding-8b"
            retriever = _initialize_dense_retriever_centralized(index_path, dense_embedding_model)
            print(f"   ✅ Dense retriever initialized")
            return retriever
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not initialize offline retriever: {e}")
        return None


def _initialize_dense_retriever_centralized(index_path: str, embedding_model: str = "qwen3-embedding-0.6b") -> Dict[str, Any]:
    """Initialize dense retriever using Tevatron (centralized version).
    
    This function loads the FAISS index (embeddings + docid lookup) and the encoder model.
    It does NOT load the document text corpus, since that's already loaded in AgentSystem and 
    passed to search_dense() as the 'corpus' parameter.
    
    Args:
        index_path: Path to the directory containing indexes
        embedding_model: Name of the embedding model/index to use (e.g., "qwen3-embedding-8b")
        
    Returns:
        Dictionary containing retriever metadata, FAISS index, encoder, etc.
        
    Raises:
        ImportError: If required packages not installed (tevatron, faiss, torch, transformers)
        FileNotFoundError: If dense index not found
        RuntimeError: If initialization fails
    """
    try:
        import faiss
        import torch
        import glob
        import pickle
        from itertools import chain
        import numpy as np
        from tevatron.retriever.arguments import ModelArguments
        from tevatron.retriever.driver.encode import DenseModel
        from tevatron.retriever.searcher import FaissFlatSearcher
        from transformers import AutoTokenizer
        
        # Map embedding model name to HuggingFace model name
        model_mapping = {
            "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
            "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
        }
        model_name = model_mapping.get(embedding_model, embedding_model)
        
        print(f"   🔧 Initializing FAISS searcher with Tevatron...")
        print(f"   📦 Model: {model_name}")
        
        # 1. Load FAISS index from pickle files (BrowseComp-Plus pattern)
        dense_index_path = os.path.join(index_path, embedding_model)
        if not os.path.exists(dense_index_path):
            raise FileNotFoundError(f"Dense index not found at {dense_index_path}")
        
        pickle_pattern = os.path.join(dense_index_path, "corpus.shard*.pkl")
        index_files = glob.glob(pickle_pattern)
        
        if not index_files:
            raise FileNotFoundError(f"No pickle files found matching {pickle_pattern}")
        
        print(f"   📁 Pattern match found {len(index_files)} files; loading them into index.")
        
        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup
        
        # Load first shard
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        retriever = FaissFlatSearcher(p_reps_0)
        
        # Load remaining shards
        lookup = []
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        for p_reps, p_lookup in shards:
            retriever.add(p_reps)
            lookup += p_lookup
        
        print(f"   📊 Loaded {retriever.index.ntotal} embeddings into FAISS index")
        
        # 2. Setup GPU (BrowseComp-Plus pattern)
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            print(f"   💻 No GPU found or using faiss-cpu. Using CPU.")
        else:
            print(f"   🚀 Using {num_gpus} GPU(s)")
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co, ngpu=num_gpus)
        
        # 3. Load model using Tevatron's DenseModel (BrowseComp-Plus exact approach)
        print(f"   🧠 Loading model: {model_name}")
        
        hf_home = os.getenv("HF_HOME")
        cache_dir = hf_home if hf_home else None
        
        model_args = ModelArguments(
            model_name_or_path=model_name,
            normalize=False,  # BrowseComp-Plus default
            pooling="eos",    # BrowseComp-Plus default for Qwen models
            cache_dir=cache_dir,
        )
        
        # Determine torch dtype
        torch_dtype_str = os.getenv("TORCH_DTYPE", "float16")
        if torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load model using Tevatron's DenseModel.load (exactly as BrowseComp-Plus)
        model = DenseModel.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            padding_side="left",  # BrowseComp-Plus default
        )
        
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ FAISS searcher initialized successfully with Tevatron")
        
        return {
            "type": "dense",
            "retriever": retriever,  # Tevatron's FaissFlatSearcher
            "lookup": lookup,
            "model": model,  # Tevatron's DenseModel
            "tokenizer": tokenizer,
            "device": device,
            "model_name": model_name,
            # BrowseComp-Plus configuration
            "task_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            "max_length": 8192,
        }
        
    except ImportError as e:
        raise ImportError(
            f"Required packages not installed: {e}. "
            "Install with: pip install tevatron faiss-gpu transformers torch\n"
            "Note: Tevatron installation: pip install git+https://github.com/texttron/tevatron.git"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize dense retriever: {e}")


def _initialize_cache():
    """Initialize SQLite persistent cache for webpage summaries."""
    try:
        cache_db_path = os.path.join(os.getcwd(), "data/browsecomp_plus/summary_cache.db")
        os.makedirs(os.path.dirname(cache_db_path), exist_ok=True)
        
        # Open connection with thread-safe settings
        cache_db = sqlite3.connect(
            cache_db_path,
            check_same_thread=False,  # Allow access from multiple threads
            timeout=10.0  # Wait up to 10s for locks
        )
        
        # Enable Write-Ahead Logging for better concurrency
        cache_db.execute("PRAGMA journal_mode=WAL")
        cache_db.execute("PRAGMA synchronous=NORMAL")
        
        # Create cache table if not exists
        cache_db.execute("""
            CREATE TABLE IF NOT EXISTS summary_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on created_at for time-based queries
        cache_db.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_cache_created_at 
            ON summary_cache(created_at)
        """)
        
        cache_db.commit()
        
        # Get cache stats
        cursor = cache_db.execute("SELECT COUNT(*) FROM summary_cache")
        cache_count = cursor.fetchone()[0]
        print(f"   💾 Persistent cache initialized: {cache_count} summaries stored")
        
        return cache_db
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not initialize persistent cache: {e}")
        return None

