import os

class Config:
    """Global configuration settings - OPTIMIZED"""

    GOOGLE_API_KEY = os.getenv(
        "GOOGLE_API_KEY", ""
    )
    TAVILY_API_KEY = os.getenv(
        "TAVILY_API_KEY", ""
    )

    # Storage directories
    CHROMA_PERSIST_DIR = "chroma_db"
    MEMORY_DIR = "conversation_memory"

    # Model settings
    EMBEDDING_MODEL = "models/embedding-001"
    PRIMARY_LLM = "gemini-2.0-flash"
    FALLBACK_LLM = "gemini-1.5-flash"

    # Memory settings - OPTIMIZED
    MAX_MEMORY_EXCHANGES = 8  # Reduced from 10
    MAX_SUMMARY_TOKENS = 1500  # Reduced from 2000

    # Performance settings
    ENABLE_PARALLEL_SEARCH = True
    SEARCH_TIMEOUT = 10  # seconds
    CACHE_TTL = 300  # 5 minutes cache
    MAX_RAG_RESULTS = 2  # Reduced from 3
    MAX_WEB_RESULTS = 2  # Reduced from 3
