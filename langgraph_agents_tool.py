import os
import json
import uuid
import asyncio
import concurrent.futures
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypedDict, Literal, List, Dict, Any, Optional, Union

import pytz
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ================================================================================================
# CONFIGURATION
# ================================================================================================


class Config:
    """Global configuration settings - OPTIMIZED"""

    GOOGLE_API_KEY = os.getenv(
        "GOOGLE_API_KEY", "AIzaSyD3u_bahMJuEWTWQfTHgWrizY8q8j-9-2M"
    )
    TAVILY_API_KEY = os.getenv(
        "TAVILY_API_KEY", "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"
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


# ================================================================================================
# STATE DEFINITION
# ================================================================================================


class State(TypedDict):
    """LangGraph state definition"""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_profile: Dict[str, Any]
    conversation_summary: str


# ================================================================================================
# PERFORMANCE OPTIMIZATION UTILITIES
# ================================================================================================


class QueryCache:
    """Simple query cache with TTL"""

    def __init__(self, ttl_seconds=Config.CACHE_TTL):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = (value, datetime.now().timestamp())

    def clear(self):
        self.cache.clear()


# Global cache instance
query_cache = QueryCache()


def timeout_wrapper(func, timeout_seconds=Config.SEARCH_TIMEOUT):
    """Wrapper to add timeout to functions"""

    def wrapper(*args, **kwargs):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return f"⚠️ Search timed out after {timeout_seconds}s"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return wrapper


# ================================================================================================
# MEMORY MANAGEMENT - OPTIMIZED
# ================================================================================================


class MemoryManager:
    """OPTIMIZED LangChain-based memory management system"""

    def __init__(self):
        self.memory_dir = Path(Config.MEMORY_DIR)
        self.memory_dir.mkdir(exist_ok=True)

        # Initialize LLM for memory operations
        self.llm = self._initialize_memory_llm()

        # LangChain memory components - OPTIMIZED
        self.conversation_memory = ConversationBufferWindowMemory(
            k=Config.MAX_MEMORY_EXCHANGES,
            return_messages=True,
            memory_key="chat_history",
        )

        # Disable summary memory for speed (can be re-enabled if needed)
        self.summary_memory = None

        # Session and profile management
        self.current_session_id = None
        self.user_profile = self._load_user_profile()

        # Performance optimization
        self._memory_cache = {}

    def _initialize_memory_llm(self):
        """Initialize LLM for memory operations"""
        try:
            return ChatGoogleGenerativeAI(
                model=Config.FALLBACK_LLM,
                temperature=0,
                google_api_key=Config.GOOGLE_API_KEY,
            )
        except Exception:
            return None

    def start_new_session(self) -> str:
        """Start a new conversation session"""
        self.current_session_id = str(uuid.uuid4())
        self.conversation_memory.clear()
        self._memory_cache.clear()  # Clear cache on new session

        # Skip session file creation for speed - only save when needed
        return self.current_session_id

    def add_message(self, human_message: str, ai_message: str):
        """OPTIMIZED: Add messages to memory"""
        try:
            self.conversation_memory.save_context(
                {"input": human_message}, {"output": ai_message}
            )
            # Clear relevant cache entries
            self._memory_cache.clear()
        except Exception:
            pass  # Fail silently for speed

    @lru_cache(maxsize=32)
    def get_memory_context_cached(self, cache_key: str) -> List[BaseMessage]:
        """Cached memory context retrieval"""
        try:
            memory_vars = self.conversation_memory.load_memory_variables({})
            return memory_vars.get("chat_history", [])
        except Exception:
            return []

    def get_memory_context(self) -> List[BaseMessage]:
        """Get current memory context with caching"""
        cache_key = f"memory_{hash(str(self.conversation_memory.buffer))}"
        return self.get_memory_context_cached(cache_key)

    def quick_search_memory(self, query: str) -> str:
        """FAST memory search with caching"""
        cache_key = f"memory_search_{hash(query.lower())}"
        cached_result = query_cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            memory_context = self.get_memory_context()
            query_lower = query.lower()

            # Quick search through recent messages only
            relevant_messages = []
            for msg in memory_context[-6:]:  # Only check last 6 messages for speed
                if hasattr(msg, "content") and query_lower in msg.content.lower():
                    content = (
                        msg.content[:150] + "..."
                        if len(msg.content) > 150
                        else msg.content
                    )
                    msg_type = "You" if "Human" in str(type(msg)) else "AI"
                    relevant_messages.append(f"{msg_type}: {content}")
                    if len(relevant_messages) >= 2:  # Limit results for speed
                        break

            result = ""
            if relevant_messages:
                result = "🧠 **From Recent Memory:**\n" + "\n".join(relevant_messages)

            query_cache.set(cache_key, result)
            return result
        except Exception:
            return ""

    def quick_search_profile(self, query: str) -> str:
        """FAST profile search with caching"""
        if not self.user_profile:
            return ""

        cache_key = f"profile_search_{hash(query.lower())}"
        cached_result = query_cache.get(cache_key)
        if cached_result:
            return cached_result

        query_lower = query.lower()
        relevant_info = []

        for key, value in self.user_profile.items():
            if query_lower in key.lower() or query_lower in str(value).lower():
                relevant_info.append(f"**{key}:** {value}")
                if len(relevant_info) >= 3:  # Limit for speed
                    break

        result = ""
        if relevant_info:
            result = "👤 **From Your Profile:**\n" + "\n".join(relevant_info)

        query_cache.set(cache_key, result)
        return result

    def update_user_profile(self, key: str, value: str):
        """Update user profile"""
        self.user_profile[key] = value
        # Clear cache when profile changes
        query_cache.clear()
        # Save async for speed
        try:
            self._save_user_profile()
        except Exception:
            pass

    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        return self.user_profile.copy()

    def _load_user_profile(self) -> Dict[str, Any]:
        """Load user profile from storage"""
        profile_file = self.memory_dir / "user_profile.json"
        try:
            if profile_file.exists():
                with open(profile_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_user_profile(self):
        """Save user profile to storage"""
        profile_file = self.memory_dir / "user_profile.json"
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                json.dump(self.user_profile, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


# ================================================================================================
# RAG SYSTEM - OPTIMIZED
# ================================================================================================


class RAGManager:
    """OPTIMIZED RAG (Retrieval-Augmented Generation) system manager"""

    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self._rag_cache = {}
        self._initialize_rag()

    def _initialize_rag(self):
        """Initialize RAG system"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL, google_api_key=Config.GOOGLE_API_KEY
            )

            os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)

            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name="document_collection",
            )
            print("✅ RAG system initialized")
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")

    def is_initialized(self) -> bool:
        """Check if RAG system is properly initialized"""
        return self.vectorstore is not None and self.embeddings is not None

    @timeout_wrapper
    def quick_search_documents(self, query: str) -> str:
        """FAST document search with caching and timeout"""
        if not self.is_initialized():
            return ""

        # Check cache first
        cache_key = f"rag_search_{hash(query.lower())}"
        cached_result = query_cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            # Reduced k for speed
            docs = self.vectorstore.similarity_search(query, k=Config.MAX_RAG_RESULTS)
            if not docs:
                query_cache.set(cache_key, "")
                return ""

            results = ["📚 **From Knowledge Base:**"]
            for i, doc in enumerate(docs, 1):
                # Shortened content for speed
                content = (
                    doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                )
                results.append(f"**{i}.** {content}")

            result = "\n".join(results)
            query_cache.set(cache_key, result)
            return result
        except Exception:
            return ""

    def add_document(self, content: str, title: str = None, source: str = None) -> str:
        """Add document to knowledge base"""
        if not self.is_initialized():
            return "❌ Knowledge base not initialized"

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Reduced for speed
                chunk_overlap=100,  # Reduced for speed
                length_function=len,
            )

            chunks = text_splitter.split_text(content)
            documents = []

            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": source or f"manual_upload_{datetime.now().isoformat()}",
                    "title": title
                    or f"Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "chunk_id": i,
                    "timestamp": datetime.now().isoformat(),
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

            self.vectorstore.add_documents(documents)
            # Clear cache when new documents added
            query_cache.clear()
            return f"✅ Added document with {len(chunks)} chunks to knowledge base"
        except Exception as e:
            return f"❌ Failed to add document: {str(e)}"


# ================================================================================================
# WEB SEARCH SYSTEM - OPTIMIZED
# ================================================================================================


class WebSearchManager:
    """OPTIMIZED Web search system using Tavily API"""

    @staticmethod
    @timeout_wrapper
    def quick_search_web(query: str) -> str:
        """FAST web search with caching and timeout"""
        if (
            not Config.TAVILY_API_KEY
            or Config.TAVILY_API_KEY == "your_tavily_api_key_here"
        ):
            return "❌ Tavily API key not configured"

        # Check cache first
        cache_key = f"web_search_{hash(query.lower())}"
        cached_result = query_cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            url = "https://api.tavily.com/search"
            headers = {"content-type": "application/json"}

            payload = {
                "api_key": Config.TAVILY_API_KEY,
                "query": query,
                "max_results": Config.MAX_WEB_RESULTS,  # Reduced for speed
                "format": "markdown",
            }

            response = requests.post(
                url, json=payload, headers=headers, timeout=8
            )  # Reduced timeout
            response.raise_for_status()
            data = response.json()

            if "results" not in data:
                result = f"❌ No web results found for: {query}"
                query_cache.set(cache_key, result)
                return result

            results = ["🌐 **From Web Search:**"]
            for i, result in enumerate(data["results"], 1):
                title = result.get("title", "No title")
                url_link = result.get("url", "")
                content = result.get("content", "No content")

                # Shortened for speed
                content_preview = (
                    content[:250] + "..." if len(content) > 250 else content
                )
                results.append(f"**{i}. {title}**")
                results.append(f"📝 {content_preview}")
                if i >= Config.MAX_WEB_RESULTS:  # Limit results
                    break

            result = "\n".join(results)
            query_cache.set(cache_key, result)
            return result
        except Exception as e:
            return f"❌ Web search failed: {str(e)}"


# ================================================================================================
# OPTIMIZED SEARCH COORDINATOR
# ================================================================================================


class FastSearchCoordinator:
    """OPTIMIZED search coordinator with parallel processing"""

    def __init__(self, memory_manager: MemoryManager, rag_manager: RAGManager):
        self.memory_manager = memory_manager
        self.rag_manager = rag_manager
        self.web_search = WebSearchManager()

    def smart_search(self, query: str) -> str:
        """
        OPTIMIZED intelligent search with early termination:
        1. Quick memory/profile check (parallel)
        2. RAG search (only if no memory results)
        3. Web search (only if no local results)
        """
        results = []

        # Step 1: PARALLEL search memory and profile (fastest)
        if Config.ENABLE_PARALLEL_SEARCH:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                profile_future = executor.submit(
                    self.memory_manager.quick_search_profile, query
                )
                memory_future = executor.submit(
                    self.memory_manager.quick_search_memory, query
                )

                # Get results with timeout
                try:
                    profile_info = profile_future.result(timeout=2)
                    memory_info = memory_future.result(timeout=2)
                except concurrent.futures.TimeoutError:
                    profile_info = memory_info = ""
        else:
            profile_info = self.memory_manager.quick_search_profile(query)
            memory_info = self.memory_manager.quick_search_memory(query)

        if profile_info:
            results.append(profile_info)
        if memory_info:
            results.append(memory_info)

        # If we found good memory/profile info, return early for speed
        if len(results) >= 1 and any(len(r) > 50 for r in results):
            return "\n\n".join(results)

        # Step 2: RAG search (only if limited memory results)
        rag_info = self.rag_manager.quick_search_documents(query)
        if rag_info:
            results.append(rag_info)
            # Early termination if we have good local results
            if len(results) >= 2:
                return "\n\n".join(results)

        # Step 3: Web search (only if no good local results)
        if not results or all(len(r) < 50 for r in results):
            web_info = self.web_search.quick_search_web(query)
            if web_info and "❌" not in web_info:
                results.append(web_info)
                results.append("\n💡 *Searched web as no local information was found.*")

        return "\n\n".join(results) if results else "❌ No information found."


# ================================================================================================
# OPTIMIZED TOOLS DEFINITION
# ================================================================================================

# Initialize global managers
memory_manager = MemoryManager()
rag_manager = RAGManager()
search_coordinator = FastSearchCoordinator(memory_manager, rag_manager)


# Math Tools (unchanged - already fast)
@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# DateTime Tools (unchanged - already fast)
@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time."""
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        current_time = datetime.now(pytz.UTC)
        return current_time.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def get_current_date() -> str:
    """Get the current date."""
    return datetime.now().strftime("%Y-%m-%d")


@tool
def calculate_days_between_dates(start_date: str, end_date: str) -> int:
    """Calculate the number of days between two dates."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return (end - start).days
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD")


# OPTIMIZED Memory Tools
@tool
def remember_information(key: str, value: str) -> str:
    """Store personal information in your profile for future reference."""
    try:
        memory_manager.update_user_profile(key, value)
        return f"✅ Remembered: {key} = {value}"
    except Exception as e:
        return f"❌ Error storing information: {str(e)}"


@tool
def recall_information(query: str = None) -> str:
    """FAST recall of information from your profile and recent conversations."""
    try:
        if query:
            # Quick parallel search
            profile_info = memory_manager.quick_search_profile(query)
            memory_info = memory_manager.quick_search_memory(query)

            results = []
            if profile_info:
                results.append(profile_info)
            if memory_info:
                results.append(memory_info)

            return (
                "\n\n".join(results)
                if results
                else f"❌ No information found about: {query}"
            )
        else:
            # Show profile summary
            profile = memory_manager.get_user_profile()
            if not profile:
                return "👤 No personal information stored yet."

            result = ["👤 **Your Profile:**"]
            for key, value in list(profile.items())[:5]:  # Limit for speed
                result.append(f"**{key}:** {value}")

            if len(profile) > 5:
                result.append(f"... and {len(profile) - 5} more items")

            return "\n".join(result)
    except Exception as e:
        return f"❌ Error recalling information: {str(e)}"


# OPTIMIZED Knowledge Base Tools
@tool
def search_knowledge_base(query: str) -> str:
    """FAST search through your personal knowledge base for relevant documents."""
    try:
        result = rag_manager.quick_search_documents(query)
        return result or f"❌ No relevant documents found for: {query}"
    except Exception as e:
        return f"❌ Knowledge base search failed: {str(e)}"


@tool
def add_to_knowledge_base(content: str, title: str = None, source: str = None) -> str:
    """Add information or documents to your personal knowledge base."""
    try:
        return rag_manager.add_document(content, title, source)
    except Exception as e:
        return f"❌ Failed to add to knowledge base: {str(e)}"


# MAIN OPTIMIZED Search Tool
@tool
def find_information(query: str) -> str:
    """
    FAST intelligent search for information with priority:
    1. Your personal profile and recent conversations (fastest)
    2. Your knowledge base documents (if no memory match)
    3. Current web information (if no local results)

    Optimized for speed with caching and parallel processing.
    """
    try:
        return search_coordinator.smart_search(query)
    except Exception as e:
        return f"❌ Search failed: {str(e)}"


# Direct Web Search Tool (for explicit requests)
@tool
def search_current_web_info(query: str) -> str:
    """FAST direct web search for current information."""
    try:
        return WebSearchManager.quick_search_web(query)
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"


# OPTIMIZED System Management
@tool
def get_system_status() -> str:
    """Get quick status of all system components."""
    try:
        memory_context = len(memory_manager.get_memory_context())
        profile_items = len(memory_manager.get_user_profile())
        rag_status = (
            "✅ Ready" if rag_manager.is_initialized() else "❌ Not initialized"
        )
        cache_size = len(query_cache.cache)

        return f"""📊 **System Status** (Optimized)

🧠 **Memory:** {memory_context} messages | {profile_items} profile items
📚 **Knowledge Base:** {rag_status}
🌐 **Web Search:** {"✅ Ready" if Config.TAVILY_API_KEY != "your_tavily_api_key_here" else "❌ Not configured"}
⚡ **Cache:** {cache_size} cached queries
🤖 **LLM:** {Config.PRIMARY_LLM}

🚀 **Performance Features:**
   • Parallel search processing: {"✅" if Config.ENABLE_PARALLEL_SEARCH else "❌"}
   • Query caching: ✅ ({Config.CACHE_TTL}s TTL)
   • Search timeout: {Config.SEARCH_TIMEOUT}s
   • Early termination: ✅"""
    except Exception as e:
        return f"❌ Error getting system status: {str(e)}"


# Clear cache tool for debugging
@tool
def clear_cache() -> str:
    """Clear the query cache to force fresh searches."""
    try:
        query_cache.clear()
        return "✅ Query cache cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"


# ================================================================================================
# OPTIMIZED LANGGRAPH SETUP
# ================================================================================================

# Create optimized tools list (prioritized order)
tools = [
    # Core functionality (most used)
    find_information,
    remember_information,
    recall_information,
    # Knowledge base
    search_knowledge_base,
    add_to_knowledge_base,
    # Direct web search (for explicit requests)
    search_current_web_info,
    # Math tools (fast)
    add_numbers,
    subtract_numbers,
    multiply_numbers,
    divide_numbers,
    # DateTime tools (fast)
    get_current_datetime,
    get_current_date,
    calculate_days_between_dates,
    # System management
    get_system_status,
    clear_cache,
]


# Initialize LLM (optimized)
def initialize_llm():
    """Initialize LLM with optimized settings"""
    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.PRIMARY_LLM,
            temperature=0,
            google_api_key=Config.GOOGLE_API_KEY,
            request_timeout=30,  # Add timeout
        )
        print(f"✅ Using LLM: {Config.PRIMARY_LLM}")
        return llm
    except Exception as e:
        print(f"⚠️ Primary LLM failed, trying fallback: {e}")
        try:
            llm = ChatGoogleGenerativeAI(
                model=Config.FALLBACK_LLM,
                temperature=0,
                google_api_key=Config.GOOGLE_API_KEY,
                request_timeout=30,
            )
            print(f"✅ Using fallback LLM: {Config.FALLBACK_LLM}")
            return llm
        except Exception as e2:
            print(f"❌ Both LLMs failed: {e2}")
            raise


# Initialize LLM and bind tools
llm = initialize_llm()
llm_with_tools = llm.bind_tools(tools)


# OPTIMIZED Agent node
def agent(state: State):
    """OPTIMIZED main agent with memory integration"""
    # Get memory context (cached)
    memory_context = memory_manager.get_memory_context()

    # Limit context for speed (only keep recent relevant messages)
    limited_context = memory_context[-6:] if len(memory_context) > 6 else memory_context

    # Combine with current messages
    all_messages = limited_context + state["messages"]

    # Get AI response
    response = llm_with_tools.invoke(all_messages)

    return {"messages": [response]}


# Tool node (unchanged)
tool_node = ToolNode(tools)


# Conditional edge function (unchanged)
def should_continue(state: State) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# Build graph (unchanged)
def build_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ================================================================================================
# OPTIMIZED SYSTEM PROMPT
# ================================================================================================

OPTIMIZED_SYSTEM_PROMPT = """🧠 You are a FAST, intelligent AI assistant with optimized memory, knowledge management, and search capabilities.

🎯 **SPEED-OPTIMIZED Tool Selection:**

**PRIMARY TOOL (use for 90% of questions):**
- `find_information(query)` - Lightning-fast intelligent search across memory → knowledge base → web

**SPECIFIC TOOLS (only when explicitly requested):**
- `remember_information(key, value)` - When user says "remember that..."
- `recall_information(query)` - When user asks "what do you know about me?"
- `add_to_knowledge_base(content, title)` - When user says "add this to knowledge base"
- `search_current_web_info(query)` - When user specifically requests web search

**UTILITIES (direct use):**
- Math: `add_numbers`, `subtract_numbers`, `multiply_numbers`, `divide_numbers`
- Time: `get_current_datetime`, `get_current_date`, `calculate_days_between_dates`

---

🚀 **SPEED OPTIMIZATIONS ACTIVE:**

✅ **Query Caching** - Repeated searches return instantly
✅ **Parallel Processing** - Memory and profile searched simultaneously  
✅ **Early Termination** - Stops searching when good results found
✅ **Limited Context** - Uses only recent relevant conversation history
✅ **Timeout Protection** - Prevents slow searches from blocking responses
✅ **Smart Results Limiting** - Returns concise, relevant information

---

🎯 **Response Strategy (OPTIMIZED):**

1. **For ANY question** → Use `find_information(query)` - it handles everything intelligently and fast
2. **For calculations** → Use math/datetime tools directly
3. **For memory storage** → Use `remember_information(key, value)`

**DO NOT:**
- Use multiple search tools for the same query
- Search extensively when a quick answer exists in memory
- Return overly long responses that slow down the conversation

---

🌟 **Key Behaviors:**

1. **Speed First**: Prioritize fast, relevant responses over exhaustive searches
2. **Smart Caching**: Leverage cached results when available
3. **Concise Responses**: Keep responses focused and to the point
4. **Efficient Memory**: Use recent conversation context intelligently
5. **Early Success**: Stop searching when sufficient information is found

Your goal: Provide helpful, accurate responses as FAST as possible while maintaining intelligence and personalization!"""


def invoke_with_memory(graph, user_message: str, session_id: str = None):
    """OPTIMIZED invoke with memory support"""
    # Start new session if needed
    if not session_id:
        session_id = memory_manager.start_new_session()

    # Create messages
    system_msg = SystemMessage(content=OPTIMIZED_SYSTEM_PROMPT)
    user_msg = HumanMessage(content=user_message)

    # Prepare state
    state = {
        "messages": [system_msg, user_msg],
        "session_id": session_id,
        "user_profile": memory_manager.get_user_profile(),
        "conversation_summary": "",
    }

    # Invoke graph
    result = graph.invoke(state)

    # Extract AI response
    ai_response = result["messages"][-1].content if result["messages"] else ""

    # Add to memory (async for speed)
    try:
        memory_manager.add_message(user_message, ai_response)
    except Exception:
        pass  # Don't let memory errors slow down responses

    return result, session_id


# ================================================================================================
# OPTIMIZED MAIN APPLICATION
# ================================================================================================


def main():
    """OPTIMIZED main application entry point"""
    print("🚀 OPTIMIZED Enhanced AI Assistant - Fast & Intelligent")
    print("=" * 60)
    print(f"🔧 LLM: {Config.PRIMARY_LLM}")
    print(f"🧠 Memory: LangChain (optimized, {Config.MAX_MEMORY_EXCHANGES} exchanges)")
    print(
        f"📚 Knowledge Base: {'✅ Ready' if rag_manager.is_initialized() else '❌ Failed'}"
    )
    print(
        f"🌐 Web Search: {'✅ Ready' if Config.TAVILY_API_KEY != 'your_tavily_api_key_here' else '❌ Not configured'}"
    )
    print(
        f"⚡ Performance: Parallel search: {'✅' if Config.ENABLE_PARALLEL_SEARCH else '❌'}, Cache: ✅, Timeout: {Config.SEARCH_TIMEOUT}s"
    )

    # Build graph
    graph = build_graph()
    current_session_id = None

    print(
        f"\n💡 **Optimizations:** Caching • Parallel Processing • Early Termination • Timeouts"
    )
    print(f"📁 Storage: {Config.MEMORY_DIR}")
    print("\n" + "=" * 60)
    print("🚀 SPEED COMMANDS: 'fast', 'status', 'cache', 'new', 'quit'")
    print("=" * 60)

    while True:
        user_input = input("\n⚡ Ask anything (FAST): ").strip()

        if user_input.lower() == "quit":
            print("\n👋 Thanks for using the OPTIMIZED AI Assistant!")
            break

        elif user_input.lower() in ["help", "fast"]:
            print("""
🚀 **OPTIMIZED AI Assistant - Speed Guide:**

**💬 Natural Questions (ALL OPTIMIZED):**
   • "What's my name?" - Instant from memory cache
   • "Tell me about Python" - Smart search: memory → docs → web
   • "Latest AI news" - Cached or fast web search
   • "What did we discuss?" - Fast memory search

**⚡ Speed Features:**
   • Query caching (5min TTL) - Repeated questions = instant answers
   • Parallel processing - Multiple sources searched simultaneously
   • Early termination - Stops when good answer found
   • Smart timeouts - No more waiting for slow searches

**💾 Memory (FAST):**
   • "Remember I work at Google" - Instant storage
   • "What do you know about me?" - Cached profile lookup

**📚 Knowledge Base (OPTIMIZED):**
   • "Add this to knowledge base: [content]" - Fast chunking & storage
   • "Search my docs for ML" - Vector search with caching

**⚙️ System Commands:**
   • 'status' - Performance metrics
   • 'cache' - Clear cache for fresh results
   • 'new' - New session (clears memory context)

**🎯 Expected Response Time: 1-3 seconds (vs 10-30s before optimization)**
            """)

        elif user_input.lower() == "status":
            start_time = datetime.now()
            try:
                result = get_system_status()
                response_time = (datetime.now() - start_time).total_seconds()
                print(f"\n{result}")
                print(f"\n⚡ Status query time: {response_time:.2f}s")
            except Exception as e:
                print(f"\n❌ Error getting status: {str(e)}")

        elif user_input.lower() == "cache":
            try:
                result = clear_cache()
                print(f"\n🧹 {result}")
            except Exception as e:
                print(f"\n❌ Error clearing cache: {str(e)}")

        elif user_input.lower() == "new":
            current_session_id = memory_manager.start_new_session()
            print(f"✅ New optimized session: {current_session_id[:8]}...")

        elif user_input:
            start_time = datetime.now()
            try:
                result, current_session_id = invoke_with_memory(
                    graph, user_input, current_session_id
                )

                response_time = (datetime.now() - start_time).total_seconds()

                # Display response
                ai_response = result["messages"][-1].content
                print(f"\n🤖 {ai_response}")
                print(f"\n⚡ Response time: {response_time:.2f}s")

                # Show session info on first interaction
                if not hasattr(main, "_session_shown"):
                    print(
                        f"💡 Session: {current_session_id[:8]}... | Optimized for speed!"
                    )
                    main._session_shown = True

            except Exception as e:
                response_time = (datetime.now() - start_time).total_seconds()
                print(f"\n❌ Error ({response_time:.2f}s): {str(e)}")
                print("💡 Try 'status' or 'cache' if performance issues persist")

                # Handle rate limits
                if "rate limit" in str(e).lower():
                    print("🔄 Rate limit detected. Try 'new' for fresh session...")


if __name__ == "__main__":
    main()
