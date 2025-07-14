import pytz
from datetime import datetime
from langchain_core.tools import tool
from memory.memory_manager import memory_manager
from rag_system import rag_manager
from search_coordinator import FastSearchCoordinator
from web_search import WebSearchManager
from utils import query_cache
from config.config import Config

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
        return f"Remembered: {key} = {value}"
    except Exception as e:
        return f"Error storing information: {str(e)}"


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
                else f"No information found about: {query}"
            )
        else:
            # Show profile summary
            profile = memory_manager.get_user_profile()
            if not profile:
                return "No personal information stored yet."

            result = ["Your Profile:"]
            for key, value in list(profile.items())[:5]:  # Limit for speed
                result.append(f"**{key}:** {value}")

            if len(profile) > 5:
                result.append(f"... and {len(profile) - 5} more items")

            return "\n".join(result)
    except Exception as e:
        return f"Error recalling information: {str(e)}"


# OPTIMIZED Knowledge Base Tools
@tool
def search_knowledge_base(query: str) -> str:
    """FAST search through your personal knowledge base for relevant documents."""
    try:
        result = rag_manager.quick_search_documents(query)
        return result or f"No relevant documents found for: {query}"
    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"


@tool
def add_to_knowledge_base(content: str, title: str = None, source: str = None) -> str:
    """Add information or documents to your personal knowledge base."""
    try:
        return rag_manager.add_document(content, title, source)
    except Exception as e:
        return f"Failed to add to knowledge base: {str(e)}"


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
        return f"Search failed: {str(e)}"


# Direct Web Search Tool (for explicit requests)
@tool
def search_current_web_info(query: str) -> str:
    """FAST direct web search for current information."""
    try:
        return WebSearchManager.quick_search_web(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# OPTIMIZED System Management
@tool
def get_system_status() -> str:
    """Get quick status of all system components."""
    try:
        memory_context = len(memory_manager.get_memory_context())
        profile_items = len(memory_manager.get_user_profile())
        rag_status = "Ready" if rag_manager.is_initialized() else "Not initialized"
        cache_size = len(query_cache.cache)

        return f"""System Status (Optimized)

Memory: {memory_context} messages | {profile_items} profile items
Knowledge Base: {rag_status}
Web Search: {"Ready" if Config.TAVILY_API_KEY != "your_tavily_api_key_here" else "Not configured"}
Cache: {cache_size} cached queries
LLM: {Config.PRIMARY_LLM}

Performance Features:
   • Parallel search processing: {"Enabled" if Config.ENABLE_PARALLEL_SEARCH else "Disabled"}
   • Query caching: Enabled ({Config.CACHE_TTL}s TTL)
   • Search timeout: {Config.SEARCH_TIMEOUT}s
   • Early termination: Enabled"""
    except Exception as e:
        return f"Error getting system status: {str(e)}"


# Clear cache tool for debugging
@tool
def clear_cache() -> str:
    """Clear the query cache to force fresh searches."""
    try:
        query_cache.clear()
        return "Query cache cleared successfully!"
    except Exception as e:
        return f"Error clearing cache: {str(e)}"


def get_tools():
    """Returns a list of all available tools."""
    return [
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
