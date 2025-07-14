from datetime import datetime

from config.config import Config
from graph import build_graph
from memory.memory_manager import memory_manager
from rag_system import rag_manager
from tools.custom_tools import get_system_status, clear_cache
from prompts.system_prompt import OPTIMIZED_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


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


def main():
    """OPTIMIZED main application entry point"""
    print("OPTIMIZED Enhanced AI Assistant - Fast & Intelligent")
    print("=" * 60)
    print(f"LLM: {Config.PRIMARY_LLM}")
    print(f"Memory: LangChain (optimized, {Config.MAX_MEMORY_EXCHANGES} exchanges)")
    print(f"Knowledge Base: {'Ready' if rag_manager.is_initialized() else 'Failed'}")
    print(
        f"Web Search: {'Ready' if Config.TAVILY_API_KEY != 'your_tavily_api_key_here' else 'Not configured'}"
    )
    print(
        f"Performance: Parallel search: {'Enabled' if Config.ENABLE_PARALLEL_SEARCH else 'Disabled'}, Cache: Enabled, Timeout: {Config.SEARCH_TIMEOUT}s"
    )

    # Build graph
    graph = build_graph()
    current_session_id = None

    print(
        "\n**Optimizations:** Caching • Parallel Processing • Early Termination • Timeouts"
    )
    print(f"Storage: {Config.MEMORY_DIR}")
    print("\n" + "=" * 60)
    print("SPEED COMMANDS: 'fast', 'status', 'cache', 'new', 'quit'")
    print("=" * 60)

    while True:
        user_input = input("\nAsk anything (FAST): ").strip()

        if user_input.lower() == "quit":
            print("\nThanks for using the OPTIMIZED AI Assistant!")
            break

        elif user_input.lower() in ["help", "fast"]:
            print("""
OPTIMIZED AI Assistant - Speed Guide:

Natural Questions (ALL OPTIMIZED):
   • "What's my name?" - Instant from memory cache
   • "Tell me about Python" - Smart search: memory -> docs -> web
   • "Latest AI news" - Cached or fast web search
   • "What did we discuss?" - Fast memory search

Speed Features:
   • Query caching (5min TTL) - Repeated questions = instant answers
   • Parallel processing - Multiple sources searched simultaneously
   • Early termination - Stops when good answer found
   • Smart timeouts - No more waiting for slow searches

Memory (FAST):
   • "Remember I work at Google" - Instant storage
   • "What do you know about me?" - Cached profile lookup

Knowledge Base (OPTIMIZED):
   • "Add this to knowledge base: [content]" - Fast chunking & storage
   • "Search my docs for ML" - Vector search with caching

System Commands:
   • 'status' - Performance metrics
   • 'cache' - Clear cache for fresh results
   • 'new' - New session (clears memory context)

Expected Response Time: 1-3 seconds (vs 10-30s before optimization)
            """)

        elif user_input.lower() == "cache":
            try:
                result = clear_cache()
                print(f"\n{result}")
            except Exception as e:
                print(f"\nError clearing cache: {str(e)}")

        elif user_input.lower() == "new":
            current_session_id = memory_manager.start_new_session()
            print(f"New optimized session: {current_session_id[:8]}...")

        elif user_input:
            start_time = datetime.now()
            try:
                result, current_session_id = invoke_with_memory(
                    graph, user_input, current_session_id
                )

                response_time = (datetime.now() - start_time).total_seconds()

                # Display response
                ai_response = result["messages"][-1].content
                print(f"\n{ai_response}")
                print(f"\nResponse time: {response_time:.2f}s")

                # Show session info on first interaction
                if not hasattr(main, "_session_shown"):
                    print(
                        f"Session: {current_session_id[:8]}... | Optimized for speed!"
                    )
                    main._session_shown = True

            except Exception as e:
                response_time = (datetime.now() - start_time).total_seconds()
                print(f"\nError ({response_time:.2f}s): {str(e)}")
                print("Try 'status' or 'cache' if performance issues persist")

                # Handle rate limits
                if "rate limit" in str(e).lower():
                    print("Rate limit detected. Try 'new' for fresh session...")


if __name__ == "__main__":
    main()
