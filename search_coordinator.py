import concurrent.futures

from config.config import Config
from memory.memory_manager import MemoryManager
from rag_system import RAGManager
from web_search import WebSearchManager

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
