import requests

from config.config import Config
from utils import query_cache, timeout_wrapper

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
