from datetime import datetime
from typing import Optional
import concurrent.futures

class QueryCache:
    """Simple query cache with TTL"""

    def __init__(self, ttl_seconds=300):
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

def timeout_wrapper(func, timeout_seconds=10):
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