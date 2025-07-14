import json
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from utils import query_cache


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


memory_manager = MemoryManager()
