import os
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.config import Config
from utils import query_cache, timeout_wrapper


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
            print("RAG system initialized")
        except Exception as e:
            print(f"RAG initialization failed: {e}")

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


rag_manager = RAGManager()
