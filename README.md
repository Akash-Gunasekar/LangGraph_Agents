# 🧠 LangGraph Agents: Ultra-Fast Modular AI Agent Framework

Welcome to **LangGraph Agents** — a next-generation, blazing-fast, modular AI agent framework that fuses memory, RAG, web search, and tool orchestration for intelligent, context-aware automation. Built for extensibility, speed, and real-world utility.

---

## ✨ Why Is This Unique?

- **True Parallel Search:** Simultaneously queries memory, profile, knowledge base, and the web for instant answers.
- **Early Termination:** Stops searching as soon as a high-quality answer is found.
- **Smart Caching:** All queries are cached for lightning-fast repeated access.
- **Personalized Memory:** User profile and recent conversations are always at your agent's fingertips.
- **RAG-Driven Knowledge:** Integrates vector search for deep, document-level retrieval.
- **Web-Connected:** Seamlessly pulls in up-to-date information from the web (Tavily API).
- **Extensible Tooling:** Add your own tools (math, datetime, system, etc.) in seconds.
- **Timeout Protection:** Never hangs—every search is time-bounded for responsiveness.

---

## 🏗️ **How it works:**

- User input is processed by the agent node (LLM with tool bindings).
- If tool calls are needed, the tool node executes the appropriate tool (memory, RAG, web, math, etc.).
- The agent leverages memory, knowledge base, and web search in a prioritized, parallel, and cached manner.
- The process repeats until a final response is ready—always as fast as possible.

---

## 🚀 Core Innovations

| Component                                      | Description                                                    |
| :--------------------------------------------- | :------------------------------------------------------------- |
| **LangGraph Orchestration**              | State machine graph for robust, flexible agent workflows.      |
| **LangChain Memory**                     | Windowed conversation memory and persistent user profile.      |
| **RAG (Retrieval-Augmented Generation)** | Chroma vector DB + Google embeddings for deep document search. |
| **Web Search**                           | Tavily API for real-time, up-to-date answers.                  |
| **Custom Tools**                         | Add your own tools (math, datetime, system, etc.) in seconds.  |
| **Performance First**                    | Caching, parallelism, and timeouts everywhere.                 |

---

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Akash-Gunasekar/LangGraph_Agents.git
   cd LangGraph_Agents
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```
3. **Install dependencies:**

   ```bash
   pip install langchain langgraph chromadb tavily-python
   ```

---

## 🔑 Configuration

This project requires API keys for certain services.

1. **Tavily API Key:**

   - Sign up at [Tavily AI](https://tavily.com/) to get your API key.
   - Set it as an environment variable:

     ```bash
     export TAVILY_API_KEY="your_tavily_api_key_here"
     ```

     (On Windows, use `set TAVILY_API_KEY="your_tavily_api_key_here"`)
2. **Google API Key (for embeddings):**

   - Obtain a Google API key from the [Google AI Sudio](https://aistudio.google.com/apikey/).
   - Enable the Gemini API.
   - Set it as an environment variable:

     ```bash
     export GOOGLE_API_KEY="your_google_api_key_here"
     ```

     (On Windows, use `set GOOGLE_API_KEY="your_google_api_key_here"`)

---

## 🚀 Usage

To start the interactive CLI agent, run:

```bash
python main.py
```

You can then interact with the agent by typing your queries.

---

## 🧩 Project Structure

```text
LangGraph_Agents/

├── main.py                # CLI entry point, session & user interaction
├── graph.py               # LangGraph workflow definition
├── memory/
│   └── memory_manager.py  # Optimized memory/profile management
├── rag_system.py          # RAG (vector search) manager
├── tools/
│   └── custom_tools.py    # All agent tools (search, math, datetime, etc.)
├── web_search.py          # Web search integration (Tavily API)
├── config/
│   └── config.py          # Centralized configuration
├── prompts/
│   └── system_prompt.py   # Optimized system prompt
├── utils.py               # Caching, timeouts, helpers
└── ...
```

---

## ⚡ Example Interactions

- **Ask anything:**

  - "What do you know about my recent conversations?"
  - "Add this document to my knowledge base."
  - "Search the web for the latest AI news."

---

## 🛠️ Extending & Customizing

- Add new tools in `tools/custom_tools.py` (just decorate with `@tool`)
- Update RAG logic in `rag_system.py`
- Customize memory/profile in `memory/memory_manager.py`
- Adjust workflow in `graph.py`

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Tavily Web Search](https://docs.tavily.com/)
- [Chroma Vector DB](https://www.trychroma.com/)

**Made with ❤️ by Akash Gunasekar**

*Star ⭐ this repo if it helped you build something awesome!*
