from typing import Annotated, TypedDict, Literal, List, Dict, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from config.config import Config
from tools.custom_tools import get_tools
from memory.memory_manager import memory_manager


class State(TypedDict):
    """LangGraph state definition"""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_profile: Dict[str, Any]
    conversation_summary: str


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
        print(f"Using LLM: {Config.PRIMARY_LLM}")
        return llm
    except Exception as e:
        print(f"Primary LLM failed, trying fallback: {e}")
        try:
            llm = ChatGoogleGenerativeAI(
                model=Config.FALLBACK_LLM,
                temperature=0,
                google_api_key=Config.GOOGLE_API_KEY,
                request_timeout=30,
            )
            print(f"Using fallback LLM: {Config.FALLBACK_LLM}")
            return llm
        except Exception as e2:
            print(f"Both LLMs failed: {e2}")
            raise


# Initialize LLM and bind tools
llm = initialize_llm()
llm_with_tools = llm.bind_tools(get_tools())


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
tool_node = ToolNode(get_tools())


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