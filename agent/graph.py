"""
LangGraph graph construction for the Seller Listing Assistant.

Graph topology:
  START
    └─► compress_context
          └─► classify_intent
                ├─► process_new_listing ──┐
                ├─► process_correction  ──┤
                └─► (general)            ┤
                                         ▼
                                   generate_response
                                         └─► END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    classify_intent,
    compress_context,
    generate_response,
    process_correction,
    process_new_listing,
)
from .state import AgentState


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def _route_intent(state: AgentState) -> str:
    """Route after intent classification."""
    intent = state.get("next_action", "general")
    listing_state = state.get("listing_state")

    if intent == "new_listing":
        return "process_new_listing"

    if intent == "correction":
        # Only enter correction path if there is an active correction_loop
        if listing_state and listing_state.get("status") == "correction_loop":
            return "process_correction"
        # Otherwise treat as new listing attempt or general message
        return "process_new_listing" if listing_state is None else "generate_response"

    return "generate_response"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("compress_context", compress_context)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("process_new_listing", process_new_listing)
    graph.add_node("process_correction", process_correction)
    graph.add_node("generate_response", generate_response)

    # Edges
    graph.add_edge(START, "compress_context")
    graph.add_edge("compress_context", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "process_new_listing": "process_new_listing",
            "process_correction": "process_correction",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("process_new_listing", "generate_response")
    graph.add_edge("process_correction", "generate_response")
    graph.add_edge("generate_response", END)

    return graph


# Compiled singleton — import this in main.py
compiled_graph = build_graph().compile()
