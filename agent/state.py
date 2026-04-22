"""
LangGraph state definitions for the Seller Listing Assistant.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ListingState(TypedDict):
    """Per-listing state tracked across the correction loop."""
    listing_id: str
    listing: dict                   # Current listing dict (mutated by corrections)
    issues: list[dict]              # Active blocking issues after latest check
    corrections_applied: list[dict] # History of every field updated by seller
    correction_rounds: int          # Number of fix-and-recheck cycles completed
    status: str                     # pending | correction_loop | resubmit_requested
                                    # | published | escalated


class AgentState(TypedDict):
    """Full agent state passed through the LangGraph graph each turn."""
    session_id: str
    messages: Annotated[list, add_messages]  # Full conversation messages
    conversation_summary: str                # Summary of older (trimmed) messages
    listing_state: Optional[ListingState]    # None until a listing is submitted
    next_action: str                         # Routing hint set by classify_intent node
    turn: int
    traces: list[dict]                       # Structured per-turn trace records
    pending_tool_calls: list[dict]           # Tool call records for current turn
