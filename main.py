"""
Seller Listing Assistant — interactive CLI entry point.

Run:
  python main.py

The script maintains per-session state in memory and persists structured
turn traces to traces/session_<id>.jsonl.
"""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy

from langchain_core.messages import HumanMessage

from agent import AgentState, compiled_graph
from agent.config import DEBUG
from agent.trace import append_trace, build_trace

# Configure logging: only show errors from libraries, INFO from main
logging.basicConfig(
    level=logging.ERROR,
    format="%(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def _new_session() -> AgentState:
    return AgentState(
        session_id=f"SES-{uuid.uuid4().hex[:6].upper()}",
        messages=[],
        conversation_summary="",
        listing_state=None,
        next_action="",
        turn=0,
        traces=[],
        pending_tool_calls=[],
    )


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _get_last_ai_message(state: AgentState) -> str:
    from langchain_core.messages import AIMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""


def main() -> None:
    print("=" * 60)
    print("  Seller Listing Assistant")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type 'status' to see the current listing state.")
    print("=" * 60)
    print()

    state = _new_session()
    session_id = state["session_id"]
    print(f"Session ID: {session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "status":
            ls = state.get("listing_state")
            if ls:
                print(json.dumps(
                    {k: v for k, v in ls.items() if not k.startswith("_")},
                    indent=2,
                    default=str,
                ))
            else:
                print("No active listing.")
            print()
            continue

        # Append the human message to state before invoking
        state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
        turn_num = state.get("turn", 0) + 1

        # Invoke graph
        try:
            new_state: AgentState = compiled_graph.invoke(state)
        except Exception as exc:
            logger.error("Error on turn %d: %s", turn_num, str(exc))
            print(
                "Assistant: I encountered an unexpected error. "
                "Please try again or contact support.\n"
            )
            continue

        # --- Build trace ---
        tool_calls = new_state.get("pending_tool_calls", [])
        assistant_reply = _get_last_ai_message(new_state)
        listing_snapshot = deepcopy(
            {k: v for k, v in (new_state.get("listing_state") or {}).items()
             if not k.startswith("_")}
        )

        trace = build_trace(
            session_id=session_id,
            turn=new_state.get("turn", 0),
            user_message=user_input,
            tool_calls=tool_calls,
            assistant_message=assistant_reply,
            listing_state=listing_snapshot,
        )
        append_trace(session_id, trace)

        # Print assistant reply
        print(f"\nAssistant: {assistant_reply}\n")

        # Reset pending tool calls and propagate updated state for next turn
        new_state["pending_tool_calls"] = []
        state = new_state


if __name__ == "__main__":
    main()
