"""
Structured trace logging — every conversation turn emits one trace record.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRACES_DIR = Path(__file__).parent.parent / "traces"


def _ensure_dir() -> None:
    _TRACES_DIR.mkdir(exist_ok=True)


def build_trace(
    *,
    session_id: str,
    turn: int,
    user_message: str,
    tool_calls: list[dict],
    assistant_message: str,
    listing_state: Any,
) -> dict:
    """Construct a structured turn trace record."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "turn": turn,
        "user": user_message,
        "tool_calls": tool_calls,
        "assistant": assistant_message,
        "listing_state": listing_state,
    }


def append_trace(session_id: str, trace: dict) -> None:
    """Append a trace record to the session's JSONL file."""
    _ensure_dir()
    path = _TRACES_DIR / f"session_{session_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(trace) + "\n")
