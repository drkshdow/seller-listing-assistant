# Seller Listing Assistant

A conversational AI agent that helps e-commerce sellers get their product
listings published by guiding them through validation, policy compliance, and
content corrections.

---

## Quick Start

### 1. Install dependencies

```bash
cd seller_listing_assistent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the agent

```bash
python main.py
```

Enable verbose traces:

```bash
DEBUG=true python main.py
```

Traces are also written to `traces/session_<id>.jsonl` automatically.

---

## Architecture

```
seller_listing_assistent/
├── agent/
│   ├── config.py      # Environment settings and tuning knobs
│   ├── tools.py       # Mock backend tools (validate, policy, update, publish, escalate)
│   ├── state.py       # LangGraph TypedDict state definitions
│   ├── nodes.py       # All graph node implementations
│   ├── graph.py       # Graph topology and compiled singleton
│   └── trace.py       # Structured per-turn trace logging
├── main.py            # Interactive CLI entry point
├── category_rules.json
├── prohibited_items.json
├── requirements.txt
└── traces/            # Auto-created; one .jsonl per session
```

### LangGraph Topology

```
START
  └─► compress_context        (trims/summarises old messages)
        └─► classify_intent   (LLM: new_listing | correction | general)
              ├─► process_new_listing ──┐
              ├─► process_correction   ─┤
              └─► (general)            ─┤
                                        ▼
                                 generate_response
                                        └─► END
```

### Node Responsibilities

| Node | Responsibility |
|---|---|
| `compress_context` | Rolling summarisation — keeps context window bounded |
| `classify_intent` | LLM classifies intent: new_listing / correction / general |
| `process_new_listing` | Extracts listing from NL or JSON, closes any prior in-progress listing, runs validate + policy checks, triages issues |
| `process_correction` | Parses NL corrections into field updates, applies them, re-runs checks, publishes or escalates |
| `generate_response` | LLM drafts the natural-language reply with full state context |

---

## Context Window Management

**Strategy: Sliding window + rolling LLM summary**

- The last `MAX_RECENT_MESSAGES` (default 10) messages are kept in full.
- When history exceeds this limit, `compress_context` asks the LLM to summarise
  the oldest messages into concise bullet points stored in `conversation_summary`.
- The summary is prepended to every subsequent system prompt.

**Rationale:** Pure truncation loses critical earlier context. Full summarisation
every turn is expensive. The sliding window with on-demand summarisation balances
cost, latency, and fidelity.

---

## Key Design Decisions

### 1. Programmatic triage, not LLM triage
Issue counting and the >3 / ≤3 branch are enforced in Python — not by prompting
the LLM — so the triage is deterministic and cannot be bypassed via prompt injection.

### 2. Field-path allowlist
Before calling `update_listing`, every `field_path` produced by the LLM is checked
against a hardcoded allowlist. Unknown paths are rejected to prevent injection attacks.

### 3. Anti-jailbreak system prompt
The system prompt states that validation is mandatory and cannot be waived.
The LLM is instructed never to claim a listing passed unless tools confirm it.

### 4. Tool registry pattern
All tools live in `agent/tools.py` as `@tool`-decorated functions exported via
`TOOL_REGISTRY`. Adding a new tool requires only implementing the function and
decorating it — no other file changes needed.

### 5. Previous listing handling
When a seller submits a new listing while one is still open, the in-progress
listing is automatically escalated (not silently dropped), and the seller is
notified of the escalation ticket before the new listing is processed.

---

## Structured Trace Format

Every turn appends one JSON record to `traces/session_<session_id>.jsonl`:

```json
{
  "timestamp": "2026-04-22T10:30:00Z",
  "session_id": "SES-A1B2C3",
  "turn": 3,
  "user": "Here is my updated title: Premium SoundMax Headphones",
  "tool_calls": [
    {"name": "update_listing",  "args": {"listing_id": "LST-40012", "field_path": "title", "new_value": "..."}, "result": {"updated": true}},
    {"name": "validate_listing","args": {"listing": {}}, "result": {"passed": true, "missing_fields": []}},
    {"name": "screen_policy",   "args": {"listing": {}}, "result": {"passed": true, "violations": []}},
    {"name": "publish_listing", "args": {"listing_id": "LST-40012"},  "result": {"status": "published"}}
  ],
  "assistant": "Great news — your listing is now live on the marketplace!",
  "listing_state": {
    "listing_id": "LST-40012",
    "status": "published",
    "issues": [],
    "correction_rounds": 1
  }
}
```

---

## Correction Round Logic

| Scenario | Agent Behaviour |
|---|---|
| 0 issues after checks | Publish immediately |
| 1–3 issues | Enter conversational correction loop |
| > 3 issues | List all issues, ask seller to resubmit from scratch |
| ≤ 3 issues, rounds exhausted (default: 3) | Escalate to human reviewer |
| New listing submitted while one is in-progress | Close/escalate prior listing, start new one |
