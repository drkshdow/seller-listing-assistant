"""
Mock backend tools for the Seller Listing Assistant.

Each tool is implemented as a plain Python function AND registered as a
LangChain @tool so the LLM can call them via function-calling.

Adding a new tool:
  1. Implement the function here.
  2. Decorate with @tool (or add to TOOL_REGISTRY manually).
  3. The graph will automatically pick it up from TOOL_REGISTRY.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Load static data once at import time
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent
_CATEGORY_RULES: dict = json.loads((_DATA_DIR / "category_rules.json").read_text())
_PROHIBITED: dict = json.loads((_DATA_DIR / "prohibited_items.json").read_text())

# In-memory listing store: listing_id -> dict
_LISTING_STORE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _generate_listing_id() -> str:
    suffix = "".join(random.choices(string.digits, k=5))
    return f"LST-{suffix}"


def _get_nested(obj: dict, path: str) -> Any:
    """Retrieve a dot-notation field from a nested dict."""
    parts = path.split(".")
    for part in parts:
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def _set_nested(obj: dict, path: str, value: Any) -> None:
    """Set a dot-notation field in a nested dict (mutates in place)."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


def store_listing(listing: dict) -> str:
    """Persist a listing dict in-memory and return its ID."""
    listing_id = listing.get("listing_id") or _generate_listing_id()
    listing["listing_id"] = listing_id
    _LISTING_STORE[listing_id] = listing
    return listing_id


def get_listing(listing_id: str) -> dict | None:
    return _LISTING_STORE.get(listing_id)


# ---------------------------------------------------------------------------
# Tool 1 — validate_listing
# ---------------------------------------------------------------------------

@tool
def validate_listing(listing: dict) -> dict:
    """
    Validate a product listing against category-specific field requirements.
    Returns a result dict with passed status and any missing/invalid fields.
    """
    listing_id = listing.get("listing_id", "UNKNOWN")
    category = listing.get("category", "")
    rules = _CATEGORY_RULES.get(category)

    missing_fields: list[str] = []
    errors: list[str] = []

    if rules is None:
        # Category missing or invalid - still check universal required fields
        errors.append(f"Unknown or unsupported category: '{category}'. Supported categories: {list(_CATEGORY_RULES.keys())}")
        
        # Check universal required fields even without category rules
        universal_required = ["title", "description", "category", "price", "images"]
        for field in universal_required:
            value = listing.get(field)
            if value is None or value == "" or value == []:
                missing_fields.append(field)
        
        # Check for images count
        images = listing.get("images", [])
        if len(images) < 2:  # Minimum 2 images as universal rule
            if "images" not in missing_fields:  # Don't duplicate if already marked missing
                errors.append(f"At least 2 images required; {len(images)} provided.")
        
        # Check for attributes (at least some should exist)
        attributes = listing.get("attributes", {})
        if not attributes or not isinstance(attributes, dict) or len(attributes) == 0:
            missing_fields.append("attributes")
        
        return {
            "listing_id": listing_id,
            "stage": "validation",
            "passed": False,
            "missing_fields": missing_fields,
            "errors": errors,
        }

    # Category is valid - check category-specific requirements
    # Check required fields
    for field in rules["required_fields"]:
        value = _get_nested(listing, field)
        if value is None or value == "" or value == []:
            missing_fields.append(field)

    # Check min images
    images = listing.get("images", [])
    if len(images) < rules["min_images"]:
        errors.append(
            f"At least {rules['min_images']} images required; {len(images)} provided."
        )

    # Check description length
    desc = listing.get("description", "")
    if len(desc) < rules["min_description_length"]:
        errors.append(
            f"Description must be at least {rules['min_description_length']} characters; "
            f"{len(desc)} provided."
        )

    passed = not missing_fields and not errors
    return {
        "listing_id": listing_id,
        "stage": "validation",
        "passed": passed,
        "missing_fields": missing_fields,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Tool 2 — screen_policy
# ---------------------------------------------------------------------------

@tool
def screen_policy(listing: dict) -> dict:
    """
    Screen a listing for policy violations: prohibited keywords,
    restricted categories, and protected brand variants (counterfeit indicators).
    """
    listing_id = listing.get("listing_id", "UNKNOWN")
    violations: list[dict] = []

    # Concatenate all text fields for keyword scanning
    text_corpus = " ".join(
        filter(None, [
            listing.get("title", ""),
            listing.get("description", ""),
            str(listing.get("attributes", {})),
        ])
    ).lower()

    # Check prohibited keywords
    for kw in _PROHIBITED["prohibited_keywords"]:
        if kw.lower() in text_corpus:
            violations.append({
                "type": "prohibited_keyword",
                "detail": f"Listing contains prohibited keyword: '{kw}'",
                "field": "title/description",
            })

    # Check restricted categories
    category = listing.get("category", "").lower()
    for rc in _PROHIBITED["restricted_categories"]:
        if rc.lower() in category:
            violations.append({
                "type": "restricted_category",
                "detail": f"Category '{listing.get('category')}' is restricted.",
                "field": "category",
            })

    # Check protected brand variants (counterfeit detection)
    brand_value = str(_get_nested(listing, "attributes.brand") or "").lower()
    for entry in _PROHIBITED["protected_brands"]:
        for variant in entry["known_variants"]:
            if variant.lower() == brand_value:
                violations.append({
                    "type": "counterfeit_brand",
                    "detail": (
                        f"Brand '{brand_value}' appears to be a variant of protected brand "
                        f"'{entry['original']}'. Counterfeit indicators are not allowed."
                    ),
                    "field": "attributes.brand",
                })

    passed = len(violations) == 0
    return {
        "listing_id": listing_id,
        "stage": "policy_screening",
        "violations": violations,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Tool 3 — update_listing
# ---------------------------------------------------------------------------

@tool
def update_listing(listing_id: str, field_path: str, new_value: Any) -> dict:
    """
    Apply a partial update to an in-memory listing using dot-notation field path.
    E.g. field_path='attributes.brand', new_value='SoundMax'
    """
    listing = _LISTING_STORE.get(listing_id)
    if listing is None:
        return {"error": f"Listing '{listing_id}' not found.", "updated": False}

    old_value = _get_nested(listing, field_path)
    _set_nested(listing, field_path, new_value)

    return {
        "listing_id": listing_id,
        "field_path": field_path,
        "old_value": old_value,
        "new_value": new_value,
        "updated": True,
    }


# ---------------------------------------------------------------------------
# Tool 4 — publish_listing
# ---------------------------------------------------------------------------

@tool
def publish_listing(listing_id: str) -> dict:
    """Publish a listing to the catalog after all checks pass."""
    listing = _LISTING_STORE.get(listing_id)
    if listing is None:
        return {"error": f"Listing '{listing_id}' not found.", "status": "error"}

    listing["_status"] = "published"
    return {
        "listing_id": listing_id,
        "status": "published",
        "message": "Your listing is now live on the marketplace.",
    }


# ---------------------------------------------------------------------------
# Tool 5 — escalate_to_reviewer
# ---------------------------------------------------------------------------

@tool
def escalate_to_reviewer(listing_id: str, summary: str) -> dict:
    """
    Hand the listing to a human reviewer when issues cannot be resolved
    conversationally. The summary must describe issues found, corrections
    attempted, and why the agent could not resolve them.
    """
    listing = _LISTING_STORE.get(listing_id)
    if listing is None:
        return {"error": f"Listing '{listing_id}' not found."}

    listing["_status"] = "escalated"
    ticket_suffix = "".join(random.choices(string.digits, k=5))
    ticket_id = f"RVW-{ticket_suffix}"

    return {
        "listing_id": listing_id,
        "ticket_id": ticket_id,
        "message": (
            "Your listing has been sent for manual review. "
            "A marketplace specialist will review it within 24 hours."
        ),
    }


# ---------------------------------------------------------------------------
# Tool registry — single source of truth for all tools
# ---------------------------------------------------------------------------

TOOL_REGISTRY: list = [
    validate_listing,
    screen_policy,
    update_listing,
    publish_listing,
    escalate_to_reviewer,
]

TOOL_BY_NAME: dict = {t.name: t for t in TOOL_REGISTRY}
