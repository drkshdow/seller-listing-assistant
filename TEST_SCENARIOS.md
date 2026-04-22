# Test Scenarios for Seller Listing Assistant

This document contains test sequences to validate all agent flows.

---

## Scenario 1: Happy Path - Perfect Listing (No Issues)
**Expected Flow**: new_listing → validation passes → publish immediately

```
You: hello

Expected: Greeting + invitation to submit listing

You: {
  "title": "Premium Wireless Headphones",
  "description": "High-quality wireless headphones with active noise cancellation and 30-hour battery life",
  "category": "Electronics > Audio > Headphones",
  "price": 2999,
  "currency": "INR",
  "images": ["front.jpg", "side.jpg", "box.jpg", "packaging.jpg"],
  "attributes": {
    "brand": "AudioMax",
    "color": "Black",
    "connectivity": "Bluetooth"
  }
}

Expected: ✅ Listing validated and PUBLISHED immediately
Tool calls: validate_listing, screen_policy, publish_listing
Status: published
```

---

## Scenario 2: Correction Loop (1-3 Issues)
**Expected Flow**: new_listing → 2 issues → correction_loop → corrections → publish

```
You: {
  "title": "Gaming Laptop",
  "description": "High-performance laptop for gaming",
  "category": "Electronics > Smartphones",
  "images": ["laptop1.jpg", "laptop2.jpg", "laptop3.jpg"]
}

Expected: ❌ Category mismatch, missing price
Status: correction_loop
Assistant should list ALL remaining issues together

You: {
  "category": "Electronics > Cameras > Action Cameras",
  "price": 89999,
  "currency": "INR"
}

Expected: ✅ All issues resolved → PUBLISHED
Tool calls: update_listing (category), update_listing (price), validate_listing, screen_policy, publish_listing
Status: published
```

---

## Scenario 3: Resubmit Requested (>3 Issues)
**Expected Flow**: new_listing → >3 issues → resubmit_requested → full resubmission → publish

```
You: {
  "title": "Smartphone"
}

Expected: ❌ >3 issues (description, category, price, images, attributes)
Status: resubmit_requested
Assistant should ask for COMPLETE resubmission, not piecemeal corrections

You: {
  "title": "Latest 5G Smartphone",
  "description": "Latest flagship smartphone with 5G connectivity, 128GB storage, and triple camera setup",
  "category": "Electronics > Smartphones",
  "price": 45999,
  "currency": "INR",
  "images": ["phone_front.jpg", "phone_back.jpg", "phone_box.jpg"],
  "attributes": {
    "brand": "TechBrand",
    "color": "Silver",
    "storage": "128GB"
  }
}

Expected: ✅ All issues resolved → PUBLISHED
Tool calls: validate_listing, screen_policy, publish_listing
Status: published
Listing ID should remain the same
Correction rounds: 1
```

---

## Scenario 4: Escalation After 2 Correction Rounds
**Expected Flow**: new_listing → correction_loop → corrections → still has issues → round 2 → escalated

```
You: {
  "title": "Men's T-Shirt",
  "description": "Cotton t-shirt",
  "category": "Clothing > Men > Casual",
  "price": 599,
  "images": ["tshirt.jpg", "tshirt2.jpg"]
}

Expected: ❌ Missing attributes.brand, only 2 images (needs 3)
Status: correction_loop
Correction rounds: 0

You: {
  "attributes": {
    "brand": "FashionBrand"
  }
}

Expected: ❌ Still missing 1 image (need 3 total)
Status: correction_loop
Correction rounds: 1
Assistant should ask for remaining fields

You: just use the 2 images I provided

Expected: ❌ Still need 3 images minimum
Status: escalated (after 2 rounds with unresolved issues)
Tool calls: escalate_to_reviewer
Assistant should inform about escalation with ticket ID
```

---

## Scenario 5: Policy Violation
**Expected Flow**: new_listing → policy screening fails → correction_loop or resubmit_requested

```
You: {
  "title": "Hunting Knife Set",
  "description": "Professional hunting knives with sharp blades",
  "category": "Home & Kitchen > Kitchen Tools > Knives",
  "price": 2999,
  "currency": "INR",
  "images": ["knife1.jpg", "knife2.jpg", "knife3.jpg"],
  "attributes": {
    "brand": "KnifePro",
    "material": "Steel"
  }
}

Expected: ❌ Policy violation (contains "weapon" keyword)
Status: correction_loop (1 issue only)
Tool calls: validate_listing, screen_policy
Should explain the policy violation clearly
```

---

## Scenario 6: General Conversation
**Expected Flow**: general messages without listing submission

```
You: hello

Expected: Friendly greeting + invitation to submit listing
Status: none

You: what categories do you support?

Expected: List of supported categories from category_rules.json
Status: none

You: how do I submit a listing?

Expected: Instructions on providing all required fields together
Status: none
```

---

## Scenario 7: Piecemeal Submission (Should Batch Requests)
**Expected Flow**: Agent should ask for ALL missing fields together, not one at a time

```
You: {
  "title": "Bluetooth Speaker"
}

Expected: ❌ >3 issues
Status: resubmit_requested
Assistant should list ALL missing fields in one response:
- description
- category
- price
- images
- attributes
NOT ask for them one by one!
```

---

## Scenario 8: Multiple Resubmissions Leading to Escalation
**Expected Flow**: >3 issues → resubmit → still >3 issues → resubmit → escalated

```
You: {
  "title": "Camera"
}

Expected: ❌ >3 issues
Status: resubmit_requested
Correction rounds: 0

You: {
  "title": "Action Camera 4K",
  "description": "Waterproof action camera",
  "category": "Electronics > Cameras > Action Cameras",
  "price": 12999,
  "images": ["cam.jpg"]
}

Expected: ❌ Still has issues (only 1 image, need 3; missing attributes)
Status: resubmit_requested
Correction rounds: 1

You: {
  "title": "Action Camera 4K Pro",
  "description": "Waterproof action camera with 4K recording",
  "category": "Electronics > Cameras > Action Cameras",
  "price": 13999,
  "images": ["cam.jpg", "cam2.jpg"]
}

Expected: ❌ Still has issues (only 2 images, need 3; missing attributes)
Status: escalated (after 2 resubmission attempts)
Tool calls: escalate_to_reviewer
Correction rounds: 2
```

---

## Scenario 9: Abandoned Listing (New Listing While One In Progress)
**Expected Flow**: correction_loop → new listing submitted → previous escalated + new one starts

```
You: {
  "title": "Laptop Backpack",
  "description": "Durable laptop backpack",
  "category": "Clothing > Men > Casual",
  "price": 1999,
  "images": ["bag.jpg", "bag2.jpg"]
}

Expected: ❌ Category mismatch, only 2 images, missing attributes
Status: correction_loop
Note the listing_id (e.g., LST-ABC123)

You: {
  "title": "Wireless Mouse",
  "description": "Ergonomic wireless mouse",
  "category": "Electronics > Accessories > Cables",
  "price": 899,
  "currency": "INR",
  "images": ["mouse1.jpg", "mouse2.jpg", "mouse3.jpg"],
  "attributes": {
    "brand": "TechMouse",
    "color": "Black"
  }
}

Expected: ✅ Previous listing (LST-ABC123) escalated automatically
         ❌ New listing has category issue (mouse not in Cables category)
Status: correction_loop for new listing
Tool calls: escalate_to_reviewer (for old), validate_listing, screen_policy (for new)
Should see message about previous listing being escalated
```

---

## Scenario 10: Status Command
**Expected Flow**: Check listing status mid-conversation

```
You: {
  "title": "Smart Watch",
  "description": "Fitness tracker smart watch",
  "category": "Electronics > Smartphones",
  "price": 4999,
  "images": ["watch.jpg", "watch2.jpg"]
}

Expected: ❌ Category issue, only 2 images, missing attributes
Status: correction_loop

You: status

Expected: Display current listing state:
- listing_id
- status: correction_loop
- issues: [list of issues]
- corrections_applied: []
- correction_rounds: 0
```

---

## Quick Validation Checklist

After running tests, verify:

- ✅ **Greeting handling**: Agent responds naturally to "hello", "hi", etc.
- ✅ **Batching**: Agent asks for ALL missing fields together, not one at a time
- ✅ **>3 issues**: Agent asks for FULL resubmission, not corrections
- ✅ **≤3 issues**: Agent enters correction loop
- ✅ **Resubmission tracking**: Same listing_id maintained, correction_rounds incremented
- ✅ **Escalation**: After 2 rounds with unresolved issues
- ✅ **Policy screening**: Catches prohibited items/keywords
- ✅ **Publishing**: Only happens when all checks pass (0 issues)
- ✅ **Abandoned listings**: Previous in-progress listings escalated when new one submitted
- ✅ **Message ordering**: No AWS Bedrock validation errors about message ordering
- ✅ **Clean logging**: Minimal terminal output, no debug spam
- ✅ **Trace quality**: JSONL traces contain only essential info (no prompt_messages)

---

## Expected Tool Call Sequences

### Perfect Listing → Publish
```
1. validate_listing
2. screen_policy
3. publish_listing
```

### Correction Loop → Publish
```
1. validate_listing, screen_policy (initial)
2. update_listing (for each correction)
3. validate_listing, screen_policy (re-check)
4. publish_listing (if all resolved)
```

### Escalation
```
1. validate_listing, screen_policy (initial)
2. update_listing (corrections)
3. validate_listing, screen_policy (re-check, still has issues)
4. update_listing (more corrections)
5. validate_listing, screen_policy (re-check, still has issues)
6. escalate_to_reviewer (after 2 rounds)
```

### Resubmission
```
1. validate_listing, screen_policy (initial, >3 issues)
2. validate_listing, screen_policy (on resubmission)
3. publish_listing (if resolved) OR escalate_to_reviewer (if 2 resubmissions failed)
```
