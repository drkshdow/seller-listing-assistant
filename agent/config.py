from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# AWS Bedrock configuration
AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL: str = os.getenv("BEDROCK_MODEL", "amazon.nova-pro-v1:0")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# Context window management: keep this many recent messages in full;
# everything older is collapsed into a rolling summary.
MAX_RECENT_MESSAGES: int = 10

# After this many correction rounds without resolution, escalate.
MAX_CORRECTION_ROUNDS: int = 2
