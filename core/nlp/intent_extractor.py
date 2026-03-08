"""
Customer Intent Extractor
--------------------------
Uses Claude (claude-haiku-4-5-20251001) to extract structured service intent
from free-form customer text (chat logs, email, verbal description).

Intent categories (aligned with automotive service domain):
    regular_service | repair | insurance_claim | inspection
    warranty_claim  | emergency | general_enquiry
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

import anthropic

from config import settings
from api.schemas import IntentResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert automotive service advisor AI.
Your job is to analyse customer text and extract their service intent.

Always respond with valid JSON only — no extra text, no markdown fences.

Intent categories (pick the single best match):
  regular_service  – routine oil change, tyre rotation, scheduled maintenance
  repair           – fix a mechanical/electrical fault that isn't accident-related
  insurance_claim  – damage from an accident being reported to insurance
  inspection       – pre-purchase check, emission test, safety inspection
  warranty_claim   – fault covered by manufacturer or extended warranty
  emergency        – brake failure, engine seizure, vehicle stuck / unsafe to drive
  general_enquiry  – pricing, availability, general questions

Urgency levels:
  high   – emergency, safety risk, insurance deadline
  medium – repair needed soon, warranty about to expire
  low    – routine service, inspection, general question

Sentiment:
  positive | neutral | negative
"""

USER_TEMPLATE = """\
Customer message:
\"\"\"
{text}
\"\"\"

Extract and return:
{{
  "customer_intent": "<category>",
  "urgency": "<high|medium|low>",
  "key_concerns": ["<concern 1>", "<concern 2>", ...],
  "sentiment": "<positive|neutral|negative>",
  "reasoning": "<1-2 sentence explanation>"
}}
"""

VALID_INTENTS = {
    "regular_service", "repair", "insurance_claim",
    "inspection", "warranty_claim", "emergency", "general_enquiry",
}


class IntentExtractor:
    """
    Extracts customer service intent from text using the Claude LLM.

    Usage::

        extractor = IntentExtractor()
        result: IntentResult = extractor.extract("My car was rear-ended, I need an insurance claim.")
    """

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def extract(self, customer_text: str) -> IntentResult:
        """Parse customer_text and return a structured IntentResult."""
        prompt = USER_TEMPLATE.format(text=customer_text.strip())

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        data = _parse_json(raw)

        intent = data.get("customer_intent", "general_enquiry")
        if intent not in VALID_INTENTS:
            intent = "general_enquiry"

        return IntentResult(
            customer_intent=intent,
            urgency=data.get("urgency", "low"),
            key_concerns=data.get("key_concerns", []),
            sentiment=data.get("sentiment", "neutral"),
        )


def _parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON object from surrounding text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise
