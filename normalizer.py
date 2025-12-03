import json
from typing import Any, Dict

from llm_engine import generate_completion
from schema import ClaimAttributes


PROMPT_TEMPLATE = """
You are an expert insurance claim analyst.

Your task:
Given a free-text insurance claim description, extract the following fields
and return them as a **single JSON object** only, with no explanation text:

- loss_type: short text describing the type/cause of loss
  (e.g. "Vehicle", "Fire", "Water Damage", "Theft", "Device", "Injury")
- severity: one of ["Low", "Medium", "High", "Critical"]
- asset: what asset or property was affected (e.g. "Car", "Building", "Laptop")
- estimated_loss: short string for approximate monetary loss if mentioned, else "Unknown"
- incident_date: date or time phrase if mentioned, else "Unknown"
- location: city/area/building, if mentioned, else "Unknown"
- confidence: one of ["Low", "Medium", "High"] describing your confidence
- explanation: ONE short sentence explaining your reasoning.

Rules:
- Output MUST be valid JSON.
- Do not include any text before or after the JSON.
- Do not use comments.
- If a field is not explicitly stated, set it to "Unknown".

Example 1:
Claim description:
"Minor rear-end collision on the highway, bumper scratched. No injuries."

JSON:
{{
  "loss_type": "Vehicle",
  "severity": "Low",
  "asset": "Car",
  "estimated_loss": "Unknown",
  "incident_date": "Unknown",
  "location": "Highway (unspecified)",
  "confidence": "High",
  "explanation": "This is a low severity vehicle claim affecting a car bumper with no injuries."
}}

Example 2:
Claim description:
"Fire broke out in the kitchen last night in our Mumbai apartment, burning cabinets and ceiling."

JSON:
{{
  "loss_type": "Fire",
  "severity": "High",
  "asset": "Kitchen (apartment)",
  "estimated_loss": "Unknown",
  "incident_date": "Last night",
  "location": "Mumbai apartment",
  "confidence": "High",
  "explanation": "Kitchen fire in a residential apartment with significant damage to cabinets and ceiling."
}}

Now analyze the following claim:

Claim description:
"{claim_text}"

JSON:
"""


def _extract_json_segment(raw_text: str) -> str:
    """
    Try to safely pull out the JSON object part from the LLM output.
    """
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end != -1:
        return raw_text[start:end].strip()
    return raw_text.strip()


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON robustly. On failure, return an empty dict for fallback defaults.
    """
    cleaned = _extract_json_segment(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def normalize_claim(claim_text: str) -> ClaimAttributes:
    """
    Main function: takes free-text claim, returns structured ClaimAttributes.
    """
    prompt = PROMPT_TEMPLATE.format(claim_text=claim_text.strip())
    llm_output = generate_completion(prompt, max_tokens=256, temperature=0.1)
    parsed = _safe_parse_json(llm_output)

    # Use Pydantic to apply defaults for missing fields
    claim = ClaimAttributes(**parsed)
    return claim


def normalize_claim_as_json_str(claim_text: str) -> str:
    """
    Helper for simple JSON string output (for APIs or basic UI).
    """
    claim = normalize_claim(claim_text)
    return json.dumps(claim.to_display_dict(), indent=2)
