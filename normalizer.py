import json
from typing import Dict, Any, Tuple

from llm_engine import generate_completion
from schema import ClaimAttributes

# ---------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------

PROMPT_TEMPLATE = """
You are an expert insurance claim analyst.

Your task:
Given a free-text insurance claim description, extract the following fields
and return them as a single JSON object only, with no explanation text:

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

SHORT_TEXT_CLASSIFIER_PROMPT = """
You are a classifier that determines whether a short text is an insurance claim
or a normal conversational message.

Return a JSON object in the following format:

{{
  "is_claim": true | false,
  "reason": "short explanation why"
}}

Rules:
- If the text describes damage, loss, fire, theft, accident, leakage, injury,
  breakage, repairs, or financial harm -> is_claim = true.
- If the text is conversational (greetings, jokes, questions, commands,
  unrelated chat requests) -> is_claim = false.
- Respond with only JSON.

Text:
"{text}"
"""

DOCUMENT_CLASSIFICATION_PROMPT = """
You are an insurance document classifier.

You will receive the FULL TEXT of a PDF document extracted as plain text.

Classify the document into ONE of these types:

- "CLAIM": A specific incident is being reported. This includes:
    - Fire, electrical short circuit, smoke damage
    - Water leakage, flooding, burst pipes, urban floods, property damage
    - Vehicle accidents or collisions (motor claims)
    - Theft, burglary, break-in, stolen items
    - Machinery or equipment breakdown
    - Travel-related incidents such as baggage loss, mishandled luggage,
      missing bags, airline PIR (Property Irregularity Report) references,
      or compensation requests for lost items
    - Health/medical incidents such as hospitalization, surgery, medical
      treatment, reimbursement for hospital bills

- "POLICY": Policy wording, coverage terms, exclusions, premium tables,
  general product brochures, or any document that explains rules/coverage
  rather than reporting a concrete loss event.

- "OTHER": Anything else that is not an insurance claim or policy wording,
  such as email correspondence, letters only about missing documents or
  claim rejection WITHOUT describing the underlying incident.

Examples that should be classified as OTHER:
- Letters saying "your health claim is rejected due to document mismatch"
  without describing the actual accident or illness.
- Requests to resubmit KYC or identity documents.

Respond with a SINGLE JSON object:

{{
  "type": "CLAIM" | "POLICY" | "OTHER",
  "reason": "short explanation of why you chose this type"
}}

Rules:
- Always return valid JSON and nothing else.
- If the document clearly describes ANY incident or loss (accident, damage,
  theft, fire, baggage loss, hospitalization, etc.), classify as CLAIM.
- Only classify as POLICY when it contains rules/coverage details and NO
  incident description.
- Only classify as OTHER when it is correspondence or administrative communication.

Document text:
---
{document_text}
---
Return only the JSON object.
"""


DOCUMENT_EXTRACTION_PROMPT = """
You are helping with insurance claim processing.

You will receive the FULL TEXT of a document, which may include greetings,
policy details, background, and other noise.

Your task:
- Identify the MAIN INCIDENT DESCRIPTION that describes what happened,
  what was damaged, and how.
- Return it as a short, self-contained summary (1 to 3 sentences).
- If there are multiple incidents, focus on the PRIMARY one.
- If no incident is found, return exactly: "Unknown".

Respond with plain text only.

Document text:
---
{document_text}
---
Now write ONLY the concise incident description or "Unknown".
"""

MAX_DOC_CHARS = 4000

# Heuristics for document-level detection (PDFs)
DOC_CLAIM_KEYWORDS = [
    # generic claim words
    "insurance claim", "claim report", "claim form", "incident report",
    "loss report", "intimate claim", "intimation",

    # motor accident
    "motor accident", "road accident", "collision", "rear-end", "vehicle damage",
    "bumper damaged", "bumper damage", "headlamp broken", "traffic signal",

    # health / medical
    "hospitalization", "hospitalisation", "admitted", "discharge summary",
    "surgery", "operation", "inpatient", "medical expenses",
    "reimbursement claim", "cashless claim", "treatment details",

    # property / fire / burglary / flood
    "fire incident", "fire broke out", "short circuit", "burnt", "smoke damage",
    "kitchen fire", "house burglary", "break-in", "burglary occurred", "stolen items",
    "property damage", "wall damage", "ceiling damage", "water leakage",
    "burst pipe", "flood water", "urban flood", "inundation", "waterlogging",

    # travel / baggage
    "baggage loss", "lost baggage", "missing baggage", "checked-in baggage",
    "baggage not delivered", "baggage mishandling", "property irregularity report",
    "pir report", "airport", "flight", "airline acknowledged",
]

DOC_POLICY_KEYWORDS = [
    "terms and conditions", "coverage details", "scope of cover", "exclusions",
    "premium payable", "sum insured", "policy wording", "this policy document",
    "renewal notice", "endorsement", "schedule of benefits",
]

DOC_NONCLAIM_LETTER_KEYWORDS = [
    "claim rejected", "rejection letter", "deficiency of documents",
    "mismatch in documents", "kindly submit", "required documents",
    "discrepancy in submitted documents",
]

# ---------------------------------------------------------------------
# Cheap heuristic to avoid extra LLM calls on obvious claims
# ---------------------------------------------------------------------


def _looks_like_claim(text: str) -> bool:
    """Fast keyword-based check: does this text look like a claim at all?"""
    t = text.lower()
    keywords = [
        "accident",
        "damage",
        "damaged",
        "fire",
        "burnt",
        "burned",
        "theft",
        "stolen",
        "burglary",
        "collision",
        "leakage",
        "leak",
        "flood",
        "storm",
        "hail",
        "crack",
        "broken",
        "dent",
        "injury",
        "injuries",
        "claim",
        "insurance",
    ]
    return any(k in t for k in keywords)


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------


def _extract_json_segment(raw_text: str) -> str:
    """
    Try to safely pull out the JSON object part from the LLM output.
    """
    if not raw_text:
        return ""

    # If there are markdown fences, strip them first
    txt = raw_text.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()

    start = txt.find("{")
    end = txt.rfind("}") + 1
    if start != -1 and end != -1:
        return txt[start:end].strip()
    return txt


def _safe_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Parse JSON robustly. On failure, return an empty dict for fallback defaults.
    """
    cleaned = _extract_json_segment(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Classifiers & document utilities
# ---------------------------------------------------------------------


def classify_document_type(document_text: str) -> Tuple[bool, str, str]:
    """
    Classify a long document as CLAIM / POLICY / OTHER using the LLM,
    with keyword-based heuristics to avoid obvious misclassification.
    Returns (is_claim, doc_type, reason).
    """
    document_text = (document_text or "").strip()
    if not document_text:
        return False, "OTHER", "Document text is empty."

    truncated = document_text[:MAX_DOC_CHARS]
    lower = truncated.lower()

    # --- Heuristic signals ---
    claim_hit = any(k in lower for k in DOC_CLAIM_KEYWORDS)
    policy_hit = any(k in lower for k in DOC_POLICY_KEYWORDS)
    nonclaim_letter_hit = any(k in lower for k in DOC_NONCLAIM_LETTER_KEYWORDS)

    # 1) If it is clearly a "documents mismatch / rejection" letter,
    #    and there is no detailed incident section, prefer OTHER.
    if nonclaim_letter_hit and not claim_hit:
        return False, "OTHER", "Letter about document mismatch/rejection, not a full claim report."

    # 2) Ask the LLM to classify (for borderline / complex cases)
    prompt = DOCUMENT_CLASSIFICATION_PROMPT.format(document_text=truncated)
    raw = generate_completion(prompt, max_tokens=128, temperature=0.1)
    parsed = _safe_parse_json(raw)

    doc_type = (parsed.get("type") or "OTHER").upper().strip()
    reason = (parsed.get("reason") or "").strip()

    if doc_type not in {"CLAIM", "POLICY", "OTHER"}:
        doc_type = "OTHER"

    # 3) Use heuristics to correct obvious mistakes:
    #    - If we see strong claim language but model did NOT say CLAIM -> override to CLAIM.
    if claim_hit and doc_type != "CLAIM":
        doc_type = "CLAIM"
        if not reason:
            reason = "Contains clear incident and loss-related keywords (heuristic override)."

    #    - If we see only policy wording and no claim language, and model said CLAIM -> downgrade.
    if policy_hit and not claim_hit and doc_type == "CLAIM":
        doc_type = "POLICY"
        if not reason:
            reason = "Contains only policy wording/coverage keywords and no incident description."

    if not reason:
        reason = "Model did not provide a reason."

    is_claim = doc_type == "CLAIM"
    return is_claim, doc_type, reason



def classify_short_text(text: str) -> Tuple[bool, str]:
    """
    LLM-based classifier for short inputs (free-text box).
    Returns (is_claim, reason).
    """
    text = (text or "").strip()
    if not text:
        return False, "Empty text."

    prompt = SHORT_TEXT_CLASSIFIER_PROMPT.format(text=text)
    raw = generate_completion(prompt, max_tokens=64, temperature=0.1)

    parsed = _safe_parse_json(raw)
    is_claim = bool(parsed.get("is_claim", False))
    reason = parsed.get("reason", "No reason provided.")

    return is_claim, reason


def extract_primary_claim_text(document_text: str) -> str:
    """
    From a long document (e.g., PDF), extract the main incident description.
    """
    document_text = (document_text or "").strip()
    if not document_text:
        return "Unknown"

    truncated = document_text[:MAX_DOC_CHARS]
    prompt = DOCUMENT_EXTRACTION_PROMPT.format(document_text=truncated)

    summary = generate_completion(prompt, max_tokens=128, temperature=0.1)
    summary = (summary or "").strip()

    if summary.startswith('"') and summary.endswith('"'):
        summary = summary[1:-1].strip()

    return summary if summary else "Unknown"


def normalize_document_text(document_text: str) -> ClaimAttributes:
    """
    For long/complex documents (like PDFs), first extract the main incident
    description, then run the standard claim normalizer on that.
    """
    document_text = (document_text or "").strip()
    if not document_text:
        return ClaimAttributes()

    incident_summary = extract_primary_claim_text(document_text)
    print("[DEBUG] Extracted incident summary:", incident_summary)

    if incident_summary == "Unknown":
        # fall back to truncating the original text
        incident_summary = document_text[:MAX_DOC_CHARS]

    return normalize_claim(incident_summary)


# ---------------------------------------------------------------------
# Core normalizer
# ---------------------------------------------------------------------


def normalize_claim(claim_text: str) -> ClaimAttributes:
    """
    Main function: takes free-text claim (short) and returns structured ClaimAttributes.
    Also handles non-claim conversational inputs gracefully.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        return ClaimAttributes()

    # 1) Cheap heuristic: if it doesn't look like a claim, ask the classifier.
    if not _looks_like_claim(claim_text):
        is_claim, reason = classify_short_text(claim_text)
        if not is_claim:
            # Structured JSON saying this is not a claim
            claim = ClaimAttributes(
                loss_type="Not a claim",
                severity="N/A",
                asset="N/A",
                estimated_loss="N/A",
                incident_date="N/A",
                location="N/A",
                confidence="High",
                explanation=(
                    "The provided text is not an insurance claim. "
                    f"Reason: {reason}"
                ),
            )
            print("[DEBUG] Short text classified as NON-CLAIM:", reason)
            return claim

    # 2) It is (or strongly looks like) a claim -> run the structured extractor
    prompt = PROMPT_TEMPLATE.format(claim_text=claim_text)
    llm_output = generate_completion(prompt, max_tokens=192, temperature=0.1)

    print("\n[DEBUG] Raw LLM output:\n", repr(llm_output), "\n")

    parsed = _safe_parse_json(llm_output)
    print("[DEBUG] Parsed dict:", parsed)

    claim = ClaimAttributes(**parsed)
    print("[DEBUG] Claim model:", claim.model_dump())

    return claim


def normalize_claim_as_json_str(claim_text: str) -> str:
    """
    Helper for simple JSON string output (for APIs or basic UI).
    """
    claim = normalize_claim(claim_text)
    # schema.ClaimAttributes should provide to_display_dict(); fall back to model_dump
    if hasattr(claim, "to_display_dict"):
        payload = claim.to_display_dict()
    else:
        payload = claim.model_dump()
    return json.dumps(payload, indent=2)
