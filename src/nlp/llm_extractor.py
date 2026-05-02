"""
llm_extractor.py - Symptom extraction using Google Gemini API.

Sends the user's free-text description along with the full symptom list
to Gemini, which returns a structured JSON of matched symptoms.

Requires
--------
- ``GEMINI_API_KEY`` environment variable (get one at https://aistudio.google.com/apikey).
- ``pip install google-generativeai``
"""

import os
import json
import re
from typing import List, Dict

from src.nlp.base import BaseExtractor

# Gemini model to use (fast + cheap)
GEMINI_MODEL = "gemini-2.0-flash"


def _get_client():
    """Lazy-import and configure the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com/apikey "
            "and set it: export GEMINI_API_KEY=your_key"
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


# System prompt that instructs Gemini to act as a medical symptom extractor
SYSTEM_PROMPT = """\
You are a medical symptom extraction assistant. Your job is to read a patient's \
free-text description of how they feel and identify which symptoms from a known \
list are present.

RULES:
1. ONLY return symptoms that appear in the provided SYMPTOM LIST.
2. Match semantically — e.g. "my head hurts" → "headache", "can't sleep" → "insomnia".
3. Return a JSON object with exactly this schema:
   {
     "matched_symptoms": ["symptom1", "symptom2", ...],
     "reasoning": "Brief explanation of why each symptom was matched"
   }
4. If no symptoms match, return {"matched_symptoms": [], "reasoning": "No recognizable symptoms found"}.
5. Do NOT add symptoms that the patient did not describe.
6. Be generous with matching — if the description is vague but plausibly matches, include it with low confidence.
7. Return ONLY valid JSON, no markdown fences, no extra text.
"""


class LLMExtractor(BaseExtractor):
    """NLP extractor powered by Google Gemini API.

    How it works
    ------------
    1. Build a prompt containing the symptom list + user text.
    2. Call Gemini to extract matched symptoms from the text.
    3. Validate the response against the known symptom list.
    4. Return structured results.
    """

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = _get_client()

    def extract_symptoms(self, text: str, symptom_list: List[str]) -> Dict:
        """Extract symptoms from free text via Gemini LLM."""
        if not text or not text.strip():
            return {
                "matched_symptoms": [],
                "confidence": 0.0,
                "details": [],
                "backend": "llm",
            }

        self._ensure_client()

        # Build the user prompt
        symptom_block = "\n".join(f"- {s}" for s in sorted(symptom_list))
        user_prompt = (
            f"SYMPTOM LIST (only match from these):\n{symptom_block}\n\n"
            f"PATIENT DESCRIPTION:\n\"{text}\"\n\n"
            f"Extract the matching symptoms as JSON."
        )

        try:
            response = self._client.generate_content(
                [
                    {"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + user_prompt]},
                ],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
            )

            raw = response.text.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            matched_raw = result.get("matched_symptoms", [])
            reasoning = result.get("reasoning", "")

        except json.JSONDecodeError:
            # If Gemini returns non-JSON, try to extract symptom names
            matched_raw = self._fallback_parse(response.text, symptom_list)
            reasoning = "Parsed from non-JSON response"
        except Exception as exc:
            return {
                "matched_symptoms": [],
                "confidence": 0.0,
                "details": [],
                "backend": "llm",
                "error": str(exc),
            }

        # Validate against known symptoms (case-insensitive)
        symptom_set = {s.lower() for s in symptom_list}
        validated = []
        for s in matched_raw:
            s_lower = s.lower().strip()
            if s_lower in symptom_set:
                # Return the original-cased version from symptom_list
                original = next(orig for orig in symptom_list if orig.lower() == s_lower)
                validated.append(original)

        details = [{"symptom": s, "score": 0.95} for s in validated]
        avg_conf = 0.95 if validated else 0.0

        return {
            "matched_symptoms": validated,
            "confidence": round(avg_conf, 4),
            "details": details,
            "reasoning": reasoning,
            "backend": "llm",
        }

    @staticmethod
    def _fallback_parse(text: str, symptom_list: List[str]) -> List[str]:
        """Try to extract symptom names from unstructured LLM output."""
        text_lower = text.lower()
        found = []
        for s in symptom_list:
            if s.lower() in text_lower:
                found.append(s)
        return found
