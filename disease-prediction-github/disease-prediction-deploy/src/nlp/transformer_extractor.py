"""
transformer_extractor.py - Offline symptom extraction using sentence-transformers.

Uses semantic similarity between the user's text (split into clauses) and
each known symptom name to decide which symptoms are present.

Model: ``all-MiniLM-L6-v2`` (~80 MB, downloads on first run).
No API key required.
"""

import re
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, util

from src.nlp.base import BaseExtractor

# Default similarity threshold – a symptom is considered "matched"
# when the best clause-to-symptom cosine sim exceeds this value.
DEFAULT_THRESHOLD = 0.55

# Maximum number of symptoms to return (ranked by score)
MAX_MATCHES = 15

# Model singleton (loaded once, reused across requests)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        print("[NLP] Loading sentence-transformer model (first time only)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[NLP] Sentence-transformer model loaded.")
    return _model


def _split_into_clauses(text: str) -> List[str]:
    """Split user text into meaningful chunks for better matching.

    Examples
    --------
    >>> _split_into_clauses("I have a headache, nausea and blurry vision")
    ['I have a headache', 'nausea', 'blurry vision']
    """
    # Split on commas, semicolons, "and", "also", newlines
    clauses = re.split(r"[,;\n]|\band\b|\balso\b|\bplus\b", text, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c.strip()]

    # Also include the full text as one chunk for context
    if len(clauses) > 1:
        clauses.append(text.strip())

    return clauses


class TransformerExtractor(BaseExtractor):
    """Offline NLP extractor using sentence-transformer cosine similarity.

    How it works
    ------------
    1. Pre-encode all 377 symptom names into embedding vectors.
    2. Split the user's text into clauses.
    3. Encode each clause.
    4. Compute cosine similarity between every clause and every symptom.
    5. For each symptom, take the max similarity across clauses.
    6. Return symptoms whose max score exceeds the threshold.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._symptom_embeddings = None
        self._symptom_names: List[str] = []

    def _ensure_embeddings(self, symptom_list: List[str]):
        """Encode symptom names once and cache the embeddings."""
        if self._symptom_embeddings is not None and self._symptom_names == symptom_list:
            return

        model = _get_model()

        # Create descriptive phrases for better matching
        # e.g. "abdominal pain" -> "abdominal pain"
        # e.g. "nausea" -> "nausea"
        descriptions = []
        for s in symptom_list:
            # Clean underscores / formatting artefacts
            clean = s.replace("_", " ").strip()
            descriptions.append(clean)

        self._symptom_names = list(symptom_list)
        self._symptom_embeddings = model.encode(
            descriptions, convert_to_tensor=True, show_progress_bar=False
        )

    def extract_symptoms(self, text: str, symptom_list: List[str]) -> Dict:
        """Extract symptoms from free text via semantic similarity."""
        if not text or not text.strip():
            return {
                "matched_symptoms": [],
                "confidence": 0.0,
                "details": [],
                "backend": "transformer",
            }

        self._ensure_embeddings(symptom_list)
        model = _get_model()

        # Split input into clauses and encode
        clauses = _split_into_clauses(text)
        clause_embeddings = model.encode(
            clauses, convert_to_tensor=True, show_progress_bar=False
        )

        # Compute cosine similarity: clauses × symptoms
        sim_matrix = util.cos_sim(clause_embeddings, self._symptom_embeddings)
        # shape: (num_clauses, num_symptoms)

        # For each symptom, take the best (max) similarity across all clauses
        max_scores = sim_matrix.max(dim=0).values.cpu().numpy()

        # Collect matches above threshold
        details = []
        for idx in np.argsort(-max_scores):
            score = float(max_scores[idx])
            if score < self.threshold:
                break
            if len(details) >= MAX_MATCHES:
                break
            details.append({
                "symptom": self._symptom_names[idx],
                "score": round(score, 4),
            })

        matched = [d["symptom"] for d in details]
        avg_conf = float(np.mean([d["score"] for d in details])) if details else 0.0

        return {
            "matched_symptoms": matched,
            "confidence": round(avg_conf, 4),
            "details": details,
            "backend": "transformer",
        }
