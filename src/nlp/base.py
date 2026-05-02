"""
base.py - Abstract interface and factory for NLP symptom extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import os


class BaseExtractor(ABC):
    """Interface that every NLP symptom extractor must implement."""

    @abstractmethod
    def extract_symptoms(self, text: str, symptom_list: List[str]) -> Dict:
        """Parse free-text and return matched symptoms.

        Parameters
        ----------
        text : str
            The user's natural-language symptom description.
        symptom_list : list[str]
            The full list of known symptom names (lowercase).

        Returns
        -------
        dict with keys:
            matched_symptoms : list[str]
                Symptom names that were identified in the text.
            confidence : float
                Overall confidence score (0-1).
            details : list[dict]
                Per-symptom info: ``{"symptom": str, "score": float}``.
            backend : str
                Which extractor produced the result ("transformer" | "llm").
        """
        ...


def create_extractor(backend: str | None = None) -> BaseExtractor:
    """Factory – instantiate the requested NLP backend.

    Parameters
    ----------
    backend : str, optional
        ``"transformer"`` or ``"llm"``.
        Falls back to the ``NLP_BACKEND`` env var, then ``"transformer"``.
    """
    backend = (backend or os.getenv("NLP_BACKEND", "transformer")).lower().strip()

    if backend == "llm":
        from src.nlp.llm_extractor import LLMExtractor
        return LLMExtractor()
    else:
        from src.nlp.transformer_extractor import TransformerExtractor
        return TransformerExtractor()
