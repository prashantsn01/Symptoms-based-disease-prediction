"""
src/nlp - Natural Language Processing module for symptom extraction.

Two implementations are provided:

1. **TransformerExtractor** (``src.nlp.transformer_extractor``)
   - Fully offline, uses ``sentence-transformers`` for semantic similarity.
   - No API key needed.  Model downloads once (~80 MB).

2. **LLMExtractor** (``src.nlp.llm_extractor``)
   - Uses Google Gemini API for accurate free-text understanding.
   - Requires a ``GEMINI_API_KEY`` environment variable.

Switch between them via the ``NLP_BACKEND`` env var or ``create_extractor()``.
"""

from src.nlp.base import BaseExtractor, create_extractor

__all__ = ["BaseExtractor", "create_extractor"]
