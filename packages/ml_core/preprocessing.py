"""
Text Pre-processing — §4.D.4

spaCy tokenization, normalization, and linguistic preprocessing
prior to LLM semantic evaluation.
"""

from __future__ import annotations


class TextPreprocessor:
    """
    §4.D.4 — spaCy NLP preprocessing pipeline.

    Performs tokenization, normalization, and linguistic preprocessing
    on transcribed text before semantic evaluation.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = None  # Lazy-loaded

    def load_model(self) -> None:
        """Load spaCy model."""
        # TODO: Implement — import spacy
        raise NotImplementedError

    def preprocess(self, text: str) -> str:
        """
        Normalize and clean transcribed text.

        Args:
            text: Raw ASR transcription output.

        Returns:
            Cleaned, normalized text string.
        """
        # TODO: Implement per §4.D.4
        raise NotImplementedError
