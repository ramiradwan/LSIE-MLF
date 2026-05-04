"""
Text Pre-processing — §4.D.4

spaCy tokenization, normalization, and linguistic preprocessing
prior to LLM semantic evaluation.
"""

from __future__ import annotations

from typing import Any


class TextPreprocessor:
    """
    §4.D.4 — spaCy NLP preprocessing pipeline.

    Performs tokenization, normalization, and linguistic preprocessing
    on transcribed text before semantic evaluation.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Any = None  # Lazy-loaded spaCy Language

    def load_model(self) -> None:
        """Load spaCy model (§4.D.4)."""
        import spacy

        try:
            self._nlp = spacy.load(self.model_name)
        except OSError:
            self._nlp = spacy.blank("en")

    def preprocess(self, text: str) -> str:
        """
        Normalize and clean transcribed text.

        §4.D.4 — spaCy tokenization and normalization prior to
        LLM semantic evaluation.

        Args:
            text: Raw ASR transcription output.

        Returns:
            Cleaned, normalized text string.
        """
        if self._nlp is None:
            self.load_model()

        # §4.D.4 — Tokenize and normalize: lowercase, strip punctuation,
        # remove stopwords and whitespace tokens, lemmatize
        doc: Any = self._nlp(text)
        tokens: list[str] = [
            (token.lemma_ or token.text).lower()
            for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]

        return " ".join(tokens)
