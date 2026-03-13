"""
Tests for packages/ml_core/preprocessing.py — Phase 1 validation.

Verifies TextPreprocessor against §4.D.4:
spaCy tokenization, normalization, stopword/punctuation removal.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from packages.ml_core.preprocessing import TextPreprocessor


def _make_token(
    lemma: str,
    is_punct: bool = False,
    is_space: bool = False,
    is_stop: bool = False,
) -> MagicMock:
    """Helper to create a mock spaCy Token."""
    tok = MagicMock()
    tok.lemma_ = lemma
    tok.is_punct = is_punct
    tok.is_space = is_space
    tok.is_stop = is_stop
    return tok


@pytest.fixture()
def mock_spacy(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install mock spacy into sys.modules."""
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "spacy", mock)
    return mock


class TestTextPreprocessor:
    """§4.D.4 — spaCy NLP preprocessing pipeline."""

    def test_load_model(self, mock_spacy: MagicMock) -> None:
        """§4.D.4 — Loads en_core_web_sm by default."""
        proc = TextPreprocessor()
        proc.load_model()
        mock_spacy.load.assert_called_once_with("en_core_web_sm")

    def test_preprocess_removes_stopwords_and_punctuation(
        self, mock_spacy: MagicMock
    ) -> None:
        """§4.D.4 — Strips punctuation, stopwords, whitespace tokens."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = [
            _make_token("hello"),
            _make_token(",", is_punct=True),
            _make_token(" ", is_space=True),
            _make_token("the", is_stop=True),
            _make_token("world"),
        ]
        mock_spacy.load.return_value = mock_nlp

        proc = TextPreprocessor()
        result = proc.preprocess("Hello, the world")

        assert result == "hello world"

    def test_preprocess_lemmatizes(self, mock_spacy: MagicMock) -> None:
        """§4.D.4 — Uses lemma form of tokens."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = [
            _make_token("run"),
            _make_token("fast"),
        ]
        mock_spacy.load.return_value = mock_nlp

        proc = TextPreprocessor()
        result = proc.preprocess("running faster")

        assert result == "run fast"

    def test_preprocess_lowercases(self, mock_spacy: MagicMock) -> None:
        """§4.D.4 — Output is lowercased."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = [_make_token("HELLO")]
        mock_spacy.load.return_value = mock_nlp

        proc = TextPreprocessor()
        result = proc.preprocess("HELLO")

        assert result == "hello"

    def test_preprocess_empty_input(self, mock_spacy: MagicMock) -> None:
        """§4.D.4 — Empty input returns empty string."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = []
        mock_spacy.load.return_value = mock_nlp

        proc = TextPreprocessor()
        result = proc.preprocess("")

        assert result == ""

    def test_preprocess_noisy_asr_output(self, mock_spacy: MagicMock) -> None:
        """§4.D.4 — Handles noisy transcription with filler words."""
        mock_nlp = MagicMock()
        mock_nlp.return_value = [
            _make_token("um", is_stop=True),
            _make_token("...", is_punct=True),
            _make_token("like", is_stop=True),
            _make_token("greeting"),
        ]
        mock_spacy.load.return_value = mock_nlp

        proc = TextPreprocessor()
        result = proc.preprocess("Um... like greeting")

        assert result == "greeting"
