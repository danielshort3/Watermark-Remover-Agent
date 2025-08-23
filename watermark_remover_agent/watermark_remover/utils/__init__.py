"""
Utility subpackage.

This package exposes various helper functions used by the Watermark Remover
agent.  Currently it contains transposition utilities for musical key
suggestions.
"""

from .transposition_utils import (
    normalize_key,
    get_transposition_suggestions,
    KEY_TO_SEMITONE,
    SEMITONE_TO_KEY,
    INSTRUMENT_TRANSPOSITIONS,
)

__all__ = [
    "normalize_key",
    "get_transposition_suggestions",
    "KEY_TO_SEMITONE",
    "SEMITONE_TO_KEY",
    "INSTRUMENT_TRANSPOSITIONS",
]