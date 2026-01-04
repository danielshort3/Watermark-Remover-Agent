"""Tests for key normalization in order-of-worship extraction."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from watermark_remover.agent import order_of_worship_graph as order_graph


def test_normalize_extracted_key_modulation() -> None:
    assert order_graph._normalize_extracted_key("F-G") == "F"
    assert order_graph._normalize_extracted_key("F - G") == "F"
    assert order_graph._normalize_extracted_key(f"f\u2013g") == "F"


def test_normalize_extracted_key_non_modulation() -> None:
    assert order_graph._normalize_extracted_key("Bb") == "Bb"
    assert order_graph._normalize_extracted_key("G/B") == "G/B"
