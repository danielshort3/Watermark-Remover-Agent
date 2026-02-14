"""Regression tests for tolerant LLM JSON parsing in the order-of-worship graph."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from watermark_remover.agent import order_of_worship_graph as order_graph


def test_extract_json_from_llm_output_accepts_python_dict_numeric_keys() -> None:
    raw = (
        '{"date": "03_30_2025", "songs": {0: {"title": "Made To Worship", '
        '"artist": "Default Arrangement", "key": "G"}}}'
    )
    parsed = order_graph._extract_json_from_llm_output(raw)
    assert isinstance(parsed, dict)
    assert parsed.get("date") == "03_30_2025"
    songs = parsed.get("songs")
    assert isinstance(songs, dict)
    assert 0 in songs
    assert songs[0]["title"] == "Made To Worship"

