"""Tests for the PDF resolution logic in order_of_worship_graph."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from watermark_remover.agent import order_of_worship_graph as order_graph


def _make_state(input_dir: Path) -> dict[str, str]:
    return {"input_dir": str(input_dir)}


def test_resolve_pdf_handles_single_digit_day(tmp_path: Path) -> None:
    """A request for November 09 should match November 9 within the input folder."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf_path = input_dir / "November 9, 2025.pdf"
    pdf_path.write_bytes(b"dummy")

    state = _make_state(input_dir)
    candidates = order_graph._list_pdf_candidates(state)

    resolved, meta = order_graph._resolve_pdf_from_instruction(
        state,
        "November 09, 2025.pdf",
        "Download the songs in 'November 09, 2025.pdf'",
        candidates,
    )

    assert resolved == str(pdf_path)
    assert meta is not None
    assert meta["resolved_name"] == pdf_path.name
    assert "11_09_2025" in meta.get("matched_dates", [])


def test_resolve_pdf_uses_pdf_text_when_filename_unclear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If filenames lack dates, fall back to parsing PDF contents (via date extraction)."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf_path = input_dir / "service.pdf"
    pdf_path.write_bytes(b"dummy")

    state = _make_state(input_dir)
    candidates = order_graph._list_pdf_candidates(state)

    def fake_determine(state_arg, pdf_candidate: str) -> str:
        assert state_arg is state
        assert pdf_candidate == str(pdf_path)
        return "11_09_2025"

    monkeypatch.setattr(order_graph, "_determine_order_folder_from_pdf", fake_determine)

    resolved, meta = order_graph._resolve_pdf_from_instruction(
        state,
        "",
        "Download the songs from the November 9, 2025 service",
        candidates,
    )

    assert resolved == str(pdf_path)
    assert meta is not None
    assert meta.get("matched_pdf_date") == "11_09_2025"
