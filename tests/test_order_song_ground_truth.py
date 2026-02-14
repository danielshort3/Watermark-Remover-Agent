"""Regression tests for deterministic song extraction and reconciliation."""

import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from watermark_remover.agent import order_of_worship_graph as order_graph


_RAW_TEXT = """
5:02 Only King Forever [ Elevation Worship in A ]
3:15 All Hail The Power Of Jesus' Name [ Default Arrangement in F-G ]
4:00 Worthy Is The Lamb [ Default Arrangement in G ]
3:28 Worthy Of It All [ CeCe Winans in D ]
1:30 Worthy [ Elevation Worship in D ]
2:30 We Fall Down [ Lifeway Worship in D ]
I Exalt Thee [ Default Arrangement in D ]
2:00 We Fall Down [ Lifeway Worship in D ]
It Really Is Amazing Grace [ Default Arrangement in G ]
"""


def test_extract_bracket_songs_from_text_preserves_order_and_duplicates() -> None:
    songs = order_graph._extract_bracket_songs_from_text(_RAW_TEXT)
    assert len(songs) == 9
    assert songs[0]["title"] == "Only King Forever"
    assert songs[0]["artist"] == "Elevation Worship"
    assert songs[0]["key"] == "A"
    assert songs[1]["key"] == "F"  # F-G modulation should normalize to F.
    assert songs[5]["title"] == "We Fall Down"
    assert songs[7]["title"] == "We Fall Down"  # duplicate title retained


def test_reconcile_song_map_with_ground_truth_restores_missing_and_shifted_entries() -> None:
    ground_truth = order_graph._extract_bracket_songs_from_text(_RAW_TEXT)
    llm_partial = {
        0: {"title": "All Hail The Power Of Jesus' Name", "artist": "Default Arrangement", "key": "F"},
        1: {"title": "Worthy Is The Lamb", "artist": "Default Arrangement", "key": "G"},
        2: {"title": "We Fall Down", "artist": "Lifeway Worship", "key": "D"},
        3: {"title": "I Exalt Thee", "artist": "Default Arrangement", "key": "D"},
        4: {"title": "It Really Is Amazing Grace", "artist": "Default Arrangement", "key": "G"},
    }
    reconciled, backfilled, replaced = order_graph._reconcile_song_map_with_ground_truth(
        llm_songs=llm_partial,
        ground_truth_songs=ground_truth,
    )
    assert len(reconciled) == 9
    assert reconciled[0]["title"] == "Only King Forever"
    assert set(backfilled) == {5, 6, 7, 8}
    assert 0 in replaced


def test_has_song_selection_constraints_detects_filters() -> None:
    assert order_graph._has_song_selection_constraints("Download the songs for the French Horn.") is False
    assert order_graph._has_song_selection_constraints("Only download the fourth song.") is True
    assert order_graph._has_song_selection_constraints("Do not download It Really Is Amazing Grace.") is True


def test_deduplicate_song_entries_drops_exact_duplicates_only() -> None:
    songs = {
        0: {"title": "We Fall Down", "artist": "Lifeway Worship", "key": "D"},
        1: {"title": "We Fall Down", "artist": "Lifeway Worship", "key": "D"},
        2: {"title": "We Fall Down", "artist": "Lifeway Worship", "key": "E"},
    }
    deduped, dropped = order_graph._deduplicate_song_entries(songs)
    assert dropped == [1]
    assert len(deduped) == 2
    assert deduped[0][0] == 0
    assert deduped[1][0] == 2


def test_extract_songs_node_backfills_partial_llm_output() -> None:
    llm_partial = {
        "date": "02_15_2026",
        "songs": {
            "0": {"title": "All Hail The Power Of Jesus' Name", "artist": "Default Arrangement", "key": "F"},
            "1": {"title": "Worthy Is The Lamb", "artist": "Default Arrangement", "key": "G"},
            "2": {"title": "We Fall Down", "artist": "Lifeway Worship", "key": "D"},
            "3": {"title": "I Exalt Thee", "artist": "Default Arrangement", "key": "D"},
            "4": {"title": "It Really Is Amazing Grace", "artist": "Default Arrangement", "key": "G"},
        },
    }

    original_get_pdf_text_cached = order_graph._get_pdf_text_cached
    original_run_llm_strict_json = order_graph._run_llm_strict_json
    try:
        order_graph._get_pdf_text_cached = lambda state, path: _RAW_TEXT  # type: ignore[assignment]
        order_graph._run_llm_strict_json = lambda *args, **kwargs: llm_partial  # type: ignore[assignment]

        state = {
            "pdf_name": "dummy.pdf",
            "default_instrument": "French Horn",
            "overrides": {},
            "user_input": "Download the songs for the French Horn.",
            "debug": False,
        }
        out = order_graph.extract_songs_node(state)
        songs = out.get("songs", {})

        assert len(songs) == 8
        assert songs[0]["title"] == "Only King Forever"
        assert songs[7]["title"] == "It Really Is Amazing Grace"
        assert sum(1 for item in songs.values() if item.get("title") == "We Fall Down") == 1
    finally:
        order_graph._get_pdf_text_cached = original_get_pdf_text_cached  # type: ignore[assignment]
        order_graph._run_llm_strict_json = original_run_llm_strict_json  # type: ignore[assignment]
