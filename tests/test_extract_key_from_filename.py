import pytest

# Skip this test module if langchain_core is not available, since the tools
# module depends on it at import time.
pytest.importorskip("langchain_core")

from watermark_remover.agent.tools import extract_key_from_filename


@pytest.mark.parametrize(
    "fname,expected",
    [
        ("my_lighthouse_frenchhorn12_C_001.png", "C"),
        ("your_name_frenchhorn12_A_001.png", "A"),
        ("who_else_frenchhorn12_Ab_001.png", "Ab"),
        ("song_altosax_Gb_010.jpg", "Gb"),
        ("piece_trumpet_F#_123.jpeg", "F#"),
        ("some_random_file.png", None),
        ("no_key_here_001.txt", None),
    ],
)
def test_extract_key_from_filename(fname, expected):
    assert extract_key_from_filename(fname) == expected
