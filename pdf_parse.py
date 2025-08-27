import json
import re
from typing import Dict, List, Tuple

# --- PDF text extraction helpers ------------------------------------------------

def _read_pdf_text(pdf_path: str) -> str:
    """
    Try multiple libraries to extract text from a PDF. Uses the first one available.
    """
    # 1) pdfplumber (if present)
    try:
        import pdfplumber  # type: ignore
        text_chunks: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks)
    except Exception:
        pass

    # 2) PyPDF2 (commonly available)
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        text_chunks = []
        for page in reader.pages:
            # page.extract_text() returns a string or None
            txt = page.extract_text() or ""
            text_chunks.append(txt)
        return "\n".join(text_chunks)
    except Exception:
        pass

    # 3) pdfminer.six (if present)
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(pdf_path) or ""
    except Exception:
        pass

    raise RuntimeError(
        "Could not read PDF text. Please install one of: pdfplumber, PyPDF2, or pdfminer.six."
    )

# --- Parsing logic ---------------------------------------------------------------

# Matches lines like:
# "4:06 Holy Water [ We the Kingdom in F ]"
# "Trust And Obey [ Default Arrangement in C ]"
# "Come Thou Fount (Above All Else) [ Shane & Shane in C ]"
SONG_LINE_RE = re.compile(
    r"""^\s*
        (?:\d{1,2}:\d{2}\s+)?                # optional leading duration
        (?P<title>.+?)                       # song title (non-greedy)
        \s*\[\s*
        (?P<artist>.+?)                      # artist/arranger text up to ' in '
        \s+in\s+
        (?P<key>[A-Ga-g](?:[#b])?(?:\s*(?:maj(?:or)?|min(?:or)?|m))?)  # musical key
        \s*\]\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _normalize_key(k: str) -> str:
    """
    Normalize musical key text to a compact form:
    - Keep base note + optional #/b
    - Append 'm' if minor is indicated
    Examples:
        'F' -> 'F', 'Bb major' -> 'Bb', 'C# minor' -> 'C#m', 'G m' -> 'Gm'
    """
    s = k.strip().replace("♭", "b").replace("♯", "#")
    # Detect base like A, Bb, C#, etc.
    base_match = re.match(r"^[A-Ga-g](?:[#b])?", s)
    base = (base_match.group(0).upper() if base_match else s.strip().upper())

    # Minor?
    is_minor = bool(re.search(r"(?:\bmin(?:or)?\b|\bm\b)$", s, re.IGNORECASE))
    return base + ("m" if is_minor else "")

def extract_songs_from_text(text: str) -> Dict[int, Dict[str, str]]:
    """
    Parse raw text and return the index-keyed song dict.
    """
    songs: List[Dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "[" not in line or "]" not in line:
            continue
        m = SONG_LINE_RE.match(line)
        if not m:
            continue
        title = m.group("title").strip()
        artist = m.group("artist").strip()
        key = _normalize_key(m.group("key"))
        songs.append({"title": title, "artist": artist, "key": key})

    # Index from 0
    return {i: song for i, song in enumerate(songs)}

def extract_songs_from_pdf(pdf_path: str, save_to: str = None) -> Dict[int, Dict[str, str]]:
    """
    Extract songs from the given PDF. Optionally save the dict as JSON.
    """
    text = _read_pdf_text(pdf_path)
    result = extract_songs_from_text(text)
    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result

# --- Example usage ---------------------------------------------------------------
if __name__ == "__main__":
    # Update the path to your file as needed:
    path = "August 24, 2025.pdf"
    songs = extract_songs_from_pdf(path, save_to=None)
    print(json.dumps(songs, ensure_ascii=False, indent=2))
