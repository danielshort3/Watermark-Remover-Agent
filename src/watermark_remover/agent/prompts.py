"""Centralized LLM prompt builders for the Watermark Remover agent.

This module gathers all prompt templates used across tools and graphs in one
place to make debugging and updates easier. Callers should use the small
builder functions below to generate the exact prompt strings needed.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import time
import re

# simple global sequence for prompt files within a process
_PROMPT_SEQ = 0


def _sanitize_filename(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (value or "").strip())
    return re.sub(r"_+", "_", s).strip("_")


def log_prompt(label: str, prompt: str) -> None:
    """Persist the rendered prompt under output/logs/<RUN_TS>/prompts/.

    - Creates the directory if needed.
    - File name format: <seq>_<label>_<ms>.txt
    - Best-effort; any error is swallowed to avoid impacting the pipeline.
    """
    global _PROMPT_SEQ
    try:
        base = os.environ.get("WMRA_LOG_DIR")
        if not base:
            # Fallback to output/logs/<RUN_TS>
            run_ts = os.environ.get("RUN_TS") or time.strftime("%Y%m%d_%H%M%S")
            base = os.path.join(os.getcwd(), "output", "logs", run_ts)
        folder = os.path.join(base, "prompts")
        os.makedirs(folder, exist_ok=True)
        _PROMPT_SEQ += 1
        fname = f"{_PROMPT_SEQ:03d}_{_sanitize_filename(label)}_{int(time.time()*1000)}.txt"
        path = os.path.join(folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt if isinstance(prompt, str) else str(prompt))
    except Exception:
        # Do not fail the pipeline due to prompt logging issues
        return


def log_llm_response(label: str, content: Any) -> None:
    """Persist the raw LLM response alongside prompts for side-by-side debugging."""
    try:
        base = os.environ.get("WMRA_LOG_DIR")
        if not base:
            run_ts = os.environ.get("RUN_TS") or time.strftime("%Y%m%d_%H%M%S")
            base = os.path.join(os.getcwd(), "output", "logs", run_ts)
        folder = os.path.join(base, "prompts")
        os.makedirs(folder, exist_ok=True)
        fname = f"{_sanitize_filename(label)}_response_{int(time.time()*1000)}.txt"
        path = os.path.join(folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))
    except Exception:
        return


def build_candidate_selection_prompt(title: str, req_artist: str, items: List[Dict[str, Any]]) -> str:
    """Prompt for selecting the best search candidate by index.

    Priority rules enforce main-title equality so hymn variants like
    "Amazing Grace (Simplified)" count as exact matches, while phrase
    matches like "This Is Amazing Grace" do not.
    """
    rules = (
        "You are matching hymn/worship song search results to a user request.\n"
        "Choose the SINGLE best candidate by index.\n"
        "STRICT RULES (highest priority first):\n"
        "1) MAIN TITLE match dominates all else. The candidate's MAIN title text must equal the requested title (case-insensitive).\n"
        "   - Accept short arrangement descriptors appended in parentheses or after a hyphen as the same composition (e.g., '(Simplified)', '(Hymn)', '(Traditional)', '(Hymn Sheet)').\n"
        "   - Do NOT treat phrase-containing or retitled songs as matches when the MAIN title differs (e.g., 'This Is Amazing <Title>', '... (My Chains Are Gone)').\n"
        "2) Among MAIN-TITLE matches: prefer exact equality (no extra text); next prefer arrangement descriptors from the allowlist above; next benign performance descriptors (e.g., 'Live', 'Acoustic'); finally other appended text.\n"
        "3) Use artist only as a tie-breaker AFTER title/variant preference.\n"
        "4) Prefer canonical worship arrangements over covers/lead sheets/scores when still tied.\n"
        "Return STRICT JSON: {\"index\": <int>, \"confidence\": <float 0..1>, \"reason\": \"short\"}."
    )
    return (
        f"{rules}\nRequested Title: {title}\nRequested Artist: {req_artist or 'N/A'}\nCANDIDATES: {items}\n"
    )


def build_error_doctor_prompt(context: Dict[str, Any]) -> str:
    """Prompt for the automation error doctor to choose one recovery action."""
    import json as _json

    return (
        "You are an automation error doctor for a sheet-music site.\n"
        "Given the failure context, choose ONE recovery action that is most likely to succeed.\n"
        "Allowed actions: wait_longer, scroll_to_key_button, refresh_page, open_parts_then_key, back_to_results, "
        "scroll_to_orchestration, click_chords_then_orchestration, scroll_to_image, scroll_to_next_button, reopen_orchestration.\n"
        "Return STRICT JSON: {\"action\": <one>, \"why\": \"short reason\"}.\n\n"
        f"Context: {_json.dumps(context)[:1800]}\n"
    )


def build_instrument_selection_prompt(
    requested_instrument: str,
    req_canon: str,
    options_source: List[str],
    req_is_part: bool,
) -> str:
    """Prompt for selecting the best instrument option from a dropdown."""

    def _canon(s: str) -> str:
        # minimal local canonicalization for display context only
        import re as _re
        t = (s or "").strip().lower()
        t = _re.sub(r"[^a-z0-9\s/+-]", " ", t)
        t = _re.sub(r"\s+", " ", t).strip()
        return t

    indexed = [{"i": i, "label": lab, "canon": _canon(lab)} for i, lab in enumerate(options_source)]
    if req_is_part:
        rules_extra = (
            "- Prefer sectioned labels (e.g., '1/2', '1 & 2') when multiple horn options exist.\n"
            "- DO NOT choose generic or choral/piano options unless the request was for those.\n"
        )
    else:
        rules_extra = (
            "- If the request is 'Piano/Vocal' or 'Choir', you may choose those exact categories.\n"
        )
    return (
        "You are selecting an instrument option from a dropdown.\n"
        f"Requested instrument (raw): {requested_instrument}\n"
        f"Requested instrument (canonical): {req_canon}\n"
        "Options (choose EXACTLY one by index or label; you MUST select a label whose canonical form equals the requested canonical if present):\n"
        f"{indexed}\n"
        "Rules:\n"
        "- STRICT: Only select a label whose canonical form equals requested canonical when such labels exist.\n"
        f"{rules_extra}"
        "- Return STRICT JSON as either {\"index\": <int>} or {\"label\": \"<exact option label>\"}."
    )


def build_key_choice_prompt(requested_key: str, available_labels: List[str]) -> str:
    """Prompt for choosing the best key label from available buttons."""
    mapping = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
        "A#": 10, "Bb": 10, "B": 11
    }
    options = "\n- " + "\n- ".join(available_labels)
    return (
        "Task: Select the SINGLE best key label from the provided list.\n"
        f"Requested key: {requested_key}\n"
        f"Available labels (choose exactly one of these):\n{options}\n\n"
        "Rules:\n"
        "- Choose the label whose pitch class is the smallest chromatic distance (0–6) from the requested key.\n"
        "- If two are equally close, prefer the label ABOVE the requested key (positive direction).\n"
        "- Treat enharmonics as equal (a label like 'A#/Bb' represents both).\n"
        "Return STRICT JSON: {\"label\": <one of the provided labels verbatim>}"
        "\nMapping (for your reasoning): " + str(mapping)
    )


def build_alternate_readings_ranking_prompt(items: List[Dict[str, Any]]) -> str:
    """Prompt to rank alternate instrument/key candidates (best-first)."""
    return (
        "Task: Rank alternate parts a French horn player can READ DIRECTLY (no extra transposition).\n"
        "Prefer BRASS first, then SAX. Avoid others unless nothing else exists.\n"
        f"Candidates: {items}\n"
        "Return STRICT JSON: {\"order\": [indices in best-first order]}"
    )


def build_order_parser_prompt(instruction: str) -> str:
    """Prompt for parsing PDF instruction into structured order metadata."""
    parser_system = (
        "You are a precise data extraction assistant. "
        "Given an instruction about processing an 'order of worship' PDF, you MUST return "
        "a single strict JSON object with exactly these keys: "
        "pdf_name (string, must include '.pdf'), "
        "default_instrument (string), "
        "overrides (object mapping zero-based integer indices to instrument strings), "
        "order_folder (string: folder name for this run). "
        "Do not include any explanation text. If something is missing, return an empty string or empty object for that key."
    )
    return (
        parser_system
        + "\n\nInstruction:\n"
        + instruction
        + "\n\nReturn ONLY the JSON object."
    )


def build_song_extractor_prompt(pdf_text: str, user_instruction: str | None = None) -> str:
    """Prompt for extracting songs from order-of-worship raw text, honoring user constraints.

    If user_instruction includes constraints like "do not download <title>",
    "only download the fourth song", or specific indices/titles to include,
    the model must apply them strictly to the returned set.
    """
    song_extract_system = (
        "You are a precise parser for 'order of worship' text. "
        "Input is RAW TEXT extracted from a PDF. Identify the list of songs and return a single strict JSON object "
        "mapping zero-based integer indices to objects with exactly: title (string), artist (string), key (string). "
        "If details cannot be found for a field, use an empty string. Do not fabricate songs not present in the text. "
        "Return only JSON, no extra commentary."
    )

    # NEW: pattern-first guidance for bracketed lines and robust filtering
    pattern_hints = (
        "\n\nDETECTION HINTS (use these rules in order):\n"
        "1) Treat any line that contains a left bracket '[' and a right bracket ']' with the form:\n"
        "   <Title> [ <ArtistOrTag> in <Key> ]\n"
        "   as a STRONG song candidate. The <Title> is everything before the ' ['; trim leading timecodes like '4:56 ' or '1:00 '.\n"
        "   Inside the brackets, split on the last occurrence of ' in ' (case-insensitive) to get <ArtistOrTag> and <Key>.\n"
        "   - If <ArtistOrTag> equals or contains phrases like 'Default Arrangement', 'Arrangement', or 'Default', set artist=\"\".\n"
        "   - Normalize <Key> to a concise musical key token (A–G with optional #/b and optional 'm' for minor).\n"
        "     Examples: A, Bb, F#, Dm. If the key cannot be deduced, use an empty string.\n"
        "2) Ignore non-song operational rows: lines containing words like 'Scripture', 'Sermon', 'Announcements', 'Offering', "
        "'Prayer', 'Welcome', 'Length', 'Rehearsal', or section headings/tracks of song form (e.g., 'Intro, V1, C1, B×2').\n"
        "3) Titles may include parentheses and punctuation—keep them verbatim (e.g., 'Firm Foundation (He Won't)').\n"
        "4) If a song appears more than once, keep each occurrence in appearance order (they may have different keys).\n"
        "5) Only extract songs you can anchor to the bracket pattern; do not infer songs from adjacent lines.\n"
        "6) Keys may be written with Unicode flats/sharps. Prefer ASCII forms: 'B♭'→'Bb', 'E♭'→'Eb', 'F♯'→'F#'.\n"
    )

    # Keep your strict selection logic, unchanged, but mention order mapping to the detected list
    rules = (
        "\n\nSELECTION RULES (apply strictly if present):\n"
        "- If the user requests to exclude songs by title (e.g., 'do not download Amazing Grace'), omit those titles.\n"
        "- If the user requests to include only certain songs by ordinal (first/second/third/fourth/...), interpret ordinals in the appearance order in RAW TEXT and return only those.\n"
        "- If the user specifies explicit indices, treat them as 1-based unless clearly marked zero-based, and include only those.\n"
        "- If both include and exclude constraints are provided, apply exclusions first, then include filters.\n"
        "- If constraints produce zero songs, return an empty JSON object.\n"
        "- Never fabricate songs; only pick from titles present in RAW TEXT.\n"
    )

    # Tiny examples to nudge the model toward correct slicing
    examples = (
        "\n\nEXAMPLES (pattern application):\n"
        "Line: '4:56 Praise [ Elevation Worship in E ]' → {\"title\":\"Praise\",\"artist\":\"Elevation Worship\",\"key\":\"E\"}\n"
        "Line: '3:25 Because He Lives (Amen) [ Matt Maher in A ]' → {\"title\":\"Because He Lives (Amen)\",\"artist\":\"Matt Maher\",\"key\":\"A\"}\n"
        "Line: '4:00 Firm Foundation (He Won't) [ Default Arrangement in D ]' → {\"title\":\"Firm Foundation (He Won't)\",\"artist\":\"\",\"key\":\"D\"}\n"
    )

    ui = (user_instruction or "").strip()
    ui_block = ("\n\nUSER REQUESTS:\n" + ui) if ui else ""

    return (
        song_extract_system
        + pattern_hints
        + rules
        + examples
        + ui_block
        + "\n\nRAW ORDER OF WORSHIP TEXT:\n"
        + pdf_text
        + "\n\nReturn ONLY the JSON object."
    )


def build_single_song_parser_prompt(instruction: str) -> str:
    """Prompt for parsing a single natural-language instruction into title/instrument/key."""
    return (
        "You are a JSON API that extracts structured metadata from a natural-language "
        "instruction about downloading sheet music. Given the instruction, return a JSON "
        "object with the fields 'title', 'instrument' and 'key'. If any element is missing, "
        "return an empty string for that field. Do not include any extra keys or text.\n\n"
        f"Instruction: {instruction}\n\n"
        "JSON:"
    )
