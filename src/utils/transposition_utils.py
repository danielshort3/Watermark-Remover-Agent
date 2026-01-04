"""
Utility functions for musical key transposition.

This module is adapted from the original Watermark‑Remover project and
provides helper functions to normalise key names and suggest alternate
instrument/key combinations when a requested key is unavailable.  The
agent uses these helpers to offer intelligent fallbacks when scraping
sheet music.  See the original project for full context.
"""
 # Moved to src/utils for better project structure.

import re
from typing import List, Dict, Tuple

# Mapping from key names to semitone numbers (C == 0).  Enharmonic
# equivalents share the same semitone value.
KEY_TO_SEMITONE: Dict[str, int] = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11,
}

# Reverse mapping from semitone number back to a representative key name.
SEMITONE_TO_KEY: Dict[int, str] = {
    0: 'C',
    1: 'C#/Db',
    2: 'D',
    3: 'D#/Eb',
    4: 'E',
    5: 'F',
    6: 'F#/Gb',
    7: 'G',
    8: 'G#/Ab',
    9: 'A',
    10: 'A#/Bb',
    11: 'B'
}

# Default transposition offsets for common orchestral parts.  A value
# indicates how many semitones the written key is transposed relative
# to concert pitch.  For example, a Bb instrument (Clarinet) reads
# music a major second above concert pitch (−2 semitones here because
# we subtract the offset when computing the written key).
INSTRUMENT_TRANSPOSITIONS: Dict[str, int] = {
    'Rhythm Chart': 0,
    'Acoustic Guitar': 0,
    'Flute 1/2': 0,
    'Flute/Oboe 1/2/3': 0,
    'Oboe': 0,
    'Clarinet 1/2': -2,
    'Bass Clarinet': -2,
    'Bassoon': 0,
    'French Horn 1/2': -7,
    'Trumpet 1,2': -2,
    'Trumpet 3': -2,
    'Trombone 1/2': 0,
    'Trombone 3/Tuba': 0,
    'Alto Sax': -9,
    'Tenor Sax 1/2': -2,
    'Bari Sax': -9,
    'Timpani': 0,
    'Percussion': 0,
    'Violin 1/2': 0,
    'Viola': 0,
    'Cello': 0,
    'Double Bass': 0,
    'String Reduction': 0,
    'String Bass': 0,
    'Lead Sheet (SAT)': 0,
}


def normalize_key(key: str) -> str:
    """Normalise a key string (e.g. 'ab ' -> 'Ab').

    Parameters
    ----------
    key: str
        The raw key string provided by the user.

    Returns
    -------
    str
        A canonical representation of the key.  Returns an empty
        string if no key is provided.
    """
    key = key.strip()
    if not key:
        return ""
    # Handle modulations like "F-G" by taking the starting key.
    mod_match = re.match(r"^([A-Ga-g](?:#|b)?)\s*-\s*([A-Ga-g](?:#|b)?)$", key)
    if mod_match:
        key = mod_match.group(1)
    # Split on whitespace and take the first token in case of
    # additional descriptors (e.g. "C major").
    key = key.split()[0]
    note = key[0].upper()
    accidental = key[1:].replace("B", "b")
    return note + accidental


def get_interval_name(semitones: int) -> str:
    """Return a musical interval name for a given semitone difference.

    Parameters
    ----------
    semitones: int
        The absolute distance in semitones between two notes.

    Returns
    -------
    str
        A human‑friendly name for the interval (e.g. 'Perfect Fifth').
    """
    intervals = {
        0: 'Perfect Unison',
        1: 'Minor Second',
        2: 'Major Second',
        3: 'Minor Third',
        4: 'Major Third',
        5: 'Perfect Fourth',
        6: 'Tritone',
        7: 'Perfect Fifth',
        8: 'Minor Sixth',
        9: 'Major Sixth',
        10: 'Minor Seventh',
        11: 'Major Seventh',
        12: 'Octave'
    }
    return intervals.get(semitones % 12, f'{semitones} semitones')


def _modular_distance(a: int, b: int) -> Tuple[int, str]:
    """Return the smallest distance and direction from a to b on the circle of fifths.

    Returns the absolute number of semitones and a string describing
    whether the interval is 'above', 'below' or 'none' (for unison).
    """
    up = (b - a) % 12
    down = (a - b) % 12
    if up < down:
        return up, 'above' if up != 0 else 'none'
    if down < up:
        return down, 'below' if down != 0 else 'none'
    # up == down -> symmetrical (e.g. tritone or unison)
    return up, 'none' if up != 0 else 'none'


def get_transposition_suggestions(
    available_keys: List[str], selected_instrument: str, target_key: str
) -> Dict[str, List[dict]]:
    """Suggest alternate instrument/key combinations when a target key is unavailable.

    This helper looks at the list of available keys and computes direct
    matches (where another instrument playing a different written key
    would still sound in the target concert key) and closest matches
    based on minimal semitone differences.

    Parameters
    ----------
    available_keys: list of str
        Keys that exist for the selected song (as discovered in your
        library).
    selected_instrument: str
        The instrument originally requested by the user.
    target_key: str
        The desired concert key (e.g. 'Bb').

    Returns
    -------
    dict
        A dictionary with two lists: ``'direct'`` for exact matches and
        ``'closest'`` for near matches.  Each entry contains the
        instrument, the key and the interval information.
    """
    target_key = normalize_key(target_key)
    # Bail out if we do not recognise the target key or instrument
    if target_key not in KEY_TO_SEMITONE:
        return {'direct': [], 'closest': []}
    if selected_instrument not in INSTRUMENT_TRANSPOSITIONS:
        return {'direct': [], 'closest': []}
    target_semitone = KEY_TO_SEMITONE[target_key]
    matches_direct: List[dict] = []
    matches_closest: List[dict] = []
    for instrument, transposition in INSTRUMENT_TRANSPOSITIONS.items():
        if instrument == selected_instrument:
            continue
        # Compute the written key (in semitones) required for this
        # instrument to sound the target concert key.
        required_written_semitone = (target_semitone - transposition) % 12
        # Convert to a named key (some names represent multiple enharmonics)
        required_written_key = SEMITONE_TO_KEY.get(required_written_semitone, 'Unknown')
        # If the required written key exists, record a direct match
        if required_written_key in available_keys:
            matches_direct.append({
                'instrument': instrument,
                'key': required_written_key,
                'difference': 0,
                'interval_direction': 'none',
                'interval': 'Perfect Unison'
            })
        else:
            # Determine the available semitone numbers
            available_semitones = [KEY_TO_SEMITONE.get(k, None) for k in available_keys if k in KEY_TO_SEMITONE]
            if not available_semitones:
                continue
            # Find the closest available key in terms of semitone distance
            diffs: List[Tuple[int, int, str]] = []
            for semitone in available_semitones:
                diff, direction = _modular_distance(required_written_semitone, semitone)
                diffs.append((diff, semitone, direction))
            diffs.sort(key=lambda x: x[0])
            closest_diff, closest_semitone, interval_direction = diffs[0]
            closest_key = SEMITONE_TO_KEY.get(closest_semitone, 'Unknown')
            interval_name = get_interval_name(closest_diff)
            matches_closest.append({
                'instrument': instrument,
                'key': closest_key,
                'difference': closest_diff,
                'interval_direction': interval_direction,
                'interval': interval_name
            })
    matches_closest.sort(key=lambda s: s['difference'])
    return {'direct': matches_direct, 'closest': matches_closest}
