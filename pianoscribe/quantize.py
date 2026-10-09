"""Map performed note times (seconds) onto a rhythmic grid (beats).

Beat times come either from a beat tracker run on the audio or from a fixed
tempo. Note times are converted to beat positions by interpolating between
neighbouring beats, so moderate tempo changes (rubato) do not throw off the
notation; positions are then rounded to the nearest grid subdivision.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .notes import Note, QuantizedNote


def estimate_beats(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return np.asarray(beats, dtype=float)


def beats_from_notes(notes: list[Note]) -> np.ndarray:
    """Estimate a beat grid from the note onsets alone (used for MIDI input
    or when no audio is available): tempo from the onset autocorrelation."""
    import librosa

    if not notes:
        return np.array([0.0, 0.5])
    sr, hop = 22050, 256
    end = max(n.offset for n in notes)
    env = np.zeros(int(end * sr / hop) + 2)
    for n in notes:
        env[int(n.onset * sr / hop)] += n.velocity / 127
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, units="time")
    return np.asarray(beats, dtype=float)


def uniform_beats(bpm: float, start: float, end: float) -> np.ndarray:
    period = 60.0 / bpm
    return start + period * np.arange(int(np.ceil((end - start) / period)) + 2)


def tempo_of(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return 120.0
    return float(60.0 / np.median(np.diff(beats)))


def normalize_tempo(beats: np.ndarray, low: float = 50.0, high: float = 160.0) -> np.ndarray:
    """Beat trackers often lock onto half or double the written tempo. Keep
    the beat in a range that gives readable notation."""
    beats = np.asarray(beats, dtype=float)
    while len(beats) >= 2 and tempo_of(beats) > high:
        beats = beats[::2]
    while len(beats) >= 2 and tempo_of(beats) < low:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
    return beats


def _to_beats(t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Seconds -> fractional beat index, extrapolating linearly at the ends."""
    idx = np.arange(len(beats), dtype=float)
    out = np.interp(t, beats, idx)
    head = beats[1] - beats[0]
    tail = beats[-1] - beats[-2]
    out = np.where(t < beats[0], (t - beats[0]) / head, out)
    out = np.where(t > beats[-1], len(beats) - 1 + (t - beats[-1]) / tail, out)
    return out


def quantize(notes: list[Note], beats: np.ndarray, subdivisions: int = 4,
             beats_per_bar: int = 4, pickup: float | None = None) -> list[QuantizedNote]:
    """Quantize ``notes`` to ``1/subdivisions`` of a beat.

    The first note is placed at the start of a bar, or ``pickup`` beats before
    the first full bar if given (an anacrusis).
    """
    if not notes:
        return []
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 2:
        beats = uniform_beats(120.0, 0.0, max(n.offset for n in notes))
    on = _to_beats(np.array([n.onset for n in notes]), beats)
    off = _to_beats(np.array([n.offset for n in notes]), beats)
    grid = subdivisions
    q_on = np.round(on * grid).astype(int)
    q_off = np.maximum(np.round(off * grid).astype(int), q_on + 1)

    # Shift by whole beats only, so the tracked beat phase is preserved.
    shift = int(q_on.min()) // grid * grid
    if pickup:
        shift -= int(round((beats_per_bar - pickup) * grid))
    out = [
        QuantizedNote(n.pitch, Fraction(int(a - shift), grid), Fraction(int(b - a), grid), n.velocity)
        for n, a, b in zip(notes, q_on, q_off)
    ]
    # Drop duplicates (same key struck twice into one grid slot).
    seen: dict[tuple[int, Fraction], QuantizedNote] = {}
    for q in out:
        key = (q.pitch, q.start)
        if key not in seen or q.duration > seen[key].duration:
            seen[key] = q
    return sorted(seen.values(), key=lambda q: (q.start, q.pitch))
