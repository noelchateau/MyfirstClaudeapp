"""Polyphonic piano transcription: audio signal -> list of :class:`Note`.

Two engines are available:

* ``dsp`` (default, no extra dependencies): a constant-Q spectrogram is analysed
  frame by frame with iterative harmonic-salience estimation and spectral
  subtraction (after Klapuri, "Multiple fundamental frequency estimation by
  summing harmonic amplitudes", 2006). Frame-level pitch activity is then
  turned into notes with gap closing, onset-based splitting of repeated notes
  and snapping to detected onsets.

* ``neural``: the ByteDance high-resolution piano transcription model
  (Kong et al., 2021) via the optional ``piano_transcription_inference``
  package. It is considerably more accurate on real recordings but requires
  PyTorch and a ~165 MB model download on first use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .notes import Note

LOWEST_KEY = 21  # A0
N_KEYS = 88
BINS_PER_SEMITONE = 3
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
HOP = 256  # ~11.6 ms at 22.05 kHz


@dataclass
class DSPParams:
    n_harmonics: int = 10
    max_polyphony: int = 10
    # Fraction of a detected note's estimated partials removed from the
    # spectrum before looking for further notes. Below 1 leaves room for
    # notes whose partials coincide with it (octaves, twelfths).
    subtraction: float = 0.6
    # A candidate is accepted when its salience exceeds both thresholds.
    abs_threshold: float = 0.15  # relative to the loudest salience in the piece
    rel_threshold: float = 0.2  # relative to the strongest pitch in the same frame
    min_note_ms: float = 45.0
    gap_fill_ms: float = 35.0
    onset_snap_ms: float = 40.0
    # Minimum rise (dB) of a key's own energy at an onset for it to count as struck.
    restrike_db: float = 2.5


# ---------------------------------------------------------------------------
# Spectral front end
# ---------------------------------------------------------------------------

def _cqt(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    fmin = librosa.midi_to_hz(LOWEST_KEY) * 2 ** (-1 / BINS_PER_OCTAVE)  # centre key on middle bin
    max_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(0.45 * sr / fmin)))
    n_bins = min(max_bins, N_KEYS * BINS_PER_SEMITONE + 3 * BINS_PER_OCTAVE)
    C = np.abs(
        librosa.cqt(
            y, sr=sr, hop_length=HOP, fmin=fmin, n_bins=n_bins,
            bins_per_octave=BINS_PER_OCTAVE,
        )
    )
    return C.T.astype(np.float32)  # (frames, bins)


def _harmonic_index(n_bins: int, n_harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    """CQT bin of each harmonic of each key, plus a validity mask."""
    keys = np.arange(N_KEYS)[:, None]
    h = np.arange(1, n_harmonics + 1)[None, :]
    centre = keys * BINS_PER_SEMITONE + 1
    idx = centre + np.round(BINS_PER_OCTAVE * np.log2(h)).astype(int)
    valid = idx < n_bins - 1
    return np.where(valid, idx, 0), valid


def _local_max(R: np.ndarray) -> np.ndarray:
    """Max over +/-1 bin to tolerate tuning deviations and piano inharmonicity."""
    M = R.copy()
    M[:, 1:] = np.maximum(M[:, 1:], R[:, :-1])
    M[:, :-1] = np.maximum(M[:, :-1], R[:, 1:])
    return M


def _smooth_partials(A: np.ndarray) -> np.ndarray:
    """Spectral smoothness: cap each overtone by the mean of its two
    neighbouring partials, so energy belonging to other notes (e.g. an octave
    or a twelfth above, whose partials coincide with ours) is not attributed
    to this candidate. The fundamental is left as is."""
    out = A.copy()
    nxt = np.concatenate([A[..., 2:], A[..., -1:]], axis=-1)  # partial h+1 (last: itself)
    out[..., 1:] = np.minimum(A[..., 1:], (A[..., :-1] + nxt) / 2.0)
    return out


def frame_salience(S: np.ndarray, params: DSPParams) -> np.ndarray:
    """Iterative multi-pitch estimation. Returns (frames, 88) salience of the
    pitches selected in each frame (0 where not selected)."""
    T, n_bins = S.shape
    idx, valid = _harmonic_index(n_bins, params.n_harmonics)
    h = np.arange(1, params.n_harmonics + 1)
    weights = (1.0 / h ** 0.8)[None, None, :] * valid[None]

    R = S.copy()
    act = np.zeros((T, N_KEYS), dtype=np.float32)
    first = None
    frames = np.arange(T)
    for _ in range(params.max_polyphony):
        A = _local_max(R)[:, idx] * valid[None]  # (T, 88, H)
        A = _smooth_partials(A)
        # Require a real (not missing) fundamental region: piano bass keys have
        # weak fundamentals, so look at the strongest of the first 3 partials.
        sal = np.sum(weights * A, axis=-1) * (A[..., :3].max(axis=-1) > 0)
        sal[act > 0] = 0.0  # never pick the same key twice in a frame
        best = np.argmax(sal, axis=1)
        val = sal[frames, best]
        if first is None:
            first = val.copy()
            global_ref = max(float(first.max()), 1e-9)
        ok = (val > params.abs_threshold * global_ref) & (val > params.rel_threshold * first)
        if not ok.any():
            break
        t_sel, p_sel = frames[ok], best[ok]
        act[t_sel, p_sel] = val[ok]
        amps = A[t_sel, p_sel]  # (n, H)
        for k in range(params.n_harmonics):
            bins = idx[p_sel, k]
            m = valid[p_sel, k]
            for d in (-1, 0, 1):
                b = np.clip(bins + d, 0, n_bins - 1)
                R[t_sel[m], b[m]] = np.maximum(R[t_sel[m], b[m]] - params.subtraction * amps[m, k], 0.0)
    return act


# ---------------------------------------------------------------------------
# Note tracking
# ---------------------------------------------------------------------------

def _segments(active: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _key_energy_db(S: np.ndarray) -> np.ndarray:
    """Energy around each key's fundamental and first two overtones, in dB."""
    idx, valid = _harmonic_index(S.shape[1], 3)
    E = (_local_max(S)[:, idx] * valid[None]).sum(axis=-1)
    return 20 * np.log10(E + 1e-6)


def track_notes(act: np.ndarray, S: np.ndarray, onset_frames: np.ndarray,
                sr: int, params: DSPParams) -> list[Note]:
    """Onset-driven note tracking.

    Piano notes always begin with a hammer strike, so a note may only start at
    a detected onset, and only for keys whose own energy jumps there. It then
    lasts while the key stays active (short gaps bridged) or until that key
    is struck again.
    """
    fps = sr / HOP
    min_len = max(1, int(round(params.min_note_ms / 1000 * fps)))
    gap = int(round(params.gap_fill_ms / 1000 * fps))
    look = max(2, int(round(params.onset_snap_ms / 1000 * fps)))
    T = act.shape[0]
    E = _key_energy_db(S)
    peak = max(float(act.max()), 1e-9)
    onset_frames = np.unique(np.clip(np.asarray(onset_frames, dtype=int), 0, T - 1))

    # Bridge short dropouts in frame activity.
    active = act > 0
    for p in range(N_KEYS):
        segs = _segments(active[:, p])
        for (_, e0), (s1, _) in zip(segs, segs[1:]):
            if s1 - e0 <= gap:
                active[e0:s1, p] = True

    # Which keys are struck at each onset?
    starts: list[list[int]] = [[] for _ in range(N_KEYS)]
    for o in onset_frames:
        after = slice(o, min(o + look + 1, T))
        before = slice(max(o - look, 0), o + 1)
        struck = active[after].any(axis=0)
        jump = E[after].max(axis=0) - E[before].min(axis=0)
        for p in np.flatnonzero(struck & (jump >= params.restrike_db)):
            if not starts[p] or o - starts[p][-1] >= min_len:
                starts[p].append(int(o))

    notes: list[Note] = []
    for p in range(N_KEYS):
        col = active[:, p]
        for i, s in enumerate(starts[p]):
            limit = starts[p][i + 1] if i + 1 < len(starts[p]) else T
            first_on = s + int(np.argmax(col[s:limit])) if col[s:limit].any() else limit
            e = first_on
            while e < limit and col[e]:
                e += 1
            if e - s < min_len:
                continue
            strength = float(act[s:min(s + look + 1, e), p].max(initial=0.0)) / peak
            velocity = int(np.clip(round(25 + 100 * np.sqrt(strength)), 1, 127))
            notes.append(Note(LOWEST_KEY + p, s / fps, e / fps, velocity))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def transcribe_dsp(y: np.ndarray, sr: int = SAMPLE_RATE,
                   params: DSPParams | None = None) -> list[Note]:
    import librosa

    params = params or DSPParams()
    # Lead-in silence so a note sounding at t=0 still produces an onset.
    pad = int(0.25 * sr)
    y = np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    S = _cqt(y, sr)
    S = S / max(float(S.max()), 1e-9)
    act = frame_salience(S, params)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=False)
    notes = remove_attack_ghosts(track_notes(act, S, onset_frames, sr, params))
    shift = pad / sr
    return [Note(n.pitch, max(n.onset - shift, 0.0), max(n.offset - shift, 0.0), n.velocity)
            for n in notes if n.offset > shift]


def remove_attack_ghosts(notes: list[Note], window: float = 0.03, semitone_ratio: float = 0.75,
                         octave_ratio: float = 0.65) -> list[Note]:
    """Drop likely spurious notes struck together with a much louder note:

    * a semitone neighbour (the broadband hammer attack smears into adjacent keys);
    * an octave neighbour (left over from shared partials: above a bass note
      with a strong second partial, or below a note whose even partials
      look like the upper half of a lower note's series).
    """
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if abs(m.onset - n.onset) > window:
                continue
            d = n.pitch - m.pitch
            if (abs(d) == 1 and n.velocity < semitone_ratio * m.velocity) or (
                abs(d) == 12 and n.velocity < octave_ratio * m.velocity
            ):
                ghost = True
                break
        if not ghost:
            keep.append(n)
    return keep


# ---------------------------------------------------------------------------
# Neural engine (optional)
# ---------------------------------------------------------------------------

def neural_available() -> bool:
    try:
        import piano_transcription_inference  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe_neural(path: str, device: str | None = None) -> list[Note]:
    import os
    import tempfile

    import torch
    from piano_transcription_inference import PianoTranscription, load_audio, sample_rate

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    audio, _ = load_audio(path, sr=sample_rate, mono=True)
    model = PianoTranscription(device=torch.device(device))
    with tempfile.TemporaryDirectory() as tmp:
        result = model.transcribe(audio, os.path.join(tmp, "out.mid"))
    return sorted(
        (
            Note(int(e["midi_note"]), float(e["onset_time"]), float(e["offset_time"]),
                 int(e["velocity"]))
            for e in result["est_note_events"]
        ),
        key=lambda n: (n.onset, n.pitch),
    )


def transcribe(path: str, engine: str = "auto", y: np.ndarray | None = None,
               sr: int = SAMPLE_RATE) -> list[Note]:
    """Transcribe an audio file. ``engine`` is ``auto``, ``dsp`` or ``neural``."""
    if engine == "auto":
        engine = "neural" if neural_available() else "dsp"
    if engine == "neural":
        if not neural_available():
            raise RuntimeError(
                "The neural engine needs: pip install torch piano_transcription_inference"
            )
        return transcribe_neural(path)
    if engine != "dsp":
        raise ValueError(f"unknown engine {engine!r}")
    if y is None:
        from .audio import load_audio

        y = load_audio(path, sr)
    return transcribe_dsp(y, sr)
