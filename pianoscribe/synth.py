"""A small additive piano-like synthesizer, used for demos and tests.

It models what matters for transcription: inharmonic partials, a decaying
envelope with a sharp attack, brighter/longer tones in the bass, and release
when the key is lifted.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE
from .notes import Note


def synthesize(notes: list[Note], sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    end = max((n.offset for n in notes), default=0.0) + 1.0
    out = np.zeros(int(end * sr), dtype=np.float64)
    for n in notes:
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        B = 1e-4 * 2 ** ((n.pitch - 48) / 24)  # inharmonicity grows toward the treble
        dur = n.duration + 0.3  # release tail
        t = np.arange(int(dur * sr)) / sr
        decay = 1.0 + 3.0 * (n.pitch - 21) / 87  # higher notes decay faster
        env = np.exp(-decay * t) * np.minimum(1.0, t / 0.004)
        rel = np.clip((t - n.duration) / 0.25, 0.0, 1.0)
        env *= 1.0 - rel
        tone = np.zeros_like(t)
        for h in range(1, 16):
            fh = h * f0 * np.sqrt(1 + B * h * h)
            if fh > 0.45 * sr:
                break
            amp = (1.0 / h ** 1.1) * np.exp(-0.15 * h * t)
            if n.pitch < 40 and h == 1:
                amp *= 0.5  # weak bass fundamental, as on a real piano
            tone += amp * np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
        start = int(n.onset * sr)
        seg = (n.velocity / 127.0) * env * tone
        out[start:start + len(seg)] += seg[: len(out) - start]
    out += 1e-4 * rng.standard_normal(len(out))
    return (out / max(np.max(np.abs(out)), 1e-9) * 0.9).astype(np.float32)
