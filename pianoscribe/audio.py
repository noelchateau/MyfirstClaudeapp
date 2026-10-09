"""Audio loading helpers."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 22050


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file librosa/ffmpeg can read as a mono float32 signal at ``sr``."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)
