"""Audio loading helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 signal at ``sr``.

    WAV, FLAC, OGG, MP3, AIFF and Opus are read directly; anything else
    (M4A/AAC, WMA, video files...) is decoded with ffmpeg when it is installed.
    """
    import librosa

    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as err:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"Could not read {Path(path).name}: {err}. Install ffmpeg to open this format, "
                "or convert the file to WAV, FLAC or MP3."
            ) from err
        with tempfile.TemporaryDirectory() as tmp:
            wav = str(Path(tmp) / "decoded.wav")
            subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", wav],
                           check=True, capture_output=True)
            y, _ = librosa.load(wav, sr=sr, mono=True)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)
