"""Audio loading helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050

# Formats libsndfile reads natively; everything else goes through ffmpeg.
DIRECT_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff", ".opus"}


def find_ffmpeg() -> str | None:
    """ffmpeg on the PATH, else the copy bundled with the imageio-ffmpeg package."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _decode_with_ffmpeg(path: str, sr: int) -> np.ndarray:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            f"Could not read {Path(path).name}: this format needs ffmpeg. "
            "Run `pip install imageio-ffmpeg` or install ffmpeg, or convert the file to WAV."
        )
    with tempfile.TemporaryDirectory() as tmp:
        wav = str(Path(tmp) / "decoded.wav")
        # -vn: ignore any video stream (MP4/MOV); keep the first audio track, mono.
        proc = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(sr), wav],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not Path(wav).exists():
            detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no audio track found"
            raise RuntimeError(f"Could not extract audio from {Path(path).name}: {detail}")
        import soundfile as sf

        y, _ = sf.read(wav, dtype="float32")
    return y


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio or video file as a mono float32 signal at ``sr``.

    WAV, FLAC, OGG, MP3, AIFF and Opus are read directly. MP4/MOV video,
    M4A/AAC and anything else ffmpeg understands are decoded with ffmpeg
    (bundled via the imageio-ffmpeg dependency).
    """
    import librosa

    if not Path(path).exists():
        raise RuntimeError(f"{path} not found")
    if Path(path).suffix.lower() in DIRECT_SUFFIXES:
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception:
            # e.g. an older libsndfile without MP3 support, or an unusual encoding
            y = _decode_with_ffmpeg(path, sr)
    else:
        y = _decode_with_ffmpeg(path, sr)
    if y.size == 0:
        raise RuntimeError(f"{Path(path).name} contains no audio")
    y = y / max(float(np.max(np.abs(y))), 1e-9)
    return y.astype(np.float32)
