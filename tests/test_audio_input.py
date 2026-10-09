import subprocess

import numpy as np
import pytest
import soundfile as sf

from pianoscribe import audio
from pianoscribe.audio import SAMPLE_RATE, find_ffmpeg, load_audio
from pianoscribe.cli import main
from pianoscribe.evaluate import note_f1
from pianoscribe.notes import Note
from pianoscribe.synth import synthesize
from pianoscribe.transcribe import transcribe_dsp

REF = [Note(p, 0.5 * i, 0.5 * i + 0.4, 90) for i, p in enumerate([60, 64, 67, 72, 67, 64, 60])]


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    path = tmp_path_factory.mktemp("in") / "scale.wav"
    sf.write(path, synthesize(REF), SAMPLE_RATE)
    return path


def _ffmpeg(*args):
    subprocess.run([find_ffmpeg(), "-loglevel", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="module")
def mp4(wav):
    """A real video file: 2 fps test pattern + AAC audio, like a phone recording."""
    path = wav.with_name("performance.mp4")
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=2", "-i", wav,
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac", path)
    return path


def test_wav_and_mp4_load_the_same_audio(wav, mp4):
    a, b = load_audio(str(wav)), load_audio(str(mp4))
    assert abs(len(a) - len(b)) < 0.1 * SAMPLE_RATE  # AAC adds a little padding
    assert note_f1(REF, transcribe_dsp(b)).f1 == 1.0


def test_mp4_end_to_end(mp4, tmp_path):
    assert main([str(mp4), "-o", str(tmp_path)]) == 0
    assert (tmp_path / "performance.musicxml").stat().st_size > 0


def test_wav_end_to_end(wav, tmp_path):
    assert main([str(wav), "-o", str(tmp_path)]) == 0
    assert (tmp_path / "scale.musicxml").stat().st_size > 0


def test_stereo_44k_wav(tmp_path):
    y = synthesize(REF, sr=44100)  # CD-quality stereo, resampled on load
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([y, 0.5 * y], axis=1), 44100)
    assert note_f1(REF, transcribe_dsp(load_audio(str(path)))).f1 == 1.0


def test_mp4_without_audio_gives_clear_error(tmp_path):
    path = tmp_path / "silent_video.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=2", "-t", "1", "-c:v", "mpeg4", path)
    with pytest.raises(RuntimeError, match="Could not extract audio"):
        load_audio(str(path))
    assert main([str(path), "-o", str(tmp_path / "out")]) == 1


def test_bundled_ffmpeg_used_when_not_on_path(mp4, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert "imageio_ffmpeg" in find_ffmpeg()
    assert len(load_audio(str(mp4))) > SAMPLE_RATE
