import sys
import types

import numpy as np
import pytest

from pianoscribe.demo import demo_piece
from pianoscribe.evaluate import note_f1
from pianoscribe.notes import Note
from pianoscribe.synth import synthesize
from pianoscribe.transcribe import remove_attack_ghosts, transcribe, transcribe_dsp


def test_single_notes_across_the_keyboard():
    ref = [Note(p, 0.6 * i, 0.6 * i + 0.45, 90) for i, p in enumerate([33, 45, 57, 60, 69, 81, 93])]
    est = transcribe_dsp(synthesize(ref))
    assert note_f1(ref, est).f1 == 1.0


def test_chords():
    chords = [(48, 52, 55), (53, 57, 60), (43, 50, 55, 59), (52, 60, 67), (41, 57, 64, 69)]
    ref = [Note(p, 0.8 * i, 0.8 * i + 0.7, 85) for i, c in enumerate(chords) for p in c]
    s = note_f1(ref, transcribe_dsp(synthesize(ref)))
    assert s.recall >= 0.85 and s.precision >= 0.9, s


def test_octave_doubled_chords_known_limitation():
    # An octave doubling inside a dense chord shares every partial with the
    # lower note; the DSP engine finds the chord but often misses the upper
    # octave. (The neural engine handles this far better.) Guard against
    # regressions without pretending it is solved.
    chords = [(48, 52, 55, 60), (53, 57, 60, 65), (43, 55, 59, 62, 67)]
    ref = [Note(p, 0.8 * i, 0.8 * i + 0.7, 85) for i, c in enumerate(chords) for p in c]
    s = note_f1(ref, transcribe_dsp(synthesize(ref)))
    assert s.precision >= 0.9 and s.recall >= 0.7, s


def test_repeated_key_is_split_into_separate_notes():
    ref = [Note(64, 0.25 * i, 0.25 * i + 0.22, 90) for i in range(8)]
    est = transcribe_dsp(synthesize(ref))
    assert note_f1(ref, est).f1 >= 0.85


def test_two_hand_demo_piece():
    ref = demo_piece()
    s = note_f1(ref, transcribe_dsp(synthesize(ref)))
    assert s.f1 >= 0.85, s


def test_silence_gives_no_notes():
    assert transcribe_dsp(np.zeros(22050, dtype=np.float32)) == []


def test_attack_ghosts_removed():
    notes = [Note(60, 1.0, 2.0, 100), Note(61, 1.01, 1.2, 40), Note(64, 1.0, 2.0, 40)]
    assert [n.pitch for n in remove_attack_ghosts(notes)] == [60, 64]


def test_neural_adapter_maps_events(monkeypatch, tmp_path):
    """The neural engine adapter, with the heavy model replaced by a stub."""
    fake = types.ModuleType("piano_transcription_inference")
    fake.sample_rate = 16000
    fake.load_audio = lambda path, sr, mono: (np.zeros(sr), sr)

    class PianoTranscription:
        def __init__(self, device):
            pass

        def transcribe(self, audio, midi_path):
            return {"est_note_events": [
                {"midi_note": 67, "onset_time": 0.5, "offset_time": 1.0, "velocity": 70},
                {"midi_note": 60, "onset_time": 0.5, "offset_time": 0.9, "velocity": 64},
            ]}

    fake.PianoTranscription = PianoTranscription
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "piano_transcription_inference", fake)
    monkeypatch.setitem(sys.modules, "torch", torch)

    notes = transcribe(str(tmp_path / "x.wav"), engine="neural")
    assert notes == [Note(60, 0.5, 0.9, 64), Note(67, 0.5, 1.0, 70)]


def test_unknown_engine():
    with pytest.raises(ValueError):
        transcribe("x.wav", engine="magic", y=np.zeros(100, dtype=np.float32))
