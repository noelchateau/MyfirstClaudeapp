from pianoscribe.cli import main
from pianoscribe.demo import demo_piece
from pianoscribe.pipeline import write_performance_midi


def test_demo_end_to_end(tmp_path):
    assert main(["--demo", "-o", str(tmp_path), "--performance-midi"]) == 0
    for name in ("demo_ode_to_joy.musicxml", "demo_ode_to_joy.score.mid",
                 "demo_ode_to_joy.performance.mid", "demo_ode_to_joy.html"):
        assert (tmp_path / name).stat().st_size > 0


def test_midi_input_skips_transcription(tmp_path):
    from music21 import converter

    mid = write_performance_midi(demo_piece(), tmp_path / "ode.mid")
    assert main([str(mid), "-o", str(tmp_path / "out"), "--bpm", "96"]) == 0
    score = converter.parse(str(tmp_path / "out" / "ode.musicxml"))
    rh = [p.nameWithOctave for n in score.parts[0].recurse().notes for p in n.pitches]
    assert rh[:8] == ["E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4"]


def test_missing_file(tmp_path):
    assert main([str(tmp_path / "nope.wav")]) == 2
