from fractions import Fraction as F

import numpy as np

from pianoscribe.notes import Note, QuantizedNote
from pianoscribe.quantize import normalize_tempo, quantize, uniform_beats
from pianoscribe.score import build_score, split_hands, write_html, write_musicxml


def test_quantize_snaps_to_grid_and_follows_rubato():
    # Beats slow down from 0.5 s to 0.7 s apart; notes sit slightly off each beat.
    beats = np.cumsum([0.0] + [0.5 + 0.05 * i for i in range(5)])
    notes = [Note(60 + i, b + 0.02 * (-1) ** i, b + 0.2, 80) for i, b in enumerate(beats[:4])]
    q = quantize(notes, beats, subdivisions=4)
    assert [n.start for n in q] == [0, 1, 2, 3]


def test_quantize_pickup():
    beats = uniform_beats(120, 0.0, 4.0)
    q = quantize([Note(67, 0.0, 0.5, 80), Note(72, 0.5, 1.0, 80)], beats, beats_per_bar=4, pickup=1)
    assert [n.start for n in q] == [3, 4]


def test_normalize_tempo_halves_fast_beats():
    fast = uniform_beats(240, 0.0, 10.0)
    assert abs(60 / np.median(np.diff(normalize_tempo(fast))) - 120) < 1


def test_split_hands():
    notes = [QuantizedNote(p, F(0), F(1)) for p in (36, 43, 48, 64, 67, 72)]
    right, left = split_hands(notes)
    assert [n.pitch for n in left] == [36, 43, 48]
    assert [n.pitch for n in right] == [64, 67, 72]
    # an octave straddling the split goes one note per hand
    right, left = split_hands([QuantizedNote(52, F(0), F(1)), QuantizedNote(64, F(0), F(1))])
    assert [n.pitch for n in left] == [52] and [n.pitch for n in right] == [64]


def test_build_score_and_export(tmp_path):
    from music21 import converter

    notes = [
        QuantizedNote(72, F(0), F(1)), QuantizedNote(74, F(1), F(1)),
        QuantizedNote(76, F(2), F(1, 2)), QuantizedNote(77, F(5, 2), F(3, 2)),
        QuantizedNote(48, F(0), F(4)), QuantizedNote(55, F(0), F(4)),
        QuantizedNote(70, F(4), F(1)),  # B-flat: should be spelled with a flat in F major
        QuantizedNote(53, F(4), F(4)), QuantizedNote(65, F(5), F(3)),
    ]
    score = build_score(notes, title="Test", time_signature="4/4", bpm=100)
    xml = write_musicxml(score, tmp_path / "t.musicxml")
    back = converter.parse(str(xml))
    rh, lh = back.parts
    assert len(rh.getElementsByClass("Measure")) == 2
    assert [c.name for c in lh.recurse().getElementsByClass("Clef")][:1] == ["bass"] or \
        lh.recurse().getElementsByClass("Clef")[0].sign == "F"
    names = [p.name for n in rh.recurse().notes for p in n.pitches]
    assert "B-" in names and "A#" not in names
    html = write_html(xml, tmp_path / "t.html", "Test")
    assert "opensheetmusicdisplay" in html.read_text()
