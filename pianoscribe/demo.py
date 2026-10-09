"""A short two-hand demo piece ("Ode to Joy" over an Alberti-style bass)."""

from __future__ import annotations

from .notes import Note

# (beat, length_in_beats, midi_pitch)
_MELODY = [
    (0, 1, 64), (1, 1, 64), (2, 1, 65), (3, 1, 67),
    (4, 1, 67), (5, 1, 65), (6, 1, 64), (7, 1, 62),
    (8, 1, 60), (9, 1, 60), (10, 1, 62), (11, 1, 64),
    (12, 1.5, 64), (13.5, 0.5, 62), (14, 2, 62),
    (16, 1, 64), (17, 1, 64), (18, 1, 65), (19, 1, 67),
    (20, 1, 67), (21, 1, 65), (22, 1, 64), (23, 1, 62),
    (24, 1, 60), (25, 1, 60), (26, 1, 62), (27, 1, 64),
    (28, 1.5, 62), (29.5, 0.5, 60), (30, 2, 60),
]

# one chord per bar, broken as root-fifth-third-fifth in eighths
_CHORDS = [(48, 55, 52), (43, 50, 47), (48, 55, 52), (43, 50, 47),
           (48, 55, 52), (43, 50, 47), (43, 50, 47), (48, 55, 52)]


def demo_piece(bpm: float = 96.0) -> list[Note]:
    spb = 60.0 / bpm
    notes = [Note(p, b * spb, (b + 0.97 * d) * spb, 90) for b, d, p in _MELODY]
    for bar, (root, fifth, third) in enumerate(_CHORDS):
        for i, p in enumerate((root, fifth, third, fifth) * 2):
            b = bar * 4 + i * 0.5
            notes.append(Note(p, b * spb, (b + 0.97 * 0.5) * spb, 60))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))
