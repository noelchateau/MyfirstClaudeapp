"""Shared note data types."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Note:
    """A performed note, timed in seconds."""

    pitch: int  # MIDI note number (21 = A0 ... 108 = C8)
    onset: float  # seconds
    offset: float  # seconds
    velocity: int = 80  # 1..127

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass(frozen=True)
class QuantizedNote:
    """A note placed on the score grid, timed in quarter-note beats."""

    pitch: int
    start: Fraction  # beats from the start of the piece
    duration: Fraction  # beats
    velocity: int = 80

    @property
    def end(self) -> Fraction:
        return self.start + self.duration
