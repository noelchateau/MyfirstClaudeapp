"""Note-level accuracy metrics (onset-only, as in MIREX / mir_eval)."""

from __future__ import annotations

from dataclasses import dataclass

from .notes import Note


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    matched: int


def note_f1(reference: list[Note], estimate: list[Note], onset_tol: float = 0.05) -> Scores:
    """Greedy one-to-one matching: same pitch, onset within ``onset_tol`` seconds."""
    used: set[int] = set()
    matched = 0
    for r in reference:
        best, best_d = None, onset_tol
        for i, e in enumerate(estimate):
            if i in used or e.pitch != r.pitch:
                continue
            d = abs(e.onset - r.onset)
            if d <= best_d:
                best, best_d = i, d
        if best is not None:
            used.add(best)
            matched += 1
    p = matched / len(estimate) if estimate else 0.0
    r = matched / len(reference) if reference else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return Scores(p, r, f, matched)
