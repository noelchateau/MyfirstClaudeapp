"""Build a two-staff piano score from quantized notes, and export it."""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .notes import QuantizedNote


HAND_SPAN = 14  # semitones a hand can comfortably cover (a ninth/tenth)


def _split_chord(pitches: list[int], split: int) -> int:
    """Index into sorted ``pitches`` where the right hand begins."""
    if pitches[0] >= split:
        return 0
    if pitches[-1] < split:
        return len(pitches)
    if pitches[-1] - pitches[0] <= 7:  # small cluster across the split: one hand
        return 0 if sum(pitches) / len(pitches) >= split else len(pitches)
    # Otherwise split at the widest interval that leaves each hand playable,
    # preferring splits near the split point on ties.
    best, best_key = None, None
    for i in range(1, len(pitches)):
        if pitches[i - 1] - pitches[0] > HAND_SPAN or pitches[-1] - pitches[i] > HAND_SPAN:
            continue
        key = (pitches[i] - pitches[i - 1], -abs((pitches[i] + pitches[i - 1]) / 2 - split))
        if best_key is None or key > best_key:
            best, best_key = i, key
    if best is None:
        best = sum(p < split for p in pitches)
    return best


def split_hands(notes: list[QuantizedNote], split: int = 60) -> tuple[list[QuantizedNote], list[QuantizedNote]]:
    """Assign notes struck together to right/left hand.

    Small clusters go to one hand according to their register; wide chords
    are divided at their largest internal gap so neither hand has to stretch
    beyond about a tenth. ``split`` (MIDI number, 60 = middle C) is the
    register boundary.
    """
    right: list[QuantizedNote] = []
    left: list[QuantizedNote] = []
    by_start: dict[Fraction, list[QuantizedNote]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    for start in sorted(by_start):
        chord = sorted(by_start[start], key=lambda n: n.pitch)
        i = _split_chord([n.pitch for n in chord], split)
        left.extend(chord[:i])
        right.extend(chord[i:])
    return right, left


def _respell(pitch, key):
    """Prefer flats in flat keys, sharps in sharp (and natural) keys."""
    if pitch.accidental is None or pitch.accidental.alter == 0:
        return pitch
    wants_flats = key is not None and key.sharps < 0
    if (pitch.accidental.alter > 0) == wants_flats:
        return pitch.getEnharmonic()
    return pitch


def _fill_staff(part, notes: list[QuantizedNote], total: Fraction, key,
                max_gap: Fraction = Fraction(1, 2)) -> None:
    """Write ``notes`` into ``part`` as a single voice of chords and rests.

    Notes starting together become a chord. A chord lasts until its longest
    note ends or the next chord begins, whichever is first, which keeps the
    staff readable for typical piano textures. Gaps of at most ``max_gap``
    beats before the next chord are absorbed into the previous chord: release
    times are imprecise and such tiny rests mostly add clutter.
    """
    from music21 import chord, note, pitch as m21pitch

    groups: dict[Fraction, list[QuantizedNote]] = {}
    for n in notes:
        groups.setdefault(n.start, []).append(n)
    starts = sorted(groups)
    pos = Fraction(0)
    for i, s in enumerate(starts):
        if s > pos:
            part.append(note.Rest(quarterLength=s - pos))
        nxt = starts[i + 1] if i + 1 < len(starts) else total
        dur = min(max(n.duration for n in groups[s]), nxt - s)
        if i + 1 < len(starts) and nxt - s - dur <= max_gap:
            dur = nxt - s
        pitches = sorted({n.pitch for n in groups[s]})
        vel = max(n.velocity for n in groups[s])
        spelled = [_respell(m21pitch.Pitch(midi=p), key) for p in pitches]
        el = note.Note(spelled[0]) if len(spelled) == 1 else chord.Chord(spelled)
        el.quarterLength = dur
        el.volume.velocity = vel
        part.append(el)
        pos = s + dur
    if pos < total:
        part.append(note.Rest(quarterLength=total - pos))


def build_score(notes: list[QuantizedNote], *, title: str = "Transcription",
                time_signature: str = "4/4", bpm: float | None = None,
                split: int = 60, composer: str | None = None):
    """Create a ``music21.stream.Score`` with treble and bass staves."""
    from music21 import clef, instrument, layout, metadata, meter, stream, tempo
    from music21 import key as m21key

    ts = meter.TimeSignature(time_signature)
    bar = Fraction(ts.barDuration.quarterLength).limit_denominator(64)
    end = max((n.end for n in notes), default=bar)
    total = max(bar, -(-end // bar) * bar)  # round up to a full bar

    # Key estimate (Krumhansl-Schmuckler) for the key signature and spelling.
    detected = None
    if notes:
        probe = stream.Stream()
        from music21 import note as m21note

        for n in notes:
            probe.insert(float(n.start), m21note.Note(n.pitch, quarterLength=float(n.duration)))
        try:
            detected = probe.analyze("key")
        except Exception:  # analysis can fail on very sparse input
            detected = None

    right, left = split_hands(notes, split)
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    score.metadata.composer = composer or ""  # otherwise music21 credits itself
    staves = []
    for name, part_notes, staff_clef in (("Piano RH", right, clef.TrebleClef()),
                                          ("Piano LH", left, clef.BassClef())):
        part = stream.PartStaff(id=name)
        part.partName = "Piano" if name.endswith("RH") else ""
        part.insert(0, instrument.Piano())
        part.append(staff_clef)
        if detected is not None:
            part.append(m21key.KeySignature(detected.sharps))
        part.append(meter.TimeSignature(time_signature))
        if bpm and name.endswith("RH"):
            part.append(tempo.MetronomeMark(number=round(bpm)))
        _fill_staff(part, part_notes, total, detected)
        staves.append(part)

    for part in staves:
        part.makeMeasures(inPlace=True)
        part.makeTies(inPlace=True)
        score.insert(0, part)
    score.insert(0, layout.StaffGroup(staves, name="Piano", abbreviation="Pno.", symbol="brace"))
    score.detected_key = detected  # convenience for callers
    return score


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def write_musicxml(score, path: str | Path) -> Path:
    path = Path(path)
    score.write("musicxml", fp=str(path))
    return path


def write_midi(score, path: str | Path) -> Path:
    path = Path(path)
    score.write("midi", fp=str(path))
    return path


def find_musescore() -> str | None:
    for exe in ("mscore", "musescore", "mscore4", "musescore4", "MuseScore4", "mscore3", "musescore3"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def write_pdf(musicxml_path: str | Path, pdf_path: str | Path) -> Path | None:
    """Engrave a PDF with MuseScore if it is installed; returns None otherwise."""
    exe = find_musescore()
    if exe is None:
        return None
    subprocess.run([exe, "-o", str(pdf_path), str(musicxml_path)], check=True,
                   capture_output=True, timeout=300)
    return Path(pdf_path)


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #fdfcf8; --fg: #1d1d1f; --muted: #6b6b70; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.5 system-ui, -apple-system, sans-serif; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid #e4e2da; }}
  h1 {{ margin: 0; font-size: 20px; }}
  p {{ margin: 4px 0 0; color: var(--muted); }}
  #score {{ background: #fff; margin: 16px; padding: 8px; border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
</style>
</head>
<body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<div id="score">Loading score&hellip;</div>
<script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@2.1.3/build/opensheetmusicdisplay.min.js"></script>
<script>
  const xml = {xml_json};
  document.getElementById("score").textContent = "";
  const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {{
    autoResize: true, drawTitle: false, backend: "svg"
  }});
  osmd.load(xml).then(() => osmd.render()).catch(err => {{
    document.getElementById("score").textContent = "Could not render score: " + err;
  }});
</script>
</body>
</html>
"""


def write_html(musicxml_path: str | Path, html_path: str | Path, title: str, subtitle: str = "") -> Path:
    """A self-contained web page that renders the score in any browser."""
    import html
    import json

    xml = Path(musicxml_path).read_text(encoding="utf-8")
    page = _HTML.format(title=html.escape(title), subtitle=html.escape(subtitle),
                        xml_json=json.dumps(xml).replace("</", "<\\/"))
    Path(html_path).write_text(page, encoding="utf-8")
    return Path(html_path)
