"""End-to-end: audio (or MIDI) file -> score files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .notes import Note

MIDI_SUFFIXES = {".mid", ".midi"}


@dataclass
class Options:
    engine: str = "auto"  # auto | dsp | neural
    bpm: float | None = None  # fixed tempo instead of beat tracking
    time_signature: str = "4/4"
    subdivisions: int = 4  # grid steps per beat (4 = sixteenths, 2 = eighths)
    pickup: float | None = None  # beats of anacrusis before the first bar
    split: int = 60  # hand-split MIDI pitch
    title: str | None = None
    formats: tuple[str, ...] = ("musicxml", "midi", "html")  # + "pdf", "performance"


@dataclass
class Result:
    notes: list[Note]
    bpm: float
    key: str | None
    outputs: dict[str, Path] = field(default_factory=dict)


def read_midi_notes(path: str | Path) -> list[Note]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [Note(n.pitch, n.start, n.end, n.velocity)
             for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def write_performance_midi(notes: list[Note], path: str | Path) -> Path:
    """The raw (unquantized) transcription, useful for listening back."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Piano")
    piano.notes = [pretty_midi.Note(n.velocity, n.pitch, n.onset, n.offset) for n in notes]
    pm.instruments.append(piano)
    pm.write(str(path))
    return Path(path)


def run(input_path: str | Path, out_dir: str | Path, opts: Options | None = None,
        log=print) -> Result:
    from music21 import meter

    from . import quantize as Q
    from .score import build_score, write_html, write_midi, write_musicxml, write_pdf

    opts = opts or Options()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    title = opts.title or " ".join(stem.replace("_", " ").replace("-", " ").split())

    y = sr = None
    if input_path.suffix.lower() in MIDI_SUFFIXES:
        log(f"Reading MIDI notes from {input_path.name}")
        notes = read_midi_notes(input_path)
    else:
        from .audio import SAMPLE_RATE, load_audio
        from .transcribe import neural_available, transcribe

        engine = opts.engine
        if engine == "auto":
            engine = "neural" if neural_available() else "dsp"
        sr = SAMPLE_RATE
        log(f"Loading {input_path.name}")
        y = load_audio(str(input_path), sr)
        log(f"Transcribing {len(y) / sr:.1f}s of audio with the {engine} engine")
        notes = transcribe(str(input_path), engine=engine, y=y, sr=sr)
    log(f"Found {len(notes)} notes")
    if not notes:
        raise RuntimeError("No notes were detected in the input.")

    end = max(n.offset for n in notes)
    if opts.bpm:
        beats = Q.uniform_beats(opts.bpm, notes[0].onset, end)
    else:
        beats = Q.estimate_beats(y, sr) if y is not None else Q.beats_from_notes(notes)
        if len(beats) < 4:
            beats = Q.uniform_beats(Q.tempo_of(beats) if len(beats) >= 2 else 120.0, notes[0].onset, end)
        beats = Q.normalize_tempo(beats)
    bpm = opts.bpm or Q.tempo_of(beats)
    beats_per_bar = meter.TimeSignature(opts.time_signature).barDuration.quarterLength
    qnotes = Q.quantize(notes, beats, opts.subdivisions, int(beats_per_bar), opts.pickup)
    log(f"Tempo ~{bpm:.0f} BPM, quantized to 1/{opts.subdivisions} beat")

    score = build_score(qnotes, title=title, time_signature=opts.time_signature,
                        bpm=bpm, split=opts.split)
    key = None
    if score.detected_key is not None:
        key = f"{score.detected_key.tonic.name} {score.detected_key.mode}"
        log(f"Key: {key}")

    result = Result(notes, bpm, key)
    xml_path = write_musicxml(score, out_dir / f"{stem}.musicxml")
    result.outputs["musicxml"] = xml_path
    if "midi" in opts.formats:
        result.outputs["midi"] = write_midi(score, out_dir / f"{stem}.score.mid")
    if "performance" in opts.formats:
        result.outputs["performance"] = write_performance_midi(notes, out_dir / f"{stem}.performance.mid")
    if "html" in opts.formats:
        subtitle = f"{len(notes)} notes · ~{bpm:.0f} BPM" + (f" · {key}" if key else "")
        result.outputs["html"] = write_html(xml_path, out_dir / f"{stem}.html", title, subtitle)
    if "pdf" in opts.formats:
        pdf = write_pdf(xml_path, out_dir / f"{stem}.pdf")
        if pdf is None:
            log("PDF skipped: MuseScore not found (install it, or open the .musicxml file in any notation app)")
        else:
            result.outputs["pdf"] = pdf
    return result
