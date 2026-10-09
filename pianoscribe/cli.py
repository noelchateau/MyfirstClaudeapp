"""Command-line interface: ``pianoscribe recording.wav``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pianoscribe",
        description="Transcribe solo piano audio into sheet music (MusicXML, MIDI, HTML, PDF).",
    )
    p.add_argument("input", nargs="?", help="audio file (wav, mp3, flac, ...; also mp4/m4a) or a MIDI file")
    p.add_argument("-o", "--out-dir", default="transcriptions", help="output folder (default: %(default)s)")
    p.add_argument("--engine", choices=["auto", "dsp", "neural"], default="auto",
                   help="transcription engine; 'auto' uses neural when installed (default: %(default)s)")
    p.add_argument("--bpm", type=float, help="fixed tempo instead of automatic beat tracking")
    p.add_argument("--time-signature", "-t", default="4/4", help="e.g. 3/4, 6/8 (default: %(default)s)")
    p.add_argument("--grid", type=int, default=4, choices=[1, 2, 3, 4, 6, 8, 12],
                   help="grid steps per beat: 2=eighths, 3=triplets, 4=sixteenths (default: %(default)s)")
    p.add_argument("--pickup", type=float, help="beats of pickup (anacrusis) before the first full bar")
    p.add_argument("--split", type=int, default=60,
                   help="MIDI pitch dividing left/right hand, 60 = middle C (default: %(default)s)")
    p.add_argument("--title", help="score title (default: from the file name)")
    p.add_argument("--pdf", action="store_true", help="also engrave a PDF (requires MuseScore)")
    p.add_argument("--performance-midi", action="store_true",
                   help="also write the unquantized transcription as MIDI")
    p.add_argument("--demo", action="store_true",
                   help="synthesize a short demo piece and transcribe it (no input needed)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .pipeline import Options, run

    if args.demo:
        import soundfile as sf

        from .audio import SAMPLE_RATE
        from .demo import demo_piece
        from .synth import synthesize

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        args.input = str(out / "demo_ode_to_joy.wav")
        sf.write(args.input, synthesize(demo_piece()), SAMPLE_RATE)
        print(f"Wrote demo audio to {args.input}")
    elif not args.input:
        _parser().error("an input file is required (or use --demo)")
    if not Path(args.input).exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2

    formats = ["musicxml", "midi", "html"]
    if args.pdf:
        formats.append("pdf")
    if args.performance_midi:
        formats.append("performance")
    opts = Options(engine=args.engine, bpm=args.bpm, time_signature=args.time_signature,
                   subdivisions=args.grid, pickup=args.pickup, split=args.split,
                   title=args.title, formats=tuple(formats))
    try:
        result = run(args.input, args.out_dir, opts)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("\nWrote:")
    for kind, path in result.outputs.items():
        print(f"  {kind:12s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
