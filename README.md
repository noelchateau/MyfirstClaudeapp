# pianoscribe

Turn a recording of solo piano into sheet music.

```
pianoscribe my_recording.mp3
```

produces, in `transcriptions/`:

| file | what it is |
| --- | --- |
| `my_recording.musicxml` | the score — open in MuseScore, Finale, Sibelius, Dorico, Noteflight… |
| `my_recording.html` | the score rendered in your browser, no software needed |
| `my_recording.score.mid` | the quantized score as MIDI |
| `my_recording.pdf` | engraved PDF (with `--pdf`, needs [MuseScore](https://musescore.org) installed) |
| `my_recording.performance.mid` | raw, unquantized transcription (with `--performance-midi`) |

![Demo output](docs/demo_score.png)

*Output of `pianoscribe --demo`: a synthesized two-hand "Ode to Joy" transcribed back to notation.*

## Install

Requires Python 3.10+. Input: WAV, FLAC, MP3, OGG, Opus, AIFF out of the box; M4A/AAC, WMA and video files (MP4, MOV…) when [ffmpeg](https://ffmpeg.org) is installed; MIDI (`.mid`) to skip transcription.

```bash
pip install -e .            # DSP engine, no GPU or model download
pip install -e ".[neural]"  # + the neural engine (recommended for real recordings)
```

## How it works

```
audio ──► note detection ──► beat tracking & quantization ──► hand split, key, notation ──► MusicXML / MIDI / HTML / PDF
```

1. **Note detection** (`pianoscribe/transcribe.py`), two engines:
   - **`neural`** — the ByteDance *High-resolution Piano Transcription* model
     (Kong et al. 2021, ~96% note F1 on the MAESTRO benchmark). Used automatically when
     `torch` and `piano_transcription_inference` are installed; downloads a ~165 MB
     checkpoint on first use. **Use this for real recordings.**
   - **`dsp`** — a dependency-light signal-processing engine: constant-Q spectrogram →
     iterative harmonic-salience estimation with spectral subtraction (Klapuri 2006) →
     onset-driven note tracking (a note can only begin at a detected hammer strike, and
     only on keys whose energy jumps there) → removal of semitone/octave "ghost" notes.
2. **Rhythm** (`quantize.py`) — beats are tracked in the audio and note times are mapped
   onto them by interpolation, so tempo drift and rubato don't wreck the notation. Positions
   are rounded to a grid (16ths by default). Half/double-tempo beat tracking is corrected.
3. **Notation** (`score.py`) — notes are split between the hands (by register, splitting
   wide chords at their largest gap), the key is estimated (Krumhansl–Schmuckler) for the
   key signature and sharp/flat spelling, and a grand-staff score is built with
   [music21](https://web.mit.edu/music21/).

MIDI files are accepted as input too — transcription is skipped and they are just notated.

## Options

```
pianoscribe INPUT [-o OUT_DIR] [--engine auto|dsp|neural] [--bpm 92]
                  [--time-signature 3/4] [--grid 2|3|4|6|8|12] [--pickup 1]
                  [--split 60] [--title "..."] [--pdf] [--performance-midi]
pianoscribe --demo          # synthesize a demo piece and transcribe it
```

Tips for the best score:
- If the rhythm looks wrong, give the tempo with `--bpm` and the meter with `--time-signature`.
- Use `--grid 2` (eighths) for simple music, `--grid 3`/`6`/`12` for triplet feel.
- If the piece starts with a pickup, pass `--pickup <beats>` so the barlines land right.

Python API:

```python
from pianoscribe.pipeline import Options, run
result = run("etude.wav", "out/", Options(time_signature="3/4"))
print(result.bpm, result.key, result.outputs["musicxml"])
```

## Accuracy and limitations

DSP engine, note-level F1 (pitch exact, onset within 50 ms) on synthesized piano audio
(measured with `pianoscribe/evaluate.py`; the tests enforce minimum thresholds):

| material | precision | recall | F1 |
| --- | --- | --- | --- |
| two-hand demo piece | 0.97 | 0.94 | 0.95 |
| random 1–4 note chords, A2–B5 | 0.98 | 0.93 | 0.95 |
| random chords with octave doublings | 0.96 | 0.84 | 0.90 |

Be aware:
- **Synthetic audio is easier than a real piano.** Expect noticeably lower DSP accuracy on
  real recordings, especially with heavy pedal, reverb, or dense textures. The neural engine
  is the right choice there.
- The DSP engine's main weakness is an **octave doubled inside a dense chord** (every partial
  of the upper note coincides with the lower one).
- Automatic transcription gives a strong first draft, not a finished edition. Expect to fix
  things like voicing within a hand, the first downbeat, and note lengths
  (pedalled notes) in a notation editor.
- Each hand is written as a single voice of chords; independent inner voices aren't separated.

## Development

```bash
pip install -e ".[dev]"
pytest
```

`pianoscribe/synth.py` (a small additive piano synthesizer) and `pianoscribe/evaluate.py`
(note F1) make it easy to measure changes to the transcriber.
