"""Drag-and-drop desktop app: a small local web server plus a browser page.

Run ``pianoscribe-app`` (or double-click the launcher in the project folder).
Everything runs on this computer; files never leave it.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

MAX_UPLOAD = 1024 ** 3  # 1 GB
ALLOWED_SUFFIXES = {".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff",
                    ".m4a", ".mp4", ".mov", ".aac", ".mid", ".midi"}
DOWNLOAD_TYPES = {
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "midi": "audio/midi",
    "performance": "audio/midi",
    "pdf": "application/pdf",
    "html": "text/html; charset=utf-8",
}


def default_output_root() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "Pianoscribe"


@dataclass
class Job:
    id: str
    filename: str
    state: str = "running"  # running | done | error
    log: list[str] = field(default_factory=list)
    error: str | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    folder: Path | None = None
    started: float = field(default_factory=time.time)

    def as_json(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "state": self.state, "log": self.log,
            "error": self.error, "summary": self.summary,
            "folder": str(self.folder) if self.folder else None,
            "files": {k: f"/api/jobs/{self.id}/files/{k}" for k in self.outputs},
            "elapsed": round(time.time() - self.started, 1),
        }


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\- ]+", "", Path(name).stem).strip() or "recording"
    return stem[:80]


def _unique_folder(root: Path, stem: str) -> Path:
    folder = root / stem
    n = 2
    while folder.exists():
        folder = root / f"{stem} ({n})"
        n += 1
    folder.mkdir(parents=True)
    return folder


def _run_job(job: Job, audio_path: Path, options: dict) -> None:
    from .pipeline import Options, run

    try:
        formats = ["musicxml", "midi", "performance"]
        from .score import find_musescore

        if find_musescore():
            formats.append("pdf")
        opts = Options(
            time_signature=options.get("ts") or "4/4",
            bpm=float(options["bpm"]) if options.get("bpm") else None,
            subdivisions=int(options.get("grid") or 4),
            title=options.get("title") or None,
            formats=tuple(formats),
        )
        result = run(audio_path, job.folder, opts, log=job.log.append)
        job.outputs = result.outputs
        job.summary = {"notes": len(result.notes), "bpm": round(result.bpm), "key": result.key,
                       "pdf": "pdf" in result.outputs}
        job.state = "done"
    except Exception as e:  # report anything to the page rather than dying silently
        traceback.print_exc()
        job.error = str(e) or e.__class__.__name__
        job.state = "error"


class App:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def start_job(self, filename: str, data: bytes, options: dict) -> Job:
        stem = _safe_stem(filename)
        suffix = Path(filename).suffix.lower()
        job = Job(id=uuid.uuid4().hex[:12], filename=filename)
        with self.lock:
            job.folder = _unique_folder(self.output_root, stem)
            self.jobs[job.id] = job
        audio_path = job.folder / f"{stem}{suffix}"
        audio_path.write_bytes(data)
        threading.Thread(target=_run_job, args=(job, audio_path, options), daemon=True).start()
        return job


def _static(name: str) -> bytes:
    return resources.files("pianoscribe.web").joinpath(name).read_bytes()


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pianoscribe"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # -- helpers --------------------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, status: int, message: str):
            self._json({"error": message}, status)

        # -- routes ---------------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, _static("index.html"), "text/html; charset=utf-8")
            if path == "/vendor/osmd.js":
                return self._send(200, _static("vendor/opensheetmusicdisplay.min.js"),
                                  "application/javascript", {"Cache-Control": "max-age=86400"})
            m = re.fullmatch(r"/api/jobs/(\w+)", path)
            if m:
                job = app.jobs.get(m.group(1))
                return self._json(job.as_json()) if job else self._error(404, "Unknown job")
            m = re.fullmatch(r"/api/jobs/(\w+)/files/(\w+)", path)
            if m:
                job = app.jobs.get(m.group(1))
                kind = m.group(2)
                if not job or kind not in job.outputs:
                    return self._error(404, "File not found")
                f = job.outputs[kind]
                inline = "inline" in parse_qs(urlparse(self.path).query)
                disposition = "inline" if inline else f'attachment; filename="{f.name}"'
                return self._send(200, f.read_bytes(), DOWNLOAD_TYPES.get(kind, "application/octet-stream"),
                                  {"Content-Disposition": disposition})
            self._error(404, "Not found")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path == "/api/transcribe":
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                filename = unquote(q.get("name", "recording.wav"))
                if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
                    return self._error(400, "Please choose a WAV or MP3 file "
                                            "(FLAC, OGG, M4A, MP4 and MIDI also work).")
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return self._error(400, "The file is empty.")
                if length > MAX_UPLOAD:
                    return self._error(413, "That file is larger than 1 GB.")
                data = self.rfile.read(length)
                job = app.start_job(filename, data, q)
                return self._json(job.as_json())
            m = re.fullmatch(r"/api/jobs/(\w+)/open-folder", url.path)
            if m and (job := app.jobs.get(m.group(1))) and job.folder:
                _reveal(job.folder)
                return self._json({"ok": True})
            self._error(404, "Not found")

    return Handler


def _reveal(folder: Path) -> None:
    """Open a folder in Explorer / Finder / the Linux file manager."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError:
        pass


def _free_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 50), 0]:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def serve(port: int = 8765, open_browser: bool = True, output_root: Path | None = None) -> None:
    app = App(output_root or default_output_root())
    port = _free_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"Pianoscribe is running at {url}")
    print(f"Scores are saved in {app.output_root}")
    print("Keep this window open while you use it. Close it (or press Ctrl+C) to quit.")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pianoscribe-app", description="Open the drag-and-drop app.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.add_argument("--output", type=Path, help="where to save scores (default: Documents/Pianoscribe)")
    args = p.parse_args()
    serve(args.port, not args.no_browser, args.output)


if __name__ == "__main__":
    main()
