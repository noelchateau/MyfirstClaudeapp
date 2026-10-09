import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import quote

import pytest
import soundfile as sf

from pianoscribe.app import App, make_handler
from pianoscribe.audio import SAMPLE_RATE
from pianoscribe.demo import demo_piece
from pianoscribe.synth import synthesize


@pytest.fixture
def server(tmp_path):
    app = App(tmp_path / "scores")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers, r.read()


def _post(url, data=b""):
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_vendored_viewer_are_served(server):
    base, _ = server
    status, _, body = _get(base + "/")
    assert status == 200 and b"Drop a recording here" in body
    status, _, body = _get(base + "/vendor/osmd.js")
    assert status == 200 and len(body) > 100_000


def test_upload_transcribe_download(server, tmp_path):
    base, _ = server
    wav = tmp_path / "in.wav"
    sf.write(wav, synthesize(demo_piece()[:20]), SAMPLE_RATE)
    status, job = _post(f"{base}/api/transcribe?name={quote('My Piece.wav')}&ts=4/4&grid=4",
                        wav.read_bytes())
    assert status == 200
    for _ in range(300):
        _, _, body = _get(f"{base}/api/jobs/{job['id']}")
        job = json.loads(body)
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    assert job["summary"]["notes"] > 10
    assert job["folder"].endswith("My Piece")
    _, headers, body = _get(base + job["files"]["musicxml"])
    assert b"<score-partwise" in body
    assert headers["Content-Disposition"] == 'attachment; filename="My Piece.musicxml"'
    _, headers, _ = _get(base + job["files"]["musicxml"] + "?inline=1")
    assert headers["Content-Disposition"] == "inline"


def test_rejects_non_audio(server):
    base, _ = server
    status, body = _post(f"{base}/api/transcribe?name=notes.txt", b"hello")
    assert status == 400 and "WAV or MP3" in body["error"]


def test_unknown_files_are_not_served(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/jobs/nope/files/musicxml")
    assert e.value.code == 404
