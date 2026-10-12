"""Warmup drains before the scored window opens. Stub server, no GPU.

A warmup request still in flight when the scored requests start spends GPU time
inside the scored window while its audio is not counted, so RTFx reads low. The
stub records when each request arrived and when its response finished, keyed by
the request's text, so the check is on the server's clock, not the client's
bookkeeping.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from thetalker import client

PCM_BYTES = 4800  # 0.1 s at 24 kHz, 16-bit mono


class TimingServer:
    def __init__(self):
        self.events: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                start = time.perf_counter()
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                index = int(body["input"].split()[1].rstrip("."))
                # uneven service times, so a closed loop has requests in flight at any boundary
                time.sleep(0.02 + 0.03 * (index % 4))
                self.send_response(200)
                self.send_header("Content-Type", "audio/pcm")
                self.end_headers()
                self.wfile.write(b"\x01\x00" * (PCM_BYTES // 2))
                self.wfile.flush()
                with stub.lock:
                    stub.events[body["input"]] = (start, time.perf_counter())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def texts(tmp_path):
    path = tmp_path / "texts.jsonl"
    path.write_text("".join(json.dumps({"text": f"line {i}."}) + "\n" for i in range(64)))
    return path


def run(monkeypatch, port, out_dir, texts, warmup, requests, concurrency):
    monkeypatch.setattr(sys, "argv", [
        "thetalker-bench", "--port", str(port), "--texts-jsonl", str(texts), "--out-dir", str(out_dir),
        "--task-type", "CustomVoice", "--speaker", "Ryan", "--save-audio-limit", "0",
        "--concurrency", str(concurrency), "--requests", str(requests), "--warmup", str(warmup),
    ])
    assert client.main() == 0
    return json.loads((out_dir / "summary.json").read_text())


def test_scored_window_opens_after_every_warmup_request_finished(monkeypatch, tmp_path, texts):
    warmup, requests = 12, 20
    with TimingServer() as server:
        summary = run(monkeypatch, server.port, tmp_path / "out", texts, warmup, requests, concurrency=8)
    warm = [server.events[f"line {i}."] for i in range(warmup)]
    scored = [server.events[f"line {i}."] for i in range(warmup, warmup + requests)]
    assert max(end for _, end in warm) < min(start for start, _ in scored)
    assert summary["warmup_excluded"] == warmup and summary["n"] == requests
    assert summary["warmup_errors"] == 0


def test_rtfx_counts_scored_audio_over_the_scored_window(monkeypatch, tmp_path, texts):
    warmup, requests = 10, 16
    with TimingServer() as server:
        summary = run(monkeypatch, server.port, tmp_path / "out", texts, warmup, requests, concurrency=4)
    rows = summary["requests"]
    scored = [r for r in rows if not r["warmup"]]
    warm = [r for r in rows if r["warmup"]]
    window = max(r["t_submit_s"] + r["wall_s"] for r in scored) - min(r["t_submit_s"] for r in scored)
    assert summary["elapsed_seconds"] == pytest.approx(window)
    assert summary["aggregate_audio_seconds"] == pytest.approx(requests * PCM_BYTES / 2 / 24000)
    assert summary["aggregate_RTFx"] == pytest.approx(summary["aggregate_audio_seconds"] / window)
    assert max(r["t_submit_s"] + r["wall_s"] for r in warm) <= min(r["t_submit_s"] for r in scored)
    assert summary["warmup_elapsed_seconds"] <= min(r["t_submit_s"] for r in scored) + 1e-6


def test_no_warmup_runs_one_phase(monkeypatch, tmp_path, texts):
    with TimingServer() as server:
        summary = run(monkeypatch, server.port, tmp_path / "out", texts, 0, 8, concurrency=8)
    assert summary["n"] == 8 and summary["warmup_excluded"] == 0 and summary["completed"] == 8
