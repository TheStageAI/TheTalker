"""Registered-voice protocol against a stub server. No GPU, no vllm-omni.

The stub implements the two endpoints the client touches: POST /v1/audio/voices
(multipart upload, answered like vllm-omni: {"success": true, "voice": {...}})
and POST /v1/audio/speech (streams a little raw PCM). It records what it got, so
the tests check the wire, not the client's own bookkeeping.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from thetalker import client, voices

REPO = Path(__file__).resolve().parent.parent
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake reference bytes"


class StubServer:
    def __init__(self, refuse_upload: bool = False):
        self.uploads: list[dict] = []
        self.speech: list[dict] = []
        self.refuse_upload = refuse_upload
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/audio/voices":
                    message = BytesParser(policy=email_policy).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body
                    )
                    form = {}
                    for part in message.iter_parts():
                        name = part.get_param("name", header="content-disposition")
                        form[name] = part.get_content() if part.get_filename() is None else (
                            part.get_filename(), part.get_payload(decode=True))
                    stub.uploads.append(form)
                    if stub.refuse_upload:
                        self._reply(400, {"error": {"message": "Voice name is reserved"}})
                    else:
                        self._reply(200, {"success": True, "voice": {"name": form.get("name")}})
                    return
                stub.speech.append(json.loads(body))
                self.send_response(200)
                self.send_header("Content-Type", "audio/pcm")
                self.end_headers()
                self.wfile.write(b"\x01\x00" * 2400)

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
def ref_wav(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(WAV)
    return path


@pytest.fixture
def texts(tmp_path):
    path = tmp_path / "texts.jsonl"
    path.write_text("".join(json.dumps({"text": f"line {i}."}) + "\n" for i in range(3)))
    return path


def run_client(monkeypatch, port, out_dir, texts, *extra):
    argv = ["thetalker-bench", "--port", str(port), "--texts-jsonl", str(texts), "--out-dir", str(out_dir),
            "--concurrency", "2", "--requests", "4", "--save-audio-limit", "0", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return client.main()


def test_register_voice_sends_the_form_vllm_omni_reads(ref_wav):
    with StubServer() as server:
        record = voices.register_voice("127.0.0.1", server.port, "my_voice", ref_wav, " the transcript ")
    assert record == {"name": "my_voice"}
    (form,) = server.uploads
    assert form["name"] == "my_voice"
    assert form["ref_text"] == "the transcript"
    assert form["consent"] == voices.DEFAULT_CONSENT
    assert form["audio_sample"] == ("reference.wav", WAV)


def test_register_voice_raises_on_refusal(ref_wav):
    with StubServer(refuse_upload=True) as server:
        with pytest.raises(RuntimeError, match="HTTP 400"):
            voices.register_voice("127.0.0.1", server.port, "ryan", ref_wav, "x")


def test_cli_registers_and_reports(ref_wav, capsys):
    with StubServer() as server:
        assert voices.main(["--port", str(server.port), "--name", "v1", "--ref-audio", str(ref_wav)]) == 0
    assert json.loads(capsys.readouterr().out) == {"name": "v1"}
    assert "ref_text" not in server.uploads[0]


def test_client_registers_once_then_names_the_voice(monkeypatch, tmp_path, ref_wav, texts):
    with StubServer() as server:
        rc = run_client(monkeypatch, server.port, tmp_path / "out", texts,
                        "--register-voice", str(ref_wav), "--ref-text", "the transcript")
    assert rc == 0
    assert len(server.uploads) == 1
    assert server.uploads[0]["name"] == voices.DEFAULT_VOICE
    assert server.uploads[0]["ref_text"] == "the transcript"
    assert len(server.speech) == 4
    for body in server.speech:
        assert body["voice"] == voices.DEFAULT_VOICE
        assert body["task_type"] == "Base"
        assert "ref_audio" not in body and "ref_text" not in body
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["voice"] == voices.DEFAULT_VOICE and summary["voice_registered"] == str(ref_wav)
    assert summary["ref_audio"] is None


def test_client_voice_alone_does_not_upload(monkeypatch, tmp_path, texts):
    with StubServer() as server:
        assert run_client(monkeypatch, server.port, tmp_path / "out", texts, "--voice", "earlier") == 0
    assert server.uploads == []
    assert {body["voice"] for body in server.speech} == {"earlier"}


def test_client_inline_reference_still_works(monkeypatch, tmp_path, ref_wav, texts):
    with StubServer() as server:
        assert run_client(monkeypatch, server.port, tmp_path / "out", texts,
                          "--ref-audio", str(ref_wav), "--ref-text", "t") == 0
    assert server.uploads == []
    for body in server.speech:
        assert body["ref_audio"].startswith("data:audio/wav;base64,") and body["ref_text"] == "t"
        assert "voice" not in body


def test_client_refused_upload_sends_no_speech(monkeypatch, tmp_path, ref_wav, texts, capsys):
    with StubServer(refuse_upload=True) as server:
        assert run_client(monkeypatch, server.port, tmp_path / "out", texts,
                          "--register-voice", str(ref_wav)) == 1
    assert server.speech == []
    assert "voice upload refused" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--voice", "v", "--ref-audio", "x.wav"],
    ["--register-voice", "x.wav", "--ref-audio", "x.wav"],
    ["--voice", "v", "--task-type", "CustomVoice", "--speaker", "Ryan"],
    [],
])
def test_client_rejects_ambiguous_reference(monkeypatch, tmp_path, texts, extra):
    with pytest.raises(SystemExit) as excinfo:
        run_client(monkeypatch, 1, tmp_path / "out", texts, *extra)
    assert excinfo.value.code == 2


def test_customvoice_body_is_unchanged():
    args = argparse.Namespace(body_profile="vllm-omni", task_type="CustomVoice", speaker="Ryan", instruct=None,
                              language="English", max_new_tokens=256, stream_format="audio", model="",
                              ref_text="", voice=None)
    body = client.build_body(args, "hi")
    assert body["speaker"] == "Ryan" and "voice" not in body and "ref_audio" not in body


def _run_streaming():
    spec = importlib.util.spec_from_file_location("run_streaming", REPO / "examples" / "run_streaming.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_streaming_calls_a_registered_voice(monkeypatch, tmp_path):
    module = _run_streaming()
    monkeypatch.setattr(sys, "argv", ["run_streaming.py", "--voice", "my_voice", "--out", str(tmp_path / "o.wav")])
    args = module.parse_args()
    body = json.loads(module.build_body(args))
    assert body["voice"] == "my_voice" and "ref_audio" not in body
    monkeypatch.setattr(sys, "argv", ["run_streaming.py", "--voice", "v", "--ref-audio", "x.wav"])
    with pytest.raises(SystemExit):
        module.parse_args()


def _sweep(server, out_dir, ref_wav, extra_env=None):
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    bin_dir = out_dir.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    python = bin_dir / "python"
    # a wrapper, not a symlink: a symlinked venv interpreter loses its site-packages
    python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
           "REF_AUDIO": str(ref_wav), "REF_TEXT": "the transcript", "REQUESTS": "2", "WARMUP": "0"}
    for key in ("VOICE", "INLINE_REF", "TASK_TYPE"):
        env.pop(key, None)
    env.update(extra_env or {})
    return subprocess.run(["bash", str(REPO / "examples" / "run_sweep.sh"), str(out_dir),
                           f"127.0.0.1:{server.port}", "2", "4"],
                          capture_output=True, text=True, env=env, timeout=120, stdin=subprocess.DEVNULL)


def test_sweep_registers_once_and_names_the_voice(tmp_path, ref_wav):
    with StubServer() as server:
        result = _sweep(server, tmp_path / "sweep", ref_wav, {"VOICE": "sweep_voice"})
    assert result.returncode == 0, result.stderr
    assert [u["name"] for u in server.uploads] == ["sweep_voice"]
    assert len(server.speech) == 4
    assert {body["voice"] for body in server.speech} == {"sweep_voice"}
    assert not any("ref_audio" in body for body in server.speech)


def test_sweep_inline_ref_is_opt_in(tmp_path, ref_wav):
    with StubServer() as server:
        result = _sweep(server, tmp_path / "sweep", ref_wav, {"INLINE_REF": "1"})
    assert result.returncode == 0, result.stderr
    assert server.uploads == []
    assert all("ref_audio" in body and "voice" not in body for body in server.speech)
