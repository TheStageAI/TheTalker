"""Register a reference voice with the server once, then call it by name.

vllm-omni stores an uploaded voice (``POST /v1/audio/voices``, multipart form:
``name``, ``consent``, ``ref_text`` and the ``audio_sample`` file) and resolves
``"voice": <name>`` in a speech request to that stored reference. This is how a
production voice-cloning service is called, and what public TTS benchmarks
measure: the reference is uploaded once, not sent with every request.

CLI:
    thetalker-voice --port 8091 --name my_voice --ref-audio reference.wav \\
        --ref-text "transcript of the reference"

Uploading a name that exists replaces it (vllm-omni's default registration
policy). Exit status is 1 when the server refuses the upload.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import uuid
from pathlib import Path

DEFAULT_VOICE = "thetalker_ref"
DEFAULT_CONSENT = "thetalker-benchmark"
VOICES_PATH = "/v1/audio/voices"


def multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    """A multipart/form-data body with text fields and one wav file; returns (body, content type)."""
    boundary = uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        for key, value in fields.items()
    ]
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
        + file_path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def register_voice(host: str, port: int, name: str, ref_audio: Path, ref_text: str = "", *,
                   consent: str = DEFAULT_CONSENT, path: str = VOICES_PATH, timeout_s: float = 120.0) -> dict:
    """Upload ``ref_audio`` (+ its transcript) as voice ``name``; return the server's voice record.

    Raises RuntimeError when the server refuses it.
    """
    fields = {"name": name, "consent": consent}
    if ref_text.strip():
        fields["ref_text"] = ref_text.strip()
    body, content_type = multipart_body(fields, "audio_sample", Path(ref_audio))
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": content_type})
        resp = conn.getresponse()
        raw = resp.read()
    finally:
        conn.close()
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        data = {}
    if resp.status != 200 or not data.get("success"):
        detail = raw[:500].decode("utf-8", "replace")
        raise RuntimeError(f"voice upload refused: HTTP {resp.status}: {detail}")
    return data.get("voice", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--name", default=DEFAULT_VOICE)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", default="", help="transcript of --ref-audio, word for word")
    parser.add_argument("--consent", default=DEFAULT_CONSENT, help="consent recording id the server stores")
    args = parser.parse_args(argv)
    try:
        voice = register_voice(args.host, args.port, args.name, args.ref_audio, args.ref_text,
                               consent=args.consent)
    except (RuntimeError, OSError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(voice, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
