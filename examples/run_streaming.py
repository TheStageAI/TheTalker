#!/usr/bin/env python3
"""One streaming TTS request: writes the audio and reports what a player would hear.

Sends a single `POST /v1/audio/speech` with `stream: true`, records the arrival
time of every chunk, writes the PCM to a WAV and prints the three numbers a
streaming deployment is judged on:

  * first-byte TTFA -- request sent to first audio byte received
  * audible TTFA    -- plus the leading silence the model itself generates
  * continuity      -- whether a player started at the first chunk would have
                       run dry before the stream ended

The metric math is thetalker/metrics.py, the same code the benchmark client uses.

Usage:
    python examples/run_streaming.py --ref-audio reference.wav \\
        --ref-text "transcript of the reference wav" \\
        --text "The quick brown fox jumps over the lazy dog."
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thetalker.metrics import (  # noqa: E402
    audible_onset_offset_s,
    audio_seconds_from_bytes,
    compute_continuity_stats,
    inter_chunk_gaps_s,
)


def add_request_args(parser: argparse.ArgumentParser) -> None:
    """The request options examples/run_long_text.py shares with this script."""
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--path", default="/v1/audio/speech")
    parser.add_argument("--task-type", default="Base", choices=["Base", "CustomVoice", "VoiceDesign"])
    parser.add_argument("--ref-audio", type=Path, default=None,
                        help="reference wav; required for the Base (voice clone) task")
    parser.add_argument("--ref-text", default="", help="transcript of --ref-audio")
    parser.add_argument("--speaker", default=None, help="CustomVoice preset voice")
    parser.add_argument("--instruct", default=None, help="VoiceDesign style description")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="cap on the codec tokens generated for one request")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--read-size", type=int, default=8192)
    parser.add_argument("--timeout-s", type=float, default=600.0)


def check_request_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """What a task type needs beyond the defaults, for both scripts."""
    if args.task_type == "Base" and args.ref_audio is None:
        parser.error("--ref-audio is required for --task-type Base")
    if args.task_type == "VoiceDesign" and not args.instruct:
        parser.error("--task-type VoiceDesign requires --instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_request_args(parser)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--continuity-threshold-s", type=float, default=0.1,
                        help="underrun budget in seconds; 0.1 matches vllm-omni")
    parser.add_argument("--out", type=Path, default=Path("out.wav"))
    args = parser.parse_args()
    check_request_args(parser, args)
    return args


def reference_bytes(args: argparse.Namespace) -> bytes | None:
    """The --ref-audio file, read once; None for a task that carries no reference."""
    return args.ref_audio.read_bytes() if args.task_type == "Base" else None


def build_body(args: argparse.Namespace, ref_bytes: bytes | None = None) -> str:
    body = {
        "input": args.text,
        "task_type": args.task_type,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    if args.speaker:
        body["speaker"] = args.speaker
    if args.instruct:
        body["instructions"] = args.instruct
    if args.task_type == "Base":
        if ref_bytes is None:
            ref_bytes = reference_bytes(args)
        encoded = base64.b64encode(ref_bytes).decode("ascii")
        body["ref_audio"] = "data:audio/wav;base64," + encoded
        body["ref_text"] = args.ref_text
    return json.dumps(body)


def stream_request(
    args: argparse.Namespace, payload: str | None = None
) -> tuple[list[float], list[int], bytes]:
    """Send one request; return per-chunk arrival times, chunk sizes and the PCM."""
    if payload is None:
        payload = build_body(args)
    arrivals: list[float] = []
    sizes: list[int] = []
    buf = bytearray()

    conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout_s)
    conn.connect()
    sent_at = time.perf_counter()
    conn.request("POST", args.path, body=payload, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        detail = response.read(500).decode("utf-8", "replace")
        conn.close()
        raise SystemExit(f"HTTP {response.status}: {detail}")
    while True:
        # read1, not read: read() waits for the full buffer and bunches the
        # arrival timestamps, which destroys the chunk timeline.
        chunk = response.read1(args.read_size)
        if not chunk:
            break
        if not arrivals and (chunk[:4] == b"RIFF" or chunk[:5] == b"data:"):
            conn.close()
            raise SystemExit("server did not return raw PCM; ask for response_format=pcm")
        arrivals.append(time.perf_counter() - sent_at)
        sizes.append(len(chunk))
        buf.extend(chunk)
    conn.close()
    if not arrivals:
        raise SystemExit("no audio received")
    return arrivals, sizes, bytes(buf[: len(buf) // 2 * 2])


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def main() -> None:
    args = parse_args()
    arrivals, sizes, pcm = stream_request(args)
    write_wav(args.out, pcm, args.sample_rate)

    audio_s = audio_seconds_from_bytes(len(pcm), args.sample_rate)
    leading_silence_s = audible_onset_offset_s(pcm, args.sample_rate)
    stats = compute_continuity_stats(
        chunk_arrival_times_s=arrivals,
        chunk_bytes=sizes,
        sample_rate=args.sample_rate,
        threshold_s=args.continuity_threshold_s,
    )
    gaps = inter_chunk_gaps_s(arrivals)

    print(f"wrote {args.out} ({audio_s:.2f} s of audio in {len(arrivals)} chunks)")
    print(f"TTFA first byte : {arrivals[0]:.3f} s")
    print(f"TTFA audible    : {arrivals[0] + leading_silence_s:.3f} s "
          f"(leading silence {leading_silence_s:.3f} s)")
    print(f"longest gap     : {max(gaps):.3f} s" if gaps else "longest gap     : n/a")
    if stats.is_continuous:
        print(f"playback        : no stall (worst underrun {stats.max_underrun_s:.3f} s, "
              f"budget {args.continuity_threshold_s:.3f} s)")
    else:
        print(f"playback        : WOULD HAVE STALLED -- worst underrun "
              f"{stats.max_underrun_s:.3f} s over {stats.underrun_event_count} gap(s), "
              f"budget {args.continuity_threshold_s:.3f} s")


if __name__ == "__main__":
    main()
