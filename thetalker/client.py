"""Streaming TTS load client (POST /v1/audio/speech, stream=true, raw PCM).

Drives a vllm-omni (or OpenAI-`audio/speech`-compatible) TTS server with a wave
or closed-loop or Poisson-arrival load, records the arrival time and byte size
of every streamed audio chunk, and derives RTFx / TTFA / continuity from that
timeline via thetalker/metrics.py -- no metric math lives in this file.

Highlights:
  - per-chunk (arrival, bytes) recorded; records.jsonl is the primary artifact
  - connect cost measured separately and excluded from TTFA
  - resp.read1() instead of resp.read(): read() blocks until the full 8192 bytes
    are available, which bunches arrival timestamps and fakes the chunk timeline
  - RIFF/container guard: response_format must be raw PCM or the timeline is corrupt
  - warmup is a separate phase that drains (every warmup request finished)
    before the scored window opens, and its requests are marked in records.jsonl
  - optional per-row overrides in --texts-jsonl (ref_audio abs path, ref_text,
    language) for voice-clone sets; rows without them use the CLI globals
  - --task-type / --speaker / --instruct for CustomVoice and VoiceDesign
    checkpoints (upstream body fields "speaker" and "instructions"); those task
    types never send ref_audio/ref_text, and --ref-audio is only required for
    the Base task
  - Base task, registered voice (the published protocol): --register-voice
    <wav> uploads the reference once before the first request (POST
    /v1/audio/voices, thetalker/voices.py) and every request then names it,
    "voice": <--voice>; --voice alone calls a voice registered earlier. Without
    either, every request carries the reference inline (ref_audio/ref_text)
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import random
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from thetalker.metrics import (
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_WIDTH,
    build_request_record,
    percentile,
)
from thetalker.voices import DEFAULT_VOICE, register_voice

BACKEND = "openai_audio_speech"
RUN = "thetalker"


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_bytes)


def load_prompt_rows(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        text = str(data.get("text") or data.get("reference") or "").strip()
        row = {
            "sample_id": str(data.get("sample_id") or data.get("id") or f"prompt_{index:04d}"),
            "text": text,
            "group": str(data.get("group") or ""),
        }
        for key in ("ref_audio", "ref_text", "language"):
            if data.get(key):
                row[key] = str(data[key])
        rows.append(row)
    if not rows:
        raise ValueError(f"no prompt rows in {path}")
    return rows


_ref_b64_cache: dict[str, str] = {}
_ref_b64_lock = threading.Lock()


def ref_audio_b64_for(path: str) -> str:
    """B64-encode a per-row ref wav once; hundreds of rows share few distinct refs."""
    with _ref_b64_lock:
        if path not in _ref_b64_cache:
            _ref_b64_cache[path] = (
                "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
            )
        return _ref_b64_cache[path]


def build_body(args, text: str, *, row: dict | None = None, ref_audio_b64: str | None = None) -> dict:
    if getattr(args, "body_profile", "vllm-omni") == "nari":
        # Nari's request schema forbids unknown fields; send only the ones it defines.
        return {"input": text, "voice": args.speaker, "language": args.language or "English", "stream": True,
                "response_format": "pcm", "non_streaming_mode": False, "max_new_tokens": args.max_new_tokens}

    row = row or {}
    task_type = getattr(args, "task_type", "Base") or "Base"
    body = {
        "input": text,
        "task_type": task_type,
        "language": row.get("language", args.language),
        "max_new_tokens": args.max_new_tokens,
        "stream": True,
        "stream_format": args.stream_format,
        "response_format": "pcm",
    }
    speaker = getattr(args, "speaker", None)
    if speaker:
        body["speaker"] = speaker  # upstream alias of "voice" (OpenAICreateSpeechRequest)
    instruct = getattr(args, "instruct", None)
    if instruct:
        body["instructions"] = instruct
    voice = getattr(args, "voice", None)
    if task_type not in ("CustomVoice", "VoiceDesign") and voice:
        # registered voice: the server resolves the stored reference by name
        body["voice"] = voice
    elif task_type not in ("CustomVoice", "VoiceDesign"):
        # only the Base (voice clone) task carries a reference
        body["ref_audio"] = ref_audio_b64_for(row["ref_audio"]) if row.get("ref_audio") else ref_audio_b64
        body["ref_text"] = row.get("ref_text", args.ref_text)
    if args.model:
        body["model"] = args.model
    return body


def run_one(args, barrier, prompt_rows, ref_audio_b64, index: int, run_t0: float, delay: float) -> dict:
    prompt = prompt_rows[index % len(prompt_rows)]
    request_id = f"req_{index:04d}"
    body = build_body(args, prompt["text"], row=prompt, ref_audio_b64=ref_audio_b64)
    payload = json.dumps(body)
    if barrier is not None:
        barrier.wait()
    if delay > 0:
        time.sleep(delay)

    chunk_arrival_s: list[float] = []
    chunk_bytes: list[int] = []
    buf = bytearray()
    errors: list[dict] = []
    connect_ms = None
    sent_at = None
    finish_reason = None
    connect_started = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout_s)
        conn.connect()
        connect_ms = (time.perf_counter() - connect_started) * 1000.0
        sent_at = time.perf_counter()
        conn.request("POST", args.path, body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            errors.append({"type": f"HTTP{resp.status}", "message": resp.read(500).decode("utf-8", "replace")})
            finish_reason = "error"
        else:
            while True:
                # read1: return as soon as bytes are available, so the arrival timestamp
                # is the transport's, not the client's buffering
                chunk = resp.read1(args.read_size)
                if not chunk:
                    break
                now = time.perf_counter()
                if not chunk_arrival_s and chunk[:4] == b"RIFF":
                    errors.append({"type": "ContainerFormat", "message": "RIFF header: not raw pcm"})
                # stream=true alone yields text/event-stream with base64 audio inside a
                # JSON envelope: byte counts then overstate the audio by ~1.36x and every
                # byte-derived metric (audio_s, deficit, throughput) is wrong.
                if not chunk_arrival_s and (chunk[:6] == b"event:" or chunk[:5] == b"data:"):
                    errors.append({"type": "ContainerFormat", "message": "SSE envelope: not raw pcm"})
                chunk_arrival_s.append(now - sent_at)
                chunk_bytes.append(len(chunk))
                buf.extend(chunk)
            finish_reason = "stream_end"
        conn.close()
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
        finish_reason = "error"
    end = time.perf_counter()
    submit_at = connect_started if sent_at is None else sent_at
    wall_s = end - submit_at

    pcm = bytes(buf[: len(buf) // 2 * 2])
    wav_rel = None
    if pcm and index < args.save_audio_limit:
        wav_rel = f"audio/request_{index:04d}.wav"
        write_wav(args.out_dir / wav_rel, pcm, args.sample_rate)

    return build_request_record(
        backend=BACKEND,
        run=RUN,
        run_id=args.run_id,
        profile=args.profile,
        point=args.point,
        repeat=0,
        warmup=index < args.warmup,
        request_index=index,
        request_id=request_id,
        sample_id=prompt["sample_id"],
        group=prompt.get("group", ""),
        text=prompt["text"],
        ok=not errors and len(pcm) > 0,
        finish_reason=finish_reason,
        errors=errors,
        t_submit_s=submit_at - run_t0,
        t_connect_ms=connect_ms,
        chunk_arrival_s=chunk_arrival_s,
        chunk_bytes=chunk_bytes,
        wall_s=wall_s,
        pcm_bytes=pcm,
        sample_rate=args.sample_rate,
        sample_width=DEFAULT_SAMPLE_WIDTH,
        channels=DEFAULT_CHANNELS,
        threshold_s=args.continuity_threshold_s,
        wav=str(args.out_dir / wav_rel) if wav_rel else None,
        server_meta={},
    )


def scored_window_s(scored: list[dict]) -> float:
    """Wall-clock span from the first scored submit to the last scored finish; the warmup is outside it."""
    if not scored:
        return 0.0
    start = min(r["t_submit_s"] for r in scored)
    end = max(r["t_submit_s"] + r["wall_s"] for r in scored)
    return end - start


def summarize(rows: list[dict], elapsed: float, extra: dict) -> dict:
    scored = [r for r in rows if not r["warmup"]]
    ok_rows = [r for r in scored if r["ok"]]
    audio = sum(r["audio_s"] for r in ok_rows)
    window = scored_window_s(scored)
    summary = {
        "n": len(scored),
        "warmup_excluded": len(rows) - len(scored),
        "completed": len(ok_rows),
        "errors": len(scored) - len(ok_rows),
        "elapsed_seconds": window,
        "run_elapsed_seconds": elapsed,
        "aggregate_audio_seconds": audio,
        "aggregate_RTFx": (audio / window) if window else None,
        "ttfa_p50": percentile([r["ttfa_s"] for r in ok_rows], 0.50),
        "ttfa_p95": percentile([r["ttfa_s"] for r in ok_rows], 0.95),
        "ttfa_audible_p50": percentile([r["ttfa_audible_s"] for r in ok_rows], 0.50),
        "ttfa_audible_p95": percentile([r["ttfa_audible_s"] for r in ok_rows], 0.95),
        "wall_p50": percentile([r["wall_s"] for r in ok_rows], 0.50),
        "wall_p95": percentile([r["wall_s"] for r in ok_rows], 0.95),
        "buffer_deficit_p50": percentile([r["buffer_deficit_s"] for r in ok_rows], 0.50),
        "buffer_deficit_p95": percentile([r["buffer_deficit_s"] for r in ok_rows], 0.95),
        "continuity_ok_pct": (
            100.0 * sum(1 for r in ok_rows if r["continuity_ok"]) / len(ok_rows) if ok_rows else None
        ),
        "gap_p95_max": max((r["gap_p95_s"] for r in ok_rows if r["gap_p95_s"] is not None), default=None),
        "audio_s_p50": percentile([r["audio_s"] for r in ok_rows], 0.50),
    }
    summary.update(extra)
    summary["requests"] = rows
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--path", default="/v1/audio/speech")
    parser.add_argument("--model", default="")
    parser.add_argument("--body-profile", choices=["vllm-omni", "nari"], default="vllm-omni",
                        help="request body shape; nari = Nari Qwen3-TTS engine")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=0,
                        help="scored requests, issued after the --warmup ones; 0 = one wave of --concurrency. "
                             ">concurrency = closed loop")
    parser.add_argument("--texts-jsonl", type=Path, required=True)
    parser.add_argument("--task-type", default="Base", choices=["Base", "CustomVoice", "VoiceDesign"],
                        help="request body task_type; CustomVoice/VoiceDesign never send ref_audio/ref_text")
    parser.add_argument("--speaker", default=None,
                        help="CustomVoice preset voice (e.g. vivian); body field 'speaker'")
    parser.add_argument("--instruct", default=None,
                        help="voice style/description; body field 'instructions' (required by VoiceDesign)")
    parser.add_argument("--ref-audio", type=Path, default=None,
                        help="reference wav sent inline with every Base (voice clone) request; "
                             "the Base task needs it, --register-voice or --voice")
    parser.add_argument("--ref-text", default="Hello. This is a short TTS runtime smoke test.")
    parser.add_argument("--voice", default=None,
                        help="Base task: name a voice registered on the server instead of sending the "
                             "reference inline")
    parser.add_argument("--register-voice", type=Path, default=None, metavar="WAV",
                        help="Base task: upload WAV with --ref-text once as voice --voice (default "
                             f"{DEFAULT_VOICE}) before the first request, then call it by name")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="server-side cap on generated codec tokens; 256 is the "
                             "value the published tables were measured with")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--read-size", type=int, default=8192)
    parser.add_argument("--stream-format", default="audio", choices=["audio", "sse"],
                        help="audio = raw pcm bytes per code2wav chunk; sse = OpenAI "
                             "speech.audio.delta events, base64 in JSON")
    parser.add_argument("--save-audio-limit", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--arrival-rate", type=float, default=0.0, help="req/s Poisson; 0 = wave")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0,
                        help="N warmup requests run as their own phase at --concurrency and drained before "
                             "the scored ones start; excluded from the stats "
                             "and from the RTFx window")
    parser.add_argument("--continuity-threshold-s", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=0.05,
                        help="fail the run when more than this share of scored requests "
                             "errored (0 = tolerate none, 1 = never fail)")
    args = parser.parse_args()

    scored_requests = args.requests or args.concurrency
    n_requests = scored_requests + args.warmup
    args.profile = "poisson" if args.arrival_rate > 0 else ("closed_loop" if scored_requests > args.concurrency else "wave")
    args.point = f"r{args.arrival_rate:g}" if args.arrival_rate > 0 else f"c{args.concurrency}"
    args.run_id = f"{RUN}__{args.profile}__{args.point}"

    ref_less_task = args.task_type in ("CustomVoice", "VoiceDesign")
    if (args.register_voice or args.voice) and ref_less_task:
        parser.error("--voice / --register-voice are for --task-type Base")
    if (args.register_voice or args.voice) and args.ref_audio:
        parser.error("--register-voice / --voice and --ref-audio are exclusive: registered or inline reference")
    if args.register_voice and not args.voice:
        args.voice = DEFAULT_VOICE
    if args.ref_audio is None and not args.voice and not ref_less_task:
        parser.error("--task-type Base needs --register-voice, --voice or --ref-audio")
    if args.task_type == "VoiceDesign" and not args.instruct:
        parser.error("--task-type VoiceDesign requires --instruct")
    if args.body_profile == "nari" and not args.speaker:
        parser.error("--body-profile nari requires --speaker")
    ref_audio_b64 = None
    if args.ref_audio is not None and not ref_less_task:
        ref_audio_b64 = "data:audio/wav;base64," + base64.b64encode(args.ref_audio.read_bytes()).decode("ascii")
    prompt_rows = load_prompt_rows(args.texts_jsonl)
    if args.voice and any(row.get("ref_audio") for row in prompt_rows):
        parser.error("--texts-jsonl rows carry their own ref_audio; a registered voice is one reference")
    voice_record = None
    if args.register_voice:
        try:
            voice_record = register_voice(args.host, args.port, args.voice, args.register_voice, args.ref_text)
        except (RuntimeError, OSError) as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows), encoding="utf-8"
    )

    # Warmup is its own phase and drains completely before the scored phase starts:
    # a warmup request still in flight would spend GPU time inside the scored window
    # while its audio is not counted, which reads RTFx low.
    rng = random.Random(args.seed)
    warmup_indices = list(range(args.warmup))
    scored_indices = list(range(args.warmup, n_requests))
    delays = [0.0] * n_requests
    if args.arrival_rate > 0:
        t = 0.0
        for i in scored_indices:
            t += rng.expovariate(args.arrival_rate)
            delays[i] = t
        barrier = None
        workers = scored_requests
    else:
        # a barrier only makes sense for a single wave; in a closed loop the pool size
        # (max_workers) is what keeps exactly --concurrency requests in flight
        barrier = threading.Barrier(args.concurrency) if scored_requests == args.concurrency else None
        workers = args.concurrency
    (args.out_dir / "schedule.jsonl").write_text(
        "".join(json.dumps({"request_index": i, "phase": "warmup" if i < args.warmup else "scored",
                            "offset_s": d}) + "\n" for i, d in enumerate(delays)),
        encoding="utf-8",
    )

    run_t0 = time.perf_counter()
    rows = []

    def run_phase(indices, phase_barrier, max_workers):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one, args, phase_barrier, prompt_rows, ref_audio_b64, i, run_t0, delays[i])
                for i in indices
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    if warmup_indices:
        # closed loop at --concurrency; returns only when every warmup request has finished
        run_phase(warmup_indices, None, min(args.concurrency, len(warmup_indices)))
    warmup_elapsed = time.perf_counter() - run_t0
    run_phase(scored_indices, barrier, workers)
    elapsed = time.perf_counter() - run_t0
    rows.sort(key=lambda row: row["request_index"])

    (args.out_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_rows = [
        {
            "sample_id": f"{row['sample_id']}__req{row['request_index']:04d}",
            "request_index": row["request_index"],
            "group": row["group"],
            "text": row["text"],
            "reference": row["text"],
            "wav": row["wav"],
            "audio_seconds": row["audio_s"],
        }
        for row in rows if row.get("wav")
    ]
    (args.out_dir / "quality_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    summary = summarize(rows, elapsed, {
        "backend": BACKEND,
        "run": RUN,
        "run_id": args.run_id,
        "profile": args.profile,
        "point": args.point,
        "prompt_count": len(prompt_rows),
        "texts_jsonl": str(args.texts_jsonl),
        "url": f"http://{args.host}:{args.port}{args.path}",
        "task_type": args.task_type,
        "speaker": args.speaker,
        "instruct": args.instruct,
        "ref_audio": str(args.ref_audio) if args.ref_audio is not None and not ref_less_task else None,
        "voice": args.voice,
        "voice_registered": str(args.register_voice) if args.register_voice else None,
        "voice_record": voice_record,
        "max_new_tokens": args.max_new_tokens,
        "read_size": args.read_size,
        "stream_format": args.stream_format,
        "arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "requests_issued": n_requests,
        "warmup_requests": args.warmup,
        "warmup_phase": "separate, drained before the scored window",
        "warmup_elapsed_seconds": warmup_elapsed,
        "warmup_errors": sum(1 for row in rows if row["warmup"] and not row["ok"]),
        "max_new_tokens_cap": args.max_new_tokens,
        "max_error_rate": args.max_error_rate,
    })
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    headline = {k: v for k, v in summary.items() if k != "requests"}
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    return run_status(summary, args.max_error_rate)


def run_status(summary: dict, max_error_rate: float) -> int:
    """0 if the point is usable, 1 otherwise, with the reason on stderr.

    A point where every request failed still produces a well-formed summary full
    of zeros; exiting 0 on it invites a reader to treat "server not up yet" as a
    measurement.
    """
    scored = summary["n"]
    completed = summary["completed"]
    if scored == 0:
        print("FAILED: no scored requests (all of them were warmup?)", file=sys.stderr)
        return 1
    if completed == 0:
        print(
            f"FAILED: all {scored} scored requests errored, so every number in "
            f"summary.json is zero; is the server up and streaming raw PCM?",
            file=sys.stderr,
        )
        return 1
    error_rate = (scored - completed) / scored
    if error_rate > max_error_rate:
        print(
            f"FAILED: {scored - completed} of {scored} scored requests errored "
            f"({error_rate:.1%} > --max-error-rate {max_error_rate:.1%}); "
            f"the point is not comparable to one without errors",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
