#!/usr/bin/env python3
"""Synthesize a long text as parallel per-sentence requests and join the audio.

The OpenAI speech API serves one request as one stream, so a long text sent
whole is synthesized sequentially at single-client speed. This script splits
the text into sentences (thetalker/text.py), sends them to the server with
--concurrency requests in flight, streams each part into memory, joins the PCM
in text order and writes one WAV. The request body is the one
examples/run_streaming.py sends.

Per part it prints the characters, the time the first byte arrived and the
time the part finished, both counted from the start of the run. At the end it
prints the wall time, the audio length and the time one request after another
would take: an upper bound, the sum of the parts' own request times, which were
measured while the parts competed for the GPU, or, with --compare-sequential, a
measured second pass that sends the same parts sequentially -- a pass with a
failed part prints no time, because a pass that died is not a comparison. A part
that fails is reported in its row and left out of the WAV, and the script exits
non-zero.

Usage:
    python examples/run_long_text.py --ref-audio reference.wav \\
        --ref-text "<its transcript>" --text-file book_chapter.txt \\
        --concurrency 8 --out chapter.wav
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_streaming import (  # noqa: E402
    add_request_args,
    build_body,
    check_request_args,
    reference_bytes,
    stream_request,
    write_wav,
)
from thetalker.metrics import audio_seconds_from_bytes  # noqa: E402
from thetalker.text import split_sentences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_request_args(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="the text to synthesize")
    source.add_argument("--text-file", type=Path, help="a UTF-8 file with the text")
    parser.add_argument("--max-chars", type=int, default=300,
                        help="longest part; longer sentences are split at a comma or space")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--compare-sequential", action="store_true",
                        help="also send the same parts one after another and time that")
    parser.add_argument("--out", type=Path, default=Path("long_text.wav"))
    args = parser.parse_args()
    check_request_args(parser, args)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.max_chars < 1:
        parser.error("--max-chars must be at least 1")
    return args


def run_parts(args: argparse.Namespace, payloads: list[str], concurrency: int) -> tuple[list[dict], float]:
    """Send every payload with `concurrency` in flight; results keep the input order."""
    run_t0 = time.perf_counter()

    def one(payload: str) -> dict:
        started = time.perf_counter() - run_t0
        try:
            arrivals, _, pcm = stream_request(args, payload)
        except (SystemExit, OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            # a server that dies mid-stream raises IncompleteRead, which is not an
            # OSError; one part must not cost the run the audio of every other part
            return {"error": str(exc), "request_s": 0.0, "pcm": b""}
        return {
            "first_byte_s": started + arrivals[0],
            "done_s": started + arrivals[-1],
            "request_s": arrivals[-1],
            "pcm": pcm,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, payloads))
    return results, time.perf_counter() - run_t0


def main() -> int:
    args = parse_args()
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    parts = split_sentences(text, args.max_chars)
    if not parts:
        raise SystemExit("the text is empty")

    ref_bytes = reference_bytes(args)
    payloads = [
        build_body(argparse.Namespace(**{**vars(args), "text": part}), ref_bytes)
        for part in parts
    ]

    results, wall_s = run_parts(args, payloads, args.concurrency)
    done = [result for result in results if "error" not in result]
    pcm = b"".join(result["pcm"] for result in results)
    if pcm:
        write_wav(args.out, pcm, args.sample_rate)

    print(f"{'part':>4}  {'chars':>5}  {'first byte s':>12}  {'done s':>7}")
    for index, (part, result) in enumerate(zip(parts, results)):
        if "error" in result:
            print(f"{index:>4}  {len(part):>5}  {'failed':>12}  {result['error']}")
        else:
            print(f"{index:>4}  {len(part):>5}  {result['first_byte_s']:>12.3f}  {result['done_s']:>7.3f}")
    print()
    print(f"wrote {args.out}" if pcm else "wrote nothing: every part failed")
    print(f"parts                 : {len(parts)} at concurrency {args.concurrency}"
          f"{'' if len(done) == len(parts) else f', {len(parts) - len(done)} failed'}")
    print(f"total wall time       : {wall_s:.3f} s")
    print(f"total audio           : {audio_seconds_from_bytes(len(pcm), args.sample_rate):.2f} s")
    if done:
        print(f"first audio byte      : {min(r['first_byte_s'] for r in done):.3f} s")
    failed = len(parts) - len(done)
    if args.compare_sequential:
        sequential, sequential_s = run_parts(args, payloads, 1)
        # a pass whose parts died fast is quick, not fast: it is no comparison
        errors = [result["error"] for result in sequential if "error" in result]
        failed += len(errors)
        if errors:
            print(f"sequential wall time  : not comparable, {len(errors)} of "
                  f"{len(payloads)} parts failed in the sequential pass ({errors[0]})")
        else:
            print(f"sequential wall time  : {sequential_s:.3f} s (measured, one part after another)")
    else:
        estimate_s = sum(result["request_s"] for result in results)
        print(f"sequential upper bound: {estimate_s:.3f} s (sum of the parts' request times, "
              f"which were measured while the parts competed for the GPU; "
              f"--compare-sequential measures the real thing)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
