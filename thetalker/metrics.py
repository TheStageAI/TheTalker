"""Shared serving metrics for this benchmark.

ONE implementation of every metric. A client adapter may only produce a raw
timeline -- (chunk_arrival_s, chunk_bytes, pcm) -- and must call
build_request_record() for anything derived, so every metric is computed
exactly once, in one place.

Buffer deficit ("underrun") is a VERBATIM port of vllm-omni's
  vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats
(function introduced in PR #3618, "[Bench] Add audio-streaming continuity
metric for TTS"), so our number is definitionally identical to their
audio_underrun metric. Do not "improve" it: any change breaks comparability
with published vllm-omni numbers.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_log = logging.getLogger(__name__)

# /2 renamed the record's "arm" field to "run"; the arithmetic is unchanged.
SCHEMA_VERSION = "tts-serving-record/2"

# PCM defaults, matching vllm_omni/metrics/definitions.py (DEFAULT_AUDIO_SAMPLE_RATE /
# DEFAULT_AUDIO_CHANNELS) and Qwen3-TTS 12Hz s16le mono output.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_CHANNELS = 1
SECONDS_PER_FRAME = 1.0 / 12.5  # Qwen3-TTS 12Hz codec frame = 80 ms of audio

# vllm_omni/metrics/definitions.py::AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S
AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S = 0.1
# vllm-omni's override knob; honoured here so both stacks read the same threshold.
_VLLM_OMNI_THRESHOLD_ENV = "VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S"
_LOCAL_THRESHOLD_ENV = "TTS_BENCH_CONTINUITY_THRESHOLD_S"

# Secondary thresholds reported alongside the 0.1 s default (design sec. 4.1).
CONTINUITY_REPORT_THRESHOLDS_S = (0.05, 0.1, 0.2)


# --------------------------------------------------------------------------------------
# vllm-omni port -- keep names, defaults and arithmetic identical
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuityStats:
    """Result of a single-request continuity analysis.

    Attributes:
        max_underrun_s: Worst-case wall-clock seconds the player was starved.
        underrun_event_count: Inter-chunk intervals during which the buffer
            went negative (one per gap, not per starved millisecond).
        is_continuous: ``max_underrun_s <= threshold_s`` for the request.
    """

    max_underrun_s: float
    underrun_event_count: int
    is_continuous: bool


def compute_continuity_stats(
    chunk_arrival_times_s: list[float],
    chunk_bytes: list[int],
    sample_rate: int,
    sample_width: int = 2,
    channels: int = 1,
    threshold_s: float = 0.1,
) -> ContinuityStats:
    """Compute the continuity stats for one streaming response.

    Args:
        chunk_arrival_times_s: Wall-clock seconds (since request start) at
            which each non-empty audio chunk's *last byte* arrived. Must be
            monotonically non-decreasing.
        chunk_bytes: Byte count of each chunk in the same order.
        sample_rate: PCM sample rate (Hz). 24 000 for Qwen3-TTS / VoxCPM2.
        sample_width: Bytes per sample (2 = s16le).
        channels: PCM channel count (1 = mono).
        threshold_s: Underrun budget. The default 0.1 s matches the
            commonly-cited "audible gap" threshold for streaming TTS.

    Returns:
        A :class:`ContinuityStats` summarising the worst-case deficit and
        whether it stayed under the threshold.

    Listener model: the playback clock starts at the FIRST BYTE (``t0 = arrivals[0]``),
    i.e. no pre-roll. ``audible_onset_offset_s`` below assumes a different listener --
    one who hears nothing until the audible onset. The two are not interchangeable and
    the 0.1 s test is not invariant to the difference, so a continuity verdict must be
    labelled with the playback model it was taken under. Do not change ``t0`` here: this
    function is a verbatim port and its number must stay comparable to vllm-omni's.
    """
    n = len(chunk_arrival_times_s)
    if n == 0 or n != len(chunk_bytes):
        return ContinuityStats(0.0, 0, True)

    bytes_per_s = sample_rate * sample_width * channels
    if bytes_per_s <= 0:
        return ContinuityStats(0.0, 0, True)

    t0 = chunk_arrival_times_s[0]
    received_before = 0
    max_underrun_s = 0.0
    event_count = 0
    for i in range(n):
        if i > 0:
            played_bytes = (chunk_arrival_times_s[i] - t0) * bytes_per_s
            deficit_bytes = played_bytes - received_before
            if deficit_bytes > 0:
                deficit_s = deficit_bytes / bytes_per_s
                if deficit_s > max_underrun_s:
                    max_underrun_s = deficit_s
                event_count += 1
        received_before += chunk_bytes[i]

    return ContinuityStats(
        max_underrun_s=max_underrun_s,
        underrun_event_count=event_count,
        is_continuous=max_underrun_s <= threshold_s,
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def continuity_threshold_s(default: float = AUDIO_CONTINUITY_DEFAULT_THRESHOLD_S) -> float:
    """Underrun budget in seconds; env override kept name-compatible with vllm-omni."""
    for name in (_LOCAL_THRESHOLD_ENV, _VLLM_OMNI_THRESHOLD_ENV):
        raw = os.environ.get(name)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                return value
    return default


def bytes_per_second(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    channels: int = DEFAULT_CHANNELS,
) -> int:
    return int(sample_rate) * int(sample_width) * int(channels)


def audio_seconds_from_bytes(
    n_bytes: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    channels: int = DEFAULT_CHANNELS,
) -> float:
    bps = bytes_per_second(sample_rate, sample_width, channels)
    return (float(n_bytes) / bps) if bps > 0 else 0.0


def percentile(values: Iterable[float | None], q: float) -> float | None:
    """Ilya's percentile convention, kept so old summaries stay comparable."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if q == 0.5:
        return statistics.median(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(len(xs) * q) - 1))]


def pcm_stats(pcm_bytes: bytes | bytearray | memoryview | None, sample_rate: int) -> dict:
    """PCM sanity block (s16le). numpy is imported lazily so tests stay dependency-free."""
    raw = bytes(pcm_bytes or b"")
    raw = raw[: len(raw) // 2 * 2]
    if not raw:
        return {
            "samples": 0,
            "audio_seconds": 0.0,
            "rms": 0.0,
            "peak": 0,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
        }
    import numpy as np

    pcm = np.frombuffer(raw, dtype="<i2")
    abs_values = np.abs(pcm.astype(np.int32))
    return {
        "samples": int(pcm.size),
        "audio_seconds": float(pcm.size / sample_rate),
        "rms": float(math.sqrt(float(np.mean(pcm.astype(np.float64) ** 2)))),
        "peak": int(abs_values.max()),
        "silence_ratio": float(np.mean(abs_values <= 32)),
        "clipping_ratio": float(np.mean(abs_values >= 32760)),
    }


def inter_chunk_gaps_s(chunk_arrival_s: Sequence[float]) -> list[float]:
    return [b - a for a, b in zip(chunk_arrival_s, chunk_arrival_s[1:])]


def audible_onset_offset_s(
    pcm: bytes | bytearray | memoryview,
    sample_rate: int,
    sample_width: int = 2,
    channels: int = 1,
    window_ms: float = 5.0,
    rel_threshold: float = 0.05,
    sustain_ms: float = 20.0,
) -> float:
    """Seconds from the start of `pcm` to the first sustained sound.

    Returns 0.0 for an empty buffer, for a buffer that opens with sound, and for a
    buffer that never crosses the threshold -- the metric may only ever move TTFA
    later, never earlier, and an all-silent capture is a quality finding, not a
    latency one. A stream that is not 16-bit PCM also reads 0.0, but logs a warning
    rather than passing silently.

    The three constants (5 ms window, 5% of the file's own peak, 20 ms sustain) are
    this benchmark's definition of onset, not a standard: a published "audible TTFA"
    from another stack is not known to share them.
    """
    if not pcm:
        return 0.0
    if sample_width != 2:
        # Every backend in this benchmark streams s16le; a 24-bit or float stream would
        # otherwise be scored as "no leading silence" and quietly re-published as audible
        # TTFA == first-byte TTFA.
        _log.warning(
            "audible_onset_offset_s: sample_width=%d is not 16-bit PCM; leading silence "
            "reported as 0.0 and ttfa_audible_s will equal ttfa_s for this stream",
            sample_width,
        )
        return 0.0
    import numpy as np

    raw = bytes(pcm)
    samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
    if channels > 1:
        samples = samples[::channels]
    win = max(1, int(sample_rate * window_ms / 1000.0))
    need = max(1, int(sustain_ms / window_ms))
    n_win = samples.size // win
    if n_win == 0:
        return 0.0
    peak = int(np.abs(samples.astype(np.int32)).max()) or 1
    thr = peak * rel_threshold
    blocks = samples[: n_win * win].astype(np.float64).reshape(n_win, win)
    loud = np.sqrt((blocks * blocks).mean(axis=1)) >= thr
    if need == 1:
        hits = np.flatnonzero(loud)
    else:
        # first window of the first run of `need` consecutive loud windows
        counts = np.concatenate(([0], np.cumsum(loud)))
        hits = np.flatnonzero(counts[need:] - counts[:-need] == need)
    if hits.size == 0:
        return 0.0
    return max(0.0, float(hits[0] * win) / sample_rate)


# --------------------------------------------------------------------------------------
# the one per-request record
# --------------------------------------------------------------------------------------

#: Exact key set of a record. Both clients emit this and nothing else, so a records.jsonl
#: from either backend parses with the same reader.
REQUEST_RECORD_KEYS: tuple[str, ...] = (
    "schema_version",
    "backend",
    "run",
    "run_id",
    "profile",
    "point",
    "repeat",
    "warmup",
    "request_index",
    "request_id",
    "sample_id",
    "group",
    "text",
    "text_chars",
    "ok",
    "finish_reason",
    "errors",
    "t_submit_s",
    "t_connect_ms",
    "ttfa_s",
    "ttfa_with_connect_s",
    "leading_silence_s",
    "ttfa_audible_s",
    "wall_s",
    "chunks",
    "chunk_arrival_s",
    "chunk_bytes",
    "chunk_audio_s",
    "audio_s",
    "frames",
    "sample_rate",
    "sample_width",
    "channels",
    "continuity_threshold_s",
    "buffer_deficit_s",
    "continuity_ok",
    "underrun_event_count",
    "continuity_ok_by_threshold",
    "delivery_rtf_wall_over_audio",
    "delivery_rtfx_audio_over_wall",
    "gap_p95_s",
    "gap_max_s",
    "audio_s_per_chunk_median",
    "normalised_gap_p95",
    "samples",
    "audio_seconds",
    "rms",
    "peak",
    "silence_ratio",
    "clipping_ratio",
    "wav",
    "server_meta",
)


def build_request_record(
    *,
    backend: str,
    run: str,
    run_id: str,
    profile: str,
    point: str,
    repeat: int,
    warmup: bool,
    request_index: int,
    request_id: str,
    sample_id: str,
    group: str,
    text: str,
    ok: bool,
    finish_reason: str | None,
    errors: list[dict],
    t_submit_s: float,
    t_connect_ms: float | None,
    chunk_arrival_s: Sequence[float],
    chunk_bytes: Sequence[int],
    wall_s: float,
    pcm_bytes: bytes | bytearray | memoryview | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    channels: int = DEFAULT_CHANNELS,
    threshold_s: float | None = None,
    wav: str | None = None,
    server_meta: dict[str, Any] | None = None,
) -> dict:
    """Build the per-request record. All derived metrics are computed HERE, once.

    chunk_arrival_s is measured from the moment the request was SENT on an already
    established connection; connect cost is reported separately (t_connect_ms) so the
    WS and HTTP transports stay comparable.
    """
    arrivals = [float(t) for t in chunk_arrival_s]
    sizes = [int(b) for b in chunk_bytes]
    threshold = continuity_threshold_s() if threshold_s is None else float(threshold_s)

    stats = compute_continuity_stats(
        chunk_arrival_times_s=arrivals,
        chunk_bytes=sizes,
        sample_rate=sample_rate,
        sample_width=sample_width,
        channels=channels,
        threshold_s=threshold,
    )
    by_threshold = {
        f"{thr:g}": compute_continuity_stats(
            chunk_arrival_times_s=arrivals,
            chunk_bytes=sizes,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
            threshold_s=thr,
        ).is_continuous
        for thr in CONTINUITY_REPORT_THRESHOLDS_S
    }

    total_bytes = sum(sizes)
    audio_s = audio_seconds_from_bytes(total_bytes, sample_rate, sample_width, channels)
    chunk_audio_s = [
        audio_seconds_from_bytes(b, sample_rate, sample_width, channels) for b in sizes
    ]
    gaps = inter_chunk_gaps_s(arrivals)
    gap_p95 = percentile(gaps, 0.95)
    per_chunk_median = percentile(chunk_audio_s, 0.5)
    ttfa = arrivals[0] if arrivals else None
    leading_silence_s = (
        audible_onset_offset_s(pcm_bytes, sample_rate, sample_width, channels)
        if pcm_bytes
        else 0.0
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "run": run,
        "run_id": run_id,
        "profile": profile,
        "point": point,
        "repeat": int(repeat),
        "warmup": bool(warmup),
        "request_index": int(request_index),
        "request_id": request_id,
        "sample_id": sample_id,
        "group": group,
        "text": text,
        "text_chars": len(text),
        "ok": bool(ok),
        "finish_reason": finish_reason,
        "errors": list(errors or []),
        "t_submit_s": float(t_submit_s),
        "t_connect_ms": None if t_connect_ms is None else float(t_connect_ms),
        "ttfa_s": ttfa,
        "ttfa_with_connect_s": (
            None if ttfa is None or t_connect_ms is None else ttfa + float(t_connect_ms) / 1000.0
        ),
        "leading_silence_s": leading_silence_s,
        "ttfa_audible_s": (ttfa + leading_silence_s) if (ok and ttfa is not None) else None,
        "wall_s": float(wall_s),
        "chunks": len(arrivals),
        "chunk_arrival_s": arrivals,
        "chunk_bytes": sizes,
        "chunk_audio_s": chunk_audio_s,
        "audio_s": audio_s,
        "frames": audio_s / SECONDS_PER_FRAME,
        "sample_rate": int(sample_rate),
        "sample_width": int(sample_width),
        "channels": int(channels),
        "continuity_threshold_s": threshold,
        "buffer_deficit_s": stats.max_underrun_s,
        "continuity_ok": stats.is_continuous,
        "underrun_event_count": stats.underrun_event_count,
        "continuity_ok_by_threshold": by_threshold,
        # both directions, spelled out: the skill quotes audio/wall (higher better), the
        # design doc and vllm-omni's audio_rtf quote wall/audio (lower better, <1 = realtime)
        "delivery_rtf_wall_over_audio": (wall_s / audio_s) if audio_s > 0 else None,
        "delivery_rtfx_audio_over_wall": (audio_s / wall_s) if wall_s > 0 else None,
        "gap_p95_s": gap_p95,
        "gap_max_s": max(gaps) if gaps else None,
        "audio_s_per_chunk_median": per_chunk_median,
        "normalised_gap_p95": (
            gap_p95 / per_chunk_median if gap_p95 is not None and per_chunk_median else None
        ),
        "wav": wav,
        "server_meta": dict(server_meta or {}),
    }
    record.update(pcm_stats(pcm_bytes, sample_rate) if pcm_bytes is not None else {
        "samples": int(total_bytes // (sample_width * channels)),
        "audio_seconds": audio_s,
        "rms": None,
        "peak": None,
        "silence_ratio": None,
        "clipping_ratio": None,
    })
    missing = set(REQUEST_RECORD_KEYS) - set(record)
    extra = set(record) - set(REQUEST_RECORD_KEYS)
    if missing or extra:
        raise AssertionError(f"record schema drift: missing={sorted(missing)} extra={sorted(extra)}")
    return record
