"""Smoke tests for thetalker/metrics.py -- no server, no GPU.

Run with: pytest benchmark/test_metrics.py
"""

from __future__ import annotations

from thetalker.metrics import (
    build_request_record,
    compute_continuity_stats,
    percentile,
)

SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_S = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


def test_percentile_basic():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.5) is None
    assert percentile([None, None], 0.9) is None


def test_continuity_stats_no_gap():
    """Chunks arrive faster than playback consumes them: no underrun."""
    chunk_bytes = [BYTES_PER_S // 10] * 5  # 100 ms of audio per chunk
    # arrivals well ahead of the 100 ms/chunk playback clock
    chunk_arrival_s = [0.0, 0.02, 0.04, 0.06, 0.08]
    stats = compute_continuity_stats(
        chunk_arrival_times_s=chunk_arrival_s,
        chunk_bytes=chunk_bytes,
        sample_rate=SAMPLE_RATE,
        sample_width=SAMPLE_WIDTH,
        channels=CHANNELS,
        threshold_s=0.1,
    )
    assert stats.is_continuous
    assert stats.max_underrun_s == 0.0
    assert stats.underrun_event_count == 0


def test_continuity_stats_underrun():
    """A long gap between chunk 0 and chunk 1 starves playback."""
    chunk_bytes = [BYTES_PER_S // 10] * 3  # 100 ms of audio per chunk
    chunk_arrival_s = [0.0, 1.0, 1.1]  # 1 s gap after the first chunk
    stats = compute_continuity_stats(
        chunk_arrival_times_s=chunk_arrival_s,
        chunk_bytes=chunk_bytes,
        sample_rate=SAMPLE_RATE,
        sample_width=SAMPLE_WIDTH,
        channels=CHANNELS,
        threshold_s=0.1,
    )
    assert not stats.is_continuous
    assert stats.max_underrun_s > 0.8
    assert stats.underrun_event_count >= 1


def test_build_request_record_schema_and_continuity():
    chunk_bytes = [BYTES_PER_S // 10] * 5
    chunk_arrival_s = [0.0, 0.02, 0.04, 0.06, 0.08]
    pcm = b"\x00\x01" * (sum(chunk_bytes) // 2)
    record = build_request_record(
        backend="test",
        run="test_run_label",
        run_id="test_run",
        profile="wave",
        point="c1",
        repeat=0,
        warmup=False,
        request_index=0,
        request_id="req_0000",
        sample_id="sample_0",
        group="",
        text="hello world",
        ok=True,
        finish_reason="stream_end",
        errors=[],
        t_submit_s=0.0,
        t_connect_ms=1.5,
        chunk_arrival_s=chunk_arrival_s,
        chunk_bytes=chunk_bytes,
        wall_s=0.08,
        pcm_bytes=pcm,
        sample_rate=SAMPLE_RATE,
        sample_width=SAMPLE_WIDTH,
        channels=CHANNELS,
        threshold_s=0.1,
        wav=None,
        server_meta={},
    )
    assert record["ok"] is True
    assert record["continuity_ok"] is True
    assert record["chunks"] == 5
    assert record["ttfa_s"] == 0.0
    assert record["audio_s"] > 0.0
    assert record["delivery_rtfx_audio_over_wall"] > 0.0
