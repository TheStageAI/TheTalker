"""The shipped benchmark reference clip is the exact file the published numbers used."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference"
SHA256 = "331699b7166bc525ea61cad994e75ce46556998d31aa99ac4a4dfb14ceddb549"


def test_reference_clip_is_the_measured_file():
    assert hashlib.sha256((REFERENCE / "ref.wav").read_bytes()).hexdigest() == SHA256


def test_reference_clip_format():
    with wave.open(str(REFERENCE / "ref.wav")) as handle:
        assert (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) == (1, 24000, 2)
        assert abs(handle.getnframes() / handle.getframerate() - 4.64) < 0.01


def test_reference_transcript():
    assert (REFERENCE / "ref.txt").read_text().strip() == "Hello. This is a short Qwen TTS runtime smoke test."
