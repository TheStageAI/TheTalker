"""Locust file for a streaming TTS endpoint (/v1/audio/speech), Qwen3-TTS shown.

Configuration is at the top of the file; the reference clip and its transcript
come from the environment, since the checkpoint below is served in the
voice-cloning task type.

    pip install "thetalker[load]"
    python examples/get_reference.py --out reference.wav
    export REF_AUDIO=reference.wav REF_TEXT="<the transcript it printed>"
    locust -f examples/load_test_locust.py --headless -u 32 -r 32 -t 5m

It reports two request types per call: `ttfa` is the time to the first audio
byte, `inference` the whole streamed response. The numbers in this repository's
Benchmarks table are not taken with locust -- use benchmark/run_benchmark.py for
those; this file is for holding a server under open-loop load and watching it.
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from locust import User, between, events, task

# Insert your inference URL here
INFERENCE_URL = "http://127.0.0.1:8091/v1/audio/speech"
ENDPOINT_PATH = urlparse(INFERENCE_URL).path

# Insert your authorization token here, e.g. "Bearer <apitoken>"
# If you didn't set any authorization, set AUTHORIZATION = ""
AUTHORIZATION = ""
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Voice-clone reference: a clean mono wav of 5-15 s and its word-for-word
# transcript. examples/get_reference.py downloads an openly licensed one.
REF_AUDIO = os.environ.get("REF_AUDIO")
REF_TEXT = os.environ.get("REF_TEXT")
TASK_TYPE = "Base"
LANGUAGE = "English"
MAX_NEW_TOKENS = 100

TEXTS_JSONL = os.environ.get(
    "TEXTS_JSONL",
    str(Path(__file__).resolve().parent.parent / "benchmark" / "prompts" / "seed_tts_eval_en_100.jsonl"),
)

MIN_WAIT = 0
MAX_WAIT = 0

READ_SIZE = 8192
TIMEOUT = 600

HEADERS = {
    "Authorization": AUTHORIZATION,
    "Content-Type": "application/json",
}


def _load_texts() -> list[str]:
    texts = []
    for line in Path(TEXTS_JSONL).read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(json.loads(line)["text"])
    if not texts:
        raise RuntimeError(f"no prompts in {TEXTS_JSONL}")
    return texts


def _ref_audio_b64() -> str:
    if not REF_AUDIO or not REF_TEXT:
        raise RuntimeError(
            f"task type {TASK_TYPE} needs a reference clip: set REF_AUDIO and REF_TEXT "
            "(python examples/get_reference.py --out reference.wav)"
        )
    return "data:audio/wav;base64," + base64.b64encode(Path(REF_AUDIO).read_bytes()).decode("ascii")


TEXTS = _load_texts()


class TTSUser(User):
    wait_time = between(MIN_WAIT, MAX_WAIT)

    def on_start(self) -> None:
        self.ref_audio_b64 = _ref_audio_b64() if TASK_TYPE == "Base" else None

    @task
    def infer(self) -> None:
        payload = {
            "model": MODEL_NAME,
            "input": random.choice(TEXTS),
            "task_type": TASK_TYPE,
            "language": LANGUAGE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        }
        if self.ref_audio_b64 is not None:
            payload["ref_audio"] = self.ref_audio_b64
            payload["ref_text"] = REF_TEXT

        start_time = time.perf_counter()
        ttfa_ms = None
        audio_bytes = 0
        try:
            with requests.post(
                INFERENCE_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=TIMEOUT,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=READ_SIZE):
                    if not chunk:
                        continue
                    if ttfa_ms is None:
                        ttfa_ms = int((time.perf_counter() - start_time) * 1000)
                        events.request.fire(
                            request_type="ttfa",
                            name=ENDPOINT_PATH,
                            response_time=ttfa_ms,
                            response_length=len(chunk),
                            exception=None,
                        )
                    audio_bytes += len(chunk)
            events.request.fire(
                request_type="inference",
                name=ENDPOINT_PATH,
                response_time=int((time.perf_counter() - start_time) * 1000),
                response_length=audio_bytes,
                exception=None,
            )
        except Exception as exc:
            events.request.fire(
                request_type="inference",
                name=ENDPOINT_PATH,
                response_time=int((time.perf_counter() - start_time) * 1000),
                response_length=audio_bytes,
                exception=exc,
            )
