#!/usr/bin/env python3
"""Download one openly licensed reference utterance for the voice-cloning path.

The Base task type every command in this repository uses is voice cloning: a
request carries a recording of the voice to imitate and that recording's exact
transcript. This script fetches one utterance from LibriSpeech test-clean,
writes it as a mono 16 kHz WAV and prints the transcript to paste into
--ref-text.

    pip install "thetalker[examples]"
    python examples/get_reference.py --out reference.wav

Source: LibriSpeech ASR corpus, `openslr/librispeech_asr` on the Hugging Face
Hub, the test split of the `clean` config. LibriSpeech is derived from LibriVox
public-domain
audiobooks and is distributed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/); cite
V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "Librispeech: an ASR corpus
based on public domain audio books", ICASSP 2015. No audio is redistributed by
this repository -- the clip is downloaded on your machine, at your own
acceptance of those terms.

Any clean mono WAV of 5 to 15 seconds of a single speaker works just as well;
this script only exists so the commands in the README are runnable without
supplying one.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import wave
from pathlib import Path

DATASET = "openslr/librispeech_asr"
CONFIG = "clean"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reference.wav"))
    parser.add_argument("--index", type=int, default=0,
                        help="which utterance of the split to take")
    parser.add_argument("--min-seconds", type=float, default=5.0,
                        help="skip utterances shorter than this")
    parser.add_argument("--max-seconds", type=float, default=15.0,
                        help="skip utterances longer than this")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face datasets cache")
    return parser.parse_args()


def test_split_name() -> str:
    """The test split of the clean config; releases name it `test` or `test.clean`."""
    from datasets import get_dataset_split_names

    names = get_dataset_split_names(DATASET, CONFIG)
    for name in names:
        if name.startswith("test"):
            return name
    raise SystemExit(f"no test split in {DATASET}/{CONFIG}; found {names}")


def load_utterances(cache_dir: str | None):
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        raise SystemExit(
            'datasets is not installed; run: pip install "thetalker[examples]"'
        )
    split = test_split_name()
    # streaming: the split is hundreds of MB and only one utterance is needed.
    dataset = load_dataset(
        DATASET, CONFIG, split=split, streaming=True, cache_dir=cache_dir
    )
    # decode=False keeps the raw bytes, so soundfile decodes them rather than
    # the datasets audio backend, which is an extra heavyweight dependency.
    return split, dataset.cast_column("audio", Audio(decode=False))


def decode(raw: bytes):
    try:
        import io

        import soundfile as sf
    except ImportError:
        raise SystemExit(
            'soundfile is not installed; run: pip install "thetalker[examples]"'
        )
    samples, sample_rate = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, int(sample_rate)


def write_wav(path: Path, samples, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def pick_utterance(args: argparse.Namespace) -> tuple[str, dict | None]:
    """First utterance in the length window, at --index among those."""
    split, utterances = load_utterances(args.cache_dir)
    # The streaming iterator holds a background download; leaving it suspended
    # until interpreter shutdown crashes on GIL release, so close it explicitly.
    iterator = iter(utterances)
    kept = 0
    found = None
    try:
        for row in iterator:
            samples, sample_rate = decode(row["audio"]["bytes"])
            seconds = len(samples) / sample_rate
            if not args.min_seconds <= seconds <= args.max_seconds:
                continue
            if kept < args.index:
                kept += 1
                continue
            found = {
                "samples": samples,
                "sample_rate": sample_rate,
                "seconds": seconds,
                "text": " ".join(str(row["text"]).split()).capitalize(),
            }
            break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator, utterances
        gc.collect()
    return split, found


def main() -> int:
    args = parse_args()
    split, found = pick_utterance(args)
    if found is None:
        print(
            f"no utterance between {args.min_seconds} s and {args.max_seconds} s "
            f"found at index {args.index}",
            file=sys.stderr,
        )
        return 1

    write_wav(args.out, found["samples"], found["sample_rate"])
    print(f"wrote {args.out} ({found['seconds']:.1f} s, {found['sample_rate']} Hz mono)")
    print(f"source: {DATASET} config {CONFIG} split {split}, {LICENSE}")
    print()
    print("transcript, pass this as --ref-text:")
    print(found["text"])
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # The streaming download leaves an async filesystem thread that some
    # datasets/fsspec combinations abort on during interpreter finalization,
    # after the wav is already written and closed. Leave without finalizing.
    os._exit(code)
