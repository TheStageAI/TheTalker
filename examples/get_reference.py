#!/usr/bin/env python3
"""Download one openly licensed reference utterance for the voice-cloning path.

The Base task type every command in this repository uses is voice cloning: a
request carries a recording of the voice to imitate and that recording's exact
transcript. This script fetches one utterance from LibriTTS-R test-clean,
writes it as a mono 24 kHz WAV and prints the transcript to paste into
--ref-text.

    pip install "thetalker[examples]"
    python examples/get_reference.py --out reference.wav

The reference sets the bandwidth of the cloned voice: a model asked to imitate
a 16 kHz recording returns a voice with nothing above 8 kHz, whatever rate it
serves at. LibriTTS-R is 24 kHz, so the clip does not cap the models it is
given to.

Source: LibriTTS-R, `mythicinfinity/libritts_r` on the Hugging Face Hub, the
test split of the `clean` config. It is derived from LibriVox public-domain
audiobooks and is distributed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/); cite
Y. Koizumi et al., "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech
Corpus", Interspeech 2023, and H. Zen et al., "LibriTTS: A Corpus Derived from
LibriSpeech for Text-to-Speech", Interspeech 2019. No audio is redistributed by
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

DATASET = "mythicinfinity/libritts_r"
CONFIG = "clean"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reference.wav"))
    parser.add_argument("--index", type=int, default=0,
                        help="which utterance of the split to take")
    parser.add_argument("--min-seconds", type=float, default=5.0,
                        help="skip utterances shorter than this")
    parser.add_argument("--max-seconds", type=float, default=12.0,
                        help="skip utterances longer than this")
    parser.add_argument("--any-transcript", action="store_true",
                        help="keep utterances whose transcript carries dialogue "
                             "punctuation or a stammer, which are skipped by default")
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


def transcript_of(row) -> str:
    """LibriTTS-R carries both a normalized and a verbatim transcript."""
    raw = row.get("text_normalized") or row.get("text_original") or row.get("text") or ""
    return " ".join(str(raw).split())


def usable_transcript(text: str) -> bool:
    """A reference prompt the cloner can follow: one plain spoken sentence.

    Dialogue quotes and ellipsis stammers read as pauses and repeated words, so
    the voice they produce is uneven and the --ref-text a caller pastes no
    longer matches what the clip says.
    """
    if not text:
        return False
    if any(ch in text for ch in '"\u201c\u201d'):
        return False
    return ". . ." not in text and "..." not in text


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
            text = transcript_of(row)
            if not args.any_transcript and not usable_transcript(text):
                continue
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
                "text": text,
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
