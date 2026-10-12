"""Adversarial checks for thetalker/text.py and the joining in run_long_text.py.

No server, no GPU. Run with: pytest benchmark/test_text_adversarial.py
"""

from __future__ import annotations

import argparse
import http.client
import importlib
import json
import random
import re
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from thetalker.text import split_sentences

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "benchmark" / "prompts"
MAX_CHARS = (300, 120, 60)
_WORD = re.compile(r"\w")


def _prompt_texts() -> list[str]:
    texts: list[str] = []
    for name in ("seed_tts_eval_en_100.jsonl", "seed_tts_eval_en_200.jsonl"):
        with (PROMPTS / name).open(encoding="utf-8") as handle:
            texts += [json.loads(line)["text"] for line in handle if line.strip()]
    return texts


def _prompt_groups(count: int = 400, seed: int = 20260922) -> list[str]:
    rng = random.Random(seed)
    texts = _prompt_texts()
    groups = []
    for index in range(count):
        picked = rng.sample(texts, rng.randint(1, 12))
        separator = "\n\t" if index % 5 == 0 else " "
        groups.append(separator.join(picked))
    return groups


# built once: every case below indexes into it, and it reads both prompt files.
PROMPT_GROUPS = _prompt_groups()

ADVERSARIAL = {
    "money": "It cost $3.50. She paid the rest in cash.",
    "decimal": "Pi is 3.14159 and e is 2.71828. Both are irrational.",
    "thousands": "The city holds 1,234,567 people. That is a lot of buses.",
    "ellipsis_dots": "I waited... Nobody came. Then the bell rang.",
    "ellipsis_char": "I waited\u2026 Nobody came. Then the bell rang.",
    "initials": "J. K. Rowling wrote it. Then she rested.",
    "initials_run": "A. B. C. Smith signed the form. D. E. Jones did not.",
    "abbrev_mid": "We met Dr. Smith and Mrs. Brown in Ohio. They left at once.",
    "abbrev_end": "She grew up in the U.S. She left at twenty.",
    "abbrev_etc": "Bring bread, milk, etc. Then come home.",
    "quote_period_inside": "He said \"Go home.\" Then he left.",
    "quote_period_outside": "He called it \"the end\". Then he left.",
    "quote_nested": "She said, \"He told me 'Run.' and vanished.\" Nobody moved.",
    "quote_curly": "\u201cStop,\u201d she said. \u201cIt is over.\u201d",
    "parens_span": "He left (it was late. Very late) and never came back.",
    "em_dash": "He paused \u2014 a long pause \u2014 then spoke. She did not.",
    "url": "Visit www.example.com/a.b for the details. It is free.",
    "email": "Write to a.b@c.de today. He answers fast.",
    "filename": "Open config.yaml first. Then run it twice.",
    "version": "We shipped v1.2.3 today. Tomorrow we rest.",
    "numbered_list": "1. First 2. Second 3. Third",
    "roman": "Chapter IV. The end came quickly.",
    "bang_question": "Really!? I cannot believe it.",
    "repeated_punct": "What?!?! No way!!! Go.",
    "newlines_tabs": "First line\n\tsecond line. Third one\there.",
    "nbsp": "One\u00a0two three. Four\u00a0five six.",
    "runon_5000": " ".join(["the road went on and on past the mill"] * 120),
    "runon_no_space": "a" * 5000,
    "punct_only": "... !!! ---",
    "punct_only_char": "\u2026",
    "punct_run_inside": "Start. " + "." * 450 + " End.",
    "empty": "",
    "whitespace": "   \n\t ",
    "colon_then_list": "Items: 1. eggs 2. milk",
    "digit_after_period": "He owed 5. 6 dollars were left.",
    "no_final_period": "The last sentence has no terminator",
    "single_word": "Hello",
    "semicolons": "It was gone; all was dark; nothing moved.",
    "comma_flood": "one, " * 200 + "and done.",
    "mixed_scripts": "He read \u041f\u0440\u0438\u0432\u0435\u0442 aloud. Then he stopped.",
}

# Nothing to speak: split_sentences returns no parts, so the round trip does not apply.
NO_SPEECH = {"punct_only", "punct_only_char", "empty", "whitespace"}


def _strip_ws(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def _longest_token(text: str) -> int:
    return max((len(token) for token in text.split()), default=0)


def _round_trip_failures(text: str, parts: list[str]) -> list[str]:
    failures = []
    if _strip_ws("".join(parts)) != _strip_ws(text):
        failures.append("joined parts differ from the input by more than whitespace")
    if _words(" ".join(parts)) != _words(text):
        failures.append("word sequence changed")
    return failures


def _invariant_failures(text: str, parts: list[str], max_chars: int) -> list[str]:
    failures = []
    for part in parts:
        if not part.strip():
            failures.append("empty part")
        if not _WORD.search(part):
            failures.append("part is punctuation only: %r" % part[:40])
    failures += _over_cap_failures(parts, max_chars)
    failures += _cut_failures(text, parts, max_chars)
    return failures


def _over_cap_failures(parts: list[str], max_chars: int) -> list[str]:
    """A part may pass max_chars only when no neighbour has room to take the excess.

    A punctuation run longer than a whole part has to sit somewhere, and every
    part must keep a word in it; anything else over the cap is a packing bug.
    """
    failures = []
    for index, part in enumerate(parts):
        if len(part) <= max_chars:
            continue
        neighbours = parts[max(index - 1, 0):index] + parts[index + 1:index + 2]
        if any(len(other) + 1 < max_chars for other in neighbours):
            failures.append("part of %d chars over max_chars %d beside a part with room"
                            % (len(part), max_chars))
    return failures


def _cut_failures(text: str, parts: list[str], max_chars: int) -> list[str]:
    """Every part boundary must fall between two input tokens, not inside one."""
    tokens = text.split()
    owner = [index for index, token in enumerate(tokens) for _ in token]
    dense = "".join(tokens)
    failures = []
    position = 0
    for part in parts:
        squeezed = "".join(part.split())
        if not dense.startswith(squeezed, position):
            failures.append("part is not the next run of the input: %r" % part[:40])
            break
        position += len(squeezed)
        if position < len(dense) and owner[position - 1] == owner[position]:
            if len(tokens[owner[position]]) <= max_chars:
                failures.append("cut mid-word: %r" % dense[max(0, position - 20):position + 20])
    return failures


def _check(name: str, text: str) -> list[str]:
    problems = []
    for max_chars in MAX_CHARS:
        parts = split_sentences(text, max_chars)
        found = _invariant_failures(text, parts, max_chars)
        if _longest_token(text) <= max_chars:
            found += _round_trip_failures(text, parts)
        problems += ["%s [max_chars=%d] %s" % (name, max_chars, item) for item in found]
    return problems


@pytest.mark.parametrize("index", range(len(PROMPT_GROUPS)))
def test_prompt_groups_round_trip_and_invariants(index):
    assert _check("group_%d" % index, PROMPT_GROUPS[index]) == []


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_round_trip_and_invariants(name):
    text = ADVERSARIAL[name]
    parts = split_sentences(text)
    if name in NO_SPEECH:
        assert parts == []
        return
    assert parts, "expected at least one part"
    assert _check(name, text) == []


def test_prompt_corpus_still_exercises_the_splitter():
    """The 400 cases are only worth running while the corpus behind them varies."""
    assert len(_prompt_texts()) == 300, "a prompt file changed size"
    lengths = [len(group) for group in PROMPT_GROUPS]
    assert min(lengths) < 100 < 500 < max(lengths), "the groups no longer span short and long"
    assert any("\t" in group for group in PROMPT_GROUPS), "no group carries the tab separator"
    split = [split_sentences(group, 60) for group in PROMPT_GROUPS]
    assert all(split), "a group yielded no part"
    assert sum(len(parts) > 1 for parts in split) > len(PROMPT_GROUPS) // 2, (
        "hardly any group splits; the corpus no longer reaches the splitting paths"
    )


@pytest.mark.parametrize(
    "text", ["", "   \n\t ", "... !!! ---", "\u2026", "-- ... --", "!!!"]
)
def test_text_without_words_yields_no_parts(text):
    assert split_sentences(text) == []


def test_punctuation_run_does_not_become_its_own_part():
    parts = split_sentences("Start. " + "." * 450 + " End.", 300)
    assert all(_WORD.search(part) for part in parts)
    assert _words(" ".join(parts)) == ["Start", "End"]
    assert all(len(part) <= 300 for part in parts), [len(part) for part in parts]


def test_unbreakable_token_is_the_only_mid_word_cut():
    parts = split_sentences("short words " + "z" * 700, 300)
    assert parts[0] == "short words"
    assert "".join(parts[1:]) == "z" * 700
    assert split_sentences("a" * 5000, 300) == ["a" * 300] * 16 + ["a" * 200]


def test_run_on_without_punctuation_splits_at_spaces():
    text = ADVERSARIAL["runon_5000"]
    assert len(text) > 4000 and not re.search(r"[.!?;:]", text)
    parts = split_sentences(text, 300)
    assert len(parts) > 10
    assert all(len(part) <= 300 for part in parts)
    assert " ".join(parts) == text


# examples/run_long_text.py: the parts must be joined in text order.


def _run_long_text():
    examples = str(REPO / "examples")
    if examples not in sys.path:
        sys.path.insert(0, examples)
    return importlib.import_module("run_long_text")


def _fake_args() -> argparse.Namespace:
    return argparse.Namespace(host="127.0.0.1", port=1, path="/v1/audio/speech",
                              timeout_s=1.0, read_size=8192)


def _marker_stream(delay_for):
    """Stand-in for stream_request: marker PCM per payload, finishing out of order."""

    def stream_request(args, payload=None):
        index = int(payload)
        time.sleep(delay_for(index))
        return [0.0, 0.01], [2, 2], bytes([index, index])

    return stream_request


def test_parts_are_joined_in_text_order_not_completion_order(monkeypatch):
    module = _run_long_text()
    payloads = [str(index) for index in range(8)]
    finished: list[int] = []
    lock = threading.Lock()

    def stream_request(args, payload=None):
        index = int(payload)
        time.sleep(0.02 * (len(payloads) - index))
        with lock:
            finished.append(index)
        return [0.0, 0.01], [2, 2], bytes([index, index])

    monkeypatch.setattr(module, "stream_request", stream_request)
    results, wall_s = module.run_parts(_fake_args(), payloads, concurrency=8)

    assert finished == sorted(finished, reverse=True), "the fake did not finish out of order"
    assert b"".join(result["pcm"] for result in results) == bytes(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    )
    assert wall_s > 0


def test_more_parts_than_concurrency_keeps_text_order(monkeypatch):
    module = _run_long_text()
    payloads = [str(index) for index in range(20)]
    monkeypatch.setattr(module, "stream_request",
                        _marker_stream(lambda index: 0.01 if index % 2 else 0.0))
    results, _ = module.run_parts(_fake_args(), payloads, concurrency=3)

    assert len(results) == len(payloads)
    assert b"".join(result["pcm"] for result in results) == bytes(
        byte for index in range(20) for byte in (index, index)
    )


def test_single_part_text_is_sent_as_one_request(monkeypatch):
    module = _run_long_text()
    parts = split_sentences("hello there quiet world", 300)
    assert parts == ["hello there quiet world"]
    sent: list[str] = []

    def stream_request(args, payload=None):
        sent.append(payload)
        return [0.0], [2], b"\x01\x02"

    monkeypatch.setattr(module, "stream_request", stream_request)
    results, _ = module.run_parts(_fake_args(), parts, concurrency=8)
    assert sent == parts
    assert b"".join(result["pcm"] for result in results) == b"\x01\x02"


# examples/run_long_text.py: a server that dies must not cost the other parts.


def _main_args(tmp_path, text: str, **overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        host="127.0.0.1", port=1, path="/v1/audio/speech", task_type="CustomVoice",
        ref_audio=None, ref_text="", speaker="alice", instruct=None, language="English",
        max_new_tokens=64, sample_rate=24000, read_size=8192, timeout_s=1.0,
        text=text, text_file=None, max_chars=300, concurrency=1,
        compare_sequential=False, out=tmp_path / "out.wav",
    )
    vars(args).update(overrides)
    return args


@pytest.mark.parametrize("exc", [
    http.client.IncompleteRead(b"\x01"),
    http.client.BadStatusLine("garbage on the wire"),
    http.client.RemoteDisconnected("closed without a response"),
    http.client.HTTPException("protocol error"),
    json.JSONDecodeError("bad response", "{", 0),
    OSError("connection reset by peer"),
    SystemExit("HTTP 500: out of memory"),
])
def test_a_dead_server_costs_only_its_own_part(monkeypatch, exc):
    module = _run_long_text()
    payloads = ["0", "1", "2"]

    def stream_request(args, payload=None):
        if payload == "1":
            raise exc
        index = int(payload)
        return [0.0, 0.01], [2, 2], bytes([index, index])

    monkeypatch.setattr(module, "stream_request", stream_request)
    results, _ = module.run_parts(_fake_args(), payloads, concurrency=3)

    assert results[1]["error"], "the failed part carries no reason"
    assert b"".join(result["pcm"] for result in results) == bytes([0, 0, 2, 2])


def test_failed_part_is_reported_and_the_others_are_written(tmp_path, monkeypatch, capsys):
    module = _run_long_text()
    text = "First part here. Second part here. Third part here."

    def stream_request(args, payload=None):
        if json.loads(payload)["input"] == "Second part here.":
            raise http.client.IncompleteRead(b"\x01")
        return [0.0, 0.01], [2, 2], b"\x02\x03"

    monkeypatch.setattr(module, "parse_args", lambda: _main_args(tmp_path, text))
    monkeypatch.setattr(module, "stream_request", stream_request)
    assert module.main() == 1

    out = capsys.readouterr().out
    assert "failed" in out and "1 failed" in out
    with wave.open(str(tmp_path / "out.wav")) as handle:
        assert handle.readframes(handle.getnframes()) == b"\x02\x03" * 2


def test_compare_sequential_reports_the_failed_pass_instead_of_its_time(
    tmp_path, monkeypatch, capsys
):
    module = _run_long_text()
    text = "First part here. Second part here."
    calls = []

    def stream_request(args, payload=None):
        calls.append(payload)
        if len(calls) > 2:  # the sequential pass, after both parallel parts
            raise http.client.IncompleteRead(b"")
        return [0.0, 0.01], [2, 2], b"\x02\x03"

    monkeypatch.setattr(module, "parse_args",
                        lambda: _main_args(tmp_path, text, compare_sequential=True))
    monkeypatch.setattr(module, "stream_request", stream_request)
    assert module.main() == 1

    out = capsys.readouterr().out
    assert "sequential wall time  : not comparable" in out
    assert "one part after another" not in out


@pytest.mark.parametrize("flag", [["--max-chars", "0"], ["--max-chars", "-5"],
                                  ["--concurrency", "0"]])
def test_cli_rejects_a_limit_below_one(monkeypatch, flag):
    module = _run_long_text()
    monkeypatch.setattr(sys, "argv", ["run_long_text.py", "--task-type", "CustomVoice",
                                      "--speaker", "alice", "--text", "Hello there."] + flag)
    with pytest.raises(SystemExit) as excinfo:
        module.parse_args()
    assert excinfo.value.code == 2
