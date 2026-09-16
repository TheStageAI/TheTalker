"""Tests for thetalker/text.py sentence splitting -- no server, no GPU.

Run with: pytest benchmark/test_text.py
"""

from __future__ import annotations

import pytest

from thetalker.text import split_sentences


def test_two_plain_sentences():
    assert split_sentences("The fox ran away. The dog stayed.") == [
        "The fox ran away.",
        "The dog stayed.",
    ]


def test_decimal_is_not_a_boundary():
    assert split_sentences("Pi is about 3.14 in most schoolbooks. It never ends.") == [
        "Pi is about 3.14 in most schoolbooks.",
        "It never ends.",
    ]


def test_abbreviation_is_not_a_boundary():
    assert split_sentences("Dr. Smith went home. He slept.") == [
        "Dr. Smith went home.",
        "He slept.",
    ]


def test_ellipsis_is_not_a_boundary():
    assert split_sentences("I waited... Nobody came.") == ["I waited... Nobody came."]


def test_initials_are_not_a_boundary():
    assert split_sentences("J. K. Rowling wrote it. Then she rested.") == [
        "J. K. Rowling wrote it.",
        "Then she rested.",
    ]


def test_other_terminators_and_whitespace():
    text = "  Stop!   Who goes there?\nA friend; \"Come in.\"  "
    assert split_sentences(text) == ["Stop!", "Who goes there?", "A friend;", "\"Come in.\""]


def test_run_on_sentence_splits_at_commas():
    clause = "and then the long road went on past the old mill"
    text = ", ".join([clause] * 14) + ", and home."
    assert len(text) > 700
    parts = split_sentences(text, max_chars=300)
    assert len(parts) >= 3
    assert all(len(part) <= 300 for part in parts)
    assert all(part.endswith(",") for part in parts[:-1])
    assert " ".join(parts) == text


def test_long_sentence_without_commas_splits_at_spaces():
    text = " ".join(["word"] * 100)
    parts = split_sentences(text, max_chars=50)
    assert all(len(part) <= 50 for part in parts)
    assert " ".join(parts) == text


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   \n\t ") == []


def test_cyrillic_sentences_split():
    parts = ["\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440.",
             "\u041a\u0430\u043a \u0434\u0435\u043b\u0430?",
             "\u0425\u043e\u0440\u043e\u0448\u043e."]
    assert split_sentences(" ".join(parts)) == parts


def test_accented_capital_starts_a_sentence():
    parts = ["Bonjour le monde.", "\u00c7a va bien.", "\u00c9lan vital."]
    assert split_sentences(" ".join(parts)) == parts


def test_chinese_full_stop_is_not_a_boundary():
    """Documented limit: no letter case and no space between sentences, so the
    rule cannot see a boundary and the text comes back whole."""
    text = "\u4f60\u597d\u4e16\u754c\u3002\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002"
    assert split_sentences(text) == [text]


def test_list_number_stays_with_its_item():
    assert split_sentences("1. First item. 2. Second item.") == [
        "1. First item.",
        "2. Second item.",
    ]


def test_said_no_ends_a_sentence():
    assert split_sentences("He said no. Then he left.") == ["He said no.", "Then he left."]


def test_saint_abbreviation_keeps_the_sentence_together():
    """Known ambiguity: "st" is kept for "St. Paul", so a sentence ending in the
    street abbreviation is not split."""
    assert split_sentences("We met at St. Paul. He was late.") == [
        "We met at St. Paul.",
        "He was late.",
    ]
    assert split_sentences("He lives on Main St. Then he moved.") == [
        "He lives on Main St. Then he moved."
    ]


@pytest.mark.parametrize("max_chars", [0, -1])
def test_non_positive_max_chars_is_rejected(max_chars):
    with pytest.raises(ValueError):
        split_sentences("Any text at all.", max_chars)
