"""Sentence splitter for sending a long text as several parallel requests.

Regex-based on purpose: no model download, predictable on English prose. It
splits after `.`, `!`, `?`, `;` or `:` followed by whitespace and an uppercase
letter of any script, a digit or an opening quote, and keeps decimals, common
abbreviations, initials, list numbers and ellipses intact.

A script that has no letter case and writes sentences without spaces between
them (Chinese, Japanese) has no boundary this rule can see: such a text is
returned whole, or cut by length alone.
"""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "prof", "jr", "sr",
    "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
})

_BOUNDARY = re.compile(
    r"([.!?;:]+)[\"')\]\u201d\u2019]*\s+(?=[\"'(\[\u201c\u2018]?(\w))"
)
# A single letter or dotted letters such as "J" or "U.S": initials and acronyms.
_INITIALS = re.compile(r"^(?:[A-Za-z]\.)*[A-Za-z]$")
# "1." in "1. First item." numbers the item; it is not a sentence of its own.
_LIST_NUMBER = re.compile(r"^\d+$")
_WORD = re.compile(r"\w")


def _is_boundary(text: str, match: re.Match) -> bool:
    after = match.group(2)
    if not (after.isupper() or after.isdigit()):
        return False
    punct = match.group(1)
    if ".." in punct:
        return False
    if punct != ".":
        return True
    before = text[: match.start()].rsplit(None, 1)
    word = before[-1].lstrip("\"'([\u201c\u2018") if before else ""
    if word.lower() in ABBREVIATIONS or _LIST_NUMBER.match(word):
        return False
    return not _INITIALS.match(word)


def _split_long(sentence: str, max_chars: int) -> list[str]:
    parts = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(",", 0, max_chars)
        if cut > 0:
            cut += 1
        else:
            cut = sentence.rfind(" ", 0, max_chars + 1)
            if cut <= 0:
                cut = max_chars
        parts.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    parts.append(sentence)
    return [part for part in parts if part]


def _merge_wordless(parts: list[str], max_chars: int) -> list[str]:
    """A part with no word character is punctuation the model would read aloud.

    It joins a neighbour instead of becoming a request of its own. A run that
    does not fit fills the part before it and the rest rides on the part after,
    so the merge stays inside max_chars.
    """
    merged: list[str] = []
    pending = ""
    for part in parts:
        if pending:
            part, pending = f"{pending} {part}", ""
        if not _WORD.search(part):
            room = max(max_chars - len(merged[-1]) - 1, 0) if merged else 0
            if room:
                merged[-1] = f"{merged[-1]} {part[:room]}"
            pending = part[room:]
            continue
        merged.append(part)
    if pending and merged:
        merged[-1] = f"{merged[-1]} {pending}"
    return merged


def split_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Split `text` into sentences of at most `max_chars` characters."""
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")
    text = " ".join(text.split())
    if not text:
        return []
    sentences = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _is_boundary(text, match):
            sentences.append(text[start: match.end()].strip())
            start = match.end()
    sentences.append(text[start:].strip())
    parts = []
    for sentence in sentences:
        if sentence:
            parts.extend(_split_long(sentence, max_chars))
    return _merge_wordless(parts, max_chars)
