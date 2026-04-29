"""Input sanitation for the interactive CLI.

Stage 1 of the flow:
    raw text -> sanitize -> list[str] -> analyzer -> ...

Every interactive entrypoint funnels its raw text through one of these
functions before anything else touches it. The decomposer and downstream
stages only ever see sanitized, tr_lower'd, punctuation-free input.
"""
import re
from typing import List

from util.word_methods import tr_lower

_APOSTROPHE_RE = re.compile(r"['’‘]")
_PUNCT_RE = re.compile(r"[^\w\s]|_")


def sanitize_word(raw: str) -> str:
    """Canonical form of a single-word input."""
    return tr_lower(_APOSTROPHE_RE.sub("", raw.strip()))


def sanitize_sentence(raw: str) -> List[str]:
    """Split a sentence into sanitized words (punct stripped, tr_lower'd)."""
    s = _APOSTROPHE_RE.sub("", raw)
    s = _PUNCT_RE.sub(" ", s)
    return tr_lower(s).split()
