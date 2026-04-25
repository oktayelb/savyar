"""
Turkish numerals as closed-class words.

Numerals are treated as a closed class for analysis purposes, but their
tokens carry no learnable signal (they are constant patterns). Downstream
training/evaluation drops cc_numeral entries; they exist in the decomposer
only so that suffixes attached to a numeral (e.g. "beş+e") can be validated
through the normal suffix pipeline.

The set of numeral *lexemes* is small and fixed (sıfır..dokuz, on..doksan,
yüz, bin, milyon, milyar, trilyon). Any integer is expanded to a *sequence*
of these lexemes via `number_to_turkish`.
"""

from typing import List, Dict, Tuple

from util.words.closed_class import ClosedClassWord


class Numeral(ClosedClassWord):
    """A Turkish numeral lexeme (e.g. "bir", "on", "bin", "milyon")."""

    def __init__(self, word: str, value: int):
        # Numerals can accept inflectional suffixes ("beşe", "onuncu", etc.).
        # The can_take_suffixes flag is consulted by closed-class consumers;
        # the regular decomposer still needs the root to be in words.txt for
        # suffix-bearing forms, which is already the case for all numerals below.
        super().__init__(word, pos="noun", category="numeral", can_take_suffixes=True)
        self.value = value


# ----------------------------------------------------------------------------
# Numeral lexemes
# ----------------------------------------------------------------------------

_ONES: List[Tuple[str, int]] = [
    ("sıfır", 0), ("bir", 1), ("iki", 2), ("üç", 3), ("dört", 4),
    ("beş", 5), ("altı", 6), ("yedi", 7), ("sekiz", 8), ("dokuz", 9),
]
_TENS: List[Tuple[str, int]] = [
    ("on", 10), ("yirmi", 20), ("otuz", 30), ("kırk", 40),
    ("elli", 50), ("altmış", 60), ("yetmiş", 70), ("seksen", 80), ("doksan", 90),
]
_SCALES: List[Tuple[str, int]] = [
    ("yüz", 100),
    ("bin", 1_000),
    ("milyon", 1_000_000),
    ("milyar", 1_000_000_000),
    ("trilyon", 1_000_000_000_000),
]

ALL_NUMERALS: List[Numeral] = (
    [Numeral(w, v) for w, v in _ONES]
    + [Numeral(w, v) for w, v in _TENS]
    + [Numeral(w, v) for w, v in _SCALES]
)

# Set of bare numeral surface forms (for quick membership checks).
NUMERAL_WORDS: set = {n.word for n in ALL_NUMERALS}


# ----------------------------------------------------------------------------
# Integer → Turkish word list
# ----------------------------------------------------------------------------

def _below_thousand(n: int) -> List[str]:
    """Turkish words for 0 < n < 1000. Returns [] for n == 0."""
    if n <= 0:
        return []
    words: List[str] = []
    h = n // 100
    if h > 0:
        # "yüz" (not "bir yüz"), "iki yüz", "üç yüz" …
        if h > 1:
            words.append(_ONES[h][0])
        words.append("yüz")
    rest = n % 100
    t = rest // 10
    if t > 0:
        # _TENS is 0-indexed from "on" (10); index = t-1.
        words.append(_TENS[t - 1][0])
    ones = rest % 10
    if ones > 0:
        words.append(_ONES[ones][0])
    return words


def number_to_turkish(n: int) -> List[str]:
    """Convert a non-negative integer to the list of Turkish words that spell it.

    Examples:
        0     -> ["sıfır"]
        2015  -> ["iki", "bin", "on", "beş"]
        1000  -> ["bin"]
        2000  -> ["iki", "bin"]
    """
    if n < 0:
        # Negative numbers aren't part of the expected input, but keep the
        # function total — callers decide how to present the sign.
        return ["eksi"] + number_to_turkish(-n)
    if n == 0:
        return ["sıfır"]

    result: List[str] = []
    # Scale down from trilyon → milyar → milyon → bin.
    for scale_word, divisor in (
        ("trilyon", 1_000_000_000_000),
        ("milyar",  1_000_000_000),
        ("milyon",  1_000_000),
        ("bin",     1_000),
    ):
        chunk = n // divisor
        if chunk == 0:
            continue
        if chunk == 1 and scale_word == "bin":
            # "bin" (not "bir bin") for exactly one thousand.
            result.append("bin")
        else:
            result.extend(_below_thousand(chunk))
            result.append(scale_word)
        n %= divisor

    if n > 0:
        result.extend(_below_thousand(n))

    return result
