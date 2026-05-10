from typing import List
from util.suffix import Suffix


class Word:
    """Base class representing a Turkish word with its morphological state."""

    def __init__(self, word: str, pos: str):
        self.word = word
        self.pos = pos
        self.root = word
        self.root_pos = pos
        self.suffix_list: List[Suffix] = []


    def __repr__(self):
        return f"Word({self.word!r}, pos={self.pos!r})"
