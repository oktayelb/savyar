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

    def add_suffix(self, suffix: Suffix) -> None:
        current_chain = list(self.suffix_list)
        candidate_forms = suffix.form(self.word, current_chain=current_chain)

        self.suffix_list.append(suffix)

        if candidate_forms:
            selected_form = candidate_forms[0]
            self.word = self.word + selected_form
            self.pos = suffix.makes

    def __repr__(self):
        return f"Word({self.word!r}, pos={self.pos!r})"
