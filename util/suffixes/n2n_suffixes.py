from util.suffix import Suffix, Type, SuffixGroup


from util.suffixes.n2n.case_suffixes            import CASESUFFIX
from util.suffixes.n2n.possessive_suffix         import POSSESSIVE_SUFFIX
from util.suffixes.n2n.plural_suffix            import PLURALS
from util.suffixes.n2n.derivationals            import DERIVATIONALS
from util.suffixes.n2n.conjugation_suffixes     import CONJUGATIONS
from util.suffixes.n2n.copula                   import COPULA
from util.suffixes.n2n.marking_suffix           import MARKINGS
from util.suffixes.n2n.adverbials               import ADVERBIALS
from util.suffixes.n2n.with_le                  import WITH_LE


NOUN2NOUN = (
    CASESUFFIX + POSSESSIVE_SUFFIX + PLURALS + DERIVATIONALS
    + CONJUGATIONS + COPULA + MARKINGS + ADVERBIALS + WITH_LE
)
