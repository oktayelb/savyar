"""
Closed-class (kapalı sınıf) words of Turkish.

These are function words that belong to fixed, non-productive categories:
pronouns, conjunctions, postpositions, adverbs, determiners, interjections, particles.

Unlike open-class words (nouns/verbs looked up from words.txt), closed-class words
are enumerated here exhaustively. They can still accept inflectional suffixes
(e.g. "benim", "senden") but do not undergo productive derivation.
"""

from typing import List, Dict, Optional
from util.words.words import Word


class ClosedClassWord(Word):
    """
    A word from a closed (fixed-membership) grammatical class.

    Attributes:
        category: The grammatical category (e.g. "pronoun", "conjunction").
        can_take_suffixes: Whether this word can accept inflectional suffixes.
            Pronouns and postpositions can; conjunctions and interjections typically cannot.
    """

    def __init__(self, word: str, pos: str, category: str, can_take_suffixes: bool = True):
        super().__init__(word, pos)
        self.category = category
        self.can_take_suffixes = can_take_suffixes

    def __repr__(self):
        return f"{self.__class__.__name__}({self.word!r}, cat={self.category!r})"


# ============================================================================
# PRONOUNS (Zamirler)
# ============================================================================

class Pronoun(ClosedClassWord):
    """
    Turkish pronoun. Accepts case/possessive suffixes like nouns.

    Personal pronouns have suppletive (irregular) case forms that cannot be
    derived from the nominative by regular suffix rules — e.g. "ben" → "bana"
    (dative), not *"bene". These irregular forms are stored in the `forms` dict,
    keyed by case name.

    Case keys used in `forms`:
        nominative  — yalın hal   (ben, sen, o …)
        genitive    — iyelik hali (benim, senin, onun …)
        accusative  — belirtme hali (beni, seni, onu …)
        dative      — yönelme hali (bana, sana, ona …)
        locative    — bulunma hali (bende, sende, onda …)
        ablative    — ayrılma hali (benden, senden, ondan …)

    For pronouns without irregularities the `forms` dict may be empty;
    regular suffix rules then apply.
    """

    def __init__(self, word: str, pronoun_type: str, forms: Dict[str, str] = None):
        super().__init__(word, pos="noun", category="pronoun")
        self.pronoun_type = pronoun_type  # personal, demonstrative, interrogative, reflexive, indefinite
        # Map case → surface form.  Nominative always equals self.word.
        self.forms: Dict[str, str] = {"nominative": word}
        if forms:
            self.forms.update(forms)

    def get_form(self, case: str) -> Optional[str]:
        """Returns the stored form for the given case, or None if not known."""
        return self.forms.get(case)

    def all_surface_forms(self) -> List[str]:
        """Returns all known surface forms for lookup purposes."""
        return list(self.forms.values())


# ============================================================================
# Personal pronoun irregular case tables
# ============================================================================
# Turkish personal pronouns are the only pronoun group with systematic
# suppletive forms.  Every other pronoun type (demonstrative, interrogative
# etc.) follows regular vowel-harmony suffix rules.
#
# Irregular patterns:
#   ben/sen  — dative stem changes to ban-/san- (not ben-/sen-)
#            — genitive/accusative/locative/ablative are regular from ben/sen
#   o        — oblique stem is on-  (ona, onu, onun, onda, ondan)
#   biz/siz  — fully regular (bize, bizi, bizim, bizde, bizden)
#   onlar    — fully regular plural (onlara, onları, onların, onlarda, onlardan)

_PERSONAL_FORMS = {
    "ben": {
        "nominative": "ben",
        "genitive":   "benim",
        "accusative": "beni",
        "dative":     "bana",      # irregular: ban- stem
        "locative":   "bende",
        "ablative":   "benden",
    },
    "sen": {
        "nominative": "sen",
        "genitive":   "senin",
        "accusative": "seni",
        "dative":     "sana",      # irregular: san- stem
        "locative":   "sende",
        "ablative":   "senden",
    },
    "o": {
        "nominative": "o",
        "genitive":   "onun",      # oblique stem on-
        "accusative": "onu",
        "dative":     "ona",
        "locative":   "onda",
        "ablative":   "ondan",
    },
    "biz": {
        "nominative": "biz",
        "genitive":   "bizim",
        "accusative": "bizi",
        "dative":     "bize",
        "locative":   "bizde",
        "ablative":   "bizden",
    },
    "siz": {
        "nominative": "siz",
        "genitive":   "sizin",
        "accusative": "sizi",
        "dative":     "size",
        "locative":   "sizde",
        "ablative":   "sizden",
    },
    "onlar": {
        "nominative": "onlar",
        "genitive":   "onların",
        "accusative": "onları",
        "dative":     "onlara",
        "locative":   "onlarda",
        "ablative":   "onlardan",
    },
}

# --- Personal pronouns (Kişi zamirleri) ---
PERSONAL_PRONOUNS = [
    Pronoun("ben",   "personal", _PERSONAL_FORMS["ben"]),
    Pronoun("sen",   "personal", _PERSONAL_FORMS["sen"]),
    Pronoun("o",     "personal", _PERSONAL_FORMS["o"]),
    Pronoun("biz",   "personal", _PERSONAL_FORMS["biz"]),
    Pronoun("siz",   "personal", _PERSONAL_FORMS["siz"]),
    Pronoun("onlar", "personal", _PERSONAL_FORMS["onlar"]),
]

# --- Demonstrative pronouns (İşaret zamirleri) ---
DEMONSTRATIVE_PRONOUNS = [ 
    # Pronoun("bu", "demonstrative"),       # this
    # Pronoun("şu", "demonstrative"),       # that (nearby)
    # Pronoun("o", "demonstrative"),        # that (far)
    # Pronoun("bunlar", "demonstrative"),   # these
    # Pronoun("şunlar", "demonstrative"),   # those (nearby)
    # Pronoun("onlar", "demonstrative"),    # those (far)
    # Pronoun("burası", "demonstrative"),   # this place
    # Pronoun("şurası", "demonstrative"),   # that place (nearby)
    # Pronoun("orası", "demonstrative"),    # that place (far)
]


# --- Interrogative pronouns (Soru zamirleri) ---
# INTERROGATIVE_PRONOUNS = [
#     Pronoun("kim", "interrogative"),      # who
#     Pronoun("ne", "interrogative"),       # what
#     Pronoun("nere", "interrogative"),     # where (stem, inflects as nereden, nereye...)
#     Pronoun("hangisi", "interrogative"),  # which one
#     Pronoun("kaçı", "interrogative"),     # how many (of them)
# ]

# --- Reflexive pronoun (Dönüşlü zamir) ---
REFLEXIVE_PRONOUNS = [
     Pronoun("kendi", "reflexive"),        # self
 ]

# --- Indefinite pronouns (Belgisiz zamirler) ---
# INDEFINITE_PRONOUNS = [
#     Pronoun("herkes", "indefinite"),      # everyone
#     Pronoun("hepsi", "indefinite"),       # all of them
#     Pronoun("birisi", "indefinite"),      # someone
#     Pronoun("hiçbiri", "indefinite"),     # none of them
#     Pronoun("bazısı", "indefinite"),      # some of them
#     Pronoun("birçoğu", "indefinite"),     # most of them
#     Pronoun("kimse", "indefinite"),       # anyone / no one
#     Pronoun("hep", "indefinite"),         # always/all
#     Pronoun("biri", "indefinite"),        # one (of them)
#     Pronoun("öteki", "indefinite"),       # the other
#     Pronoun("öbürü", "indefinite"),       # the other one
#     Pronoun("tümü", "indefinite"),        # all of it
#     Pronoun("cümlesi", "indefinite"),     # all (literary)
#     Pronoun("her biri", "indefinite"),    # each one
# ]

ALL_PRONOUNS = (PERSONAL_PRONOUNS + DEMONSTRATIVE_PRONOUNS)
# + INTERROGATIVE_PRONOUNS
# + REFLEXIVE_PRONOUNS
# + INDEFINITE_PRONOUNS


# ============================================================================
# CONJUNCTIONS (Bağlaçlar)
# ============================================================================

class Conjunction(ClosedClassWord):
    """Turkish conjunction. Generally does not take suffixes."""

    def __init__(self, word: str, conjunction_type: str):
        super().__init__(word, pos="noun", category="conjunction", can_take_suffixes=False)
        self.conjunction_type = conjunction_type  # coordinating, subordinating, correlative


COORDINATING_CONJUNCTIONS = [
    Conjunction("ve", "coordinating"),         # and
    Conjunction("hem", "coordinating"),        # both / also
    Conjunction("ne", "coordinating"),         # neither...nor (part of ne...ne)
    Conjunction("da", "coordinating"),         # also / too
    Conjunction("de", "coordinating"),         # also / too
    # Conjunction("ile", "coordinating"),        # with / and
    # Conjunction("ama", "coordinating"),        # but
    # Conjunction("fakat", "coordinating"),      # but
    # Conjunction("ancak", "coordinating"),      # however
    # Conjunction("lakin", "coordinating"),      # but (literary)
    # Conjunction("veya", "coordinating"),       # or
    # Conjunction("ya da", "coordinating"),      # or
    # Conjunction("yoksa", "coordinating"),      # or else
    # Conjunction("oysa", "coordinating"),       # whereas
    # Conjunction("oysaki", "coordinating"),     # whereas (emphatic)
    # Conjunction("halbuki", "coordinating"),    # whereas
    # Conjunction("meğer", "coordinating"),      # apparently
    # Conjunction("meğerki", "coordinating"),    # apparently (emphatic)
]

SUBORDINATING_CONJUNCTIONS = [
    Conjunction("çünkü", "subordinating"),     # because
    Conjunction("zira", "subordinating"),      # because (formal)
    Conjunction("ki", "subordinating"),        # that / because
    Conjunction("iken", "subordinating"),      # while
    Conjunction("diye", "subordinating"),      # so that / saying
    # Conjunction("eğer", "subordinating"),      # if
    # Conjunction("şayet", "subordinating"),     # if (formal)
    # Conjunction("madem", "subordinating"),     # since / given that
    # Conjunction("mademki", "subordinating"),   # since (emphatic)
    # Conjunction("gerçi", "subordinating"),     # although
    # Conjunction("her ne kadar", "subordinating"),  # although
    # Conjunction("rağmen", "subordinating"),    # despite
    # Conjunction("karşın", "subordinating"),    # despite
    # Conjunction("nitekim", "subordinating"),   # indeed / as a matter of fact
    # Conjunction("sanki", "subordinating"),     # as if
    # Conjunction("güya", "subordinating"),      # supposedly
    # Conjunction("yeter ki", "subordinating"),  # as long as
]

ALL_CONJUNCTIONS = COORDINATING_CONJUNCTIONS + SUBORDINATING_CONJUNCTIONS


# ============================================================================
# POSTPOSITIONS (Sonçekimler / Edatlar)
# ============================================================================

class Postposition(ClosedClassWord):
    """Turkish postposition. Some can take limited suffixes."""

    def __init__(self, word: str, governs: str = "none"):
        super().__init__(word, pos="noun", category="postposition", can_take_suffixes=False)
        self.governs = governs  # which case the preceding noun takes: nominative, dative, ablative, genitive


ALL_POSTPOSITIONS = [
    # Governs nominative (yalın hal)
    Postposition("için", "nominative"),       # for
    Postposition("gibi", "nominative"),       # like
    Postposition("kadar", "nominative"),      # as much as / until
    Postposition("ile", "nominative"),        # with (also conjunction)

    # Governs dative (-e hali)
    Postposition("doğru", "dative"),          # towards
    Postposition("karşı", "dative"),          # against
    Postposition("göre", "dative"),           # according to
    Postposition("rağmen", "dative"),         # despite
    Postposition("dek", "dative"),            # until
    Postposition("değin", "dative"),          # until
    Postposition("dair", "dative"),           # about / regarding

    # Governs ablative (-den hali)
    Postposition("beri", "ablative"),         # since
    Postposition("önce", "ablative"),         # before
    Postposition("sonra", "ablative"),        # after
    Postposition("dolayı", "ablative"),       # because of
    Postposition("ötürü", "ablative"),        # because of
    Postposition("başka", "ablative"),        # other than
    Postposition("itibaren", "ablative"),     # starting from
    Postposition("yana", "ablative"),         # since (temporal)

    # Governs genitive (-in hali) — used with possessive compound
    Postposition("üzerine", "genitive"),      # upon / about
    Postposition("arasında", "genitive"),     # among / between
    Postposition("sayesinde", "genitive"),    # thanks to
    Postposition("yerine", "genitive"),       # instead of
    Postposition("yüzünden", "genitive"),     # because of (negative)
    Postposition("hakkında", "genitive"),     # about / regarding
]


# ============================================================================
# ADVERBS — closed-class subset (Zarflar)
# ============================================================================

class Adverb(ClosedClassWord):
    """Closed-class Turkish adverb (non-derived, fixed form)."""

    def __init__(self, word: str, adverb_type: str):
        super().__init__(word, pos="noun", category="adverb", can_take_suffixes=False)
        self.adverb_type = adverb_type


ALL_ADVERBS = [
    # Temporal
    Adverb("şimdi", "temporal"),          # now
    Adverb("sonra", "temporal"),          # later / after
    Adverb("önce", "temporal"),           # before / earlier
    #Adverb("bugün", "temporal"),          # today
    #Adverb("dün", "temporal"),            # yesterday
    #Adverb("yarın", "temporal"),          # tomorrow
    Adverb("hâlâ", "temporal"),           # still
    Adverb("henüz", "temporal"),          # yet / just
    Adverb("artık", "temporal"),          # anymore / from now on
    Adverb("hep", "temporal"),            # always
    Adverb("hiç", "temporal"),            # never / ever
    #Adverb("bazen", "temporal"),          # sometimes
    #Adverb("sık sık", "temporal"),        # often
    #Adverb("nadiren", "temporal"),        # rarely
    #Adverb("derhal", "temporal"),         # immediately
    Adverb("hemen", "temporal"),          # immediately / right away
    Adverb("demin", "temporal"),          # just now
    #Adverb("geçen", "temporal"),          # the other (day or week or year etc)
    # Manner
    #Adverb("böyle", "manner"),            # like this
    #Adverb("şöyle", "manner"),            # like that
    #Adverb("öyle", "manner"),             # like that (far)
    #Adverb("nasıl", "manner"),            # how
    # Degree
    #Adverb("çok", "degree"),              # very / much
    #Adverb("az", "degree"),               # little / few
    #Adverb("pek", "degree"),              # quite / very
    #Adverb("en", "degree"),               # most (superlative)
    #Adverb("daha", "degree"),             # more (comparative)
    #Adverb("gayet", "degree"),            # quite
    #Adverb("oldukça", "degree"),          # rather / quite
    #Adverb("epey", "degree"),             # fairly / considerably

    # Place
    #Adverb("burada", "place"),            # here
    #Adverb("şurada", "place"),            # there (nearby)
    #Adverb("orada", "place"),             # there (far)
    #Adverb("nerede", "place"),            # where
    #Adverb("içeri", "place"),             # inside
    #Adverb("dışarı", "place"),            # outside
    #Adverb("ileri", "place"),             # forward
    #Adverb("geri", "place"),              # back
    #Adverb("yukarı", "place"),            # up
    #Adverb("aşağı", "place"),             # down
]


# ============================================================================
# DETERMINERS (Belirleyiciler)
# ============================================================================

class Determiner(ClosedClassWord):
    """Turkish determiner. Does not take suffixes."""

    def __init__(self, word: str, determiner_type: str):
        super().__init__(word, pos="noun", category="determiner", can_take_suffixes=False)
        self.determiner_type = determiner_type


ALL_DETERMINERS = [
    # Determiner("bir", "indefinite"),          # a / one
    # Determiner("bu", "demonstrative"),        # this
    # Determiner("şu", "demonstrative"),        # that (nearby)
    # Determiner("o", "demonstrative"),         # that (far)
    # Determiner("her", "universal"),           # every
    # Determiner("bazı", "indefinite"),         # some
    # Determiner("birçok", "indefinite"),       # many / several
    # Determiner("hiçbir", "negative"),         # no / none
    # Determiner("birkaç", "indefinite"),       # a few
    # Determiner("tüm", "universal"),           # all
    # Determiner("bütün", "universal"),         # all / whole
    # Determiner("hangi", "interrogative"),     # which
    # Determiner("kaç", "interrogative"),       # how many
    # Determiner("öbür", "demonstrative"),      # the other
    # Determiner("öteki", "demonstrative"),     # the other
    # Determiner("böyle", "demonstrative"),     # such (this kind)
    # Determiner("şöyle", "demonstrative"),     # such (that kind)
    # Determiner("öyle", "demonstrative"),      # such (that kind, far)
]



# ============================================================================
# INTERJECTIONS (Ünlemler)
# ============================================================================

class Interjection(ClosedClassWord):
    """Turkish interjection. Never takes suffixes."""

    def __init__(self, word: str):
        super().__init__(word, pos="noun", category="interjection", can_take_suffixes=False)


ALL_INTERJECTIONS = [
    # Interjection("evet"),         # yes
    # Interjection("hayır"),        # no
    # Interjection("yok"),          # no / there isn't
    # Interjection("var"),          # there is
    # Interjection("eyvah"),        # alas
    # Interjection("aman"),         # oh no
    # Interjection("hey"),          # hey
    # Interjection("of"),           # ugh
    # Interjection("ah"),           # ah
    # Interjection("oh"),           # oh
    # Interjection("vay"),          # wow
    # Interjection("bravo"),        # bravo
    # Interjection("yazık"),        # pity
    # Interjection("maşallah"),     # wonderful (protective)
    # Interjection("inşallah"),     # God willing
    # Interjection("lütfen"),       # please
    # Interjection("tamam"),        # okay
    # Interjection("haydi"),        # come on
    # Interjection("hadi"),         # come on
]


# ============================================================================
# PARTICLES (Edatlar / İlgeçler — small function words)
# ============================================================================

class Particle(ClosedClassWord):
    """Turkish discourse/focus particle."""

    def __init__(self, word: str, particle_type: str):
        super().__init__(word, pos="noun", category="particle", can_take_suffixes=False)
        self.particle_type = particle_type


ALL_PARTICLES = [
    Particle("mı", "question"),       # question particle (back vowel)
    Particle("mi", "question"),       # question particle (front vowel)
    Particle("mu", "question"),       # question particle (back round)
    Particle("mü", "question"),       # question particle (front round)
    # Particle("bile", "focus"),        # even
    # Particle("sadece", "focus"),      # only
    # Particle("yalnız", "focus"),      # only / just
    # Particle("işte", "discourse"),    # here it is / that's it
    # Particle("acaba", "discourse"),   # I wonder
    # Particle("belki", "discourse"),   # maybe
    # Particle("keşke", "discourse"),   # I wish
    # Particle("zaten", "discourse"),   # already / anyway
    # Particle("yine", "discourse"),    # again / still
    # Particle("gene", "discourse"),    # again (colloquial)
    # Particle("bile bile", "discourse"),  # knowingly
]


# ============================================================================
# CLOSED-CLASS MARKER  (used in suffix chain encoding for the ML model)
# ============================================================================

class ClosedClassMarker:
    """
    A lightweight sentinel placed in a suffix chain to signal "this entire word
    is a closed-class word" rather than a root + suffix sequence.

    Satisfies the interface that encode_suffix_chain and reconstruct_morphology
    expect from chain elements (name, makes, form), but encodes to a dedicated
    closed-class token in the ML vocabulary instead of a suffix token.

    Usage (inside a decomposition tuple):
        (surface_form, "cc_<category>", [ClosedClassMarker(cc_obj)], "cc_<category>")
    """

    def __init__(self, cc_word: "ClosedClassWord", surface_form: Optional[str] = None):
        self.cc_word = cc_word
        self.name    = f"cc_{cc_word.category}"
        self.surface_form = surface_form or cc_word.word
        self.makes   = None      # CC words don't have a suffix-style POS output
        self.is_unique = False

    def form(self, root: str, current_chain=None) -> List[str]:
        """CC markers produce no surface suffix — the word is already complete."""
        return []

    def __repr__(self):
        return f"ClosedClassMarker({self.surface_form!r}, cat={self.cc_word.category!r})"


# ============================================================================
# AGGREGATED COLLECTIONS
# ============================================================================

ALL_CLOSED_CLASS_WORDS: List[ClosedClassWord] = (
    ALL_PRONOUNS
    + ALL_CONJUNCTIONS
    + ALL_POSTPOSITIONS
    + ALL_ADVERBS
    + ALL_DETERMINERS
    + ALL_INTERJECTIONS
    + ALL_PARTICLES
)

# Lookup: surface form → list of ClosedClassWord objects
# (a word like "o" may appear as both pronoun and determiner)
# Inflected forms of pronouns (bana, seni, onun …) are also indexed
# so that a sentence tokenizer can recognise them as pronouns.
CLOSED_CLASS_LOOKUP: Dict[str, List[ClosedClassWord]] = {}
for _w in ALL_CLOSED_CLASS_WORDS:
    CLOSED_CLASS_LOOKUP.setdefault(_w.word, []).append(_w)
    # Index every stored inflected form of pronouns
    if isinstance(_w, Pronoun):
        for _form in _w.all_surface_forms():
            if _form != _w.word:
                CLOSED_CLASS_LOOKUP.setdefault(_form, []).append(_w)


def _build_closed_class_token_specs() -> List[tuple[str, str]]:
    specs: List[tuple[str, str]] = []
    seen = set()
    for cc_word in ALL_CLOSED_CLASS_WORDS:
        surfaces = [cc_word.word]
        if isinstance(cc_word, Pronoun):
            surfaces = cc_word.all_surface_forms()
        for surface in surfaces:
            key = (cc_word.category, surface)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
    return specs


CLOSED_CLASS_TOKEN_SPECS: List[tuple[str, str]] = _build_closed_class_token_specs()
ALL_PRONOUN_SURFACES = {
    surface
    for cc_word in ALL_PRONOUNS
    for surface in cc_word.all_surface_forms()
}
BEN_PRONOUN_SURFACES = set(_PERSONAL_FORMS["ben"].values())
NON_BEN_PRONOUN_SURFACES = ALL_PRONOUN_SURFACES - BEN_PRONOUN_SURFACES
