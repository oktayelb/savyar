from util.word_methods import tr_lower


AMBIGUOUS_VNOUN = "__AMBIGUOUS_VNOUN__"
_VNOUN_CANDIDATES = ("nounifier_iş", "infinitive_me", "infinitive_mek")


def _iter_suffix_tails(current_stem, suffix_names, suffix_by_name, limit=256):
    """Generate possible surface tails for a suffix-name sequence.

    The treebank adapters only need this to disambiguate a single verbal-noun
    slot, so a small capped search is enough.
    """
    results = set()

    def visit(stem, names, tail):
        if len(results) >= limit:
            return
        if not names:
            results.add(tail)
            return

        suffix_obj = suffix_by_name.get(names[0])
        if suffix_obj is None:
            return

        try:
            forms = suffix_obj.form(stem)
        except Exception:
            forms = [suffix_obj.suffix]

        seen_forms = set()
        for form in forms:
            if form in seen_forms:
                continue
            seen_forms.add(form)
            visit(stem + form, names[1:], tail + form)

    visit(current_stem, suffix_names, "")
    return results


def _fallback_vnoun_from_surface(surface):
    """Last-resort guess from the visible ending when full-tail matching fails."""
    surface = tr_lower(surface)

    mak_markers = (
        "maktan", "mekten", "makta", "mekte", "makla", "mekle",
        "mağa", "meğe", "mağı", "meği", "mağın", "meğin",
        "mak", "mek",
    )
    is_markers = (
        "ışları", "işleri", "uşları", "üşleri",
        "ışlar", "işler", "uşlar", "üşler",
        "ışını", "işini", "uşunu", "üşünü",
        "ışına", "işine", "uşuna", "üşüne",
        "ışında", "işinde", "uşunda", "üşünde",
        "ışından", "işinden", "uşundan", "üşünden",
        "ışıyla", "işiyle", "uşuyla", "üşüyle",
        "ışı", "işi", "uşu", "üşü",
        "ışa", "işe", "uşa", "üşe",
        "ışta", "işte", "uşta", "üşte",
        "ıştan", "işten", "uştan", "üşten",
        "ışla", "işle", "uşla", "üşle",
        "ışın", "işin", "uşun", "üşün",
        "ış", "iş", "uş", "üş",
    )

    for ending in mak_markers:
        if surface.endswith(ending):
            return "infinitive_mek"

    for ending in is_markers:
        if surface.endswith(ending):
            return "nounifier_iş"

    return "infinitive_me"


def resolve_ambiguous_vnoun_suffixes(surface, lemma, suffix_names, suffix_by_name):
    """Replace ambiguous verbal-noun placeholders by surface-matching labels."""
    if AMBIGUOUS_VNOUN not in suffix_names:
        return suffix_names

    surface_lower = tr_lower(surface)
    lemma_lower = tr_lower(lemma)
    resolved = list(suffix_names)

    for idx, name in enumerate(resolved):
        if name != AMBIGUOUS_VNOUN:
            continue

        best_name = None
        best_score = (-1, -1)
        for candidate in _VNOUN_CANDIDATES:
            test_names = list(resolved)
            test_names[idx] = candidate
            tails = _iter_suffix_tails(lemma_lower, test_names, suffix_by_name)
            score = (-1, -1)
            for tail in tails:
                if surface_lower.endswith(tail):
                    score = max(score, (len(test_names), len(tail)))
            if score > best_score:
                best_name = candidate
                best_score = score

        if best_score[0] < 0:
            best_name = _fallback_vnoun_from_surface(surface_lower)

        resolved[idx] = best_name

    return resolved


def has_unexpected_nounifier_is(root, lemma, chain_names, expected_suffixes):
    """Return True when a candidate injects nounifier_iş not requested by the treebank.

    The treebank lemma is authoritative. If the expected suffix sequence does
    not explicitly contain ``nounifier_iş``, then a candidate such as
    ``gör + nounifier_iş + dative_e`` or ``uç + nounifier_iş + dative_e`` must
    not be allowed to replace the treebank's own analysis. This keeps lexical
    stems and omitted derivations from being reintroduced during matching.
    """
    if "nounifier_iş" not in chain_names:
        return False
    if "nounifier_iş" in expected_suffixes:
        return False
    return True
