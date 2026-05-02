# Savyar Suffix Lookup

This file documents the suffix objects currently loaded by the project. The lookup is based on `util.decomposer.ALL_SUFFIXES` plus the special `pekistirme_suffix`.

## Field Meanings

- `name`: Stable suffix identifier used in decomposition logs and ML encoding.
- `base`: The underlying suffix string before harmony, hardening, buffer, or custom form logic.
- `comes_to`: The input POS accepted by the suffix: `NOUN`, `VERB`, or `BOTH`.
- `makes`: The POS after applying the suffix.
- `group`: Waterfall/order group used by `is_valid_transition`.
- `class`: Python class implementing the suffix behavior.
- `major`: Applies major vowel harmony.
- `minor`: Applies minor vowel harmony.
- `buffer_y`: Inserts `y` for vowel collisions where configured.
- `unique`: Prevents repeated use in a chain.

## Transition Groups

| Group | Purpose |
|---|---|
| `V2V_DERIVATIONAL` | Verb-to-verb derivation such as passive, causative, reflexive, reciprocal. |
| `VERB_NEGATING` | Verb negation and inability negation. |
| `VERB_COMPOUND` | Compound/auxiliary verb suffixes such as ability and continuative forms. |
| `N2V_DERIVATIONAL` | Project currently uses this group for several derivational suffix families, including many verb-to-noun items. |
| `N2N_DERIVATIONAL` | Noun-to-noun derivation group defined in `SuffixGroup`; current loaded derivationals mostly use `N2V_DERIVATIONAL`. |
| `V2N_DERIVATIONAL` | Verb-to-noun derivation group defined in `SuffixGroup`; current loaded verb-to-noun items mostly use `N2V_DERIVATIONAL`. |
| `DERIVATIONAL_LOCKING_VERB` | Verbal adverb/gerund forms that mostly terminate further derivation. |
| `PLURAL` | Plural suffix. |
| `POSSESSIVE` | Possessive agreement suffixes. |
| `CASE` | Noun case suffixes. |
| `MARKING_KI` | The post-case `ki` marker. |
| `WITH_LE` | Comitative/instrumental `ile/-le`. |
| `DERIVATIONAL_LOCKING_NOUN` | Noun-to-adverb style endings that mostly terminate further derivation. |
| `PREDICATIVE` | Copular/predicative and tense-like endings. |
| `CONJUGATION` | Person agreement suffixes. |

## Waterfall Model And Transition Rules

Savyar uses a suffix waterfall model in `util.decomposer.is_valid_transition`. Each suffix has a `SuffixGroup` integer. The normal rule is:

```text
next_group must be >= last_group
```

That means suffix chains generally flow from lower-numbered derivational layers toward higher-numbered inflectional layers:

```text
derivation -> plural -> possessive -> case -> marking/with/adverbial -> predicative -> conjugation
```

Once the chain has moved to a later layer, it usually cannot go back to an earlier layer. For example, after a case suffix, an ordinary derivational suffix should not follow. This keeps the search space small and blocks many impossible agglutination paths.

The current group order is:

| Value | Group |
|---:|---|
| 25 | `V2V_DERIVATIONAL` |
| 35 | `VERB_NEGATING` |
| 40 | `VERB_COMPOUND` |
| 50 | `N2V_DERIVATIONAL`, `N2N_DERIVATIONAL`, `V2N_DERIVATIONAL`, `V2N_DERIVATIONAL_NOUNIFIER` |
| 55 | `DERIVATIONAL_LOCKING_VERB` |
| 60 | `PLURAL` |
| 150 | `POSSESSIVE` |
| 200 | `CASE` |
| 225 | `MARKING_KI` |
| 230 | `WITH_LE` |
| 240 | `DERIVATIONAL_LOCKING_NOUN` |
| 250 | `PREDICATIVE` |
| 300 | `CONJUGATION` |

### Rule 1: `marking_ki` Post-Case Loop

```python
if last_g == SuffixGroup.MARKING_KI and next_g <= SuffixGroup.MARKING_KI:
    return True
```

After `marking_ki`, the chain may loop back into earlier noun-side material up to `MARKING_KI`. This models forms where `ki` creates a nominal/adjectival base that can receive further suffixes.

### Rule 2: Noun Derivational Self/Back Loop

```python
if last_g == SuffixGroup.N2N_DERIVATIONAL and next_g <= SuffixGroup.N2N_DERIVATIONAL:
    return True
```

After noun-to-noun derivation, earlier or same-level derivational material is allowed. The code comment describes this as an exception to the strict waterfall for derivational suffixes.

### Rule 3: Verb Compound Loop

```python
if last_g == SuffixGroup.VERB_COMPOUND and next_g <= SuffixGroup.VERB_COMPOUND:
    return True
```

After a compound verb suffix such as ability/continuative forms, earlier verb-side material can still appear. This supports chains like ability plus further verbal morphology, e.g. forms related to `gidebilmek`, `gidebilmeyen`, or `gitmeyebilmek`.

### Rule 4: Negation Before Nounifier Block

```python
if last_g == SuffixGroup.VERB_NEGATING and type(next_suffix) == Nounifier:
    return False
```

After verb negation, direct attachment of a `Nounifier` class suffix is blocked. This prevents some overgenerated negative-plus-nounifier analyses. Other verb-to-noun suffix classes can still be considered if they pass the remaining rules.

### Rule 5: Same-Group Repetition

```python
if next_g == last_g:
    if last_g in [
        SuffixGroup.N2N_DERIVATIONAL,
        SuffixGroup.V2V_DERIVATIONAL,
        SuffixGroup.PREDICATIVE,
    ]:
        return True
    return False
```

Most groups cannot repeat immediately. The allowed repeat groups are:

- `N2N_DERIVATIONAL`
- `V2V_DERIVATIONAL`
- `PREDICATIVE`

This allows productive derivational stacking and some predicative stacking, while blocking repeated plural/case/possessive-style suffixes unless another special rule allows them.

### Rule 6: Verb-Adverb Locking

```python
if last_g is SuffixGroup.DERIVATIONAL_LOCKING_VERB and next_g != SuffixGroup.CONJUGATION:
    return False
```

Gerund/converb-like verbal adverb suffixes mostly terminate the chain. The one allowed continuation is `CONJUGATION`, for limited forms noted in the code comments such as `ol-a-lım`, `ol-a-yım`, and `ol-a-sın`.

### Rule 7: Main Waterfall Check

```python
if next_g < last_g:
    return False
```

If no exception applied, the chain cannot move backward to a lower group. This is the core waterfall rule.

### Rule 8: Default Allow

```python
return True
```

If the transition passed all blocks and does not violate the waterfall, it is considered valid.

### Practical Consequences

- Derivational suffixes appear earlier than inflectional suffixes.
- Plural generally comes before possessive and case.
- Case generally comes before `ki`, `-le`, noun adverbials, predicatives, and conjugation.
- Gerund/converb endings strongly limit what can follow.
- Special loops exist because Turkish allows some productive derivational and compound-verb stacking that a strict one-way waterfall would reject.
- The model intentionally trades linguistic completeness for a smaller, more plausible decomposition search space.

## Lookup

| # | name | explanation | base | comes_to | makes | group | class | major | minor | buffer_y | unique | source |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `pekistirme` | Intensification/reduplication marker, e.g. `masmavi` style analyses. | `pekistirme` | `NOUN` | `NOUN` | none | `Suffix` | no | no | no | yes | `util.suffix` |
| 2 | `noun_compound` | Genitive/compound-linking suffix, roughly `-(n)in`. | `in` | `NOUN` | `NOUN` | `CASE` | `CaseSuffix` | yes | yes | no | no | `util.suffixes.n2n.case_suffixes` |
| 3 | `accusative_i` | Accusative case, direct-object marking. | `i` | `NOUN` | `NOUN` | `CASE` | `CaseSuffix` | yes | yes | yes | no | `util.suffixes.n2n.case_suffixes` |
| 4 | `ablative_den` | Ablative case, source/from meaning. | `den` | `NOUN` | `NOUN` | `CASE` | `CaseSuffix` | yes | no | no | no | `util.suffixes.n2n.case_suffixes` |
| 5 | `dative_e` | Dative case, goal/to meaning. | `e` | `NOUN` | `NOUN` | `CASE` | `CaseSuffix` | yes | no | yes | no | `util.suffixes.n2n.case_suffixes` |
| 6 | `locative_de` | Locative case, place/at/in meaning. | `de` | `NOUN` | `NOUN` | `CASE` | `CaseSuffix` | yes | no | no | no | `util.suffixes.n2n.case_suffixes` |
| 7 | `posessive_1sg` | First-person singular possessive, my. | `im` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | yes | no | no | `util.suffixes.n2n.posessive_suffix` |
| 8 | `posessive_2sg` | Second-person singular possessive, your. | `in` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | yes | no | no | `util.suffixes.n2n.posessive_suffix` |
| 9 | `posessive_3sg` | Third-person singular possessive, his/her/its. | `i` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | yes | no | no | `util.suffixes.n2n.posessive_suffix` |
| 10 | `posessive_1pl` | First-person plural possessive, our. | `imiz` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | yes | no | no | `util.suffixes.n2n.posessive_suffix` |
| 11 | `posessive_2pl` | Second-person plural/formal possessive, your. | `iniz` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | yes | no | no | `util.suffixes.n2n.posessive_suffix` |
| 12 | `posessive_3pl` | Third-person plural possessive, their. | `leri` | `NOUN` | `NOUN` | `POSSESSIVE` | `PosessiveSuffix` | yes | no | no | no | `util.suffixes.n2n.posessive_suffix` |
| 13 | `plural_ler` | Plural marker. | `ler` | `NOUN` | `NOUN` | `PLURAL` | `Plural` | yes | no | no | yes | `util.suffixes.n2n.plural_suffix` |
| 14 | `actor_ci` | Agent/profession/person-forming derivational suffix, `-ci/-cı/-cu/-cü`. | `ci` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 15 | `privative_siz` | Privative/lacking suffix, without X. | `siz` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 16 | `composessive_li` | Compositive/with-having suffix, with X. | `li` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 17 | `suitative_lik` | Nominal/adjectival `-lik` type derivation. | `lik` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 18 | `counting_er` | Distributive/counting `-er/-ar` style suffix. | `er` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | no | no | no | `util.suffix` |
| 19 | `cooperative_deş` | Associative/cooperative derivation, `-deş/-daş`. | `deş` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | no | no | no | `util.suffix` |
| 20 | `relative_ce` | Relative/manner suffix, `-ce/-ca`. | `ce` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | no | no | no | `util.suffix` |
| 21 | `relative_sel` | Relational/adjectival `-sel/-sal`. | `sel` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | no | no | no | `util.suffix` |
| 22 | `dimunitive_cik` | Diminutive suffix, small/dear X. | `cik` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 23 | `philicative_cil` | Inclination/affinity suffix, `-cil/-cıl`. | `cil` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 24 | `abstractifier_iyat` | Abstract/field-forming borrowed suffix, `-iyat`. | `iyat` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | no | no | no | no | `util.suffix` |
| 25 | `ideologicative_izm` | Ideology/doctrine suffix, `-izm`. | `izm` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | no | no | no | no | `util.suffix` |
| 26 | `scientist_olog` | Specialist/science-related borrowed suffix, `-olog`. | `olog` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | no | no | no | no | `util.suffix` |
| 27 | `familative_gil` | Family/group suffix, `-gil`. | `gil` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | no | no | no | no | `util.suffix` |
| 28 | `approximative_si` | Approximation/resemblance suffix. | `si` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 29 | `approximative_imtrak` | Approximation suffix, `-imtrak`. | `imtrak` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | no | no | no | no | `util.suffix` |
| 30 | `ordinal_inci` | Ordinal suffix, first/second/etc. | `inci` | `NOUN` | `NOUN` | `N2V_DERIVATIONAL` | `Suffix` | yes | yes | no | no | `util.suffix` |
| 31 | `conjugation_1sg` | First-person singular agreement. | `im` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 32 | `conjugation_2sg` | Second-person singular agreement. | `sin` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 33 | `conjugation_3sg` | Third-person singular agreement, often zero-marked. | `` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 34 | `conjugation_1pl` | First-person plural agreement. | `iz` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 35 | `conjugation_2pl` | Second-person plural/formal agreement. | `siniz` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 36 | `conjugation_3pl` | Third-person plural agreement. | `ler` | `BOTH` | `NOUN` | `CONJUGATION` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 37 | `nounaorist_dir` | Nominal copular/aorist-like `-dir`. | `dir` | `NOUN` | `NOUN` | `PREDICATIVE` | `Copula` | yes | yes | no | no | `util.suffixes.n2n.copula` |
| 38 | `pasttense_noundi` | Nominal past/copular past, `idi/-di`. | `di` | `NOUN` | `NOUN` | `PREDICATIVE` | `Copula` | yes | yes | yes | no | `util.suffixes.n2n.copula` |
| 39 | `if_se` | Conditional/copular `ise/-se`. | `se` | `NOUN` | `NOUN` | `PREDICATIVE` | `Copula` | yes | no | yes | no | `util.suffixes.n2n.copula` |
| 40 | `copula_mis` | Evidential/copular `imiş/-miş`. | `miş` | `NOUN` | `NOUN` | `PREDICATIVE` | `Copula` | yes | yes | yes | no | `util.suffixes.n2n.copula` |
| 41 | `marking_ki` | Post-case/relative `ki` marker. | `ki` | `NOUN` | `NOUN` | `MARKING_KI` | `MarkingKi` | no | no | no | yes | `util.suffixes.n2n.marking_suffix` |
| 42 | `temporative_leyin` | Temporal adverbial noun suffix, `-leyin`. | `leyin` | `NOUN` | `NOUN` | `DERIVATIONAL_LOCKING_NOUN` | `Suffix` | no | no | no | no | `util.suffix` |
| 43 | `adverbial_cesine` | Manner adverbial suffix, `-cesine/-casına`. | `cesine` | `NOUN` | `NOUN` | `DERIVATIONAL_LOCKING_NOUN` | `Suffix` | yes | no | no | no | `util.suffix` |
| 44 | `when_ken` | Adverbial `-ken`, while/as. | `ken` | `NOUN` | `NOUN` | `DERIVATIONAL_LOCKING_NOUN` | `Suffix` | no | no | no | no | `util.suffix` |
| 45 | `confactuous_le` | Comitative/instrumental `-le/-la`, with/by. | `le` | `NOUN` | `NOUN` | `WITH_LE` | `Suffix` | yes | yes | no | yes | `util.suffix` |
| 46 | `absentative_se` | Noun-to-verb derivation with `-se/-sa`, wish/desire/be affected by X. | `se` | `NOUN` | `VERB` | `N2V_DERIVATIONAL` | `VerbifierSuffix` | yes | no | no | no | `util.suffixes.n2v.verbifiers` |
| 47 | `aplicative_le` | Noun-to-verb `-le/-la`, to apply/do with X. | `le` | `NOUN` | `VERB` | `N2V_DERIVATIONAL` | `VerbifierSuffix` | yes | no | no | no | `util.suffixes.n2v.verbifiers` |
| 48 | `factative_en` | Verbal participle/adjectival `-en/-an`, one who does. | `en` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | no | yes | no | `util.suffixes.v2n.participles` |
| 49 | `pastfactative_miş` | Past/evidential participle `-miş/-mış`. | `miş` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | yes | no | no | `util.suffixes.v2n.participles` |
| 50 | `adjectifier_dik` | Adjectival/nominal participle `-dik/-dık`. | `dik` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | yes | no | no | `util.suffixes.v2n.participles` |
| 51 | `nounifier_ecek` | Future participle/nominalizer `-ecek/-acak`. | `ecek` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | no | yes | no | `util.suffixes.v2n.participles` |
| 52 | `factative_ir` | Aorist/habitual participial ending. | `ir` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | yes | no | no | `util.suffixes.v2n.participles` |
| 53 | `willing_esi` | Desiderative/necessitative nominal form, `-esi/-ası`. | `esi` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Participle` | yes | no | yes | no | `util.suffixes.v2n.participles` |
| 54 | `infinitive_me` | Short verbal noun/infinitive `-me/-ma`. | `me` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Infinitive` | yes | yes | no | no | `util.suffixes.v2n.infinitives` |
| 55 | `infinitive_mek` | Full infinitive `-mek/-mak`. | `mek` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Infinitive` | yes | yes | no | no | `util.suffixes.v2n.infinitives` |
| 56 | `nounifier_iş` | Verbal noun/action nominalizer `-iş/-ış`. | `iş` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Infinitive` | yes | yes | yes | no | `util.suffixes.v2n.infinitives` |
| 57 | `adverbial_erek` | Converb/adverbial `-erek/-arak`, by doing. | `erek` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | no | yes | no | `util.suffixes.v2n.gerunds` |
| 58 | `adverbial_ince` | Converb `-ince/-ınca`, when/upon doing. | `ince` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | yes | yes | no | `util.suffixes.v2n.gerunds` |
| 59 | `adverbial_ip` | Converb `-ip/-ıp`, and doing. | `ip` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | yes | yes | no | `util.suffixes.v2n.gerunds` |
| 60 | `adverbial_e` | Converb `-e/-a`, often in serial/reduplicated verb patterns. | `e` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | no | yes | no | `util.suffixes.v2n.gerunds` |
| 61 | `adverbial_dikçe` | Converb `-dikçe/-dıkça`, as long as/whenever. | `dikçe` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | yes | no | no | `util.suffixes.v2n.gerunds` |
| 62 | `since_eli` | Converb `-eli/-alı`, since doing. | `eli` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | no | yes | no | `util.suffixes.v2n.gerunds` |
| 63 | `undoing_meksizin` | Negative/without-doing converb `-meksizin/-maksızın`. | `meksizin` | `VERB` | `NOUN` | `DERIVATIONAL_LOCKING_VERB` | `Gerund` | yes | no | no | no | `util.suffixes.v2n.gerunds` |
| 64 | `toolative_ek` | Instrument/result nounifier from verbs. | `ek` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | yes | no | `util.suffixes.v2n.nounifiers` |
| 65 | `constofactative_gen` | Verbal adjective/nounifier `-gen/-gan` style. | `gen` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | no | no | `util.suffixes.v2n.nounifiers` |
| 66 | `constofactative_gin` | Verbal adjective/nounifier `-gin/-gın` style. | `gin` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 67 | `perfectative_ik` | Result/state nounifier `-ik/-ık`. | `ik` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | yes | no | `util.suffixes.v2n.nounifiers` |
| 68 | `nounifier_i` | Verbal nounifier `-i/-ı`. | `i` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 69 | `nounifier_gi` | Verbal nounifier `-gi/-gı`. | `gi` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 70 | `nounifier_ge` | Verbal nounifier `-ge/-ga`. | `ge` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | no | no | `util.suffixes.v2n.nounifiers` |
| 71 | `nounifier_im` | Verbal nounifier `-im/-ım`. | `im` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 72 | `nounifier_in` | Verbal nounifier `-in/-ın`. | `in` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 73 | `nounifier_it` | Verbal nounifier `-it/-ıt`. | `it` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | yes | no | `util.suffixes.v2n.nounifiers` |
| 74 | `nounifier_inç` | Verbal nounifier `-inç/-ınç`. | `inç` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | yes | no | `util.suffixes.v2n.nounifiers` |
| 75 | `nounifier_inti` | Verbal nounifier `-inti/-ıntı`. | `inti` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | yes | no | `util.suffixes.v2n.nounifiers` |
| 76 | `toolifier_geç` | Tool/agent nounifier `-geç/-gaç`. | `geç` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | no | no | `util.suffixes.v2n.nounifiers` |
| 77 | `subjectifier_giç` | Subject/tool nounifier `-giç/-gıç`. | `giç` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | yes | no | no | `util.suffixes.v2n.nounifiers` |
| 78 | `nounifier_anak` | Verbal nounifier `-anak/-enek`. | `anak` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | yes | no | `util.suffixes.v2n.nounifiers` |
| 79 | `nounifier_amak` | Verbal nounifier `-amak/-emek`. | `amak` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | yes | no | `util.suffixes.v2n.nounifiers` |
| 80 | `subjectifier_men` | Person/subject nounifier `-men/-man`. | `men` | `VERB` | `NOUN` | `N2V_DERIVATIONAL` | `Nounifier` | yes | no | no | no | `util.suffixes.v2n.nounifiers` |
| 81 | `wish_suffix` | Predicative/wish suffix `-se/-sa`. | `se` | `NOUN` | `NOUN` | `PREDICATIVE` | `Predicative` | yes | yes | no | yes | `util.suffixes.v2n.predicatives` |
| 82 | `pasttense_di` | Past tense/predicative `-di/-dı`. | `di` | `NOUN` | `NOUN` | `PREDICATIVE` | `Predicative` | yes | yes | no | yes | `util.suffixes.v2n.predicatives` |
| 83 | `reflexive_is` | Verb derivation `-iş/-ış`, reciprocal/reflexive-like. | `iş` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 84 | `active_it` | Verb derivation/causative-like `-it/-ıt`. | `it` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 85 | `active_dir` | Causative verb derivation `-dir/-dır`. | `dir` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 86 | `active_ir` | Causative verb derivation `-ir/-ır`. | `ir` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 87 | `active_er` | Causative verb derivation `-er/-ar`. | `er` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | no | no | no | `util.suffixes.v2v.verb_derivationals` |
| 88 | `passive_il` | Passive verb derivation `-il/-ıl`. | `il` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 89 | `reflexive_in` | Reflexive/passive-like verb derivation `-in/-ın`. | `in` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | yes | no | no | `util.suffixes.v2v.verb_derivationals` |
| 90 | `randomative_ele` | Verb derivation `-ele/-ala`, repeated/random action sense. | `ele` | `VERB` | `VERB` | `V2V_DERIVATIONAL` | `VerbDerivationalSuffix` | yes | no | no | no | `util.suffixes.v2v.verb_derivationals` |
| 91 | `negative_me` | Standard verb negation `-me/-ma`. | `me` | `VERB` | `VERB` | `VERB_NEGATING` | `VerbNegativeSuffix` | yes | no | no | yes | `util.suffixes.v2v.verb_negative` |
| 92 | `negative_able` | Inability/negative ability `-eme/-ama`. | `eme` | `VERB` | `VERB` | `VERB_NEGATING` | `VerbNegativeSuffix` | yes | no | yes | yes | `util.suffixes.v2v.verb_negative` |
| 93 | `continuous_iyor` | Progressive/continuous aspect `-iyor/-ıyor/-uyor/-üyor`. | `iyor` | `VERB` | `VERB` | `PREDICATIVE` | `CompoundVerb` | yes | yes | yes | no | `util.suffixes.v2v.verb_compounds` |
| 94 | `possibilitative_ebil` | Ability/possibility compound `-ebil/-abil`. | `ebil` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |
| 95 | `almostative_eyazmak` | Almost/nearly compound `-eyaz/-ayaz`. | `eyaz` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |
| 96 | `continuative_edurmak` | Continuative compound `-edur/-adur`. | `edur` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |
| 97 | `remainmative_ekalmak` | Continuative/remain compound `-ekal/-akal`. | `ekal` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |
| 98 | `persistive_egelmek` | Persistive/continue compound `-egel/-agal`. | `egel` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |
| 99 | `suddenative_ivermek` | Sudden/quick action compound `-iver/-ıver`. | `iver` | `VERB` | `VERB` | `VERB_COMPOUND` | `CompoundVerb` | yes | no | yes | no | `util.suffixes.v2v.verb_compounds` |

## Lookup By Direction

### Noun To Noun

`noun_compound`, `accusative_i`, `ablative_den`, `dative_e`, `locative_de`, `posessive_1sg`, `posessive_2sg`, `posessive_3sg`, `posessive_1pl`, `posessive_2pl`, `posessive_3pl`, `plural_ler`, `actor_ci`, `privative_siz`, `composessive_li`, `suitative_lik`, `counting_er`, `cooperative_deş`, `relative_ce`, `relative_sel`, `dimunitive_cik`, `philicative_cil`, `abstractifier_iyat`, `ideologicative_izm`, `scientist_olog`, `familative_gil`, `approximative_si`, `approximative_imtrak`, `ordinal_inci`, `nounaorist_dir`, `pasttense_noundi`, `if_se`, `copula_mis`, `marking_ki`, `temporative_leyin`, `adverbial_cesine`, `when_ken`, `confactuous_le`, `wish_suffix`, `pasttense_di`, `pekistirme`

### Noun To Verb

`absentative_se`, `aplicative_le`

### Verb To Noun

`factative_en`, `pastfactative_miş`, `adjectifier_dik`, `nounifier_ecek`, `factative_ir`, `willing_esi`, `infinitive_me`, `infinitive_mek`, `nounifier_iş`, `adverbial_erek`, `adverbial_ince`, `adverbial_ip`, `adverbial_e`, `adverbial_dikçe`, `since_eli`, `undoing_meksizin`, `toolative_ek`, `constofactative_gen`, `constofactative_gin`, `perfectative_ik`, `nounifier_i`, `nounifier_gi`, `nounifier_ge`, `nounifier_im`, `nounifier_in`, `nounifier_it`, `nounifier_inç`, `nounifier_inti`, `toolifier_geç`, `subjectifier_giç`, `nounifier_anak`, `nounifier_amak`, `subjectifier_men`

### Verb To Verb

`reflexive_is`, `active_it`, `active_dir`, `active_ir`, `active_er`, `passive_il`, `reflexive_in`, `randomative_ele`, `negative_me`, `negative_able`, `continuous_iyor`, `possibilitative_ebil`, `almostative_eyazmak`, `continuative_edurmak`, `remainmative_ekalmak`, `persistive_egelmek`, `suddenative_ivermek`

### Both To Noun

`conjugation_1sg`, `conjugation_2sg`, `conjugation_3sg`, `conjugation_1pl`, `conjugation_2pl`, `conjugation_3pl`

## Notes

- The `source` column reports the Python module that owns the suffix object's class, not always the file that declares the variable. Many simple derivational suffixes are instances of the base `Suffix` class, so their source appears as `util.suffix`.
- Some loaded verb-to-noun and noun-to-noun derivational suffixes currently use the `N2V_DERIVATIONAL` group. This document records the current code state, even where group names are historically inconsistent.
- `pasttense_di` and `wish_suffix` live under `v2n/predicatives.py`, but their current objects have `comes_to=NOUN` and `makes=NOUN`.
- `conjugation_*` suffixes accept `BOTH` because they can attach after nominalized/predicative chains as well as verb-derived chains.
