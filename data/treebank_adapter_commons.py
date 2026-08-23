import json
import os
from collections import Counter

import itertools

from util.decomposer import ALL_SUFFIXES, decompose
from util.suffix import Type
from util.words.closed_class import CLOSED_CLASS_LOOKUP
from util.word_methods import tr_lower


SUFFIX_BY_NAME = {s.name: s for s in ALL_SUFFIXES}

# Sentences carved out as the held-out test set. Re-adapting a treebank
# rebuilds its shard from the original CoNLL file, which would quietly pull
# every held-out sentence back into training, so the writer checks this file
# and withholds them again.
HELD_OUT_TEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_adapted.jsonl")


def sentence_text(entry):
    """The comparable form of a sentence: what it says, not how it was stored."""
    raw = entry.get("original_sentence")
    if not raw:
        raw = " ".join(word.get("word", "") for word in entry.get("words", []))
    return " ".join(tr_lower(raw).split())


def held_out_sentence_texts(path=HELD_OUT_TEST_PATH):
    """Sentence texts reserved for the test set. Missing file means none are."""
    texts = set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    texts.add(sentence_text(json.loads(line)))
    except FileNotFoundError:
        pass
    return texts


def withhold_held_out(entries, path=HELD_OUT_TEST_PATH):
    """Drop the sentences that belong to the test set, whichever treebank they are in."""
    held_out = held_out_sentence_texts(path)
    if not held_out:
        return entries, 0
    kept = [entry for entry in entries if sentence_text(entry) not in held_out]
    return kept, len(entries) - len(kept)


QUOTE_CHARS = "\"'`‘’“”„«»‹›"

AMBIGUOUS_VNOUN = "__AMBIGUOUS_VNOUN__"
_VNOUN_CANDIDATES = ("nounifier_iş", "infinitive_me", "infinitive_mek")


def strip_quotes(s):
    if not s:
        return s
    for q in QUOTE_CHARS:
        s = s.replace(q, "")
    return s


def make_layer(upos, xpos, features, lemma=None, surface=None, features_multi=None):
    return {
        "upos": upos,
        "xpos": xpos,
        "features": features,
        "lemma": lemma,
        "surface": surface,
        "features_multi": features_multi if features_multi is not None else [],
    }


def make_word(surface, lemma, layers, *, is_multiword=False, head_upos=None, head_xpos=None, **meta):
    head_layer = layers[-1] if layers else {}
    word = {
        "surface": surface,
        "lemma": lemma,
        "feature_layers": layers,
        "is_multiword": is_multiword,
        "head_upos": head_upos if head_upos is not None else head_layer.get("upos"),
        "head_xpos": head_xpos if head_xpos is not None else head_layer.get("xpos"),
    }
    word.update(meta)
    return word


def parse_feature_dict(field, *, keep_multi=False):
    feats = {}
    feats_multi = []
    if field and field != "_":
        for item in field.split("|"):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            feats[k] = v
            feats_multi.append((k, v))
    return (feats, feats_multi) if keep_multi else feats


def parse_conllu(filepath, *, preserve_mwt=False, keep_feature_order=False):
    """Parse CoNLL-U rows into sentence token dictionaries.

    Treebank-specific adapters still decide how to merge these rows into
    normalized words; this function only standardizes file reading.
    """
    sentences = []
    current = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            token_id = parts[0]
            if "." in token_id:
                continue
            if "-" in token_id and not preserve_mwt:
                continue

            parsed_features = parse_feature_dict(parts[5], keep_multi=keep_feature_order)
            if keep_feature_order:
                features, features_multi = parsed_features
            else:
                features = parsed_features
                features_multi = []

            token = {
                "id": token_id,
                "surface": parts[1],
                "lemma": parts[2],
                "upos": parts[3],
                "xpos": parts[4],
                "features": features,
                "features_multi": features_multi,
                "head": parts[6],
                "deprel": parts[7],
                "misc": parts[9] if len(parts) >= 10 else "_",
            }
            if "-" in token_id:
                a, b = token_id.split("-")
                token["mwt_range"] = (int(a), int(b))
            else:
                token["mwt_range"] = None
                if preserve_mwt:
                    token["id_int"] = int(token_id)
            current.append(token)

    if current:
        sentences.append(current)
    return sentences


def bare_root_entry(surface_lower):
    return {
        "word": surface_lower,
        "morphology_string": surface_lower,
        "root": surface_lower,
        "suffixes": [],
        "final_pos": "noun",
    }


def _build_suffix_siblings():
    """Suffix names that share a surface form, taken from the tables themselves.

    Turkish spells several distinct morphemes the same way, and which one is
    meant depends on whether the stem is nominal or verbal at that point:
    "-di" is pasttense_di after a verb but pasttense_noundi after a noun,
    "-dir" is the causative active_dir after a verb but the copula
    nounaorist_dir after a noun. Treebank features record the tense, not which
    of the two the decomposer would build, so the adapters emit one name for
    both and the gold chain becomes unbuildable for half the words.
    Sharing a surface is not enough: "-i" is both possessive_3sg and
    accusative_i, and "-ir" is both the aorist factative_ir and the causative
    active_ir. Those are different morphemes in different slots, and renaming
    one to the other to satisfy the decomposer would be relabelling the
    annotation rather than correcting it. Requiring the same SuffixGroup keeps
    only the pairs that compete for one position in the chain.
    """
    by_surface = {}
    for suffix in ALL_SUFFIXES:
        by_surface.setdefault((suffix.suffix, suffix.group), []).append(suffix.name)
    siblings = {}
    for names in by_surface.values():
        if len(names) < 2:
            continue
        for name in names:
            siblings[name] = [other for other in names if other != name]
    return siblings


SUFFIX_SIBLINGS = _build_suffix_siblings()
MAX_AMBIGUOUS_POSITIONS = 6


def decomposer_chains(surface, root):
    """Suffix-name chains the decomposer can actually build for this word."""
    try:
        analyses = decompose(surface)
    except Exception:
        return set()
    return {
        tuple(s.name for s in chain)
        for analysis_root, _pos, chain, _final in analyses
        if analysis_root == root
    }


def reconcile_suffix_names(surface, lemma, names):
    """Rename same-surface suffixes so the gold chain is one the decomposer builds.

    Only names are changed, never the segmentation: siblings share a surface
    form by definition, so the word still divides in exactly the same place.
    A word whose lemma the decomposer cannot reach at all is left untouched -
    that is a lexicon gap, and guessing at it here would only hide it.
    """
    names = list(names)
    positions = [idx for idx, name in enumerate(names) if name in SUFFIX_SIBLINGS]
    if not positions or len(positions) > MAX_AMBIGUOUS_POSITIONS:
        return names

    buildable = decomposer_chains(surface, lemma)
    if not buildable or tuple(names) in buildable:
        return names

    choices = [[names[idx]] + SUFFIX_SIBLINGS[names[idx]] for idx in positions]
    for combo in itertools.product(*choices):
        candidate = list(names)
        for idx, name in zip(positions, combo):
            candidate[idx] = name
        if tuple(candidate) in buildable:
            return candidate
    return names


def build_treebank_forced_entry(surface, lemma, expected_suffix_names):
    surface_lower = tr_lower(surface)
    root = tr_lower(lemma)

    suffixes = []
    current_stem = root
    accepted_chain = []
    for idx, sname in enumerate(expected_suffix_names):
        sobj = SUFFIX_BY_NAME.get(sname)
        if sobj:
            makes_str = "VERB" if sobj.makes == Type.VERB else "NOUN"
            rest = surface_lower[len(current_stem):]
            is_final_suffix = idx == len(expected_suffix_names) - 1
            can_use_surface_tail = is_final_suffix and rest and surface_lower.startswith(current_stem)
            try:
                forms = sobj.form(current_stem, current_chain=accepted_chain)
                form_str = ""
                for form in forms:
                    if form and rest.startswith(form):
                        form_str = form
                        break
                if not form_str:
                    # Gold treebank rows are authoritative. If Savyar's suffix
                    # generator lacks the final surface variant, keep the actual
                    # remaining token tail instead of inventing a default form.
                    form_str = rest if can_use_surface_tail else (forms[0] if forms else sobj.suffix)
            except Exception:
                form_str = rest if can_use_surface_tail else sobj.suffix
            suffixes.append({"name": sname, "form": form_str, "makes": makes_str})
            accepted_chain.append(sobj)
        else:
            suffixes.append({"name": sname, "form": "", "makes": "NOUN"})
        current_stem = current_stem + (suffixes[-1]["form"] or "")

    morphology_parts = [root] + [s["form"] for s in suffixes if s["form"]]
    return {
        "word": surface_lower,
        "morphology_string": " ".join(morphology_parts),
        "root": root,
        "suffixes": suffixes,
        "final_pos": "verb" if suffixes and suffixes[-1]["makes"] == "VERB" else "noun",
    }


def build_cc_entry(surface_lower, cc_category):
    cc_entries = CLOSED_CLASS_LOOKUP.get(surface_lower, [])
    if not cc_entries:
        return None
    matched_cc = next((c for c in cc_entries if c.category == cc_category), cc_entries[0])
    suffix_name = f"cc_{matched_cc.category}"
    return {
        "word": surface_lower,
        "morphology_string": surface_lower,
        "root": surface_lower,
        "suffixes": [{"name": suffix_name, "form": "", "makes": "", "cc_surface": surface_lower}],
        "final_pos": suffix_name,
    }


def record_unmapped(sink, feat_key, feat_val, word, *, note=""):
    slot = sink.setdefault(feat_key, {}).setdefault(feat_val, {
        "count": 0,
        "examples": [],
    })
    if note and "note" not in slot:
        slot["note"] = note
    elif "note" in slot and not slot["note"] and note:
        slot["note"] = note
    slot["count"] += 1
    if len(slot["examples"]) < 8:
        ex = f"{word['surface']}({word['lemma']})"
        if ex not in slot["examples"]:
            slot["examples"].append(ex)


def sort_unmapped_features(unmapped_features):
    unmapped_sorted = {}
    for fkey in sorted(unmapped_features.keys()):
        by_val = unmapped_features[fkey]
        unmapped_sorted[fkey] = dict(sorted(by_val.items(), key=lambda kv: -kv[1]["count"]))
    return unmapped_sorted


def write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_unmapped_features(path, unmapped_features, header):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "_header": header,
            "unmapped": sort_unmapped_features(unmapped_features),
        }, f, indent=2, ensure_ascii=False)


def normalize_input_paths(paths):
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def default_word_context(word):
    return [
        {"upos": l["upos"], "xpos": l["xpos"], "features": l["features"]}
        for l in word["feature_layers"]
    ]


def print_unmatched_summary(unmatched_log, *, unmappable_reason="unmappable_features", label="UNMAPPABLE WORDS"):
    if not unmatched_log:
        return

    decomp_mismatches = [
        e for e in unmatched_log
        if e.get("reason") not in (None, "", unmappable_reason)
        and not str(e.get("reason", "")).startswith("unmappable")
    ]
    unmappable_entries = [
        e for e in unmatched_log
        if e.get("reason") == unmappable_reason
        or str(e.get("reason", "")).startswith("unmappable")
    ]

    print(f"\n=== {label} ({len(unmappable_entries)} words) ===")
    feat_counts = Counter()
    for e in unmappable_entries:
        detail = e.get("detail") or str(e.get("reason", "")).replace("unmappable features: ", "")
        for f in detail.replace("unmapped: ", "").strip("[]").split(","):
            s = f.strip().strip("'\"")
            if s:
                feat_counts[s] += 1
    for feat, n in feat_counts.most_common(30):
        print(f"  {n:4d}x  {feat}")

    print(f"\n=== DECOMPOSER MISMATCH BREAKDOWN ({len(decomp_mismatches)} words) ===")
    reason_counts = Counter(e["reason"] for e in decomp_mismatches)
    for reason, count in reason_counts.most_common():
        print(f"  {count:4d}x  {reason}")


def _sentence_diagnostic(sent_idx, original_sentence, diagnostic_type, sentence_unmappable,
                         trainable_words_in_sentence, bare_root_words, skipped_words):
    why = "No token in the sentence produced a trainable suffix sequence."
    how_to_fix = "Usually not an adapter bug. These are often suffixless fragments, titles, numeric snippets, or unmappable tokens."
    if diagnostic_type == "partially_trainable_sentence":
        why = "At least one token was trainable, but one or more tokens had unmappable features or had to remain bare roots."
        how_to_fix = "Inspect the unmappable token list first. If it is empty, this sentence is only partially trainable because some tokens are bare roots or skipped POS."
    elif sentence_unmappable:
        diagnostic_type = "non_trainable_due_to_unmappable_tokens"
        why = "No token was trainable and at least one token has unmappable treebank features."
        how_to_fix = "Add the missing treebank→Savyar mapping for the listed unmappable tokens."

    return {
        "sentence_index": sent_idx,
        "original_sentence": original_sentence,
        "diagnostic_type": diagnostic_type,
        "why": why,
        "how_to_fix": how_to_fix,
        "trainable_word_count": trainable_words_in_sentence,
        "bare_root_words": bare_root_words,
        "skipped_words": skipped_words,
        "unmappable_tokens": sentence_unmappable,
    }


def adapt_normalized_treebank(
    input_paths,
    output_path,
    *,
    parse_sentences,
    words_from_sentence=None,
    translate_word,
    should_skip_word,
    closed_class_category,
    stats_path=None,
    unmatched_path=None,
    unmapped_path=None,
    sentence_diagnostics_path=None,
    include_input_files_in_stats=False,
    include_unmapped_feature_value_count=False,
    unmapped_header=None,
    word_context=default_word_context,
    unmappable_context_key="feature_layers",
    unmappable_reason="unmappable_features",
    unmappable_detail=True,
    parse_message="Parsing: {path}",
    parsed_message="  -> {count} sentences",
    stats_sentence_count=None,
    summary_unmappable_label="UNMAPPABLE WORDS",
    write_unmatched_log=True,
    write_sentence_diagnostics=True,
    write_unmapped_report=True,
):
    input_paths = normalize_input_paths(input_paths)
    all_sentences = []
    for path in input_paths:
        print(parse_message.format(path=path))
        sents = parse_sentences(path)
        print(parsed_message.format(count=len(sents)))
        all_sentences.extend(sents)

    total_words = 0
    matched_words = 0
    forced_words = 0
    unmappable_words = 0
    no_suffix_words = 0

    matched_sentences = 0
    partial_sentences = 0
    failed_sentences = 0

    output_entries = []
    unmatched_log = []
    unmapped_features = {}
    sentence_diagnostics = []

    if words_from_sentence is None:
        words_from_sentence = lambda sentence: sentence

    for sent_idx, sentence in enumerate(all_sentences):
        if sent_idx % 500 == 0:
            print(f"  Processing sentence {sent_idx}/{len(all_sentences)}...")

        words = words_from_sentence(sentence)
        if not words:
            continue

        original_sentence = " ".join(w["surface"] for w in words)

        word_entries = []
        sentence_all_matched = True
        sentence_has_any = False
        sentence_unmappable = []
        bare_root_words = []
        skipped_words = []
        trainable_words_in_sentence = 0

        for word in words:
            total_words += 1

            surface = strip_quotes(word["surface"])
            surface_lower = tr_lower(surface)
            lemma = strip_quotes(word["lemma"])

            if should_skip_word(word):
                skipped_words.append(surface_lower)
                bare_root_words.append(surface_lower)
                word_entries.append(bare_root_entry(surface_lower))
                no_suffix_words += 1
                continue

            cc_category = closed_class_category(word)
            if cc_category:
                entry = build_cc_entry(surface_lower, cc_category)
                if entry:
                    word_entries.append(entry)
                    matched_words += 1
                    sentence_has_any = True
                    trainable_words_in_sentence += 1
                else:
                    bare_root_words.append(surface_lower)
                    word_entries.append(bare_root_entry(surface_lower))
                    no_suffix_words += 1
                continue

            expected_suffixes, unmapped_feats, has_unmappable = translate_word(word, unmapped_features)

            if has_unmappable:
                unmappable_words += 1
                sentence_all_matched = False
                context = word_context(word)
                token_diagnostic = {
                    "surface": surface_lower,
                    "lemma": lemma,
                    unmappable_context_key: context,
                    "unmapped": list(unmapped_feats),
                }
                sentence_unmappable.append(token_diagnostic)

                unmatched_entry = {
                    "surface": surface_lower,
                    "lemma": lemma,
                    unmappable_context_key: context,
                    "reason": unmappable_reason,
                }
                if unmappable_detail:
                    unmatched_entry["detail"] = f"unmapped: {unmapped_feats}"
                elif unmappable_reason != "unmappable_features":
                    unmatched_entry["reason"] = f"{unmappable_reason}: {unmapped_feats}"
                unmatched_log.append(unmatched_entry)

                word_entries.append(bare_root_entry(surface_lower))
                bare_root_words.append(surface_lower)
                continue

            if not expected_suffixes:
                no_suffix_words += 1
                bare_root_words.append(surface_lower)
                word_entries.append(bare_root_entry(surface_lower))
                continue

            expected_suffixes = reconcile_suffix_names(surface_lower, lemma, expected_suffixes)
            entry = build_treebank_forced_entry(surface_lower, lemma, expected_suffixes)
            word_entries.append(entry)
            matched_words += 1
            sentence_has_any = True
            trainable_words_in_sentence += 1

        if word_entries:
            decomposed_parts = [we["morphology_string"] for we in word_entries]
            output_entries.append({
                "type": "sentence",
                "original_sentence": original_sentence,
                "decomposed_sentence": " ".join(decomposed_parts),
                "words": word_entries,
            })
            if sentence_all_matched and sentence_has_any:
                matched_sentences += 1
            elif sentence_has_any:
                partial_sentences += 1
                sentence_diagnostics.append(_sentence_diagnostic(
                    sent_idx, original_sentence, "partially_trainable_sentence",
                    sentence_unmappable, trainable_words_in_sentence,
                    bare_root_words, skipped_words,
                ))
            else:
                failed_sentences += 1
                sentence_diagnostics.append(_sentence_diagnostic(
                    sent_idx, original_sentence, "non_trainable_sentence",
                    sentence_unmappable, trainable_words_in_sentence,
                    bare_root_words, skipped_words,
                ))

    output_entries, withheld = withhold_held_out(output_entries)
    if withheld:
        print(f"Withholding {withheld} sentences that belong to the test set at {HELD_OUT_TEST_PATH}")

    print(f"\nWriting {len(output_entries)} sentences to {output_path}")
    write_jsonl(output_path, output_entries)

    if write_unmatched_log:
        if unmatched_path is None:
            unmatched_path = output_path.replace(".jsonl", "_unmatched.jsonl")
        write_jsonl(unmatched_path, unmatched_log)

    if write_sentence_diagnostics:
        if sentence_diagnostics_path is None:
            sentence_diagnostics_path = output_path.replace(".jsonl", "_sentence_diagnostics.jsonl")
        write_jsonl(sentence_diagnostics_path, sentence_diagnostics)

    if write_unmapped_report and unmapped_header is not None:
        if unmapped_path is None:
            unmapped_path = os.path.join(os.path.dirname(output_path), "unmapped_features.json")
        write_unmapped_features(unmapped_path, unmapped_features, unmapped_header)

    trainable_words = matched_words + forced_words
    stats = {}
    if include_input_files_in_stats:
        stats["input_files"] = [str(p) for p in input_paths]
    stats.update({
        "total_sentences": stats_sentence_count if stats_sentence_count is not None else len(all_sentences),
        "total_words": total_words,
        "translated_words (treebank-authoritative)": matched_words,
        "compat_words (legacy-forced)": forced_words,
        "trainable_words (total)": trainable_words,
        "unmappable_words": unmappable_words,
        "no_suffix_words": no_suffix_words,
        "trainable_rate": f"{trainable_words / max(total_words - no_suffix_words, 1) * 100:.1f}%",
        "fully_trainable_sentences": matched_sentences,
        "partially_trainable_sentences": partial_sentences,
        "non_trainable_sentences": failed_sentences,
        "sentence_diagnostics_count": len(sentence_diagnostics),
    })
    if include_unmapped_feature_value_count:
        stats["unmapped_feature_value_count"] = sum(len(v) for v in unmapped_features.values())

    print("\n=== ADAPTATION STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    print_unmatched_summary(
        unmatched_log,
        unmappable_reason=unmappable_reason,
        label=summary_unmappable_label,
    )

    if write_unmapped_report and unmapped_header is not None:
        print(f"\nUnmapped feature VALUES recorded in: {unmapped_path}")

    return stats


def _iter_suffix_tails(current_stem, suffix_names, suffix_by_name, limit=256):
    results = set()

    def visit(stem, names, tail, current_chain):
        if len(results) >= limit:
            return
        if not names:
            results.add(tail)
            return

        suffix_obj = suffix_by_name.get(names[0])
        if suffix_obj is None:
            return

        try:
            forms = suffix_obj.form(stem, current_chain=current_chain)
        except Exception:
            forms = [suffix_obj.suffix]

        seen_forms = set()
        for form in forms:
            if form in seen_forms:
                continue
            seen_forms.add(form)
            visit(stem + form, names[1:], tail + form, current_chain + [suffix_obj])

    visit(current_stem, suffix_names, "", [])
    return results


def resolve_ambiguous_suffix_by_surface(
    surface,
    lemma,
    suffix_names,
    marker,
    candidates,
    suffix_by_name=SUFFIX_BY_NAME,
    fallback=None,
):
    if marker not in suffix_names:
        return suffix_names

    surface_lower = tr_lower(surface)
    lemma_lower = tr_lower(lemma)
    resolved = list(suffix_names)

    def resolved_stem_before(end_idx):
        stem = lemma_lower
        chain = []
        for prev_name in resolved[:end_idx]:
            suffix_obj = suffix_by_name.get(prev_name)
            if suffix_obj is None:
                break
            try:
                forms = suffix_obj.form(stem, current_chain=chain)
            except Exception:
                forms = [suffix_obj.suffix]

            rest = surface_lower[len(stem):]
            form = ""
            for candidate_form in forms:
                if candidate_form and rest.startswith(candidate_form):
                    form = candidate_form
                    break
            if not form:
                form = forms[0] if forms else suffix_obj.suffix

            stem += form
            chain.append(suffix_obj)
        return stem, chain

    for idx, name in enumerate(resolved):
        if name != marker:
            continue

        best_name = None
        best_score = (-1, -1, -1)
        current_stem, current_chain = resolved_stem_before(idx)
        rest = surface_lower[len(current_stem):]
        for candidate_idx, candidate in enumerate(candidates):
            candidate_obj = suffix_by_name.get(candidate)
            if candidate_obj is not None:
                try:
                    forms = candidate_obj.form(current_stem, current_chain=current_chain)
                except Exception:
                    forms = [candidate_obj.suffix]
                for form in forms:
                    if form and rest.startswith(form):
                        score = (2, len(form), -candidate_idx)
                        if score > best_score:
                            best_name = candidate
                            best_score = score

            test_names = list(resolved)
            test_names[idx] = candidate
            tails = _iter_suffix_tails(lemma_lower, test_names, suffix_by_name)
            for tail in tails:
                if tail and surface_lower.endswith(tail):
                    score = (1, len(tail), -candidate_idx)
                    if score > best_score:
                        best_name = candidate
                        best_score = score

        if best_name is None:
            best_name = fallback if fallback is not None else candidates[0]

        resolved[idx] = best_name

    return resolved


def _fallback_vnoun_from_surface(surface):
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


def resolve_ambiguous_vnoun_suffixes(surface, lemma, suffix_names, suffix_by_name=SUFFIX_BY_NAME):
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
    if "nounifier_iş" not in chain_names:
        return False
    if "nounifier_iş" in expected_suffixes:
        return False
    return True
