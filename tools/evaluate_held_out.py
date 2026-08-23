#!/usr/bin/env python3
"""Honest end-to-end evaluation of the current model on held-out sentences.

The 10% validation slice relearn_all() holds out is a slice of *candidate sets*,
and _entries_to_sequences() emits exactly one candidate set per sentence entry
in corpus order - so replaying that loop reproduces which sentences the model
never trained on.
"""
import sys, random, time, argparse
sys.path.insert(0, ".")
from ml.config import config
from app.engine import WorkflowEngine, get_top_sentence_predictions
import app.nlp_pipeline as nlp

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", type=int, default=120)
ap.add_argument("--beam", type=int, default=50)
ap.add_argument("--model", default="")
args = ap.parse_args()

if args.model:
    config.model_path = args.model
engine = WorkflowEngine()

cached = engine._load_static_sequence_cache("training-all", label="eval")
assert cached is not None, "no preprocessing cache; run relearn first"
n_total = len(cached[0])
order = list(range(n_total))
random.Random(config.validation_seed).shuffle(order)
val_count = max(1, int(round(n_total * config.validation_split)))
val_indices = set(order[:val_count])
print(f"{n_total} candidate sets, {val_count} held out for validation")

held_out = []
seq_idx = 0
for entry in engine.data_manager.iter_valid_decomps():
    words = entry.get("words", []) if entry.get("type") == "sentence" else [entry]
    if engine._candidate_set_from_word_entries(words) is None:
        continue
    if seq_idx in val_indices:
        held_out.append(entry)
        if len(held_out) >= args.sentences:
            break
    seq_idx += 1
print(f"collected {len(held_out)} held-out sentences (scanned {seq_idx} sets)\n")

tot = correct = reachable = amb = amb_reachable = amb_correct = unparseable_words = 0
sent_exact = sent_parsed = 0
unparseable = 0
t0 = time.time()
for n, entry in enumerate(held_out, 1):
    gold_words = entry.get("words", [])
    surfaces = [w.get("word") or "" for w in gold_words]
    if not all(surfaces):
        continue
    analyses, failures = engine.analyze_sentence_with_failures(surfaces)
    if analyses is None:
        unparseable += 1
        unparseable_words += len(gold_words)
        continue
    sent_parsed += 1
    best = get_top_sentence_predictions(analyses, engine.trainer, top_k=1, beam_width=args.beam)[0]
    picks = best["combo_indices"]
    all_ok = True
    for w, a, pick in zip(gold_words, analyses, picks):
        tot += 1
        gold_idx = nlp.match_decompositions([w], a["decomps"])
        n_cand = len(a["decomps"])
        has_gold = bool(gold_idx)
        reachable += has_gold
        hit = has_gold and pick == gold_idx[0]
        correct += hit
        if n_cand > 1:
            amb += 1
            amb_reachable += has_gold
            amb_correct += hit
        all_ok &= hit
    sent_exact += all_ok
    if n % 20 == 0:
        print(f"  {n}/{len(held_out)} sentences, {time.time()-t0:.0f}s", flush=True)

pct = lambda a, b: f"{a/b:.1%}" if b else "n/a"
print(f"\n--- held-out sentences: {len(held_out)} ({time.time()-t0:.0f}s) ---")
print(f"sentences fully parseable        : {sent_parsed}/{len(held_out)}  "
      f"({unparseable} dropped: a word had no candidate at all, {unparseable_words} words)")
print(f"sentence exact match             : {pct(sent_exact, sent_parsed)}")
print(f"words (in parseable sentences)   : {tot}")
print(f"  gold reachable (ceiling)       : {pct(reachable, tot)}")
print(f"  model correct (of all words)   : {pct(correct, tot)}")
print(f"  model correct (of reachable)   : {pct(correct, reachable)}")
print(f"ambiguous words (>1 candidate)   : {amb}  ({pct(amb, tot)} of words)")
print(f"  gold reachable                 : {pct(amb_reachable, amb)}")
print(f"  model correct (of reachable)   : {pct(amb_correct, amb_reachable)}")
