# Workflow: Engine and End-to-End Flow

This document describes the current runtime flow in the code, mainly:

- `app/cli.py`: command-line interaction and user prompts.
- `app/engine.py`: orchestration, training set construction, relearn, curriculum, testing, sampling.
- `app/nlp_pipeline.py`: sanitation, morphology view-model building, suffix-chain encoding.
- `util/decomposer.py`: rule-based candidate generation.
- `ml/ml_ranking_model.py`: candidate scoring and training.

The short version is:

```text
raw input text
  -> sanitize into lowercase Turkish tokens
  -> rule-based decomposer creates root + suffix-chain candidates
  -> NLP adapter reconstructs display strings and encodes only suffix/closed-class chains
  -> ML model scores complete candidate chains
  -> user/treebank choice becomes gold
  -> engine trains gold-vs-generated-negative candidate sets
```

The important design point is that roots are used by the rule-based engine, display code, logs, and gold-candidate matching. Roots are not encoded into the ML model input.

## Startup

Creating `WorkflowEngine()` does the following:

1. Creates a `DataManager`.
   - This object owns paths for dictionaries, logs, training count, sample files, test files, and preprocessing caches.
2. Resolves the PyTorch device with `resolve_torch_device()`.
   - `config.device` defaults to `"cuda"`.
   - CPU fallback is refused unless `config.allow_cpu_fallback`, `SAVYAR_ALLOW_CPU=1`, or an explicit CPU device is used.
3. Creates `SentenceDisambiguator`.
   - The current repository has `98` suffix objects in `util.decomposer.ALL_SUFFIXES`.
   - The current closed-class vocabulary has `88` `(category, surface)` entries.
4. Creates `Trainer`.
   - The trainer builds AdamW, the scheduler, AMP scaler, and tries to load `ml/model.pt`.
   - Checkpoint loading is shape-tolerant: matching tensors are loaded, incompatible/missing tensors are left initialized.
5. Loads `training_count` from `data/training_count.txt`.
6. Creates an engine-level `decomp_cache` for repeated word analyses.

## Input Sanitation

The CLI sanitizes before calling the engine.

For a single word, `sanitize_word(raw)`:

1. Strips surrounding whitespace.
2. Removes apostrophes: `'`, `’`, `‘`.
3. Applies Turkish-aware lowercase through `tr_lower()`.
   - `İ` becomes `i`.
   - `I` becomes `ı`.

For a sentence, `sanitize_sentence(raw)`:

1. Removes apostrophes.
2. Replaces punctuation and underscores with spaces using `[^\w\s]|_`.
3. Applies Turkish-aware lowercase.
4. Splits on whitespace.

Example:

```text
"Evlerden geldim."
  -> "evlerden geldim "
  -> ["evlerden", "geldim"]
```

No tokenizer keeps punctuation, capitalization, apostrophes, or sentence-boundary symbols for the model. Those are removed before analysis.

## Word Analysis

`WorkflowEngine.analyze_word(word)` calls `nlp.analyze_word(word, include_closed_class=True)`.

### 1. Candidate Generation

`nlp.analyze_word()` calls `sfx.decompose_with_cc(word)`.

`decompose_with_cc()` first runs the open-class decomposer and then appends closed-class analyses.

The open-class decomposer, `decompose(word)`, does this:

1. Rejects non-`ben` pronoun surfaces for open-class decomposition unless `force=True`.
   - This keeps forms such as `sana`, `onu`, or `bizim` from being treated as ordinary dictionary-root analyses.
   - They can still appear through the closed-class path.
2. Starts with any `pekistirme` analyses from `get_pekistirme_analyses()`.
   - This handles intensifier reduplication such as `masmavi`.
3. Iterates every prefix of the word as a possible root.
   - For `evlerden`, prefixes are `e`, `ev`, `evl`, `evle`, ...
4. Skips a prefix if it is in `UNSUFFIXABLE_SET`.
5. If the prefix can be a noun, runs suffix search from POS `"noun"`.
6. If the prefix can be a verb, runs suffix search from POS `"verb"`.
7. If the prefix is not already a dictionary noun or verb, asks `get_root_candidates()` for restored lemma candidates.
   - This handles root-side phonological changes such as final consonant softening, vowel restoration, terminal `a/e` restoration, and consonant degemination.
   - For a restored lemma, the decomposer builds a virtual word from the lemma plus the remaining surface text, then searches suffixes.

Each successful decomposition has this shape:

```python
(root: str, start_pos: str, suffix_chain: list[Suffix], final_pos: str)
```

The root string stays in this tuple, but it is not part of the ML token sequence.

### 2. Suffix-Chain Search

`find_suffix_chain(word, start_pos, root, current_chain, visited, shared_cache)` is a recursive depth-first search.

At each recursion step:

1. `root` means "the surface/stem consumed so far".
2. `rest = word[len(root):]` is the part still unmatched.
3. If `rest` is empty, the current chain is a complete valid candidate.
4. The next suffix list comes from `SUFFIX_TRANSITIONS[start_pos]`.
   - From noun to noun: `NOUN2NOUN` plus V2N suffixes that accept both.
   - From noun to verb: `NOUN2VERB` plus V2V suffixes that accept both.
   - From verb to noun: `VERB2NOUN` plus N2N suffixes that accept both.
   - From verb to verb: `VERB2VERB` plus N2V suffixes that accept both.
5. For every suffix object, the code generates possible surface forms with `suffix.form(root, current_chain=current_chain)`.
6. If a generated form matches the beginning of `rest`, the suffix is accepted and recursion continues with:
   - a longer consumed stem,
   - the target POS,
   - the suffix appended to the chain.
7. A special contraction path handles `-a/-e` before `-iyor`.
   - If a suffix form ends in `a` or `e`, the shortened form can match when the following remainder starts with an `iyor` variant.

Suffix objects generate forms through `util/suffix.py` and custom suffix functions. The default form path applies:

- major vowel harmony,
- minor vowel harmony,
- consonant hardening at the suffix boundary,
- vowel-collision handling,
- optional `y` buffer insertion,
- suffix-final softening variants.

### 3. Active Transition Caveat

`is_valid_transition()` currently returns `True` immediately.

That means the detailed suffix waterfall rules written below that return statement are not active. In the current code, ordering restrictions are effectively not enforced by `is_valid_transition()`. The decomposer still filters candidates by POS transition, uniqueness, surface-form matching, dictionary roots, and special matching logic, but not by the later waterfall checks.

### 4. Closed-Class Candidates

After regular decompositions, `decompose_with_cc()` checks `CLOSED_CLASS_LOOKUP`.

For each matching closed-class word, it appends:

```python
(word, "cc_<category>", [ClosedClassMarker(cc_obj, surface_form=word)], "cc_<category>")
```

Closed-class examples include pronouns, conjunctions, postpositions, adverbs, and question particles. Pronoun lookup includes stored irregular forms such as `bana`, `sana`, `onu`, and `onun`.

Closed-class candidates are represented as one marker in the suffix chain. They are not decomposed into normal root + suffix pieces.

## NLP Adapter Output

For every decomposition, `nlp.analyze_word()` builds four aligned lists:

- `decomps`: the original `(root, pos, chain, final_pos)` tuples.
- `encoded_chains`: ML-ready chains.
- `vms`: display view models.
- `typing_strings`: compact strings used by manual sentence matching.

### Encoded Chains

`encode_suffix_chain(chain)` converts each suffix or closed-class marker into:

```python
(token_id, category_id, group_id, makes_id, position_in_word)
```

For a normal suffix:

- `token_id = 4 + suffix_index_in_ALL_SUFFIXES`.
- `category_id` is intended to mark noun vs verb output.
  - In generated decomposer chains, the current code checks `s.makes.name == "Verb"`.
  - Because `Type` enum names are uppercase (`"NOUN"`, `"VERB"`, `"BOTH"`), generated normal suffix tokens currently get `category_id = 0`.
  - The actual output-type signal is still present in `makes_id`.
  - Directly encoded log entries from `encode_suffix_names()` can set `category_id = 1` when the logged `makes` field is `"VERB"`.
- `group_id` comes from `SuffixGroup`; missing group is `0`.
- `makes_id` uses the same noun/verb/both mapping.
- `position_in_word` starts at `1` inside the current word.

For a closed-class marker:

- `token_id` comes after all suffix IDs.
- `category_id = 3`.
- group and `makes` are special value `0`.
- position is still set.

For a bare-root candidate with no suffixes:

```python
encoded_chain = []
```

There is no root token and no bare-root token in the encoded chain.

### Display Reconstruction

`reconstruct_morphology(word, decomposition)` builds the user-facing explanation.

For a closed-class candidate:

```text
root_str      = "<surface> (<category>)"
formation_str = "<surface> [<category>]"
has_chain     = False
```

For a bare root:

```text
root_str      = "<root> (<pos>)"
formation_str = "<root> (no suffixes)"
```

For a suffix chain:

1. Start from the root.
2. For each suffix, generate possible forms against the current stem.
3. Match a form at the current cursor in the surface word.
4. If `-iyor` is ahead, allow final `a/e` shortening.
5. If needed, try a one-character cursor rewind for softened/root-altered boundaries.
6. If no form matches, append the first generated form with `?`.
7. Add `(ø)` when a suffix contributes an empty surface form.
8. Build:
   - `suffixes_str`: surface forms, joined by ` + `,
   - `names_str`: suffix object names,
   - `formation_str`: step-by-step growing stem display.

The view model is for humans. The model does not consume `formation_str`, `suffixes_str`, or the root string.

### Typing Strings

Manual sentence training uses `typing_strings`.

If a candidate has suffixes:

```text
<root> <surface_suffix_1> <surface_suffix_2> ...
```

If it has no suffix chain:

```text
<root>
```

`(ø)` markers are ignored during manual target matching.

## ML Sorting During Analysis

If `training_count > 0`, `WorkflowEngine.analyze_word()` calls `nlp.score_and_sort()`.

`score_and_sort()`:

1. Does nothing when there is only one candidate.
2. Calls `trainer.predict(analysis["encoded_chains"])`.
3. Adds each score to the matching view model.
4. Sorts `decomps`, `encoded_chains`, `vms`, and `typing_strings` in descending score order.

For isolated word scoring, no root strings and no surrounding sentence context are passed. The scoring input is only the encoded candidate chain, converted into a short sentence sequence internally.

If model scoring raises an exception, sorting is skipped and the original decomposer order remains.

## Interactive Single-Word Training

The CLI path is:

```text
raw word
  -> sanitize_word()
  -> engine.analyze_word()
  -> display candidates
  -> user selects one or more correct candidates
  -> engine.commit_word()
```

`commit_word(analysis, selected_indices)` does this:

1. Reads the selected decompositions and encoded chains.
2. Logs each selected decomposition to `data/sentence_valid_decompositions.jsonl`.
   - Each log entry stores `word`, `root`, `suffixes`, and `final_pos`.
   - `suffixes` contains suffix names, matched surface forms, and output POS.
   - Closed-class and bare-root chains log an empty suffix list.
3. Runs dictionary cleanup logic.
   - It intends to remove the full surface word if a different root exists.
   - It also intends to remove an infinitive form derived from the root.
   - Current caveat: `wrd.delete_word()` always returns `False`, so `DataManager.delete()` normally does not persist deletion or report deletion messages.
4. Builds negative candidates from the other decomposer candidates for the same word.
   - A duplicate is detected by token-id signature only.
   - Root differences that produce the same suffix-token sequence are treated as the same ML candidate signature.
   - Negatives are capped by `config.max_negative_candidates`, currently `8`.
5. For each selected gold candidate, calls:

```python
trainer.train_sentence([gold_encoded_chain], negative_word_chains=negatives)
```

6. Increments `training_count` by `1`, regardless of how many selected candidates were trained.
7. Saves the model only if the new count is exactly on the checkpoint interval.

If a word has no negatives, `Trainer.train_sentence()` returns `0.0` and no optimizer step is run.

## Interactive Sentence Training

The CLI path for `sentence <text>` is:

```text
raw sentence
  -> sanitize_sentence()
  -> engine.analyze_sentence_with_failures()
  -> user enters a decomposition string or prefix
  -> engine.evaluate_sentence_target()
  -> user selects a legal full candidate combination
  -> engine.commit_sentence_training()
```

### Sentence Analysis

`analyze_sentence_with_failures(words)`:

1. Calls `nlp.analyze_words()` for each sanitized token.
2. Collects failures for tokens with no decompositions.
3. If any token fails, returns no sentence analysis and reports the failing word indexes.
4. Otherwise returns the per-word analysis list.

The analysis step does not decide a sentence-level parse by itself. It only gives candidates for each word.

### Manual Target Matching

`evaluate_sentence_target(word_data, target_str)` calls `find_matching_combinations()`.

The matcher:

1. Removes `(ø)` from the target.
2. Normalizes whitespace.
3. Runs DFS over one candidate per word.
4. At each partial combination, joins candidate `typing_strings`.
5. Keeps the branch only if the current joined text is still a valid prefix of the target.
   - Earlier tokens must match exactly.
   - The last target token may be a prefix of the current candidate token.
6. Tracks the furthest matching word and text to explain mismatches.
7. When a complete sentence combination is reached, accepts it if it matches the target prefix.
8. Scores every legal complete combination with `trainer.score_sentence_chains()`.
9. Sorts matches by descending score.

This means manual sentence entry is not free-form morphology parsing. The text must correspond to one of the already generated legal candidate combinations.

### Sentence Commit

`commit_sentence_training(sentence, words, word_data, correct_combo)`:

1. Takes the selected candidate index for each word.
2. Builds `confirmed_chains`, one encoded chain per word.
   - Bare roots contribute an empty chain.
   - Closed-class candidates contribute their closed-class marker chain during this interactive step.
3. Logs a sentence entry to `data/sentence_valid_decompositions.jsonl`.
   - The entry has `type: "sentence"`, `original_sentence`, `decomposed_sentence`, and `words`.
4. Builds negatives with `_single_substitution_negatives()`.
   - Keep the whole gold sentence fixed.
   - Replace exactly one word with a wrong candidate.
   - Skip duplicate token-id signatures.
   - Stop after `config.max_negative_candidates`, currently `8`.
5. Calls:

```python
trainer.train_sentence(confirmed_chains, negative_word_chains=negatives)
```

6. Increments `training_count` by the number of words in the confirmed sentence.
7. Saves only on checkpoint interval.

The engine does not build the full Cartesian product of all word candidates. It only uses single-word substitutions as negatives.

## Evaluation Sentence Flow

`eval sentence <text>` uses beam search before asking the user.

`get_top_sentence_predictions(word_data, trainer, top_k=10, beam_width=50)`:

1. Starts with an empty beam.
2. For each word, expands every current beam by every candidate for that word.
3. Converts the partial chain list into a sentence sequence.
4. Scores the partial sequence with `trainer.score_sentence_chains()`.
5. Keeps only the best `beam_width` partial beams.
6. Returns the top `top_k` complete beams.

After the user selects a prediction, training uses the same sentence commit flow as manual sentence training.

## Relearn

The `relearn` command calls `WorkflowEngine.relearn_all()`.

High-level flow:

1. Try to load cached static candidate sets for scope `"training-all"`.
2. If no compatible cache exists, load training entries from:
   - `data/sentence_valid_decompositions.jsonl`,
   - every `treebank_adapted*.jsonl` shard under `data/`,
   - excluding the configured test file.
3. Convert log/treebank entries into candidate sets with `_entries_to_sequences()`.
4. Split candidate sets into train/validation using `config.validation_split` and `config.validation_seed`.
5. Train with `trainer.train_bulk()`.
6. Save final suffix/ranking metrics and checkpoint.
7. Increment `training_count` by the number of words that made it into built candidate sets.

### Log Entry Conversion

`_candidate_parts_from_word_entries()` is the key conversion function.

For every word entry:

1. Read `suffixes`.
2. If `suffixes` is empty, skip that word entry.
   - This means bare-root words from logs/treebanks are not included in bulk relearn candidate sets.
   - Closed-class entries logged with empty suffixes are also skipped in bulk relearn.
3. Encode the logged suffix names with `nlp.encode_suffix_names()`.
   - The legacy `nondoing_meden` name expands into `infinitive_me + ablative_den`.
   - Unknown suffix names fall back to token ID `SUFFIX_OFFSET`, which is the first normal suffix-token slot, rather than raising an error.
4. Run the current decomposer again for the logged word.
5. Match the logged root and suffix-name sequence against generated decompositions.
6. If a match exists:
   - use the generated encoded chain as gold,
   - use all current generated encoded chains as candidates.
7. If no match exists:
   - use the directly encoded logged suffix names as the only candidate.
   - This usually cannot train because there is no negative.

Then `_candidate_set_from_word_entries()`:

1. Builds the gold sentence sequence.
2. Builds single-substitution negatives.
3. Returns `[gold_seq, neg_seq_1, neg_seq_2, ...]`.

`_entries_to_sequences()` only keeps candidate sets with at least two sequences and sequence lengths at or below `config.max_sequence_length`.

## Curriculum

The `curriculum` command calls `WorkflowEngine.train_curriculum()`.

Defaults from `ml/config.py`:

- `max_negative_candidates = 8`
- `hard_negative_count = 4`
- `medium_negative_count = 2`
- `easy_negative_count = 2`
- `dynamic_negative_pool_size = 100`
- `curriculum_generations = 3`
- `curriculum_warmup_epochs = 5`
- `curriculum_mining_epochs = 4`

Flow:

1. Load all valid entries.
2. Split entries into train and validation entries if enough data exists.
3. Build or load static validation candidate sets.
4. Optionally run warm-up training on static candidate sets.
5. For each generation:
   - rebuild candidate parts from the train entries,
   - build a wider single-substitution negative pool,
   - score the negative pool with the current model,
   - select hard, easy, then medium negatives,
   - train on `[gold, selected_negatives...]`,
   - save the checkpoint.
6. Save final metrics.

Hard negatives are the highest-scoring wrong candidates. Easy negatives come from the lowest-scoring wrong candidates. Medium negatives are sampled from the remaining middle pool. If slots remain, the code fills from the ranked list.

Curriculum does not write a separate mined-example report.

## Test and Diagnostics

`test` uses `data/trmor2018_treebank/trmor2018_gold_adapted.jsonl`.

Flow:

1. Load test JSONL shards.
2. Convert entries into candidate sets with the same static conversion path used by relearn.
3. Run `trainer.validate()`.
4. Report rank loss, rank accuracy, top-k rank accuracy, suffix accuracy, exact word accuracy, precision, recall, F1, and margin.

`test detail` additionally:

1. Builds diagnostic candidate sets.
2. Scores each set.
3. Finds cases where the best prediction is not candidate `0`.
4. Compares gold/predicted suffix token names.
5. Captures examples for the worst suffixes.

Per-suffix diagnostic names come only from normal suffix token IDs. Closed-class token IDs do not map to suffix names in this report.

## Sampling

`sample` processes a text word by word.

1. Tokenize the sample file with `DataManager.get_text_tokenized()`.
2. Analyze each unique word.
3. If only one decomposition exists, use it.
4. If multiple decompositions exist and `training_count > 0`, score candidates with `trainer.predict()`.
5. Write the chosen detailed decomposition for each token.

This is isolated word scoring. It does not use sentence context.

`sample sentence` processes `sample/sample_sentence.txt`.

1. Split text lines into sentences with a punctuation regex.
2. Clean and lowercase each sentence.
3. Analyze each token.
4. Run `get_top_sentence_predictions()` with `top_k=1`.
5. Write detailed decompositions for the best sentence-level beam.

## Caching

There are several cache layers:

- `WorkflowEngine.decomp_cache`: word string to decomposition list.
- `decompose()` LRU cache: full-word decomposer cache with `maxsize=100000`.
- `find_suffix_chain()` shared cache: per-decompose-call recursive suffix subtree cache.
- `data/preprocessed_sequence_cache/*.pkl`: persistent candidate-set cache for static preprocessing.

Static preprocessing cache metadata includes:

- cache version,
- preprocessing scope,
- training/source file signatures,
- dictionary dependency file signatures,
- suffix inventory,
- closed-class inventory,
- `max_negative_candidates`,
- `max_sequence_length`,
- optional entry digest for scoped entry lists.

Exact cache lookup requires full metadata equality. Compatible fallback can reuse a cache with the same paths, existence, and file sizes even if mtimes differ.

## What Is Not Happening

- The ML model is not given root strings, root IDs, lemmas, or raw surface characters.
- The ML model is not given the displayed suffix surface forms such as `ler`, `den`, or `(ø)`. It receives suffix object IDs and suffix metadata IDs.
- The ML model does not receive punctuation, capitalization, apostrophes, or original sentence text.
- The engine does not use a full Cartesian product of sentence candidates for training negatives.
- Bulk relearn skips logged word entries whose `suffixes` list is empty, so bare-root logged words are not included as context there.
- Closed-class candidates are used interactively, but logged closed-class entries with empty suffixes are skipped by the current bulk relearn conversion.
- `is_valid_transition()` currently does not enforce the waterfall rules because it returns `True` at the top of the function.
- The replay buffer is saved in checkpoints, but the current training code does not sample replay examples back into new optimizer steps.
- `config.use_class_weights` and `config.replay_k` exist, but the current trainer does not use them in the active loss path.
- Dictionary cleanup after word commits is effectively inactive because `wrd.delete_word()` always returns `False`.
- Suffix-derived verb markers in `formation_str` are effectively inactive for the same enum-name reason: the display code checks for `"Verb"`, while current enum names are uppercase.
