# ML Model: Morphological Candidate Ranking

The ML layer ranks complete morphological candidates. The rule-based decomposer proposes possible analyses; the model chooses which candidate is most plausible.

The model does not read raw Turkish text directly. It receives only encoded suffix chains and closed-class markers.

```text
word/sentence candidate
  -> per-word suffix or closed-class chains
  -> flattened sentence sequence
  -> Transformer encoder
  -> scalar rank score
```

Candidate `0` is always treated as gold during training and validation.

## What the Model Sees

The model input is built from encoded chains created in `app/nlp_pipeline.py`.

Each normal suffix token is encoded as:

```python
(token_id, group_id, position_in_word)
```

Meaning:

- `token_id`: suffix identity, starting at ID `5`.
- `group_id`: suffix group ID from `SuffixGroup`, or `0` for missing/special.
- `position_in_word`: one-based position inside the current word's suffix chain.

Closed-class words are encoded as one marker token:

- token IDs come after the suffix inventory,
- group feature is `0`,
- position still shows that it is the first token of a word-level chain.

Bare-root candidates have an empty chain:

```python
[]
```

When converted into a sentence sequence, a bare-root word contributes only a word separator. There is no learned root token.

## Special Token IDs

From `ml/ml_ranking_model.py`:

```text
0 = PAD
1 = WORD_SEP
2 = BOS
3 = MASK
4 = EOS
5 = first suffix token
```

Current inventory sizes in this repository:

- `98` suffix tokens from `util.decomposer.ALL_SUFFIXES`
- `88` closed-class tokens from `CLOSED_CLASS_TOKEN_SPECS`
- total model vocabulary size: `5 + suffix_count + closed_class_count`

These counts are dynamic at runtime. They change if suffix or closed-class inventories change.

## Sequence Flattening

`build_sentence_sequence(word_chains)` converts a list of per-word chains into three parallel integer lists.

Example shape:

```text
word_chains = [
  [plural_ler, ablative_den],
  [pasttense_di, conjugation_1sg],
]
```

Before adding the terminal sentence marker, the word-level stream is:

```text
[BOS, plural_ler, ablative_den, WORD_SEP, pasttense_di, conjugation_1sg, WORD_SEP]
```

With the explicit sentence terminator included, the actual stream is:

```text
[BOS, plural_ler, ablative_den, WORD_SEP, pasttense_di, conjugation_1sg, WORD_SEP, EOS]
```

Every feature stream has the same length:

```python
(
  suffix_ids,
  group_ids,
  word_pos_ids,
)
```

For each word:

1. Append all suffix or closed-class tokens in that word.
2. Append `WORD_SEP`.
3. After all words are processed, prepend `BOS`.
4. Append `EOS`.
5. The `group_ids` and `word_pos_ids` values for `BOS`, `WORD_SEP`, `EOS`, and padding are all the special feature value `0`.

For a bare-root word:

```text
[] -> WORD_SEP
```

So before adding the terminal marker, a one-word bare-root candidate is:

```text
[BOS, WORD_SEP]
```

With `EOS`, the full model sequence is:

```text
[BOS, WORD_SEP, EOS]
```

There is still no root identity in that sequence.

## What Is Not Encoded

The following are not model inputs:

- root string,
- lemma ID,
- dictionary row ID,
- raw word characters,
- byte/character/subword embeddings,
- surface suffix spelling such as `ler`, `lar`, `den`, `ten`,
- displayed formation strings,
- original punctuation,
- capitalization,
- apostrophe boundaries,
- external POS tags,
- syntactic parse information.

Roots affect the model only indirectly:

1. The decomposer uses roots to decide which candidate suffix chains exist.
2. Logs store roots so a future relearn can match the logged gold analysis back to a generated decomposition.
3. If two different roots create exactly the same suffix-token sequence, the ML model cannot distinguish them from the encoded sequence alone.

## Model Architecture

`SentenceDisambiguator` is a Transformer encoder over flattened morphology-token sequences.

Default config in `ml/config.py`:

- `embed_dim = 384`
- `num_layers = 6`
- `num_heads = 8`
- `dropout = 0.3`
- `max_sequence_length = 512`
- `group_embed_dim = 8`
- `wordpos_embed_dim = 16`
- `pos_embed_dim = 384`

For every sequence position, the model concatenates:

1. suffix/closed-class/special token embedding,
2. suffix-group embedding,
3. within-word-position embedding,
4. absolute position embedding.

With current defaults, the concatenated feature width is:

```text
384 token
+ 8 group
+ 16 word position
+ 384 absolute position
= 792
```

This goes through:

```text
Linear(792 -> 512)
GELU
Linear(512 -> 384)
```

Then:

```text
6-layer TransformerEncoder
```

The module has two heads:

- `rank_head`: `LayerNorm(384) -> Linear(384 -> 1)`
- `lm_head`: tied to `suffix_embed.weight`

`rank_head` is used for ranking and inference. `lm_head` is used by the auxiliary masked-token objective during training.

## Ranking Score

`rank_scores()`:

1. Embeds all feature streams.
2. Projects them to `embed_dim`.
3. Runs the Transformer.
4. Mean-pools sequence states.
   - If a pad mask exists, padded positions are excluded.
5. Applies `rank_head`.
6. Returns one scalar score per candidate sequence.

Higher score means "more preferred".

The score is not a probability by itself. During training and validation, scores inside a candidate set are converted into logits with temperature:

```python
logits = scores / config.ranking_temperature
```

`ranking_temperature` currently defaults to `0.1`.

## Candidate Sets

Training data is organized as candidate sets:

```python
[
  gold_sequence,
  wrong_sequence_1,
  wrong_sequence_2,
  ...
]
```

Candidate `0` must be gold. The model is trained with cross-entropy target `0`.

Single-word training uses one word's selected chain as gold and other decomposer candidates for that word as negatives.

Sentence training uses the selected chain for every word as gold. Negatives are made by replacing one word at a time with a wrong candidate while keeping the rest of the sentence gold.

The full Cartesian product of all word candidates is not built.

## Interactive Training Step

`Trainer.train_sentence(word_chains, negative_word_chains)`:

1. Converts the gold word chains into a flat sequence.
2. Skips the item if sequence length is less than `2`.
3. Skips the item if sequence length exceeds `config.max_sequence_length`.
4. Converts each negative word-chain list into a flat sequence.
5. Drops duplicate negatives equal to the gold sequence.
6. Drops negatives that exceed `max_sequence_length`.
7. If fewer than two sequences remain, returns `0.0`.
8. Runs `_ranking_step([candidate_set])` `config.steps_per_update` times.
   - Current default: `steps_per_update = 4`.
9. Appends the final loss to `train_history`.

## Bulk Training

`Trainer.train_bulk(all_seqs)` is used by relearn, curriculum, test folds, and similar batch flows.

Flow:

1. Normalize each item into a candidate set.
2. Keep only candidate sets with at least two sequences.
3. Split oversized candidate sets if they exceed adaptive CUDA budget limits.
4. Optionally increase batch size based on current GPU memory.
5. Build a bulk learning-rate schedule matched to total planned steps.
6. Shuffle candidate sets each epoch.
7. Build adaptive batches constrained by:
   - number of candidate sequences,
   - padded token count,
   - approximate attention-cell count.
8. Run `_ranking_step()` for each batch.
9. Validate after each epoch when validation sets are provided.

Default bulk values:

- `bulk_epochs = 5`
- `bulk_batch_size = 128`
- `max_sequence_length = 512`
- `max_candidate_sequences_per_batch = 64`
- `max_batch_padded_tokens = 8192`
- `max_batch_attention_cells = 2_000_000`

On CUDA out-of-memory, the trainer:

1. clears gradients and CUDA cache,
2. shrinks adaptive limits,
3. retries with smaller batches,
4. splits candidate sets if needed,
5. eventually skips only the item that cannot fit even as gold-vs-one-negative.

## Joint Training Loss

Each optimizer step computes two losses.

### 1. Candidate Ranking Loss

For each candidate set in the batch:

```python
group = scores[offset:offset + size]
logits = (group / ranking_temperature).unsqueeze(0)
target = tensor([0])
rank_loss = cross_entropy(logits, target)
```

The batch rank loss is the mean over candidate sets.

This is the primary disambiguation loss.

### 2. Auxiliary Masked-Token Loss

The code also trains a masked-token objective on gold sequences only.

Eligible tokens are:

- not `PAD`,
- not `WORD_SEP`,
- not `BOS`,
- not `EOS`,
- not padding by mask.

With current defaults:

- `mlm_mask_prob = 0.20`
- `mlm_ensure_one_mask = True`
- `mlm_use_bert_mix = True`
- `focal_gamma = 0.0`

Masking behavior:

1. Randomly select eligible tokens with probability `0.20`.
2. If a sequence has eligible tokens but none were selected, force one selected token.
3. For selected tokens:
   - 80% become `MASK`,
   - 10% become a random suffix/closed-class vocabulary token,
   - 10% stay unchanged.
4. The target is the original token ID at selected positions.
5. Unselected positions are set to `PAD` in the target and ignored by cross-entropy.

`focal_gamma` is currently `0.0`, so this is ordinary cross-entropy over selected tokens.

### Final Loss

```python
total_loss = rank_loss + config.mlm_weight * mlm_loss
```

Current default:

```text
mlm_weight = 0.2
```

The printed `loss` is this joint loss. The printed `rank_loss` is only the ranking component.

Validation loss is ranking loss only; validation does not add the MLM loss.

## Optimizer and Schedule

Trainer defaults:

- optimizer: `AdamW`
- `learning_rate = 3e-4`
- `weight_decay = 0.05`
- betas: `(0.9, 0.999)`
- gradient clipping: max norm `1.0`
- AMP enabled only on CUDA

Interactive fallback schedule:

1. Warm up for `config.warmup_steps`.
2. Cosine decay over `warmup_steps * 50`.
3. Floor at `config.lr_eta_min_ratio`.

Bulk training replaces the scheduler with a schedule matched to the planned number of bulk steps.

## Negative Generation

### Static Negatives

`WorkflowEngine._single_substitution_negatives()` builds negatives.

For every word position:

1. Iterate that word's candidates.
2. Skip the gold candidate.
3. Copy the gold sentence chains.
4. Replace only that one word with the wrong candidate.
5. Deduplicate by token-ID signature.
6. Stop at the configured limit.

Default limit:

```text
max_negative_candidates = 8
```

### Dynamic Curriculum Negatives

Curriculum first builds a wider pool:

```text
dynamic_negative_pool_size = 100
```

Then it scores that pool with the current model and selects:

- `hard_negative_count = 4`: highest-scoring wrong candidates,
- `easy_negative_count = 2`: lowest-scoring wrong candidates,
- `medium_negative_count = 2`: sampled from the remaining middle pool.

The selection order in code is hard first, easy second, medium third, then fill remaining slots from the ranked list if needed.

## Scoring and Inference

### Isolated Candidate Scoring

`Trainer.predict(candidates, context_chains=None)` calls `score_candidates()`.

In the current engine, isolated word prediction passes no context. It scores each candidate as:

```text
BOS + candidate_chain + WORD_SEP + EOS
```

Bare-root candidates receive:

```python
config.bare_root_prior_logprob
```

Current value:

```text
bare_root_prior_logprob = -0.75
```

This is added after the neural score. It is not learned.

### Sentence Scoring

`score_sentence_chains(word_chains)`:

1. Builds a full sentence sequence from all word chains.
2. Counts empty chains.
3. Adds `bare_root_prior_logprob` once for every empty chain.
4. Returns neural rank score plus that prior.

Sentence beam search repeatedly scores partial sentence chains and keeps the best beams.

### Context Parameters

`score_candidates()` can accept `context_chains` on the left and `right_chains` on the right. The current `predict()` wrapper exposes only `context_chains`, and the engine's isolated word scoring passes none. Full sentence scoring is done by scoring whole sentence sequences instead.

## Validation Metrics

`Trainer.validate(val_seqs)` evaluates candidate sets with at least two candidates.

For every candidate set:

1. Score all candidates.
2. Compute ranking CE loss with the same temperature as training.
3. Choose the highest-scoring candidate as prediction.
4. Compare prediction to candidate `0`.

Metrics:

- `rank_acc`: gold candidate is ranked first.
- `top2_acc`: gold candidate is in the top 2.
- `top3_acc`: gold candidate is in the top 3.
- `margin`: `score(gold) - max(score(negative))`.
- `suff_acc`: average token-position match rate between gold and predicted morphology tokens.
- `word_acc`: exact match of gold morphology-token sequence vs predicted morphology-token sequence.
- `suff_precision`, `suff_recall`, `suff_f1`: micro suffix-token precision/recall/F1.
- `suffix_metrics`: per-suffix precision/recall/F1 and counts.
- `suffix_group_metrics`: suffix metrics aggregated by suffix group.

`suff_acc` and `word_acc` compare morphology tokens after removing `PAD`, `WORD_SEP`, `BOS`, and `EOS`. Closed-class tokens can therefore affect those sequence-level metrics.

Per-suffix buckets use `_suffix_name_for_token_id()`, which maps only normal suffix IDs. Closed-class IDs do not become per-suffix names.

## Checkpointing

`save_checkpoint()` writes:

- model state,
- optimizer state,
- scheduler state,
- training history,
- validation history,
- best validation loss,
- global step,
- suffix inventory,
- feature schema version.

`load_checkpoint()`:

1. Loads the checkpoint on the resolved device.
2. Compares saved suffix inventory with the current suffix inventory.
3. Loads only model tensors whose names exist and shapes match.
4. Loads optimizer and scheduler state only if suffix inventory and feature schema both match.
5. Restores histories and global step.

If the suffix inventory or feature schema changed, optimizer state is discarded.

## Current Non-Features and Caveats

- Roots are not included in ML input.
- Surface characters are not included in ML input.
- Surface suffix allomorphs are not included in ML input.
- The model cannot distinguish two candidates that differ only by root if their encoded suffix/closed-class streams are identical.
- The previous coarse category/output-type streams are no longer encoded; final/output-type information is left to `token_id`, `group_id`, sequence position, `WORD_SEP`, and `EOS`.
- Direct log encoding does not fail on an unknown suffix name; it maps that name to `SUFFIX_OFFSET`, the first suffix-token slot.
- Full Cartesian sentence candidate products are not generated for training.
- Validation loss does not include the MLM auxiliary objective.
- Per-suffix metrics do not give named buckets for closed-class token IDs.
- Isolated word scoring does not use right context and currently receives no left context from the engine.
- Bulk relearn only trains entries that can become candidate sets with at least one negative.
