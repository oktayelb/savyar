# Workflows: Orchestration Layer

## Overview

`WorkflowEngine` (`app/workflows.py`) connects the decomposer, morphology adapter, ranking model, data files, and CLI. The decomposer generates candidates; the user/treebank identifies the gold candidate; the trainer learns to rank that gold candidate above decomposer-generated negatives.

## Initialization

Startup creates:

1. `DataManager` for logs, dictionary files, and sample files.
2. `SentenceDisambiguator` with suffix and closed-class vocabulary sizes.
3. `Trainer`, which owns optimizer, scheduler, checkpointing, and ranking updates.
4. A replay buffer rebuilt from logged decompositions when the checkpoint does not already contain one.

The replay buffer stores gold flat sequences for continuity and checkpoint compatibility. The active learning signal is the candidate-ranking loss.

## Word Flow

### Analyze

`analyze_word()`:

1. Sanitizes the word before it reaches the engine.
2. Calls `decompose_with_cc()` through `app.analyzer`.
3. Encodes every candidate suffix chain.
4. Scores candidates with the ranking head when the model has training history.
5. Returns aligned decompositions, encoded chains, view models, and typing strings.

### Commit

`commit_word()`:

1. Logs the selected decomposition.
2. Performs dictionary cleanup when a derived surface form should not remain as a standalone dictionary word.
3. Builds a ranking set:
   - gold: selected candidate
   - negatives: other decomposer candidates for that word
4. Calls `trainer.train_sentence([gold], negative_word_chains=negatives)`.

## Sentence Flow

### Analyze

`analyze_sentence()` analyzes each token and returns candidate lists for every word. It does not decide the sentence; sentence-level ranking happens later.

### Match User Target

`evaluate_sentence_target()` delegates to `find_matching_combinations()`. The matcher walks legal candidate combinations and keeps branches that match the user-entered decomposition string prefix. Matching combinations are scored by `trainer.score_sentence_chains()`.

### Commit

`commit_sentence_training()`:

1. Extracts the selected candidate for every word.
2. Logs the sentence-level gold analysis.
3. Generates negatives with `_single_substitution_negatives()`:
   - keep all gold word analyses
   - replace one word with one wrong candidate
   - cap the list with `config.max_negative_candidates`
4. Calls `trainer.train_sentence(gold_chains, negative_word_chains=negatives)`.

This gives many training comparisons from one sentence while avoiding full Cartesian-product explosion.

## Relearn

`relearn_all()` performs full bulk training from logs and adapted treebanks:

1. `DataManager.get_valid_decomps()` loads sentence/user logs plus adapted treebank JSONL files.
2. `_entries_to_sequences()` converts each entry into a ranking candidate set.
3. For each logged word with suffixes:
   - run the decomposer again
   - match the logged suffix chain to a generated candidate
   - use the match as the gold chain
   - use other generated candidates as possible negatives
4. If matching fails, encode the logged suffix names directly as a gold-only fallback; that entry is skipped for ranking unless negatives can be generated.
5. `trainer.train_bulk()` trains on candidate sets where candidate `0` is gold.

Bulk metrics are ranking metrics:

- `RankAcc`: how often gold is scored highest within its candidate set
- `margin`: gold score minus best negative score
- `loss`: cross-entropy over each candidate set

## Curriculum

`train_curriculum()` performs dynamic hard-negative training from the same logged/user/treebank entries used by `relearn`.

The important difference is when negatives are chosen:

- `relearn` chooses negatives from decomposer candidates before training and then trains on that static set.
- `curriculum` periodically asks the current model to score a wider negative pool and rebuilds the next training set from the wrong analyses the model currently finds confusing.

### High-Level Flow

1. Load all valid gold entries with `DataManager.get_valid_decomps()`.
2. Load the external validation set, or create a deterministic train/validation split.
3. Run a static warm-up phase with `_entries_to_sequences()` when `curriculum_warmup_epochs` is greater than zero.
4. For each curriculum generation:
   - call `_entries_to_dynamic_sequences()`
   - convert each entry into gold chains and decomposer candidate lists
   - create a wide negative pool with `_single_substitution_negatives(..., limit=config.dynamic_negative_pool_size)`
   - score the pool with `trainer.score_flat_sequences()`
   - select hard/medium/easy negatives with `_select_dynamic_negatives()`
   - train the mined sets with `trainer.train_bulk(..., epochs=config.curriculum_mining_epochs)`
   - save `ml/model.pt`

### What Counts as a Hard Negative

A hard negative is an incorrect candidate that receives a high score from the current model. For example, if the gold chain contains a locative suffix but the model strongly prefers a similar ablative analysis, that ablative candidate becomes a high-value training example.

The model is therefore tested against its current weaknesses before each curriculum generation. The next training batch is not a fixed worksheet; it adapts to the model's latest mistakes.

### Difficulty Mix

Curriculum mining intentionally mixes difficulty levels:

- hard negatives: highest-scoring wrong analyses
- medium negatives: sampled from the middle of the ranked wrong list
- easy negatives: low-scoring wrong analyses

This prevents the model from seeing only difficult edge cases. Easy negatives keep basic distinctions anchored, while hard negatives push the decision boundary.

The defaults are in `ml/config.py`:

- `max_negative_candidates = 10`
- `hard_negative_count = 6`
- `medium_negative_count = 2`
- `easy_negative_count = 2`
- `dynamic_negative_pool_size = 100`
- `curriculum_generations = 3`
- `curriculum_warmup_epochs = 5`
- `curriculum_mining_epochs = 4`

### Commands

```text
curriculum
```

Curriculum mode updates the model checkpoint. It does not currently write the mined hard examples to a separate report file.

## Evaluation and Sampling

`evaluate_word()` ranks candidates for one word and returns the top view model.

`sample_text()` processes unique words from a text file and picks the top-ranked candidate for ambiguous words.

`sample_sentences()` uses `get_top_sentence_predictions()` beam search. At each word, the beam is expanded by candidate analyses, complete partial sequences are scored by the ranking head, and only the best beams are retained.

## Data Flow

```
surface word/sentence
        |
        v
rule-based decomposer
        |
        v
candidate suffix chains
        |
        v
morphology adapter encodes chains
        |
        v
ranking model scores candidates
        |
        v
user/treebank gold choice
        |
        v
gold vs generated negatives training
```

## Caching

- `decomp_cache`: workflow-level word-to-decompositions cache.
- `decompose()` LRU cache: decomposer-level cache reused across calls.

Both remain useful because ranking training repeatedly asks the decomposer for candidate sets when building negatives.
