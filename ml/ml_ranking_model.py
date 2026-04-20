import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .config import config  # Direct import from sibling file

# Enable cuDNN auto-tuner — finds fastest convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True

# ============================================================================
# SPECIAL TOKENS
# ============================================================================
#
# Token ID 0 is reserved as padding (also used by Embedding padding_idx).
# The vocabulary is laid out as:
#   [0]            → PAD
#   [1]            → WORD_SEP  (boundary between words in a sentence)
#   [2]            → BOS       (beginning-of-sequence, prepended to every input)
#   [3 .. V+2]     → suffix IDs  (suffix_idx + 3, where suffix_idx is 0-based)
#   [V+3 .. ]      → closed-class word IDs
#
# Category IDs:
#   0 → Noun, 1 → Verb, 2 → SPECIAL (PAD / WORD_SEP / BOS), 3 → ClosedClass

SPECIAL_PAD           = 0
SPECIAL_WORD_SEP      = 1
SPECIAL_BOS           = 2          # beginning-of-sequence
SUFFIX_OFFSET         = 3          # suffix IDs start here
CATEGORY_SPECIAL      = 2          # category ID for PAD / WORD_SEP / BOS
CATEGORY_CLOSED_CLASS = 3          # category ID for closed-class word tokens

# CLOSED_CLASS_OFFSET is computed at runtime (SUFFIX_OFFSET + len(ALL_SUFFIXES))
# and passed in to SentenceDisambiguator as closed_class_vocab_size.


# ============================================================================
# PER-CATEGORY ACCURACY BUCKETS
# ============================================================================
# Human-meaningful suffix categories for the relearn diagnostic breakdown.
# Any suffix not explicitly listed falls into "other".
SUFFIX_CATEGORIES: List[str] = [
    "plural", "poss", "case", "conj", "copula",
    "gerund", "infin", "deriv", "other",
]
_CAT_NAME_TO_IDX = {c: i for i, c in enumerate(SUFFIX_CATEGORIES)}

_CATEGORY_TENSOR_CACHE: Dict[str, torch.Tensor] = {}


def _build_suffix_category_tensor(vocab_size: int, device: torch.device) -> torch.Tensor:
    """Return a (vocab_size,) long tensor mapping token-id → category index.
    Non-suffix tokens (PAD/BOS/WORD_SEP/CC) get -1."""
    cache_key = f"{vocab_size}:{device}"
    cached = _CATEGORY_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from util.suffixes.n2n.case_suffixes       import CASESUFFIX
    from util.suffixes.n2n.posessive_suffix    import POSESSIVE_SUFFIX
    from util.suffixes.n2n.plural_suffix       import PLURALS
    from util.suffixes.n2n.derivationals       import DERIVATIONALS as N2N_DERIVATIONALS
    from util.suffixes.n2n.conjugation_suffixes import CONJUGATIONS
    from util.suffixes.n2n.copula              import COPULA
    from util.suffixes.v2n.gerunds             import GERUNDS
    from util.suffixes.v2n.infinitives         import INFINITIVES
    from util.suffixes.v2n.nounifiers          import NOUNIFIERS
    from util.suffixes.n2v.verbifiers          import VERBIFIERS
    from util.suffixes.v2v.verb_derivationals  import VERB_DERIVATIONALS
    from util.suffixes.v2v.verb_negative       import VERB_NEGATIVES
    from util.suffixes.v2v.verb_compounds      import VERB_COMPOUNDS

    buckets = [
        ("plural",  PLURALS),
        ("poss",    POSESSIVE_SUFFIX),
        ("case",    CASESUFFIX),
        ("conj",    CONJUGATIONS),
        ("copula",  COPULA),
        ("gerund",  GERUNDS),
        ("infin",   INFINITIVES),
        ("deriv",   N2N_DERIVATIONALS + VERBIFIERS + NOUNIFIERS
                    + VERB_DERIVATIONALS + VERB_NEGATIVES + VERB_COMPOUNDS),
    ]
    name_to_cat_idx: Dict[str, int] = {}
    for cat, lst in buckets:
        idx = _CAT_NAME_TO_IDX[cat]
        for s in lst:
            name_to_cat_idx[s.name] = idx

    other_idx = _CAT_NAME_TO_IDX["other"]
    tensor = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
    all_sufs = _get_all_suffixes()
    for i, s in enumerate(all_sufs):
        tok_id = i + SUFFIX_OFFSET
        if tok_id >= vocab_size:
            break
        tensor[tok_id] = name_to_cat_idx.get(s.name, other_idx)

    _CATEGORY_TENSOR_CACHE[cache_key] = tensor
    return tensor


# ============================================================================
# HELPER: encode / decode sentence-level token sequences
# ============================================================================

def encode_chain(suffix_chain) -> List[Tuple[int, int]]:
    """
    Convert a List[Suffix] → List[(suffix_token_id, category_id)].
    An empty chain (bare root) returns an empty list; the caller must
    still emit a WORD_SEP token.
    """
    from ml.ml_ranking_model import SUFFIX_OFFSET, CATEGORY_SPECIAL  # avoid circular at module level

    suffix_to_id = {
        suffix.name: idx + SUFFIX_OFFSET
        for idx, suffix in enumerate(_get_all_suffixes())
    }
    category_to_id = {'Noun': 0, 'Verb': 1}

    encoded = []
    for s in suffix_chain:
        sid  = suffix_to_id.get(s.name, SUFFIX_OFFSET)  # unknown → first real suffix
        cid  = category_to_id.get(s.makes.name, 0)
        encoded.append((sid, cid))
    return encoded


def _get_all_suffixes():
    """Lazy import to avoid circular deps."""
    import util.decomposer as sfx
    return sfx.ALL_SUFFIXES


def _chain_tokens(
    word_chains: List[List[Tuple[int, int]]]
) -> Tuple[List[int], List[int]]:
    """
    Raw per-word tokens (suffixes + trailing WORD_SEP per word). No BOS.
    Used when concatenating fragments (ctx + candidate + right) in scoring.
    """
    suffix_ids:   List[int] = []
    category_ids: List[int] = []
    for chain in word_chains:
        for (sid, cid) in chain:
            suffix_ids.append(sid)
            category_ids.append(cid)
        suffix_ids.append(SPECIAL_WORD_SEP)
        category_ids.append(CATEGORY_SPECIAL)
    return suffix_ids, category_ids


def build_sentence_sequence(
    word_chains: List[List[Tuple[int, int]]]
) -> Tuple[List[int], List[int]]:
    """
    Full trainable sequence: BOS prefix + raw chain tokens.

    Layout for a 2-word sentence  [BOS | w1_suf1, w1_suf2, SEP | w2_suf1, SEP]:
        suffix_ids   = [BOS, w1_suf1_id, w1_suf2_id, WORD_SEP, w2_suf1_id, WORD_SEP]
        category_ids = [C_SPEC, w1_cat1, w1_cat2,    C_SPEC,   w2_cat1,    C_SPEC]

    The leading BOS gives the model a conditioning token for the very first
    suffix prediction (otherwise the first-token probability is unscorable).
    """
    s, c = _chain_tokens(word_chains)
    return [SPECIAL_BOS] + s, [CATEGORY_SPECIAL] + c


# ============================================================================
# MODEL
# ============================================================================

class SentenceDisambiguator(nn.Module):
    """
    Causal (decoder-only) Transformer that models the joint distribution
    over suffix token sequences at the sentence level.

    Trained like a language model: given a confirmed sentence decomposition
    as a flat token sequence, minimise cross-entropy of predicting each next
    token from all previous ones.

    At inference time, candidate decompositions for each word are scored by
    summing the log-probabilities assigned to their tokens in context, so
    every word's disambiguation is informed by all surrounding words.
    """

    def __init__(self, suffix_vocab_size: int, closed_class_vocab_size: int = 0):
        """
        Args:
            suffix_vocab_size: number of real suffix types (from ALL_SUFFIXES).
            closed_class_vocab_size: number of closed-class word types
                                     (from ALL_CLOSED_CLASS_WORDS). Defaults to 0
                                     for backward compatibility.
        Total token vocab layout:
            [0]                                 → PAD
            [1]                                 → WORD_SEP
            [2]                                 → BOS
            [3 .. suffix_vocab_size+2]          → suffix IDs
            [suffix_vocab_size+3 .. total-1]    → closed-class word IDs
        Category IDs:
            0 = Noun, 1 = Verb, 2 = Special (PAD/WORD_SEP/BOS), 3 = ClosedClass
        """
        super().__init__()
        self.embed_dim = config.embed_dim
        # Full token vocab: PAD + WORD_SEP + suffixes + closed-class words
        self.vocab_size = SUFFIX_OFFSET + suffix_vocab_size + closed_class_vocab_size

        # Token embeddings (shared with LM head via weight tying)
        self.suffix_embed   = nn.Embedding(self.vocab_size,          self.embed_dim, padding_idx=SPECIAL_PAD)
        # Category embedding: 0=Noun, 1=Verb, 2=Special, 3=ClosedClass  →  4 categories
        self.category_embed = nn.Embedding(4,                         self.embed_dim)
        # Positional embedding (up to 512 tokens per sentence)
        self.pos_embed      = nn.Embedding(512,                       self.embed_dim)

        # Project concatenated embeddings → model dim
        self.input_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)

        # Causal (decoder) transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)

        # Language-model head: hidden → vocab logits
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Tie weights (token embedding ↔ LM head), standard LM trick
        self.lm_head.weight = self.suffix_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'embed' not in name:
                nn.init.kaiming_normal_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i can only attend to 0..i."""
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        suffix_ids:   torch.Tensor,   # (B, L)
        category_ids: torch.Tensor,   # (B, L)
        pad_mask:     Optional[torch.Tensor] = None,  # (B, L) True = padding
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, L, vocab_size).
        Logits[b, i, :] = distribution over the token at position i+1
        given positions 0..i  (standard causal LM).
        """
        B, L = suffix_ids.shape
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)

        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.pos_embed(pos),
        ], dim=-1)                          # (B, L, embed_dim * 3)

        x = self.input_proj(x)              # (B, L, embed_dim)

        causal = self._causal_mask(L, suffix_ids.device)

        x = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)

        return self.lm_head(x)              # (B, L, vocab_size)

    def log_probs(
        self,
        suffix_ids:   torch.Tensor,   # (B, L)
        category_ids: torch.Tensor,   # (B, L)
        pad_mask:     Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Returns token-level log-probs of shape (B, L-1):
            log P(token[i] | token[0..i-1])   for i in 1..L-1.

        Used during scoring to evaluate candidate decompositions.
        """
        logits = self.forward(suffix_ids, category_ids, pad_mask=pad_mask)  # (B, L, V)
        log_p  = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
        targets = suffix_ids[:, 1:]                          # (B, L-1)
        token_log_probs = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

        if pad_mask is not None:
            # Mask out padding tokens in the target
            target_mask = pad_mask[:, 1:]
            token_log_probs.masked_fill_(target_mask, 0.0)

        return token_log_probs


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Wraps SentenceDisambiguator and handles:
      - Causal LM training on confirmed sentence decompositions
      - Context-aware candidate scoring at inference time
      - Checkpointing
    """

    def __init__(self, model: SentenceDisambiguator):
        self.model = model

        self.checkpoint_frequency = config.checkpoint_frequency
        self.path                 = str(config.model_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Mixed precision scaler (CUDA only; no-op on CPU)
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))

        # torch.compile for kernel fusion / faster execution (PyTorch 2.0+, Linux/Mac only)
        # Skipped on Windows — requires MSVC cl.exe which is rarely available.
        if not torch.cuda.is_available() or not hasattr(torch, 'compile'):
            pass  # CPU-only or old PyTorch — skip
        elif torch.version.cuda and hasattr(torch, 'compile'):
            import platform
            if platform.system() != 'Windows':
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    pass

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=config.learning_rate * 0.01
        )

        # Training state
        self.train_history: List[float] = []
        self.val_history:   List[float] = []
        self.best_val_loss  = float('inf')
        self.global_step    = 0

        # Experience replay buffer: list of (suffix_ids, category_ids) tuples.
        # Populated from confirmed examples; used to mix past data into each
        # training call so the model does not forget earlier decompositions.
        self.replay_buffer: List[Tuple[List[int], List[int]]] = []

        # Cached class-weight tensor for imbalanced cross-entropy. Invalidated
        # when the replay buffer changes (see _add_to_replay).
        self._class_weight_cache: Optional[torch.Tensor] = None

        try:
            self.load_checkpoint(self.path)
            print(f"Loaded model from {self.path}")
        except FileNotFoundError:
            print(f"Starting fresh (no checkpoint found at {self.path})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(
        self, suffix_ids: List[int], category_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert flat id lists to (1, L) tensors on device."""
        s = torch.tensor(suffix_ids,   dtype=torch.long, device=self.device).unsqueeze(0)
        c = torch.tensor(category_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return s, c

    def _get_best_index(self, scores: List[float]) -> int:
        """Pick the argmax score. With BOS prepended to every sequence,
        every candidate — including bare roots — produces a real log-prob,
        so no sentinel filtering is needed."""
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def _compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, float, float, float, Dict[str, Tuple[float, int]]]:
        """
        Suffix-level accuracy + macro-averaged P/R/F1, plus a separate
        word-boundary accuracy for diagnostic transparency.

        The caller is expected to have already filtered PAD tokens. This
        function additionally isolates real suffix tokens from the special
        WORD_SEP / BOS tokens before computing the headline metrics, because
        WORD_SEP is trivially predictable (every word ends with one) and
        would otherwise mask poor suffix-level performance.

        Returns: (suffix_acc, macro_p, macro_r, macro_f1, wordsep_acc, per_cat)
        where per_cat maps category name -> (accuracy, token_count).
        """
        empty_per_cat: Dict[str, Tuple[float, int]] = {c: (0.0, 0) for c in SUFFIX_CATEGORIES}
        if len(targets) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, empty_per_cat

        # Isolate the "real" suffix predictions (strip WORD_SEP and BOS).
        is_special = (targets == SPECIAL_WORD_SEP) | (targets == SPECIAL_BOS)
        suffix_mask = ~is_special

        suffix_preds   = preds[suffix_mask]
        suffix_targets = targets[suffix_mask]

        # Word-boundary accuracy, reported separately for visibility.
        sep_targets_mask = (targets == SPECIAL_WORD_SEP)
        if sep_targets_mask.any():
            wordsep_acc = (preds[sep_targets_mask] == SPECIAL_WORD_SEP).float().mean().item()
        else:
            wordsep_acc = 0.0

        if len(suffix_targets) == 0:
            return 0.0, 0.0, 0.0, 0.0, wordsep_acc, empty_per_cat

        suffix_acc = (suffix_preds == suffix_targets).float().mean().item()

        # Per-category breakdown (bucket each real suffix target by semantic group).
        cat_tensor = _build_suffix_category_tensor(self.model.vocab_size, suffix_targets.device)
        cat_idx    = cat_tensor[suffix_targets]          # (-1 for anything unclassified)
        valid_cat  = cat_idx >= 0
        per_cat: Dict[str, Tuple[float, int]] = {}
        if valid_cat.any():
            cidx_v  = cat_idx[valid_cat]
            correct = (suffix_preds[valid_cat] == suffix_targets[valid_cat]).float()
            n_cats  = len(SUFFIX_CATEGORIES)
            totals   = torch.bincount(cidx_v, minlength=n_cats).float()
            corrects = torch.bincount(cidx_v, weights=correct, minlength=n_cats)
            for i, name in enumerate(SUFFIX_CATEGORIES):
                tot = int(totals[i].item())
                acc = (corrects[i] / totals[i]).item() if tot > 0 else 0.0
                per_cat[name] = (acc, tot)
        else:
            per_cat = empty_per_cat

        num_classes = self.model.vocab_size
        tps_mask      = (suffix_preds == suffix_targets)
        tps           = torch.bincount(suffix_targets[tps_mask], minlength=num_classes).float()
        pred_counts   = torch.bincount(suffix_preds,             minlength=num_classes).float()
        target_counts = torch.bincount(suffix_targets,           minlength=num_classes).float()

        precision = tps / (pred_counts + 1e-9)
        recall    = tps / (target_counts + 1e-9)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-9)

        valid_classes = target_counts > 0
        if not valid_classes.any():
            return suffix_acc, 0.0, 0.0, 0.0, wordsep_acc, per_cat

        macro_p  = precision[valid_classes].mean().item()
        macro_r  = recall[valid_classes].mean().item()
        macro_f1 = f1[valid_classes].mean().item()

        return suffix_acc, macro_p, macro_r, macro_f1, wordsep_acc, per_cat

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Experience replay helpers
    # ------------------------------------------------------------------

    def _add_to_replay(self, suffix_ids: List[int], category_ids: List[int]) -> None:
        """Add a confirmed sequence to the replay buffer, evicting old entries if full."""
        self.replay_buffer.append((suffix_ids, category_ids))
        if len(self.replay_buffer) > config.replay_buffer_size:
            # Evict a random entry from the first half to keep a mix of old and recent.
            evict_idx = random.randrange(len(self.replay_buffer) // 2)
            self.replay_buffer.pop(evict_idx)
        # Class-frequency stats are now stale.
        self._class_weight_cache = None

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """Inverse-sqrt-frequency class weights computed from the replay buffer.

        Weights for classes that appear are rescaled so their mean is 1.0, which
        keeps the effective learning rate on non-rare classes roughly unchanged.
        Classes that never appear (including PAD) get weight 1.0; cross_entropy's
        ignore_index still filters PAD out of the loss entirely.
        Cached on the trainer; invalidated whenever the replay buffer changes.
        """
        if not config.use_class_weights:
            return None
        if self._class_weight_cache is not None:
            return self._class_weight_cache
        if not self.replay_buffer:
            return None

        V = self.model.vocab_size
        counts = torch.zeros(V, dtype=torch.float, device=self.device)
        for sids, _ in self.replay_buffer:
            if not sids:
                continue
            ids = torch.as_tensor(sids, dtype=torch.long, device=self.device)
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float))

        weights = torch.ones(V, dtype=torch.float, device=self.device)
        present = counts > 0
        if present.any():
            inv_sqrt = 1.0 / torch.sqrt(counts[present])
            # Normalize mean -> 1.0 across present classes.
            inv_sqrt = inv_sqrt * (inv_sqrt.numel() / inv_sqrt.sum())
            weights[present] = inv_sqrt

        weights[SPECIAL_PAD] = 1.0  # ignore_index skips PAD anyway; keep sane.
        self._class_weight_cache = weights
        return weights

    def _build_padded_batch(
        self, seqs: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad a list of (suffix_ids, category_ids) sequences to the same length.
        Returns (suffix_tensor, category_tensor, pad_mask) each of shape (B, L).
        pad_mask is True where the position is padding.
        Tensors are built on CPU then transferred in a single .to(device) call.
        """
        max_len = max(len(s) for s, _ in seqs)
        bsz = len(seqs)

        pin = self.device == 'cuda'
        s_t    = torch.full((bsz, max_len), SPECIAL_PAD,      dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_PAD,      dtype=torch.long)
        c_t    = torch.full((bsz, max_len), CATEGORY_SPECIAL, dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), CATEGORY_SPECIAL, dtype=torch.long)
        p_mask = torch.ones((bsz, max_len), dtype=torch.bool)

        for i, (sids, cids) in enumerate(seqs):
            L = len(sids)
            s_t[i, :L]    = torch.tensor(sids, dtype=torch.long)
            c_t[i, :L]    = torch.tensor(cids, dtype=torch.long)
            p_mask[i, :L] = False

        non_blocking = self.device == 'cuda'
        return (
            s_t.to(self.device, non_blocking=non_blocking),
            c_t.to(self.device, non_blocking=non_blocking),
            p_mask.to(self.device, non_blocking=non_blocking),
        )

    def _gradient_steps(
        self, seqs: List[Tuple[List[int], List[int]]], n_steps: int
    ) -> float:
        """
        Run `n_steps` gradient updates on a padded batch of sequences.
        Returns the loss from the final step.
        """
        s_t, c_t, p_mask = self._build_padded_batch(seqs)

        # Causal LM: predict token[i] from tokens[0..i-1]
        input_s  = s_t[:, :-1]
        input_c  = c_t[:, :-1]
        target   = s_t[:, 1:]
        in_mask  = p_mask[:, :-1]   # padding mask for the input side

        self.model.train()
        final_loss = 0.0
        use_amp = self.device == 'cuda'
        class_weights = self._compute_class_weights()

        for _ in range(n_steps):
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = self.model(input_s, input_c, pad_mask=in_mask)   # (B, L-1, V)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    weight=class_weights,
                    ignore_index=SPECIAL_PAD,
                )
            final_loss = loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

        return final_loss

    # ------------------------------------------------------------------
    # Training (public API)
    # ------------------------------------------------------------------

    def train_sentence(
        self,
        word_chains: List[List[Tuple[int, int]]],
        max_retries: int = None,   # kept for call-site compatibility, ignored
    ) -> float:
        """
        Train on a confirmed sentence (or single-word) decomposition.

        Strategy — experience replay:
          1. Encode the new example into a flat token sequence.
          2. Add it to the replay buffer.
          3. Sample `replay_k` past examples from the buffer.
          4. Run `steps_per_update` gradient steps on the mixed batch.

        This replaces the old "repeat 20 times on one sentence until loss < 0.05"
        loop, which caused memorisation and catastrophic forgetting.

        Args:
            word_chains: one encoded chain per word (empty chain = bare root).
        Returns:
            cross-entropy loss from the final gradient step.
        """
        suffix_ids, category_ids = build_sentence_sequence(word_chains)

        # Sequences with only a WORD_SEP token carry no learnable signal.
        if len(suffix_ids) < 2:
            return 0.0

        # 1. Add new example to replay buffer.
        self._add_to_replay(suffix_ids, category_ids)

        # 2. Sample past examples.
        batch: List[Tuple[List[int], List[int]]] = [(suffix_ids, category_ids)]
        if len(self.replay_buffer) > 1:
            k = min(config.replay_k, len(self.replay_buffer) - 1)
            others = [x for x in self.replay_buffer if x is not batch[0]]
            batch.extend(random.sample(others, k))

        # 3. Fixed gradient steps on mixed batch.
        print(f"   Training on {len(batch)} examples...", end="", flush=True)
        final_loss = self._gradient_steps(batch, config.steps_per_update)
        print(f" loss={final_loss:.4f}")

        self.scheduler.step()
        self.train_history.append(final_loss)
        return final_loss

    def train_bulk(
        self,
        all_seqs: List[Tuple[List[int], List[int]]],
        batch_size: int = 128,
        epochs: int = 60,
    ) -> float:
        """Train on a large pre-collected dataset in proper epoch-based batches.

        Used by relearn_all to avoid the overhead of per-sentence replay sampling.
        All sequences are added to the replay buffer first, then trained in
        shuffled mini-batches for `epochs` passes.

        Returns the average loss of the final epoch.
        """
        if not all_seqs:
            return 0.0

        # Populate replay buffer with all sequences
        for sids, cids in all_seqs:
            self._add_to_replay(sids, cids)

        use_amp = self.device == 'cuda'
        final_loss = 0.0
        data = list(all_seqs)

        class_weights = self._compute_class_weights()

        for epoch in range(epochs):
            random.shuffle(data)
            epoch_loss = 0.0
            n_batches = 0
            
            # Lists to store epoch predictions and targets for metric calculation
            all_epoch_preds = []
            all_epoch_targs = []

            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                s_t, c_t, p_mask = self._build_padded_batch(batch)
                input_s  = s_t[:, :-1]
                input_c  = c_t[:, :-1]
                target   = s_t[:, 1:]
                in_mask  = p_mask[:, :-1]

                self.model.train()
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = self.model(input_s, input_c, pad_mask=in_mask)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target.reshape(-1),
                        weight=class_weights,
                        ignore_index=SPECIAL_PAD,
                    )
                
                # Extract predictions for metrics without affecting computation graph.
                # Keep WORD_SEP and BOS in the tensors here; _compute_metrics splits
                # them out into suffix-level and word-boundary accuracies separately.
                with torch.no_grad():
                    preds = logits.argmax(dim=-1).reshape(-1)
                    targs = target.reshape(-1)
                    valid_mask = targs != SPECIAL_PAD
                    all_epoch_preds.append(preds[valid_mask].cpu())
                    all_epoch_targs.append(targs[valid_mask].cpu())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches:
                avg = epoch_loss / n_batches
                final_loss = avg
                
                # Calculate metrics for the epoch
                if all_epoch_targs:
                    epoch_preds_cat = torch.cat(all_epoch_preds)
                    epoch_targs_cat = torch.cat(all_epoch_targs)
                    suf_acc, prec, rec, f1, sep_acc, per_cat = self._compute_metrics(
                        epoch_preds_cat, epoch_targs_cat
                    )
                    header = (
                        f"   Bulk epoch {epoch+1}/{epochs}: loss={avg:.4f} | "
                        f"SufAcc={suf_acc:.4f} | SepAcc={sep_acc:.4f} | "
                        f"F1={f1:.4f} P={prec:.4f} R={rec:.4f} ({n_batches} batches)"
                    )
                    cat_cells = []
                    for cat in SUFFIX_CATEGORIES:
                        acc, cnt = per_cat.get(cat, (0.0, 0))
                        if cnt == 0:
                            cat_cells.append(f"{cat:>6}:  --- (    0)")
                        else:
                            cat_cells.append(f"{cat:>6}: {acc:5.3f} ({cnt:>5})")
                    # three per row keeps it readable in an 80-col terminal
                    rows = [cat_cells[i:i+3] for i in range(0, len(cat_cells), 3)]
                    breakdown = "\n".join("      " + "  ".join(r) for r in rows)
                    print(header)
                    print(breakdown)
                else:
                    print(f"   Bulk epoch {epoch+1}/{epochs}: avg_loss={avg:.4f}  ({n_batches} batches)")
                    
            self.scheduler.step()

        self.train_history.append(final_loss)
        return final_loss

    def train_persistent(
        self,
        training_data: List[Tuple],
        max_retries: int = None,
    ) -> float:
        """Legacy wrapper — converts old-style training tuples and delegates."""
        confirmed_chains = []
        for (_, candidates, correct_idx) in training_data:
            if correct_idx < len(candidates):
                confirmed_chains.append(candidates[correct_idx])
            elif candidates:
                confirmed_chains.append(candidates[0])
            else:
                confirmed_chains.append([])
        return self.train_sentence(confirmed_chains)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_candidates(
        self,
        context_chains: List[List[Tuple[int, int]]],   # already-committed words (left context)
        candidates:     List[List[Tuple[int, int]]],   # chains to score for the current word
        right_chains:   Optional[List[List[Tuple[int, int]]]] = None,  # future words (optional)
    ) -> List[float]:
        """
        Score each candidate chain for the *current* word given sentence context.

        For each candidate we build:
            context_tokens  +  candidate_tokens  +  WORD_SEP
        and sum the log-probabilities of the candidate tokens (and their SEP)
        conditioned on the context.  Higher is better.

        Args:
            context_chains: encoded chains for words already chosen (left).
            candidates:     encoded chains to rank for the current word.
            right_chains:   encoded chains for future words (right context).
                            If provided, they are appended after the candidate
                            so the model has bidirectional context during scoring.
                            (This requires the right context to already be committed,
                            e.g. in a re-ranking pass.)
        Returns:
            List of log-prob scores, one per candidate.
        """
        self.model.eval()

        # Build the fixed prefix: BOS + raw left-context tokens.
        # Using _chain_tokens (not build_sentence_sequence) avoids a double BOS
        # when we concatenate ctx + cand + right fragments below.
        ctx_s, ctx_c     = _chain_tokens(context_chains) if context_chains else ([], [])
        right_s, right_c = _chain_tokens(right_chains)   if right_chains   else ([], [])

        prefix_s = [SPECIAL_BOS]       + ctx_s
        prefix_c = [CATEGORY_SPECIAL]  + ctx_c
        prefix_len = len(prefix_s)

        scores: List[float] = []
        with torch.no_grad():
            for chain in candidates:
                # Raw candidate tokens (no BOS): chain + WORD_SEP.
                # Bare root → cand = [WORD_SEP], cand_len == 1 (still scorable).
                cand_s, cand_c = _chain_tokens([chain])
                cand_len = len(cand_s)

                full_s = prefix_s + cand_s + right_s
                full_c = prefix_c + cand_c + right_c

                s_t, c_t = self._to_tensor(full_s, full_c)
                lp = self.model.log_probs(s_t, c_t)   # (1, L-1)

                # lp[0, i] = log P(full_s[i+1] | full_s[0..i]).
                # Candidate tokens occupy positions prefix_len .. prefix_len+cand_len-1
                # in full_s, so they are predicted by lp[0, prefix_len-1 : prefix_len-1+cand_len].
                # Because BOS is always present, prefix_len >= 1, so start >= 0 with no clamping.
                start = prefix_len - 1
                end   = start + cand_len
                scores.append(lp[0, start:end].sum().item())

        return scores

    def fast_batch_predict(
        self,
        all_candidates: List[List[List[Tuple[int, int]]]],
        batch_size: int = 512
    ) -> List[int]:
        """
        Evaluates a large list of words, each having multiple candidate chains,
        using padded GPU batches to maximize throughput.
        Returns a list of best_indices (one per word).
        """
        self.model.eval()

        # Flatten into one list of jobs: (word_idx, cand_idx, suffix_seq, cat_seq).
        # build_sentence_sequence prepends BOS, so every seq is length >= 2.
        flat_jobs = []
        for w_idx, candidates in enumerate(all_candidates):
            for c_idx, chain in enumerate(candidates):
                cand_s, cand_c = build_sentence_sequence([chain])
                flat_jobs.append((w_idx, c_idx, cand_s, cand_c))

        if not flat_jobs:
            return []

        # w_idx -> list of scores (same order as candidates)
        scores_map = {w_idx: [] for w_idx in range(len(all_candidates))}

        with torch.no_grad():
            for i in range(0, len(flat_jobs), batch_size):
                batch = flat_jobs[i:i + batch_size]
                max_len = max(len(job[2]) for job in batch)
                bsz = len(batch)
                
                # Allocate tensors
                s_t = torch.full((bsz, max_len), SPECIAL_PAD, dtype=torch.long, device=self.device)
                c_t = torch.full((bsz, max_len), CATEGORY_SPECIAL, dtype=torch.long, device=self.device)
                p_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=self.device) # True = pad

                for b_idx, (_, _, seq_s, seq_c) in enumerate(batch):
                    seq_len = len(seq_s)
                    s_t[b_idx, :seq_len] = torch.tensor(seq_s, dtype=torch.long, device=self.device)
                    c_t[b_idx, :seq_len] = torch.tensor(seq_c, dtype=torch.long, device=self.device)
                    p_mask[b_idx, :seq_len] = False

                lp = self.model.log_probs(s_t, c_t, pad_mask=p_mask) # (bsz, max_len - 1)

                # Sum the log probabilities for each sequence
                sums = lp.sum(dim=1).tolist()

                for b_idx, job in enumerate(batch):
                    scores_map[job[0]].append(sums[b_idx])

        # Pick best index per word
        best_indices = []
        for w_idx in range(len(all_candidates)):
            c_scores = scores_map[w_idx]
            if not c_scores:
                best_indices.append(0)
            else:
                best_indices.append(self._get_best_index(c_scores))

        return best_indices

    def predict(
        self,
        candidates: List[List[Tuple[int, int]]],
        context_chains: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Tuple[int, List[float]]:
        """
        Pick the best candidate for a single word (with optional left context).

        Returns: (best_index, all_scores)
        """
        ctx = context_chains or []
        scores = self.score_candidates(ctx, candidates)
        best = self._get_best_index(scores)
        return best, scores

    def batch_predict(
        self,
        batch_candidates: List[List[List[Tuple[int, int]]]],
    ) -> List[Tuple[int, List[float]]]:
        """
        Score candidates for multiple words independently (no cross-word context).
        Used for initial ranking before the user has made any choices.

        For context-aware sentence-level ranking, use sentence_predict() instead.
        """
        results = []
        for candidates in batch_candidates:
            best_idx, scores = self.predict(candidates)
            results.append((best_idx, scores))
        return results

    def sentence_predict(
        self,
        all_candidates: List[List[List[Tuple[int, int]]]],
    ) -> List[Tuple[int, List[float]]]:
        """
        Greedy left-to-right sentence-level disambiguation.

        For each word in order, score its candidates given all previously
        committed choices as left context, then commit the winner.

        Returns: list of (best_idx, scores) per word.
        """
        committed: List[List[Tuple[int, int]]] = []
        results: List[Tuple[int, List[float]]] = []

        for candidates in all_candidates:
            scores  = self.score_candidates(committed, candidates)
            best    = self._get_best_index(scores)
            results.append((best, scores))
            committed.append(candidates[best])

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self):
        torch.save({
            'model_state':     self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'train_history':   self.train_history,
            'val_history':     self.val_history,
            'best_val_loss':   self.best_val_loss,
            'global_step':     self.global_step,
            'replay_buffer':   self.replay_buffer,
        }, self.path)
        print(f"Saved to {self.path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.train_history  = ckpt.get('train_history',  [])
        self.val_history    = ckpt.get('val_history',    [])
        self.best_val_loss  = ckpt.get('best_val_loss',  float('inf'))
        self.global_step    = ckpt.get('global_step',    0)
        self.replay_buffer  = ckpt.get('replay_buffer',  [])
        print(f"Loaded from {path} (step {self.global_step}, {len(self.replay_buffer)} replay entries)")