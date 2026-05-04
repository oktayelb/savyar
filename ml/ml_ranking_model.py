import math
import random
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Failed to initialize NumPy: No module named 'numpy'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False because encoder_layer\.norm_first was True.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .config import config  
from util.suffix import SuffixGroup, Type

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

# ============================================================================
# SPECIAL TOKENS
# ============================================================================

SPECIAL_PAD           = 0
SPECIAL_WORD_SEP      = 1
SPECIAL_BOS           = 2          
SPECIAL_MASK          = 3          
SUFFIX_OFFSET         = 4          
CATEGORY_SPECIAL      = 2          
CATEGORY_CLOSED_CLASS = 3          

SPECIAL_FEATURE_ID    = 0
WORD_FINAL_NO         = 0
WORD_FINAL_YES        = 1

GROUP_TO_ID = {None: SPECIAL_FEATURE_ID}
for idx, group in enumerate(SuffixGroup):
    GROUP_TO_ID[group] = idx + 1

TYPE_TO_ID = {
    None: SPECIAL_FEATURE_ID,
    Type.NOUN: 1,
    Type.VERB: 2,
    Type.BOTH: 3,
}

EncodedToken = Tuple[int, int, int, int, int, int, int]
FlatSequence = Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int]]

# ============================================================================
# HELPER: encode / decode sentence-level token sequences
# ============================================================================


def _get_all_suffixes():
    import util.decomposer as sfx
    return sfx.ALL_SUFFIXES


def _chain_tokens(
    word_chains: List[List[EncodedToken]]
) -> FlatSequence:
    suffix_ids:   List[int] = []
    category_ids: List[int] = []
    group_ids:    List[int] = []
    comes_to_ids: List[int] = []
    makes_ids:    List[int] = []
    pos_ids:      List[int] = []
    word_final:   List[int] = []
    for chain in word_chains:
        for (sid, cid, gid, comes_to_id, makes_id, pos_in_word, is_final) in chain:
            suffix_ids.append(sid)
            category_ids.append(cid)
            group_ids.append(gid)
            comes_to_ids.append(comes_to_id)
            makes_ids.append(makes_id)
            pos_ids.append(pos_in_word)
            word_final.append(is_final)
        suffix_ids.append(SPECIAL_WORD_SEP)
        category_ids.append(CATEGORY_SPECIAL)
        group_ids.append(SPECIAL_FEATURE_ID)
        comes_to_ids.append(SPECIAL_FEATURE_ID)
        makes_ids.append(SPECIAL_FEATURE_ID)
        pos_ids.append(SPECIAL_FEATURE_ID)
        word_final.append(WORD_FINAL_NO)
    return suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, pos_ids, word_final


def build_sentence_sequence(
    word_chains: List[List[EncodedToken]]
) -> FlatSequence:
    s, c, g, ct, m, p, wf = _chain_tokens(word_chains)
    return (
        [SPECIAL_BOS] + s,
        [CATEGORY_SPECIAL] + c,
        [SPECIAL_FEATURE_ID] + g,
        [SPECIAL_FEATURE_ID] + ct,
        [SPECIAL_FEATURE_ID] + m,
        [SPECIAL_FEATURE_ID] + p,
        [WORD_FINAL_NO] + wf,
    )


# ============================================================================
# MODEL
# ============================================================================

class SentenceDisambiguator(nn.Module):
    def __init__(self, suffix_vocab_size: int, closed_class_vocab_size: int = 0):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.vocab_size = SUFFIX_OFFSET + suffix_vocab_size + closed_class_vocab_size

        self.suffix_embed   = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=SPECIAL_PAD)
        
        self.category_embed = nn.Embedding(4, config.category_embed_dim)
        self.group_embed    = nn.Embedding(len(GROUP_TO_ID), config.group_embed_dim)
        self.comes_to_embed = nn.Embedding(max(TYPE_TO_ID.values()) + 1, config.comes_makes_embed_dim)
        self.makes_embed    = nn.Embedding(max(TYPE_TO_ID.values()) + 1, config.comes_makes_embed_dim)
        self.wordpos_embed  = nn.Embedding(64, config.wordpos_embed_dim)
        self.wordfinal_embed = nn.Embedding(2, config.wordfinal_embed_dim)
        
        self.pos_embed      = nn.Embedding(512, self.embed_dim)

        feature_width = (
            self.embed_dim * 2 + 
            config.category_embed_dim + 
            config.group_embed_dim + 
            config.comes_makes_embed_dim * 2 + 
            config.wordpos_embed_dim + 
            config.wordfinal_embed_dim
        )

        self.input_proj = nn.Sequential(
            nn.Linear(feature_width, 512),
            nn.GELU(),
            nn.Linear(512, self.embed_dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,   
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)

        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.suffix_embed.weight
        self.rank_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'embed' not in name:
                nn.init.kaiming_normal_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        suffix_ids:   torch.Tensor,   
        category_ids: torch.Tensor,   
        group_ids:    torch.Tensor,   
        comes_to_ids: torch.Tensor,   
        makes_ids:    torch.Tensor,   
        word_pos_ids: torch.Tensor,   
        word_final:   torch.Tensor,   
        pad_mask:     Optional[torch.Tensor] = None,  
    ) -> torch.Tensor:
        B, L = suffix_ids.shape
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)

        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.group_embed(group_ids),
            self.comes_to_embed(comes_to_ids),
            self.makes_embed(makes_ids),
            self.wordpos_embed(word_pos_ids.clamp(max=self.wordpos_embed.num_embeddings - 1)),
            self.wordfinal_embed(word_final),
            self.pos_embed(pos),
        ], dim=-1)

        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.lm_head(x)

    def rank_scores(
        self,
        suffix_ids:   torch.Tensor,
        category_ids: torch.Tensor,
        group_ids:    torch.Tensor,
        comes_to_ids: torch.Tensor,
        makes_ids:    torch.Tensor,
        word_pos_ids: torch.Tensor,
        word_final:   torch.Tensor,
        pad_mask:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = suffix_ids.shape
        pos = torch.arange(L, device=suffix_ids.device).unsqueeze(0).expand(B, L)

        x = torch.cat([
            self.suffix_embed(suffix_ids),
            self.category_embed(category_ids),
            self.group_embed(group_ids),
            self.comes_to_embed(comes_to_ids),
            self.makes_embed(makes_ids),
            self.wordpos_embed(word_pos_ids.clamp(max=self.wordpos_embed.num_embeddings - 1)),
            self.wordfinal_embed(word_final),
            self.pos_embed(pos),
        ], dim=-1)

        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        if pad_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~pad_mask).unsqueeze(-1).to(x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.rank_head(pooled).squeeze(-1)


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, model: SentenceDisambiguator, path: Optional[str] = None):
        self.model = model

        self.checkpoint_frequency = config.checkpoint_frequency
        self.path                 = path if path is not None else str(config.model_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device == 'cuda'))

        if not config.use_torch_compile or not torch.cuda.is_available() or not hasattr(torch, 'compile'):
            pass  
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
        
        # Interactive fallback scheduler
        self.scheduler = self._build_schedule(self.optimizer)

        self.train_history: List[float] = []
        self.val_history:   List[float] = []
        self.best_val_loss  = float('inf')
        self.global_step    = 0

        self.replay_buffer: List[FlatSequence] = []
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

    @staticmethod
    def _build_schedule(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        warmup      = max(1, int(config.warmup_steps))
        eta_min     = float(config.lr_eta_min_ratio)
        decay_total = max(warmup * 50, 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / decay_total)
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def _get_best_index(self, scores: List[float]) -> int:
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def _add_to_replay(
        self,
        suffix_ids: List[int],
        category_ids: List[int],
        group_ids: List[int],
        comes_to_ids: List[int],
        makes_ids: List[int],
        word_pos_ids: List[int],
        word_final: List[int],
    ) -> None:
        self.replay_buffer.append(
            (suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final)
        )
        if len(self.replay_buffer) > config.replay_buffer_size:
            evict_idx = random.randrange(len(self.replay_buffer) // 2)
            self.replay_buffer.pop(evict_idx)
        self._class_weight_cache = None


    def _build_padded_batch(
        self, seqs: List[FlatSequence]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(seq[0]) for seq in seqs)
        bsz = len(seqs)

        pin = self.device == 'cuda'
        s_t    = torch.full((bsz, max_len), SPECIAL_PAD,        dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_PAD,        dtype=torch.long)
        c_t    = torch.full((bsz, max_len), CATEGORY_SPECIAL,   dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), CATEGORY_SPECIAL,   dtype=torch.long)
        g_t    = torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long)
        ct_t   = torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long)
        m_t    = torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long)
        wp_t   = torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), SPECIAL_FEATURE_ID, dtype=torch.long)
        wf_t   = torch.full((bsz, max_len), WORD_FINAL_NO,      dtype=torch.long).pin_memory() if pin else torch.full((bsz, max_len), WORD_FINAL_NO,      dtype=torch.long)
        p_mask = torch.ones((bsz, max_len), dtype=torch.bool)

        for i, (sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final) in enumerate(seqs):
            L = len(sids)
            s_t[i, :L]    = torch.tensor(sids, dtype=torch.long)
            c_t[i, :L]    = torch.tensor(cids, dtype=torch.long)
            g_t[i, :L]    = torch.tensor(gids, dtype=torch.long)
            ct_t[i, :L]   = torch.tensor(comes_to_ids, dtype=torch.long)
            m_t[i, :L]    = torch.tensor(makes_ids, dtype=torch.long)
            wp_t[i, :L]   = torch.tensor(word_pos_ids, dtype=torch.long)
            wf_t[i, :L]   = torch.tensor(word_final, dtype=torch.long)
            p_mask[i, :L] = False

        non_blocking = self.device == 'cuda'
        return (
            s_t.to(self.device, non_blocking=non_blocking),
            c_t.to(self.device, non_blocking=non_blocking),
            g_t.to(self.device, non_blocking=non_blocking),
            ct_t.to(self.device, non_blocking=non_blocking),
            m_t.to(self.device, non_blocking=non_blocking),
            wp_t.to(self.device, non_blocking=non_blocking),
            wf_t.to(self.device, non_blocking=non_blocking),
            p_mask.to(self.device, non_blocking=non_blocking),
        )

    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        if gamma is None:
            gamma = config.focal_gamma

        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
            ignore_index=SPECIAL_PAD,
        )

        if gamma > 0.0:
            pt = torch.exp(-ce_loss)
            loss_per_tok = ((1 - pt) ** gamma) * ce_loss
        else:
            loss_per_tok = ce_loss

        valid_mask = targets.reshape(-1) != SPECIAL_PAD
        if valid_mask.any():
            return loss_per_tok[valid_mask].mean()
        return loss_per_tok.sum()

    def _mlm_mask_batch(
        self,
        s_t:    torch.Tensor,   
        p_mask: torch.Tensor,   
        mask_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_prob is None:
            mask_prob = config.mlm_mask_prob

        eligible = (
            (s_t != SPECIAL_PAD)
            & (s_t != SPECIAL_WORD_SEP)
            & (s_t != SPECIAL_BOS)
            & (~p_mask)
        )

        draws    = torch.rand_like(s_t, dtype=torch.float)
        selected = eligible & (draws < mask_prob)

        if config.mlm_ensure_one_mask:
            has_elig     = eligible.any(dim=1)                        
            has_selected = selected.any(dim=1)                        
            need_force   = has_elig & (~has_selected)                 
            if need_force.any():
                forced_draws = draws.masked_fill(~eligible, float('inf'))
                forced_pos   = forced_draws.argmin(dim=1)             
                rows         = torch.arange(s_t.size(0), device=s_t.device)
                rows         = rows[need_force]
                cols         = forced_pos[need_force]
                selected[rows, cols] = True

        loss_target = s_t.clone()
        loss_target[~selected] = SPECIAL_PAD  

        masked_s = s_t.clone()
        if config.mlm_use_bert_mix:
            role_draws = torch.rand_like(s_t, dtype=torch.float)
            mask_slot   = selected & (role_draws < 0.80)
            random_slot = selected & (role_draws >= 0.80) & (role_draws < 0.90)

            masked_s[mask_slot] = SPECIAL_MASK

            if random_slot.any():
                n_rand = int(random_slot.sum().item())
                rand_tokens = torch.randint(
                    low=SUFFIX_OFFSET,
                    high=self.model.vocab_size,
                    size=(n_rand,),
                    device=s_t.device,
                    dtype=s_t.dtype,
                )
                masked_s[random_slot] = rand_tokens
        else:
            masked_s[selected] = SPECIAL_MASK

        return masked_s, loss_target

    def _sequence_from_chains(self, word_chains: List[List[EncodedToken]]) -> FlatSequence:
        return build_sentence_sequence(word_chains)

    def _ranking_step(self, candidate_sets: List[List[FlatSequence]]) -> Tuple[float, float]:
        candidate_sets = [cands for cands in candidate_sets if len(cands) >= 2]
        if not candidate_sets:
            return 0.0, 0.0

        flat: List[FlatSequence] = []
        sizes: List[int] = []
        for cands in candidate_sets:
            sizes.append(len(cands))
            flat.extend(cands)

        try:
            s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(flat)

            self.model.train()
            self.optimizer.zero_grad()
            use_amp = self.device == 'cuda'
            
            temperature = config.ranking_temperature
            mlm_weight = config.mlm_weight

            with torch.amp.autocast('cuda', enabled=use_amp):
                # 1. Contrastive Ranking Loss
                scores = self.model.rank_scores(s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                losses = []
                offset = 0
                for size in sizes:
                    # Apply temperature scaling to the logits
                    logits = (scores[offset:offset + size] / temperature).unsqueeze(0)
                    target = torch.zeros(1, dtype=torch.long, device=self.device)
                    losses.append(F.cross_entropy(logits, target))
                    offset += size
                rank_loss = torch.stack(losses).mean()

                # 2. Masked Language Modeling Loss (on the gold sequences only)
                gold_indices = [sum(sizes[:i]) for i in range(len(sizes))]
                gold_s_t = s_t[gold_indices]
                gold_p_mask = p_mask[gold_indices]
                
                masked_s, mlm_target = self._mlm_mask_batch(gold_s_t, gold_p_mask)
                
                mlm_logits = self.model(
                    masked_s, 
                    c_t[gold_indices], 
                    g_t[gold_indices], 
                    ct_t[gold_indices], 
                    m_t[gold_indices], 
                    wp_t[gold_indices], 
                    wf_t[gold_indices], 
                    pad_mask=gold_p_mask
                )
                
                mlm_loss = self._compute_focal_loss(mlm_logits, mlm_target)

                # 3. Joint Objective
                total_loss = rank_loss + (mlm_weight * mlm_loss)

            final_loss = total_loss.item()
            final_rank = rank_loss.item()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not use_amp or self.scaler.get_scale() >= scaler_scale:
                self.scheduler.step()
            self.global_step += 1
            return final_loss, final_rank
            
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            if len(candidate_sets) <= 1:
                raise
            mid = len(candidate_sets) // 2
            left_loss, left_rank = self._ranking_step(candidate_sets[:mid])
            right_loss, right_rank = self._ranking_step(candidate_sets[mid:])
            return (left_loss + right_loss) / 2.0, (left_rank + right_rank) / 2.0

    @staticmethod
    def _candidate_batch_count(candidate_sets: List[List[FlatSequence]], batch_size: int) -> int:
        count = 0
        current_sets = 0
        current_sequences = 0
        max_sequences = max(1, int(config.max_candidate_sequences_per_batch))

        for cands in candidate_sets:
            cand_count = len(cands)
            would_exceed_sets = current_sets >= batch_size
            would_exceed_sequences = current_sets > 0 and current_sequences + cand_count > max_sequences
            if would_exceed_sets or would_exceed_sequences:
                count += 1
                current_sets = 0
                current_sequences = 0
            current_sets += 1
            current_sequences += cand_count

        return count + (1 if current_sets else 0)

    @staticmethod
    def _candidate_batches(
        candidate_sets: List[List[FlatSequence]],
        batch_size: int,
    ) -> List[List[List[FlatSequence]]]:
        batches: List[List[List[FlatSequence]]] = []
        current: List[List[FlatSequence]] = []
        current_sequences = 0
        max_sequences = max(1, int(config.max_candidate_sequences_per_batch))

        for cands in candidate_sets:
            cand_count = len(cands)
            would_exceed_sets = len(current) >= batch_size
            would_exceed_sequences = current and current_sequences + cand_count > max_sequences
            if would_exceed_sets or would_exceed_sequences:
                batches.append(current)
                current = []
                current_sequences = 0
            current.append(cands)
            current_sequences += cand_count

        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _suffix_token_accuracy(gold: FlatSequence, pred: FlatSequence) -> float:
        matches, gold_count, pred_count = Trainer._suffix_token_stats(gold, pred)
        denom = max(gold_count, pred_count)
        if denom == 0:
            return 1.0
        return matches / denom

    @staticmethod
    def _suffix_token_stats(gold: FlatSequence, pred: FlatSequence) -> Tuple[int, int, int]:
        gold_tokens = [
            tok for tok in gold[0]
            if tok not in (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS)
        ]
        pred_tokens = [
            tok for tok in pred[0]
            if tok not in (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS)
        ]
        matches = sum(
            1 for gold_tok, pred_tok in zip(gold_tokens, pred_tokens)
            if gold_tok == pred_tok
        )
        return matches, len(gold_tokens), len(pred_tokens)

    def train_sentence(
        self,
        word_chains: List[List[EncodedToken]],
        negative_word_chains: Optional[List[List[List[EncodedToken]]]] = None,
        max_retries: int = None,
    ) -> float:
        gold_seq = self._sequence_from_chains(word_chains)
        if len(gold_seq[0]) < 2:
            return 0.0

        self._add_to_replay(*gold_seq)
        negatives = negative_word_chains or []
        candidate_set = [gold_seq]
        for neg in negatives:
            neg_seq = self._sequence_from_chains(neg)
            if neg_seq != gold_seq:
                candidate_set.append(neg_seq)

        if len(candidate_set) < 2:
            return 0.0

        print(f"   Ranking gold against {len(candidate_set) - 1} negatives...", end="", flush=True)
        final_loss = 0.0
        for _ in range(config.steps_per_update):
            final_loss, _ = self._ranking_step([candidate_set])
        print(f" loss={final_loss:.4f}")

        self.train_history.append(final_loss)
        return final_loss

    def train_bulk(
        self,
        all_seqs: List,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        validation_seqs: Optional[List] = None,
    ) -> float:
        if batch_size is None:
            batch_size = config.bulk_batch_size
        if epochs is None:
            epochs = config.bulk_epochs
        if not all_seqs:
            return 0.0

        candidate_sets: List[List[FlatSequence]] = []
        for item in all_seqs:
            if not item:
                continue
            if isinstance(item, tuple) and len(item) == 7:
                candidate_sets.append([item])
            else:
                candidate_sets.append(list(item))

        for cands in candidate_sets:
            if cands:
                self._add_to_replay(*cands[0])

        trainable_sets = [cands for cands in candidate_sets if len(cands) >= 2]
        if not trainable_sets:
            return 0.0

        # Build dynamic learning rate schedule exactly matched to total steps
        total_steps = epochs * self._candidate_batch_count(trainable_sets, batch_size)
        warmup = max(1, int(config.warmup_steps))
        eta_min = float(config.lr_eta_min_ratio)

        def bulk_lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

        # Re-initialize scheduler to lock decay perfectly to bulk training timeframe
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, bulk_lr_lambda)

        final_loss = 0.0
        data = list(trainable_sets)

        for epoch in range(epochs):
            random.shuffle(data)
            epoch_loss = 0.0
            epoch_rank_loss = 0.0
            n_batches = 0
            correct = 0
            total = 0
            margins: List[float] = []

            for batch_sets in self._candidate_batches(data, batch_size):
                loss_value, rank_loss_value = self._ranking_step(batch_sets)
                if loss_value == 0.0 and rank_loss_value == 0.0:
                    continue
                epoch_loss += loss_value
                epoch_rank_loss += rank_loss_value
                n_batches += 1

                with torch.no_grad():
                    for cands in batch_sets:
                        scores = self.score_flat_sequences(cands)
                        if not scores:
                            continue
                        total += 1
                        if max(range(len(scores)), key=lambda i: scores[i]) == 0:
                            correct += 1
                        if len(scores) > 1:
                            margins.append(scores[0] - max(scores[1:]))

            if n_batches:
                avg = epoch_loss / n_batches
                avg_rank = epoch_rank_loss / n_batches
                final_loss = avg
                acc = correct / total if total else 0.0
                mean_margin = sum(margins) / len(margins) if margins else 0.0
                print(
                    f"   Bulk epoch {epoch+1}/{epochs}: loss={avg:.4f} | rank_loss={avg_rank:.4f} | "
                    f"RankAcc={acc:.4f} | margin={mean_margin:.4f} "
                    f"({n_batches} batches, {total} candidate sets)"
                )

            if validation_seqs:
                val_stats = self.validate(validation_seqs, batch_size=batch_size)
                self.val_history.append(val_stats['loss'])
                if val_stats['loss'] < self.best_val_loss:
                    self.best_val_loss = val_stats['loss']
                val_header = (
                    f"   Validation   : rank_loss={val_stats['loss']:.4f} | "
                    f"RankAcc={val_stats['rank_acc']:.4f} | "
                    f"SuffAcc={val_stats['suff_acc']:.4f} | "
                    f"SuffPrecision={val_stats['suff_precision']:.4f} | "
                    f"SuffRecall={val_stats['suff_recall']:.4f} | "
                    f"SuffF1={val_stats['suff_f1']:.4f} | "
                    f"margin={val_stats['margin']:.4f} "
                    f"(best={self.best_val_loss:.4f})"
                )
                print(val_header)

        self.train_history.append(final_loss)
        return final_loss

    def validate(
        self,
        val_seqs: List,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        empty = {
            'loss': 0.0,
            'rank_acc': 0.0,
            'suff_acc': 0.0,
            'suff_precision': 0.0,
            'suff_recall': 0.0,
            'suff_f1': 0.0,
            'margin': 0.0,
            'n_batches': 0,
        }
        if not val_seqs:
            return empty

        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        suff_acc_total = 0.0
        suff_matches = 0
        suff_gold_total = 0
        suff_pred_total = 0
        margins: List[float] = []

        with torch.no_grad():
            for start in range(0, len(val_seqs), batch_size):
                raw_batch_sets = [list(s) for s in val_seqs[start:start + batch_size] if len(s) >= 2]
                for batch_sets in self._candidate_batches(raw_batch_sets, batch_size):
                    flat = [seq for cands in batch_sets for seq in cands]
                    sizes = [len(cands) for cands in batch_sets]
                    scores = self.score_flat_sequences(flat)
                    offset = 0
                    losses = []
                    for set_idx, size in enumerate(sizes):
                        group = scores[offset:offset + size]
                        # Must apply temperature during validation loss calc for parity with training
                        logits = (torch.tensor(group, dtype=torch.float, device=self.device) / config.ranking_temperature).unsqueeze(0)
                        target = torch.zeros(1, dtype=torch.long, device=self.device)
                        losses.append(F.cross_entropy(logits, target).item())
                        total += 1
                        best_idx = max(range(len(group)), key=lambda i: group[i])
                        if best_idx == 0:
                            correct += 1
                        gold_seq = batch_sets[set_idx][0]
                        pred_seq = batch_sets[set_idx][best_idx]
                        suff_acc_total += self._suffix_token_accuracy(gold_seq, pred_seq)
                        matches, gold_count, pred_count = self._suffix_token_stats(gold_seq, pred_seq)
                        suff_matches += matches
                        suff_gold_total += gold_count
                        suff_pred_total += pred_count
                        margins.append(group[0] - max(group[1:]))
                        offset += size
                    total_loss += sum(losses) / len(losses)
                    n_batches += 1

        if n_batches == 0:
            return empty

        avg_loss = total_loss / n_batches
        suff_precision = suff_matches / suff_pred_total if suff_pred_total else 0.0
        suff_recall = suff_matches / suff_gold_total if suff_gold_total else 0.0
        suff_f1 = (
            2 * suff_precision * suff_recall / (suff_precision + suff_recall)
            if (suff_precision + suff_recall) > 0.0
            else 0.0
        )

        return {
            'loss':        avg_loss,
            'rank_acc':    correct / total if total else 0.0,
            'suff_acc':    suff_acc_total / total if total else 0.0,
            'suff_precision': suff_precision,
            'suff_recall': suff_recall,
            'suff_f1':     suff_f1,
            'margin':      sum(margins) / len(margins) if margins else 0.0,
            'n_batches':   n_batches,
        }


    def score_candidates(
        self,
        context_chains: List[List[EncodedToken]],   
        candidates:     List[List[EncodedToken]],   
        right_chains:   Optional[List[List[EncodedToken]]] = None,  
    ) -> List[float]:
        self.model.eval()

        if context_chains:
            ctx_s, ctx_c, ctx_g, ctx_ct, ctx_m, ctx_wp, ctx_wf = _chain_tokens(context_chains)
        else:
            ctx_s, ctx_c, ctx_g, ctx_ct, ctx_m, ctx_wp, ctx_wf = ([], [], [], [], [], [], [])
        if right_chains:
            right_s, right_c, right_g, right_ct, right_m, right_wp, right_wf = _chain_tokens(right_chains)
        else:
            right_s, right_c, right_g, right_ct, right_m, right_wp, right_wf = ([], [], [], [], [], [], [])

        prefix_s  = [SPECIAL_BOS]      + ctx_s
        prefix_c  = [CATEGORY_SPECIAL] + ctx_c
        prefix_g  = [SPECIAL_FEATURE_ID] + ctx_g
        prefix_ct = [SPECIAL_FEATURE_ID] + ctx_ct
        prefix_m  = [SPECIAL_FEATURE_ID] + ctx_m
        prefix_wp = [SPECIAL_FEATURE_ID] + ctx_wp
        prefix_wf = [WORD_FINAL_NO] + ctx_wf
        flat_sequences: List[FlatSequence] = []
        bare_indices: List[int] = []
        for idx, chain in enumerate(candidates):
            cand_s, cand_c, cand_g, cand_ct, cand_m, cand_wp, cand_wf = _chain_tokens([chain])
            if len(cand_s) <= 1:
                bare_indices.append(idx)
            flat_sequences.append((
                prefix_s + cand_s + right_s,
                prefix_c + cand_c + right_c,
                prefix_g + cand_g + right_g,
                prefix_ct + cand_ct + right_ct,
                prefix_m + cand_m + right_m,
                prefix_wp + cand_wp + right_wp,
                prefix_wf + cand_wf + right_wf,
            ))

        scores = self.score_flat_sequences(flat_sequences)
        for idx in bare_indices:
            scores[idx] += float(config.bare_root_prior_logprob)
        return scores

    def score_sentence_chains(self, word_chains: List[List[EncodedToken]]) -> float:
        full_sequence = build_sentence_sequence(word_chains)
        bare_root_count = sum(1 for chain in word_chains if not chain)
        prior = bare_root_count * float(config.bare_root_prior_logprob)
        if len(full_sequence[0]) < 2:
            return prior
        return self.score_flat_sequences([full_sequence])[0] + prior

    def score_flat_sequences(self, seqs: List[FlatSequence]) -> List[float]:
        if not seqs:
            return []
        self.model.eval()
        chunk_size = max(1, int(config.max_candidate_sequences_per_batch))
        all_scores: List[float] = []
        with torch.no_grad():
            for start in range(0, len(seqs), chunk_size):
                chunk = seqs[start:start + chunk_size]
                s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(chunk)
                scores = self.model.rank_scores(s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                all_scores.extend(scores.detach().cpu().tolist())
        return all_scores

    def predict(
        self,
        candidates: List[List[EncodedToken]],
        context_chains: Optional[List[List[EncodedToken]]] = None,
    ) -> Tuple[int, List[float]]:
        ctx = context_chains or []
        scores = self.score_candidates(ctx, candidates)
        best = self._get_best_index(scores)
        return best, scores


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
            'suffix_inventory': [s.name for s in _get_all_suffixes()],
        }, self.path)
        print(f"Saved to {self.path}")

    def load_checkpoint(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        current_suffix_inventory = [s.name for s in _get_all_suffixes()]
        saved_suffix_inventory = ckpt.get('suffix_inventory')
        suffix_inventory_matches = saved_suffix_inventory == current_suffix_inventory
        model_state = ckpt['model_state']
        current_state = self.model.state_dict()
        compatible_state = {
            k: v for k, v in model_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        self.model.load_state_dict(compatible_state, strict=False)
        if suffix_inventory_matches:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception:
                pass
        self.train_history  = ckpt.get('train_history',  [])
        self.val_history    = ckpt.get('val_history',    [])
        self.best_val_loss  = ckpt.get('best_val_loss',  float('inf'))
        self.global_step    = ckpt.get('global_step',    0)
        raw_replay = ckpt.get('replay_buffer', []) if suffix_inventory_matches else []
        upgraded_replay = []
        for entry in raw_replay:
            upgraded = self._upgrade_replay_entry(entry)
            if upgraded is not None:
                upgraded_replay.append(upgraded)
        self.replay_buffer = upgraded_replay
        if not suffix_inventory_matches:
            print("Checkpoint suffix inventory changed; replay buffer and optimizer state were discarded.")
        print(f"Loaded from {path} (step {self.global_step}, {len(self.replay_buffer)} replay entries)")

    def _upgrade_replay_entry(self, entry) -> Optional[FlatSequence]:
        if not isinstance(entry, (list, tuple)):
            return None
        if len(entry) == 7:
            return tuple(entry)
        if len(entry) != 2:
            return None

        suffix_ids, category_ids = entry
        if len(suffix_ids) != len(category_ids):
            return None

        group_ids = [SPECIAL_FEATURE_ID] * len(suffix_ids)
        comes_to_ids = [SPECIAL_FEATURE_ID] * len(suffix_ids)
        makes_ids = [SPECIAL_FEATURE_ID] * len(suffix_ids)
        word_pos_ids = [SPECIAL_FEATURE_ID] * len(suffix_ids)
        word_final = [WORD_FINAL_NO] * len(suffix_ids)

        current_word_positions: List[int] = []
        current_word_tokens: List[int] = []
        for idx, tok_id in enumerate(suffix_ids):
            if tok_id in (SPECIAL_BOS, SPECIAL_WORD_SEP):
                if current_word_tokens:
                    last_idx = current_word_tokens[-1]
                    word_final[last_idx] = WORD_FINAL_YES
                    current_word_positions.clear()
                    current_word_tokens.clear()
                continue

            current_word_tokens.append(idx)
            current_word_positions.append(len(current_word_positions) + 1)
            word_pos_ids[idx] = current_word_positions[-1]

        if current_word_tokens:
            word_final[current_word_tokens[-1]] = WORD_FINAL_YES

        return (
            list(suffix_ids),
            list(category_ids),
            group_ids,
            comes_to_ids,
            makes_ids,
            word_pos_ids,
            word_final,
        )
