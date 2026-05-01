import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
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

def encode_chain(suffix_chain) -> List[EncodedToken]:
    from ml.ml_ranking_model import SUFFIX_OFFSET, CATEGORY_SPECIAL  

    suffix_to_id = {
        suffix.name: idx + SUFFIX_OFFSET
        for idx, suffix in enumerate(_get_all_suffixes())
    }
    category_to_id = {'Noun': 0, 'Verb': 1}

    encoded = []
    last_idx = len(suffix_chain) - 1
    for idx, s in enumerate(suffix_chain):
        sid  = suffix_to_id.get(s.name, SUFFIX_OFFSET)  
        cid  = category_to_id.get(s.makes.name, 0)
        gid  = GROUP_TO_ID.get(getattr(s, 'group', None), SPECIAL_FEATURE_ID)
        comes_to_id = TYPE_TO_ID.get(getattr(s, 'comes_to', None), SPECIAL_FEATURE_ID)
        makes_id    = TYPE_TO_ID.get(getattr(s, 'makes', None), SPECIAL_FEATURE_ID)
        pos_in_word = idx + 1
        is_final    = WORD_FINAL_YES if idx == last_idx else WORD_FINAL_NO
        encoded.append((sid, cid, gid, comes_to_id, makes_id, pos_in_word, is_final))
    return encoded


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

    def log_probs(
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
        device = suffix_ids.device
        result = torch.zeros(B, L, dtype=torch.float, device=device)

        is_special = (
            (suffix_ids == SPECIAL_PAD)
            | (suffix_ids == SPECIAL_WORD_SEP)
            | (suffix_ids == SPECIAL_BOS)
            | (suffix_ids == SPECIAL_MASK)
        )
        if pad_mask is not None:
            is_special = is_special | pad_mask
        eligible = ~is_special

        flat_eligible = eligible.reshape(-1)
        if not flat_eligible.any():
            return result

        flat_idx   = flat_eligible.nonzero(as_tuple=False).squeeze(-1)  
        batch_ids  = flat_idx // L                                      
        pos_ids    = flat_idx %  L                                      
        K          = flat_idx.numel()

        batched_s  = suffix_ids[batch_ids].clone()
        batched_c  = category_ids[batch_ids].clone()
        batched_g  = group_ids[batch_ids].clone()
        batched_ct = comes_to_ids[batch_ids].clone()
        batched_m  = makes_ids[batch_ids].clone()
        batched_wp = word_pos_ids[batch_ids].clone()
        batched_wf = word_final[batch_ids].clone()
        row_range = torch.arange(K, device=device)
        batched_s[row_range, pos_ids] = SPECIAL_MASK

        batched_pad = pad_mask[batch_ids] if pad_mask is not None else None

        logits = self.forward(
            batched_s, batched_c, batched_g, batched_ct, batched_m, batched_wp, batched_wf,
            pad_mask=batched_pad,
        )
        slot_logits = logits[row_range, pos_ids]                           
        log_p       = F.log_softmax(slot_logits, dim=-1)                   
        true_toks   = suffix_ids[batch_ids, pos_ids]                       
        scores      = log_p.gather(1, true_toks.unsqueeze(-1)).squeeze(-1) 

        result[batch_ids, pos_ids] = scores
        return result


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

        if not torch.cuda.is_available() or not hasattr(torch, 'compile'):
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

    def _to_tensor(
        self,
        suffix_ids: List[int],
        category_ids: List[int],
        group_ids: List[int],
        comes_to_ids: List[int],
        makes_ids: List[int],
        word_pos_ids: List[int],
        word_final: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = [
            torch.tensor(values, dtype=torch.long, device=self.device).unsqueeze(0)
            for values in (suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final)
        ]
        return tuple(tensors)

    def _get_best_index(self, scores: List[float]) -> int:
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    def _compute_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, float, float, List[Dict[str, Any]]]:
        if len(targets) == 0:
            return 0.0, 0.0, 0.0, 0.0, []

        is_special = (targets == SPECIAL_WORD_SEP) | (targets == SPECIAL_BOS)
        suffix_mask = ~is_special

        suffix_preds   = preds[suffix_mask]
        suffix_targets = targets[suffix_mask]

        if len(suffix_targets) == 0:
            return 0.0, 0.0, 0.0, 0.0, []

        suffix_acc = (suffix_preds == suffix_targets).float().mean().item()

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
            return suffix_acc, 0.0, 0.0, 0.0, []

        macro_p  = precision[valid_classes].mean().item()
        macro_r  = recall[valid_classes].mean().item()
        macro_f1 = f1[valid_classes].mean().item()

        per_suffix = []
        valid_idx = valid_classes.nonzero(as_tuple=False).squeeze(-1)
        for idx in valid_idx.tolist():
            per_suffix.append({
                'id': idx,
                'p': precision[idx].item(),
                'r': recall[idx].item(),
                'f1': f1[idx].item(),
                'count': int(target_counts[idx].item())
            })

        return suffix_acc, macro_p, macro_r, macro_f1, per_suffix

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

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        if not config.use_class_weights:
            return None
        if self._class_weight_cache is not None:
            return self._class_weight_cache
        if not self.replay_buffer:
            return None

        V = self.model.vocab_size
        counts = torch.zeros(V, dtype=torch.float, device=self.device)
        for sids, *_ in self.replay_buffer:
            if not sids:
                continue
            ids = torch.as_tensor(sids, dtype=torch.long, device=self.device)
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float))

        weights = torch.ones(V, dtype=torch.float, device=self.device)
        present = counts > 0
        if present.any():
            inv_sqrt = 1.0 / torch.sqrt(counts[present])
            inv_sqrt = inv_sqrt * (inv_sqrt.numel() / inv_sqrt.sum())
            weights[present] = inv_sqrt

        weights[SPECIAL_PAD] = 1.0  
        self._class_weight_cache = weights
        return weights

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

    def _gradient_steps(
        self, seqs: List[Tuple[List[int], List[int]]], n_steps: int
    ) -> float:
        s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(seqs)

        self.model.train()
        final_loss = 0.0
        use_amp = self.device == 'cuda'

        for _ in range(n_steps):
            masked_s, target = self._mlm_mask_batch(s_t, p_mask)

            if (target != SPECIAL_PAD).sum() == 0:
                continue

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = self.model(masked_s, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                loss = self._compute_focal_loss(logits, target)

            final_loss = loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

        return final_loss

    def train_sentence(
        self,
        word_chains: List[List[EncodedToken]],
        max_retries: int = None,   
    ) -> float:
        suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final = build_sentence_sequence(word_chains)

        if len(suffix_ids) < 2:
            return 0.0

        self._add_to_replay(
            suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final
        )

        batch: List[FlatSequence] = [
            (suffix_ids, category_ids, group_ids, comes_to_ids, makes_ids, word_pos_ids, word_final)
        ]
        if len(self.replay_buffer) > 1:
            k = min(config.replay_k, len(self.replay_buffer) - 1)
            others = [x for x in self.replay_buffer if x is not batch[0]]
            batch.extend(random.sample(others, k))

        print(f"   Training on {len(batch)} examples...", end="", flush=True)
        final_loss = self._gradient_steps(batch, config.steps_per_update)
        print(f" loss={final_loss:.4f}")

        self.train_history.append(final_loss)
        return final_loss

    def train_bulk(
        self,
        all_seqs: List[FlatSequence],
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        validation_seqs: Optional[List[Tuple[List[int], List[int]]]] = None,
    ) -> float:
        if batch_size is None:
            batch_size = config.bulk_batch_size
        if epochs is None:
            epochs = config.bulk_epochs
        if not all_seqs:
            return 0.0

        for sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final in all_seqs:
            self._add_to_replay(
                sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final
            )

        # Build dynamic learning rate schedule exactly matched to total steps
        total_steps = epochs * ((len(all_seqs) + batch_size - 1) // batch_size)
        warmup = max(1, int(config.warmup_steps))
        eta_min = float(config.lr_eta_min_ratio)

        def bulk_lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
            return eta_min + 0.5 * (1.0 - eta_min) * (1.0 + math.cos(math.pi * progress))

        # Re-initialize scheduler to lock decay perfectly to bulk training timeframe
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, bulk_lr_lambda)

        use_amp = self.device == 'cuda'
        final_loss = 0.0
        data = list(all_seqs)

        for epoch in range(epochs):
            random.shuffle(data)
            epoch_loss = 0.0
            n_batches = 0
            
            all_epoch_preds = []
            all_epoch_targs = []

            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(batch)

                masked_s, target = self._mlm_mask_batch(s_t, p_mask)
                if (target != SPECIAL_PAD).sum() == 0:
                    continue  

                self.model.train()
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = self.model(masked_s, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                    loss = self._compute_focal_loss(logits, target)

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
                self.scheduler.step()
                self.global_step += 1
                epoch_loss += loss.item()
                n_batches += 1

            if n_batches:
                avg = epoch_loss / n_batches
                final_loss = avg
                
                if all_epoch_targs:
                    epoch_preds_cat = torch.cat(all_epoch_preds)
                    epoch_targs_cat = torch.cat(all_epoch_targs)
                    suf_acc, prec, rec, f1, per_suffix = self._compute_metrics(
                        epoch_preds_cat, epoch_targs_cat
                    )
                    header = (
                        f"   Bulk epoch {epoch+1}/{epochs}: loss={avg:.4f} | "
                        f"SufAcc={suf_acc:.4f} | F1={f1:.4f} "
                        f"P={prec:.4f} R={rec:.4f} ({n_batches} batches)"
                    )
                    
                    all_sufs = _get_all_suffixes()
                    id_to_name = {i + SUFFIX_OFFSET: s.name for i, s in enumerate(all_sufs)}
                    
                    filtered_sufs = [s for s in per_suffix if s['count'] >= 10]
                    if not filtered_sufs:
                        filtered_sufs = per_suffix
                    filtered_sufs.sort(key=lambda x: x['f1'])
                    lowest_sufs = filtered_sufs[:12] 
                    
                    cat_cells = []
                    for s_stat in lowest_sufs:
                        name = id_to_name.get(s_stat['id'], f"UNK_{s_stat['id']}")
                        cat_cells.append(f"{name[:14]:>14}: F1={s_stat['f1']:.2f} P={s_stat['p']:.2f} R={s_stat['r']:.2f} ({s_stat['count']:>4})")
                    
                    rows = [cat_cells[i:i+2] for i in range(0, len(cat_cells), 2)]
                    breakdown = "      Lowest Performing Suffixes (Count >= 10):\n"
                    breakdown += "\n".join("      " + "   ".join(r) for r in rows)
                    
                    print(header)
                    print(breakdown)
                else:
                    print(f"   Bulk epoch {epoch+1}/{epochs}: avg_loss={avg:.4f}  ({n_batches} batches)")

            if validation_seqs:
                val_stats = self.validate(validation_seqs, batch_size=batch_size)
                self.val_history.append(val_stats['loss'])
                if val_stats['loss'] < self.best_val_loss:
                    self.best_val_loss = val_stats['loss']
                val_header = (
                    f"   Validation   : loss={val_stats['loss']:.4f} | "
                    f"SufAcc={val_stats['suffix_acc']:.4f} | "
                    f"F1={val_stats['f1']:.4f} "
                    f"P={val_stats['precision']:.4f} R={val_stats['recall']:.4f} "
                    f"(best={self.best_val_loss:.4f})"
                )
                print(val_header)

        self.train_history.append(final_loss)
        return final_loss

    def validate(
        self,
        val_seqs: List[FlatSequence],
        batch_size: int = 64,
    ) -> Dict[str, float]:
        empty = {
            'loss': 0.0, 'suffix_acc': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_batches': 0,
        }
        if not val_seqs:
            return empty

        self.model.eval()
        use_amp = self.device == 'cuda'

        total_loss = 0.0
        n_batches = 0
        all_preds: List[torch.Tensor] = []
        all_targs: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, len(val_seqs), batch_size):
                batch = val_seqs[start:start + batch_size]
                s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(batch)

                masked_s, target = self._mlm_mask_batch(s_t, p_mask)
                if (target != SPECIAL_PAD).sum() == 0:
                    continue

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = self.model(masked_s, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                    loss = self._compute_focal_loss(logits, target)

                preds = logits.argmax(dim=-1).reshape(-1)
                targs = target.reshape(-1)
                valid_mask = targs != SPECIAL_PAD
                all_preds.append(preds[valid_mask].cpu())
                all_targs.append(targs[valid_mask].cpu())

                total_loss += loss.item()
                n_batches += 1

        if n_batches == 0:
            return empty

        avg_loss = total_loss / n_batches
        preds_cat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
        targs_cat = torch.cat(all_targs) if all_targs else torch.empty(0, dtype=torch.long)
        suf_acc, prec, rec, f1, _ = self._compute_metrics(preds_cat, targs_cat)

        return {
            'loss':        avg_loss,
            'suffix_acc':  suf_acc,
            'precision':   prec,
            'recall':      rec,
            'f1':          f1,
            'n_batches':   n_batches,
        }

    def train_persistent(
        self,
        training_data: List[Tuple],
        max_retries: int = None,
    ) -> float:
        confirmed_chains = []
        for (_, candidates, correct_idx) in training_data:
            if correct_idx < len(candidates):
                confirmed_chains.append(candidates[correct_idx])
            elif candidates:
                confirmed_chains.append(candidates[0])
            else:
                confirmed_chains.append([])
        return self.train_sentence(confirmed_chains)

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
        prefix_len = len(prefix_s)

        scores: List[float] = []
        with torch.no_grad():
            for chain in candidates:
                cand_s, cand_c, cand_g, cand_ct, cand_m, cand_wp, cand_wf = _chain_tokens([chain])
                num_cand_toks = len(cand_s) - 1
                if num_cand_toks <= 0:
                    scores.append(float(config.bare_root_prior_logprob))
                    continue

                full_s  = prefix_s  + cand_s  + right_s
                full_c  = prefix_c  + cand_c  + right_c
                full_g  = prefix_g  + cand_g  + right_g
                full_ct = prefix_ct + cand_ct + right_ct
                full_m  = prefix_m  + cand_m  + right_m
                full_wp = prefix_wp + cand_wp + right_wp
                full_wf = prefix_wf + cand_wf + right_wf
                L = len(full_s)

                base_s  = torch.tensor(full_s, dtype=torch.long, device=self.device)
                base_c  = torch.tensor(full_c, dtype=torch.long, device=self.device)
                base_g  = torch.tensor(full_g, dtype=torch.long, device=self.device)
                base_ct = torch.tensor(full_ct, dtype=torch.long, device=self.device)
                base_m  = torch.tensor(full_m, dtype=torch.long, device=self.device)
                base_wp = torch.tensor(full_wp, dtype=torch.long, device=self.device)
                base_wf = torch.tensor(full_wf, dtype=torch.long, device=self.device)

                batched_s  = base_s.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_c  = base_c.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_g  = base_g.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_ct = base_ct.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_m  = base_m.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_wp = base_wp.unsqueeze(0).expand(num_cand_toks, L).clone()
                batched_wf = base_wf.unsqueeze(0).expand(num_cand_toks, L).clone()

                positions = torch.arange(num_cand_toks, device=self.device) + prefix_len
                rows      = torch.arange(num_cand_toks, device=self.device)
                batched_s[rows, positions] = SPECIAL_MASK

                logits = self.model(
                    batched_s, batched_c, batched_g, batched_ct, batched_m, batched_wp, batched_wf
                )
                slot   = logits[rows, positions]                         
                log_p  = F.log_softmax(slot, dim=-1)
                true_toks = base_s[positions]                            
                per_tok   = log_p.gather(1, true_toks.unsqueeze(-1)).squeeze(-1)
                scores.append(per_tok.sum().item())

        return scores

    def score_sentence_chains(self, word_chains: List[List[EncodedToken]]) -> float:
        self.model.eval()
        full_s, full_c, full_g, full_ct, full_m, full_wp, full_wf = build_sentence_sequence(word_chains)
        bare_root_count = sum(1 for chain in word_chains if not chain)
        prior = bare_root_count * float(config.bare_root_prior_logprob)
        if len(full_s) < 2:
            return prior
        with torch.no_grad():
            tensors = self._to_tensor(full_s, full_c, full_g, full_ct, full_m, full_wp, full_wf)
            lp = self.model.log_probs(*tensors)
            return lp.sum().item() + prior

    def predict(
        self,
        candidates: List[List[EncodedToken]],
        context_chains: Optional[List[List[EncodedToken]]] = None,
    ) -> Tuple[int, List[float]]:
        ctx = context_chains or []
        scores = self.score_candidates(ctx, candidates)
        best = self._get_best_index(scores)
        return best, scores

    def batch_predict(
        self,
        batch_candidates: List[List[List[EncodedToken]]],
    ) -> List[Tuple[int, List[float]]]:
        results = []
        for candidates in batch_candidates:
            best_idx, scores = self.predict(candidates)
            results.append((best_idx, scores))
        return results

    def sentence_predict(
        self,
        all_candidates: List[List[List[EncodedToken]]],
    ) -> List[Tuple[int, List[float]]]:
        committed: List[List[EncodedToken]] = []
        results: List[Tuple[int, List[float]]] = []

        for candidates in all_candidates:
            scores  = self.score_candidates(committed, candidates)
            best    = self._get_best_index(scores)
            results.append((best, scores))
            committed.append(candidates[best])

        return results

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