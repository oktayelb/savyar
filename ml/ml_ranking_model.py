import math
import os
import random
import time
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
from typing import Any, List, Optional, Tuple, Dict
from .config import config  
from util.suffix import SuffixGroup, Type

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_gpu_enabled() -> bool:
    """Enable with SAVYAR_DEBUG_GPU=1 to print GPU/batch diagnostics."""
    return _env_flag("SAVYAR_DEBUG_GPU")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def resolve_torch_device(device: Optional[Any] = None) -> torch.device:
    requested = str(device or os.environ.get("SAVYAR_TORCH_DEVICE") or config.device).strip().lower()
    allow_cpu = bool(config.allow_cpu_fallback or _env_flag("SAVYAR_ALLOW_CPU"))

    if requested in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            warnings.warn("CUDA is unavailable; falling back to CPU because CPU fallback is enabled.")
            return torch.device("cpu")
        requested = "cuda"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        detail = (
            f"CUDA was requested, but PyTorch cannot use it "
            f"(torch.version.cuda={torch.version.cuda!r}, device_count={torch.cuda.device_count()})."
        )
        if allow_cpu:
            warnings.warn(f"{detail} Falling back to CPU because CPU fallback is enabled.")
            return torch.device("cpu")
        raise RuntimeError(
            f"{detail} Refusing to run training on CPU. Fix the CUDA/NVIDIA driver setup, "
            "set SAVYAR_TORCH_DEVICE=cpu, or set SAVYAR_ALLOW_CPU=1 to permit CPU fallback."
        )

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(requested)

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
    def __init__(
        self,
        suffix_vocab_size: int,
        closed_class_vocab_size: int = 0,
        device: Optional[Any] = None,
    ):
        target_device = resolve_torch_device(device)
        with torch.device(target_device):
            super().__init__()
            self.embed_dim = config.embed_dim
            self.vocab_size = SUFFIX_OFFSET + suffix_vocab_size + closed_class_vocab_size
            self.max_sequence_length = int(config.max_sequence_length)

            self.suffix_embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=SPECIAL_PAD)

            self.category_embed = nn.Embedding(4, config.category_embed_dim)
            self.group_embed = nn.Embedding(len(GROUP_TO_ID), config.group_embed_dim)
            self.comes_to_embed = nn.Embedding(max(TYPE_TO_ID.values()) + 1, config.comes_makes_embed_dim)
            self.makes_embed = nn.Embedding(max(TYPE_TO_ID.values()) + 1, config.comes_makes_embed_dim)
            self.wordpos_embed = nn.Embedding(64, config.wordpos_embed_dim)
            self.wordfinal_embed = nn.Embedding(2, config.wordfinal_embed_dim)

            self.pos_embed = nn.Embedding(self.max_sequence_length, self.embed_dim)

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
        if L > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {L} exceeds model max_sequence_length={self.max_sequence_length}"
            )
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
        if L > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {L} exceeds model max_sequence_length={self.max_sequence_length}"
            )
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
    def __init__(
        self,
        model: SentenceDisambiguator,
        path: Optional[str] = None,
        device: Optional[Any] = None,
    ):
        self.model = model

        self.checkpoint_frequency = config.checkpoint_frequency
        self.path                 = path if path is not None else str(config.model_path)

        self.device = resolve_torch_device(device)
        self.model.to(self.device)

        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        if not config.use_torch_compile or not torch.cuda.is_available() or not hasattr(torch, 'compile'):
            pass
        elif torch.version.cuda and hasattr(torch, 'compile'):
            import platform
            if platform.system() != 'Windows':
                try:
                    self.model = torch.compile(self.model)
                except Exception:
                    pass

        # IMPORTANT: use self.model here, not the original model argument.
        # If torch.compile wrapped the model above, the optimizer must track
        # the active module used in forward/backward.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Interactive fallback scheduler
        self.scheduler = self._build_schedule(self.optimizer)

        self.train_history: List[float] = []
        self.val_history:   List[float] = []
        self.best_val_loss  = float('inf')
        self.last_train_stats: Optional[Dict[str, Any]] = None
        self.last_validation_stats: Optional[Dict[str, Any]] = None
        self.last_validation_report: Optional[Dict[str, Any]] = None
        self.global_step    = 0

        self.replay_buffer: List[FlatSequence] = []
        self._class_weight_cache: Optional[torch.Tensor] = None
        self._adaptive_max_candidate_sequences = max(1, int(config.max_candidate_sequences_per_batch))
        self._adaptive_max_padded_tokens = max(1, int(config.max_batch_padded_tokens))
        self._adaptive_max_attention_cells = max(1, int(config.max_batch_attention_cells))

        try:
            self.load_checkpoint(self.path)
            print(f"Loaded model from {self.path}")
        except FileNotFoundError:
            print(f"Starting fresh (no checkpoint found at {self.path})")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

        if _debug_gpu_enabled():
            print(
                "[GPU DEBUG] Trainer initialized:",
                f"device={self.device}",
                f"cuda_available={torch.cuda.is_available()}",
                f"torch_cuda={torch.version.cuda}",
                f"device_count={torch.cuda.device_count()}",
                flush=True,
            )
            if self.device.type == "cuda":
                print(
                    "[GPU DEBUG] CUDA device:",
                    torch.cuda.get_device_name(self.device),
                    flush=True,
                )

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

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        value = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if value < 1024.0 or unit == "GiB":
                return f"{value:.1f}{unit}"
            value /= 1024.0
        return f"{value:.1f}GiB"

    def cuda_memory_report(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return f"device={self.device}"
        try:
            with torch.cuda.device(self.device):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            return (
                f"gpu_free={self._format_bytes(free_bytes)}/{self._format_bytes(total_bytes)} "
                f"allocated={self._format_bytes(allocated)} reserved={self._format_bytes(reserved)}"
            )
        except Exception:
            return f"device={self.device}"

    def _release_cuda_after_oom(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _shrink_adaptive_cuda_limits(self, flat_count: int, max_len: int) -> None:
        old_seq = self._adaptive_max_candidate_sequences
        old_tokens = self._adaptive_max_padded_tokens
        old_cells = self._adaptive_max_attention_cells

        if flat_count > 1:
            self._adaptive_max_candidate_sequences = max(1, min(old_seq, max(1, flat_count // 2)))
        if max_len > 0:
            current_tokens = max(1, flat_count * max_len)
            current_cells = max(1, flat_count * max_len * max_len)
            self._adaptive_max_padded_tokens = max(max_len, min(old_tokens, max(1, current_tokens // 2)))
            self._adaptive_max_attention_cells = max(
                max_len * max_len,
                min(old_cells, max(1, current_cells // 2)),
            )

        if (
            old_seq != self._adaptive_max_candidate_sequences
            or old_tokens != self._adaptive_max_padded_tokens
            or old_cells != self._adaptive_max_attention_cells
        ):
            print(
                "   CUDA OOM guard: shrinking future batches to "
                f"max_sequences={self._adaptive_max_candidate_sequences}, "
                f"max_padded_tokens={self._adaptive_max_padded_tokens}, "
                f"max_attention_cells={self._adaptive_max_attention_cells}.",
                flush=True,
            )

    @staticmethod
    def _empty_step_result() -> Dict[str, Any]:
        return {
            'loss_sum': 0.0,
            'rank_loss_sum': 0.0,
            'optimizer_steps': 0,
            'candidate_sets': 0,
            'correct': 0,
            'top2': 0,
            'top3': 0,
            'margin_sum': 0.0,
            'margin_count': 0,
            'skipped': 0,
            'adaptive_splits': 0,
        }

    @classmethod
    def _merge_step_results(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = cls._empty_step_result()
        for result in results:
            for key in merged:
                merged[key] += result.get(key, 0)
        return merged

    @staticmethod
    def _candidate_set_max_len(cands: List[FlatSequence]) -> int:
        return max((len(seq[0]) for seq in cands), default=0)

    def _candidate_set_exceeds_budget(self, cands: List[FlatSequence]) -> bool:
        if not cands:
            return False
        seq_count = len(cands)
        max_len = self._candidate_set_max_len(cands)
        return (
            seq_count > self._adaptive_max_candidate_sequences
            or seq_count * max_len > self._adaptive_max_padded_tokens
            or seq_count * max_len * max_len > self._adaptive_max_attention_cells
        )

    def _split_candidate_set_by_budget(self, cands: List[FlatSequence]) -> List[List[FlatSequence]]:
        if len(cands) <= 2 or not self._candidate_set_exceeds_budget(cands):
            return [cands]

        gold = cands[0]
        groups: List[List[FlatSequence]] = []
        current: List[FlatSequence] = [gold]
        for neg in cands[1:]:
            proposed = current + [neg]
            if len(current) > 1 and self._candidate_set_exceeds_budget(proposed):
                groups.append(current)
                current = [gold, neg]
            else:
                current = proposed
        if len(current) >= 2:
            groups.append(current)
        return groups or [cands]

    def _budget_candidate_sets(self, candidate_sets: List[List[FlatSequence]]) -> List[List[FlatSequence]]:
        budgeted: List[List[FlatSequence]] = []
        split_count = 0
        for cands in candidate_sets:
            groups = self._split_candidate_set_by_budget(cands)
            split_count += max(0, len(groups) - 1)
            budgeted.extend(groups)
        if split_count:
            print(
                f"   CUDA OOM guard: split {split_count} oversized candidate-set chunks "
                "before training.",
                flush=True,
            )
        return budgeted

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

        suffix_rows: List[List[int]] = []
        category_rows: List[List[int]] = []
        group_rows: List[List[int]] = []
        comes_to_rows: List[List[int]] = []
        makes_rows: List[List[int]] = []
        word_pos_rows: List[List[int]] = []
        word_final_rows: List[List[int]] = []
        pad_mask_rows: List[List[bool]] = []

        for sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final in seqs:
            L = len(sids)
            pad_count = max_len - L
            suffix_rows.append(list(sids) + [SPECIAL_PAD] * pad_count)
            category_rows.append(list(cids) + [CATEGORY_SPECIAL] * pad_count)
            group_rows.append(list(gids) + [SPECIAL_FEATURE_ID] * pad_count)
            comes_to_rows.append(list(comes_to_ids) + [SPECIAL_FEATURE_ID] * pad_count)
            makes_rows.append(list(makes_ids) + [SPECIAL_FEATURE_ID] * pad_count)
            word_pos_rows.append(list(word_pos_ids) + [SPECIAL_FEATURE_ID] * pad_count)
            word_final_rows.append(list(word_final) + [WORD_FINAL_NO] * pad_count)
            pad_mask_rows.append([False] * L + [True] * pad_count)

        return (
            torch.tensor(suffix_rows, dtype=torch.long, device=self.device),
            torch.tensor(category_rows, dtype=torch.long, device=self.device),
            torch.tensor(group_rows, dtype=torch.long, device=self.device),
            torch.tensor(comes_to_rows, dtype=torch.long, device=self.device),
            torch.tensor(makes_rows, dtype=torch.long, device=self.device),
            torch.tensor(word_pos_rows, dtype=torch.long, device=self.device),
            torch.tensor(word_final_rows, dtype=torch.long, device=self.device),
            torch.tensor(pad_mask_rows, dtype=torch.bool, device=self.device),
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

    def _ranking_step_once(self, candidate_sets: List[List[FlatSequence]]) -> Dict[str, Any]:
        candidate_sets = [cands for cands in candidate_sets if len(cands) >= 2]
        if not candidate_sets:
            return self._empty_step_result()

        flat: List[FlatSequence] = []
        sizes: List[int] = []
        for cands in candidate_sets:
            sizes.append(len(cands))
            flat.extend(cands)

        s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(flat)

        if _debug_gpu_enabled():
            print(
                "[GPU DEBUG] training batch:",
                f"sets={len(candidate_sets)}",
                f"flat_sequences={len(flat)}",
                f"tensor_shape={tuple(s_t.shape)}",
                f"device={s_t.device}",
                f"global_step={self.global_step}",
                flush=True,
            )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        use_amp = self.device.type == 'cuda'
        temperature = max(float(config.ranking_temperature), 1e-6)
        mlm_weight = float(config.mlm_weight)

        with torch.amp.autocast('cuda', enabled=use_amp):
            # 1. Contrastive Ranking Loss
            scores = self.model.rank_scores(s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
            losses = []
            correct = 0
            top2 = 0
            top3 = 0
            margin_sum = 0.0
            margin_count = 0
            gold_indices = []
            offset = 0
            for size in sizes:
                gold_indices.append(offset)
                group = scores[offset:offset + size]
                logits = (group / temperature).unsqueeze(0)
                target = torch.zeros(1, dtype=torch.long, device=self.device)
                losses.append(F.cross_entropy(logits, target))

                detached_group = group.detach()
                best_idx = int(torch.argmax(detached_group).item())
                if best_idx == 0:
                    correct += 1
                if self._topk_hit_tensor(detached_group, 2):
                    top2 += 1
                if self._topk_hit_tensor(detached_group, 3):
                    top3 += 1
                if detached_group.numel() > 1:
                    margin_sum += float((detached_group[0] - detached_group[1:].max()).item())
                    margin_count += 1
                offset += size
            rank_loss = torch.stack(losses).mean()

            # 2. Masked Language Modeling Loss (on the gold sequences only)
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

        final_loss = float(total_loss.item())
        final_rank = float(rank_loss.item())
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        scaler_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if not use_amp or self.scaler.get_scale() >= scaler_scale:
            self.scheduler.step()
        self.global_step += 1

        result = self._empty_step_result()
        result.update({
            'loss_sum': final_loss,
            'rank_loss_sum': final_rank,
            'optimizer_steps': 1,
            'candidate_sets': len(candidate_sets),
            'correct': correct,
            'top2': top2,
            'top3': top3,
            'margin_sum': margin_sum,
            'margin_count': margin_count,
        })
        return result

    def _ranking_step(self, candidate_sets: List[List[FlatSequence]], oom_depth: int = 0) -> Dict[str, Any]:
        candidate_sets = [cands for cands in candidate_sets if len(cands) >= 2]
        if not candidate_sets:
            return self._empty_step_result()

        flat_count = sum(len(cands) for cands in candidate_sets)
        max_len = max((len(seq[0]) for cands in candidate_sets for seq in cands), default=0)
        try:
            return self._ranking_step_once(candidate_sets)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            self._release_cuda_after_oom()
            self._shrink_adaptive_cuda_limits(flat_count, max_len)
            print(
                "   CUDA OOM guard: training batch did not fit "
                f"(sets={len(candidate_sets)}, seqs={flat_count}, max_len={max_len}, "
                f"{self.cuda_memory_report()}); retrying smaller.",
                flush=True,
            )

            max_retries = max(1, int(config.cuda_oom_retries))
            if oom_depth >= max_retries:
                print(
                    "   CUDA OOM guard: retry limit reached; skipping "
                    f"{len(candidate_sets)} candidate sets.",
                    flush=True,
                )
                skipped = self._empty_step_result()
                skipped['skipped'] = len(candidate_sets)
                return skipped

            if len(candidate_sets) > 1:
                mid = max(1, len(candidate_sets) // 2)
                result = self._merge_step_results([
                    self._ranking_step(candidate_sets[:mid], oom_depth + 1),
                    self._ranking_step(candidate_sets[mid:], oom_depth + 1),
                ])
                result['adaptive_splits'] += 1
                return result

            cands = candidate_sets[0]
            if len(cands) > 2:
                negs = cands[1:]
                mid = max(1, len(negs) // 2)
                parts = [
                    [cands[0]] + negs[:mid],
                    [cands[0]] + negs[mid:],
                ]
                parts = [part for part in parts if len(part) >= 2]
                result = self._merge_step_results([
                    self._ranking_step([part], oom_depth + 1)
                    for part in parts
                ])
                result['adaptive_splits'] += 1
                return result

            print(
                "   CUDA OOM guard: skipping one candidate set that does not fit even "
                f"as gold-vs-one-negative (max_len={max_len}).",
                flush=True,
            )
            skipped = self._empty_step_result()
            skipped['skipped'] = 1
            return skipped

    def _candidate_batch_count(self, candidate_sets: List[List[FlatSequence]], batch_size: int) -> int:
        return len(self._candidate_batches(candidate_sets, batch_size))

    def _candidate_batches(self, candidate_sets: List[List[FlatSequence]], batch_size: int) -> List[List[List[FlatSequence]]]:
        batches: List[List[List[FlatSequence]]] = []
        current: List[List[FlatSequence]] = []
        current_sequences = 0
        current_max_len = 0
        max_sequences = max(1, self._adaptive_max_candidate_sequences)
        max_tokens = max(1, self._adaptive_max_padded_tokens)
        max_attention_cells = max(1, self._adaptive_max_attention_cells)

        for cands in candidate_sets:
            cand_count = len(cands)
            cand_max_len = self._candidate_set_max_len(cands)
            next_sequences = current_sequences + cand_count
            next_max_len = max(current_max_len, cand_max_len)
            would_exceed_sets = len(current) >= batch_size
            would_exceed_sequences = bool(current) and next_sequences > max_sequences
            would_exceed_tokens = bool(current) and next_sequences * next_max_len > max_tokens
            would_exceed_attention = bool(current) and next_sequences * next_max_len * next_max_len > max_attention_cells
            if would_exceed_sets or would_exceed_sequences or would_exceed_tokens or would_exceed_attention:
                batches.append(current)
                current = []
                current_sequences = 0
                current_max_len = 0
            current.append(cands)
            current_sequences += cand_count
            current_max_len = max(current_max_len, cand_max_len)

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

    @staticmethod
    def _topk_hit(scores: List[float], k: int) -> bool:
        if not scores:
            return False
        topk = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, k)]
        return 0 in topk

    @staticmethod
    def _topk_hit_tensor(scores: torch.Tensor, k: int) -> bool:
        if scores.numel() == 0:
            return False
        topk = scores.topk(min(max(1, k), scores.numel())).indices
        return bool((topk == 0).any().item())

    @staticmethod
    def _suffix_name_for_token_id(token_id: int) -> Optional[str]:
        suffixes = _get_all_suffixes()
        suffix_idx = token_id - SUFFIX_OFFSET
        if 0 <= suffix_idx < len(suffixes):
            return suffixes[suffix_idx].name
        return None

    @classmethod
    def _update_suffix_metric_buckets(
        cls,
        suffix_buckets: Dict[str, Dict[str, int]],
        gold_tokens: List[int],
        pred_tokens: List[int],
    ) -> None:
        for gold_tok, pred_tok in zip(gold_tokens, pred_tokens):
            gold_name = cls._suffix_name_for_token_id(gold_tok)
            pred_name = cls._suffix_name_for_token_id(pred_tok)

            if gold_name is not None:
                bucket = suffix_buckets.setdefault(
                    gold_name,
                    {'tp': 0, 'fp': 0, 'fn': 0, 'gold_count': 0, 'pred_count': 0},
                )
                bucket['gold_count'] += 1

            if pred_name is not None:
                bucket = suffix_buckets.setdefault(
                    pred_name,
                    {'tp': 0, 'fp': 0, 'fn': 0, 'gold_count': 0, 'pred_count': 0},
                )
                bucket['pred_count'] += 1

            if gold_name is not None and gold_name == pred_name:
                suffix_buckets[gold_name]['tp'] += 1
                continue

            if gold_name is not None:
                suffix_buckets[gold_name]['fn'] += 1
            if pred_name is not None:
                suffix_buckets[pred_name]['fp'] += 1

    @classmethod
    def _finalize_suffix_metric_buckets(
        cls,
        suffix_buckets: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, float]]:
        suffixes = _get_all_suffixes()
        finalized: Dict[str, Dict[str, float]] = {}
        for suffix in suffixes:
            counts = suffix_buckets.get(
                suffix.name,
                {'tp': 0, 'fp': 0, 'fn': 0, 'gold_count': 0, 'pred_count': 0},
            )
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            gold_count = counts['gold_count']
            pred_count = counts['pred_count']
            denom = max(gold_count, pred_count)
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / gold_count if gold_count else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )
            finalized[suffix.name] = {
                'group': suffix.group.name if suffix.group else None,
                'accuracy': tp / denom if denom else 0.0,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gold_count': gold_count,
                'pred_count': pred_count,
            }
        return finalized

    @classmethod
    def _aggregate_group_metrics(
        cls,
        suffix_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        group_buckets: Dict[str, Dict[str, float]] = {}
        for suffix_name, metrics in suffix_metrics.items():
            group_name = metrics.get('group') or 'UNGROUPED'
            bucket = group_buckets.setdefault(
                group_name,
                {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'gold_count': 0.0, 'pred_count': 0.0},
            )
            bucket['tp'] += float(metrics.get('tp', 0))
            bucket['fp'] += float(metrics.get('fp', 0))
            bucket['fn'] += float(metrics.get('fn', 0))
            bucket['gold_count'] += float(metrics.get('gold_count', 0))
            bucket['pred_count'] += float(metrics.get('pred_count', 0))

        finalized: Dict[str, Dict[str, float]] = {}
        for group_name, counts in group_buckets.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            gold_count = counts['gold_count']
            pred_count = counts['pred_count']
            denom = max(gold_count, pred_count)
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / gold_count if gold_count else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0.0
                else 0.0
            )
            finalized[group_name] = {
                'accuracy': tp / denom if denom else 0.0,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gold_count': gold_count,
                'pred_count': pred_count,
            }
        return finalized

    def train_sentence(
        self,
        word_chains: List[List[EncodedToken]],
        negative_word_chains: Optional[List[List[List[EncodedToken]]]] = None,
        max_retries: int = None,
    ) -> float:
        gold_seq = self._sequence_from_chains(word_chains)
        if len(gold_seq[0]) < 2:
            return 0.0
        max_len = int(config.max_sequence_length)
        if len(gold_seq[0]) > max_len:
            print(
                f"   Skipping training example: sequence length {len(gold_seq[0])} "
                f"exceeds max_sequence_length={max_len}.",
                flush=True,
            )
            return 0.0

        self._add_to_replay(*gold_seq)
        negatives = negative_word_chains or []
        candidate_set = [gold_seq]
        for neg in negatives:
            neg_seq = self._sequence_from_chains(neg)
            if neg_seq != gold_seq and len(neg_seq[0]) <= max_len:
                candidate_set.append(neg_seq)

        if len(candidate_set) < 2:
            return 0.0

        print(f"   Ranking gold against {len(candidate_set) - 1} negatives...", end="", flush=True)
        final_loss = 0.0
        for _ in range(config.steps_per_update):
            result = self._ranking_step([candidate_set])
            if result['optimizer_steps']:
                final_loss = result['loss_sum'] / result['optimizer_steps']
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
        self.last_train_stats = None
        self.last_validation_stats = None
        self.last_validation_report = None

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

        trainable_sets = self._budget_candidate_sets([cands for cands in candidate_sets if len(cands) >= 2])

        if _debug_gpu_enabled():
            total_candidates = sum(len(cands) for cands in candidate_sets)
            trainable_candidates = sum(len(cands) for cands in trainable_sets)
            print(
                "[GPU DEBUG] train_bulk input:",
                f"device={self.device}",
                f"cuda_available={torch.cuda.is_available()}",
                f"total_sets={len(candidate_sets)}",
                f"trainable_sets={len(trainable_sets)}",
                f"skipped_sets_lt_2_candidates={len(candidate_sets) - len(trainable_sets)}",
                f"total_candidate_sequences={total_candidates}",
                f"trainable_candidate_sequences={trainable_candidates}",
                f"batch_size={batch_size}",
                f"max_candidate_sequences_per_batch={self._adaptive_max_candidate_sequences}",
                f"max_batch_padded_tokens={self._adaptive_max_padded_tokens}",
                f"max_batch_attention_cells={self._adaptive_max_attention_cells}",
                flush=True,
            )

        if not trainable_sets:
            print(
                "No trainable candidate sets found. Each training item needs at least "
                "one gold sequence and one negative/candidate sequence; otherwise no "
                "optimizer step is run and the GPU will remain idle."
            )
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
        print(
            "   Bulk training plan: "
            f"sets={len(data)}, epochs={epochs}, batch_size={batch_size}, "
            f"max_sequences={self._adaptive_max_candidate_sequences}, "
            f"max_padded_tokens={self._adaptive_max_padded_tokens}, "
            f"max_attention_cells={self._adaptive_max_attention_cells}, "
            f"{self.cuda_memory_report()}",
            flush=True,
        )
        for epoch in range(epochs):
            random.shuffle(data)
            epoch_loss = 0.0
            epoch_rank_loss = 0.0
            n_batches = 0
            correct = 0
            top2 = 0
            top3 = 0
            total = 0
            margin_sum = 0.0
            margin_count = 0
            skipped = 0
            epoch_start = time.monotonic()
            batches = self._candidate_batches(data, batch_size)
            log_interval = max(1, int(config.bulk_batch_log_interval))

            batch_idx = 0
            while batch_idx < len(batches):
                batch_sets = batches[batch_idx]
                batch_idx += 1
                batch_start = time.monotonic()
                result = self._ranking_step(batch_sets)
                if result['optimizer_steps'] == 0:
                    skipped += result.get('skipped', 0)
                    if result.get('adaptive_splits'):
                        remaining = [cands for batch in batches[batch_idx:] for cands in batch]
                        batches = batches[:batch_idx] + self._candidate_batches(remaining, batch_size)
                    continue
                epoch_loss += result['loss_sum']
                epoch_rank_loss += result['rank_loss_sum']
                n_batches += result['optimizer_steps']
                correct += result['correct']
                top2 += result['top2']
                top3 += result['top3']
                total += result['candidate_sets']
                margin_sum += result['margin_sum']
                margin_count += result['margin_count']
                skipped += result.get('skipped', 0)

                if result.get('adaptive_splits'):
                    remaining = [cands for batch in batches[batch_idx:] for cands in batch]
                    batches = batches[:batch_idx] + self._candidate_batches(remaining, batch_size)

                if batch_idx == 1 or batch_idx == len(batches) or batch_idx % log_interval == 0:
                    batch_loss = result['loss_sum'] / max(result['optimizer_steps'], 1)
                    batch_rank = result['rank_loss_sum'] / max(result['optimizer_steps'], 1)
                    batch_total = max(result['candidate_sets'], 1)
                    batch_acc = result['correct'] / batch_total
                    batch_top2 = result['top2'] / batch_total
                    batch_top3 = result['top3'] / batch_total
                    batch_margin = (
                        result['margin_sum'] / result['margin_count']
                        if result['margin_count']
                        else 0.0
                    )
                    flat_sequences = sum(len(cands) for cands in batch_sets)
                    max_len = max((len(seq[0]) for cands in batch_sets for seq in cands), default=0)
                    elapsed = time.monotonic() - batch_start
                    lr = self.optimizer.param_groups[0].get('lr', 0.0)
                    split_note = (
                        f" splits={result['adaptive_splits']}"
                        if result.get('adaptive_splits')
                        else ""
                    )
                    print(
                        f"   Epoch {epoch + 1}/{epochs} batch {batch_idx}/{len(batches)}: "
                        f"loss={batch_loss:.4f} rank_loss={batch_rank:.4f} "
                        f"RankAcc={batch_acc:.4f} Top2={batch_top2:.4f} Top3={batch_top3:.4f} "
                        f"margin={batch_margin:.4f} sets={result['candidate_sets']} "
                        f"seqs={flat_sequences} max_len={max_len} steps={result['optimizer_steps']}"
                        f"{split_note} lr={lr:.2e} time={elapsed:.1f}s {self.cuda_memory_report()}",
                        flush=True,
                    )

            if n_batches:
                avg = epoch_loss / n_batches
                avg_rank = epoch_rank_loss / n_batches
                final_loss = avg
                acc = correct / total if total else 0.0
                top2_acc = top2 / total if total else 0.0
                top3_acc = top3 / total if total else 0.0
                mean_margin = margin_sum / margin_count if margin_count else 0.0
                self.last_train_stats = {
                    'loss': avg,
                    'rank_acc': acc,
                    'top2_acc': top2_acc,
                    'top3_acc': top3_acc,
                    'margin': mean_margin,
                    'n_batches': n_batches,
                    'total': total,
                    'skipped': skipped,
                }
                print(
                    f"   Bulk epoch {epoch+1}/{epochs}: loss={avg:.4f} | rank_loss={avg_rank:.4f} | "
                    f"RankAcc={acc:.4f} | Top2={top2_acc:.4f} | Top3={top3_acc:.4f} | margin={mean_margin:.4f} "
                    f"({n_batches} optimizer steps, {len(batches)} logged batches, {total} candidate sets, "
                    f"{skipped} skipped, {time.monotonic() - epoch_start:.1f}s)"
                )

            if validation_seqs:
                val_stats = self.validate(validation_seqs, batch_size=batch_size)
                self.last_validation_stats = val_stats
                self.last_validation_report = val_stats
                self.val_history.append(val_stats['loss'])
                if val_stats['loss'] < self.best_val_loss:
                    self.best_val_loss = val_stats['loss']
                val_header = (
                    f"   Validation   : rank_loss={val_stats['loss']:.4f} | "
                    f"RankAcc={val_stats['rank_acc']:.4f} | "
                    f"Top2={val_stats['top2_acc']:.4f} | "
                    f"Top3={val_stats['top3_acc']:.4f} | "
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
    ) -> Dict[str, Any]:
        empty = {
            'loss': 0.0,
            'rank_acc': 0.0,
            'top2_acc': 0.0,
            'top3_acc': 0.0,
            'suff_acc': 0.0,
            'suff_precision': 0.0,
            'suff_recall': 0.0,
            'suff_f1': 0.0,
            'margin': 0.0,
            'n_batches': 0,
            'suffix_metrics': {},
            'suffix_group_metrics': {},
        }
        if not val_seqs:
            return empty

        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        correct = 0
        top2 = 0
        top3 = 0
        total = 0
        suff_acc_total = 0.0
        suff_matches = 0
        suff_gold_total = 0
        suff_pred_total = 0
        margins: List[float] = []
        suffix_buckets: Dict[str, Dict[str, int]] = {}

        with torch.no_grad():
            for start in range(0, len(val_seqs), batch_size):
                raw_batch_sets = [list(s) for s in val_seqs[start:start + batch_size] if len(s) >= 2]
                for batch_sets in self._candidate_batches(raw_batch_sets, batch_size):
                    flat = [seq for cands in batch_sets for seq in cands]
                    sizes = [len(cands) for cands in batch_sets]
                    scores = self.score_flat_sequences_tensor(flat)
                    offset = 0
                    losses = []
                    for set_idx, size in enumerate(sizes):
                        group = scores[offset:offset + size]
                        # Must apply temperature during validation loss calc for parity with training
                        logits = (group / config.ranking_temperature).unsqueeze(0)
                        target = torch.zeros(1, dtype=torch.long, device=self.device)
                        losses.append(F.cross_entropy(logits, target).item())
                        total += 1
                        best_idx = int(torch.argmax(group).item())
                        if best_idx == 0:
                            correct += 1
                        if self._topk_hit_tensor(group, 2):
                            top2 += 1
                        if self._topk_hit_tensor(group, 3):
                            top3 += 1
                        gold_seq = batch_sets[set_idx][0]
                        pred_seq = batch_sets[set_idx][best_idx]
                        suff_acc_total += self._suffix_token_accuracy(gold_seq, pred_seq)
                        matches, gold_count, pred_count = self._suffix_token_stats(gold_seq, pred_seq)
                        suff_matches += matches
                        suff_gold_total += gold_count
                        suff_pred_total += pred_count
                        gold_tokens = [
                            tok for tok in gold_seq[0]
                            if tok not in (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS)
                        ]
                        pred_tokens = [
                            tok for tok in pred_seq[0]
                            if tok not in (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS)
                        ]
                        self._update_suffix_metric_buckets(suffix_buckets, gold_tokens, pred_tokens)
                        if group.numel() > 1:
                            margins.append(float((group[0] - group[1:].max()).item()))
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
        suffix_metrics = self._finalize_suffix_metric_buckets(suffix_buckets)
        suffix_group_metrics = self._aggregate_group_metrics(suffix_metrics)

        return {
            'loss':        avg_loss,
            'rank_acc':    correct / total if total else 0.0,
            'top2_acc':    top2 / total if total else 0.0,
            'top3_acc':    top3 / total if total else 0.0,
            'suff_acc':    suff_acc_total / total if total else 0.0,
            'suff_precision': suff_precision,
            'suff_recall': suff_recall,
            'suff_f1':     suff_f1,
            'margin':      sum(margins) / len(margins) if margins else 0.0,
            'n_batches':   n_batches,
            'suffix_metrics': suffix_metrics,
            'suffix_group_metrics': suffix_group_metrics,
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
        return self.score_flat_sequences_tensor(seqs).detach().cpu().tolist()

    def score_flat_sequences_tensor(self, seqs: List[FlatSequence]) -> torch.Tensor:
        if not seqs:
            return torch.empty(0, dtype=torch.float, device=self.device)
        self.model.eval()
        chunk_size = max(1, self._adaptive_max_candidate_sequences)
        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            start = 0
            while start < len(seqs):
                current_size = min(chunk_size, len(seqs) - start)
                chunk = seqs[start:start + current_size]
                max_len = max((len(seq[0]) for seq in chunk), default=0)
                while current_size > 1 and (
                    current_size * max_len > self._adaptive_max_padded_tokens
                    or current_size * max_len * max_len > self._adaptive_max_attention_cells
                ):
                    current_size = max(1, current_size // 2)
                    chunk = seqs[start:start + current_size]
                    max_len = max((len(seq[0]) for seq in chunk), default=0)
                try:
                    s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, p_mask = self._build_padded_batch(chunk)
                    scores = self.model.rank_scores(s_t, c_t, g_t, ct_t, m_t, wp_t, wf_t, pad_mask=p_mask)
                    chunks.append(scores.detach())
                    start += current_size
                except RuntimeError as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    self._release_cuda_after_oom()
                    self._shrink_adaptive_cuda_limits(current_size, max_len)
                    if current_size <= 1:
                        print(
                            "   CUDA OOM guard: one sequence could not be scored; assigning a low score "
                            f"(max_len={max_len}, {self.cuda_memory_report()}).",
                            flush=True,
                        )
                        chunks.append(torch.full((1,), -1e4, dtype=torch.float, device=self.device))
                        start += 1
                    else:
                        chunk_size = max(1, current_size // 2)
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float, device=self.device)

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
