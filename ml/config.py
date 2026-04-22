from pathlib import Path
from dataclasses import dataclass

# Dynamic Path Resolution
# Assumes structure: savyar/ml/config.py
# Base dir becomes:  savyar/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@dataclass
class MLConfig:
    # --- File Paths ---
    model_path: Path = "ml/model.pt"
    training_count_file: Path = DATA_DIR / "training_count.txt"
    
    # --- Model Architecture ---
    # Vocab size is dynamic (passed at runtime), others are static
    embed_dim: int = 512        # Doubled from 128 to give minority-class suffixes more room to separate.
    num_layers: int = 6         # 6 to prevent memorizing the 21,064 sequences.
    num_heads: int = 16          # 512 / 16 = 32; divides cleanly.
    dropout: float = 0.1       # Bumped alongside embed_dim to counter the extra capacity.

    # --- Training Hyperparameters ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.01  # Bumped alongside embed_dim to counter the extra capacity.


    use_class_weights: bool = False

    # --- MLM Objective ---
    # Mask probability for each eligible (non-PAD/SEP/BOS) token. 0.15 is BERT's
    # default for 512-token docs; for short Turkish chains (~8–15 tokens/seq)
    # it leaves too few tokens to learn from, so we push it up.
    mlm_mask_prob: float = 0.30
    # BERT's 80/10/10 mix: of selected tokens, 80% → MASK, 10% → random suffix,
    # 10% → keep original. Prevents over-reliance on MASK as a predict-me signal.
    mlm_use_bert_mix: bool = True
    # Guarantee at least one masked token per sequence with eligible positions,
    # so short sentences never contribute zero gradient due to unlucky draws.
    mlm_ensure_one_mask: bool = True

    # --- Loss ---
    # Focal loss focusing parameter. 0.0 = plain cross-entropy. MLM already has
    # sparse supervision; focal γ>0 further shrinks the loss early in training
    # and slows learning. Default off.
    focal_gamma: float = 0.0

    # --- Bulk-training defaults ---
    # Used when train_bulk() is called without explicit overrides. Epochs bumped
    # significantly because MLM gives ~1/5 the per-step signal of causal LM.
    bulk_epochs: int = 600
    bulk_batch_size: int = 128

    # --- LR Schedule ---
    # Linear warmup then cosine decay to eta_min_ratio * base_lr. Bidirectional
    # models under AMP are unstable in the first few hundred steps without warmup.
    warmup_steps: int = 1000
    lr_eta_min_ratio: float = 0.01

    # --- Experience Replay ---
    # Sized to safely hold the expanded dataset of 21,064 sentences in memory.
    replay_buffer_size: int = 22000
    replay_k: int = 64
    steps_per_update: int = 4

    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 1000

# Create the global config instance
config = MLConfig()