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
    device: str = "cuda"
    allow_cpu_fallback: bool = False
    
    # --- Model Architecture ---
    # Vocab size is dynamic (passed at runtime), others are static
    embed_dim: int = 384        # Main suffix identity dimension (Increased from 256)
    num_layers: int = 4         
    num_heads: int = 8          
    dropout: float = 0.3        

    # Feature embedding dimensions scaled by cardinality to prevent overfitting
    category_embed_dim: int = 4
    group_embed_dim: int = 8
    comes_makes_embed_dim: int = 2
    wordpos_embed_dim: int = 16
    wordfinal_embed_dim: int = 2

    # --- Training Hyperparameters ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.05  

    use_class_weights: bool = True

    # --- MLM Objective (Reintroduced for Regularization) ---
    mlm_mask_prob: float = 0.20
    mlm_use_bert_mix: bool = True
    mlm_ensure_one_mask: bool = True
    focal_gamma: float = 0.0

    # --- Ranking Objective ---
    max_negative_candidates: int = 8
    max_candidate_sequences_per_batch: int = 64
    max_sequence_length: int = 512
    use_torch_compile: bool = False
    hard_negative_count: int = 4
    medium_negative_count: int = 2
    easy_negative_count: int = 2
    dynamic_negative_pool_size: int = 100
    curriculum_generations: int = 3
    curriculum_warmup_epochs: int = 5
    curriculum_mining_epochs: int = 4
    
    # Dual Objective Weights
    ranking_temperature: float = 0.1
    mlm_weight: float = 0.2

    # --- Bulk-training defaults ---
    bulk_epochs: int = 11
    bulk_batch_size: int = 32
    bulk_batch_log_interval: int = 1
    relearn_preprocess_log_interval: int = 1000
    max_batch_padded_tokens: int = 8192
    max_batch_attention_cells: int = 2_000_000
    cuda_oom_retries: int = 8

    # --- LR Schedule ---
    warmup_steps: int = 1000
    lr_eta_min_ratio: float = 0.01

    # --- Experience Replay ---
    replay_buffer_size: int = 22000
    replay_k: int = 64
    steps_per_update: int = 4

    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 1000
    bare_root_prior_logprob: float = -0.75
    validation_split: float = 0.1
    validation_seed: int = 42

# Create the global config instance
config = MLConfig()
