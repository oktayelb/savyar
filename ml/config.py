from pathlib import Path
from dataclasses import dataclass

@dataclass
class MLConfig:

    model_path: Path = "ml/model.pt"
    device: str = "cuda"
    allow_cpu_fallback: bool = False

    embed_dim: int = 384        
    num_layers: int = 6         
    num_heads: int = 8          
    dropout: float = 0.3        

    group_embed_dim: int = 8
    wordpos_embed_dim: int = 16
    pos_embed_dim: int = 384

    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    mlm_mask_prob: float = 0.20
    mlm_use_bert_mix: bool = True
    mlm_ensure_one_mask: bool = True
    focal_gamma: float = 0.0


    max_negative_candidates: int = 5            
    negatives_per_ambiguous_word: float = 2.0   
    max_negative_candidates_cap: int = 16       
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
    
    ranking_temperature: float = 0.1
    mlm_weight: float = 0.2


    bulk_epochs: int = 6
    bulk_batch_size: int = 1024
    bulk_batch_log_interval: int = 25
    relearn_preprocess_log_interval: int = 1000
    max_batch_padded_tokens: int = 8192
    max_batch_attention_cells: int = 2_000_000
    cuda_oom_retries: int = 8
    auto_gpu_batch_sizing: bool = True
    gpu_memory_target_ratio: float = 0.85
    gpu_memory_safety_margin_bytes: int = 1_073_741_824
    gpu_attention_cell_bytes: int = 16
    gpu_token_bytes: int = 4096
    gpu_sequence_bytes: int = 1_048_576
    max_auto_candidate_sequences: int = 4096
    max_auto_padded_tokens: int = 1_048_576
    max_auto_attention_cells: int = 64_000_000
    max_auto_bulk_batch_size: int = 512


    warmup_steps: int = 500
    lr_eta_min_ratio: float = 0.05


    steps_per_update: int = 6

    checkpoint_frequency: int = 4000   
    validation_split: float = 0.1
    validation_seed: int = 42

config = MLConfig()
