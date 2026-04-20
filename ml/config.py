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
    embed_dim: int = 256        # Doubled from 128 to give minority-class suffixes more room to separate.
    num_layers: int = 3         # Reduced to 3 to prevent memorizing the 7,000 sequences.
    num_heads: int = 8          # 256 / 8 = 32; divides cleanly.
    dropout: float = 0.25       # Bumped alongside embed_dim to counter the extra capacity.

    # --- Training Hyperparameters ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.08  # Bumped alongside embed_dim to counter the extra capacity.

    # --- Loss weighting ---
    # When True, _gradient_steps / train_bulk pass inverse-sqrt-frequency
    # class weights into cross_entropy so minority suffix classes aren't
    # drowned out by PAD/WORD_SEP/common-case tokens.
    use_class_weights: bool = True

    # --- Experience Replay ---
    # Sized to hold the entire dataset of 7,000 sentences in memory.
    replay_buffer_size: int = 7000   
    replay_k: int = 64
    steps_per_update: int = 4       

    # --- Interactive/Loop Settings ---
    checkpoint_frequency: int = 1000  

# Create the global config instance
config = MLConfig()