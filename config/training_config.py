"""Training configuration for GPT model."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # === Optimization ===
    learning_rate: float = 3e-4       # Peak LR (after warmup)
    min_lr: float = 3e-5              # Minimum LR (10% of peak)
    weight_decay: float = 0.1         # AdamW weight decay
    beta1: float = 0.9                # Adam beta1
    beta2: float = 0.95               # Adam beta2 (0.95 better for LLMs)
    grad_clip: float = 1.0            # Gradient clipping threshold

    # === Schedule ===
    max_iters: int = 5000             # Total training iterations
    warmup_iters: int = 100           # Linear warmup steps
    lr_decay_iters: int = 5000        # Cosine decay length (match max_iters)

    # === Evaluation ===
    eval_interval: int = 500          # Evaluate every N iters
    eval_iters: int = 200             # Batches for loss estimation

    # === Checkpointing ===
    checkpoint_interval: int = 1000   # Save every N iters
    checkpoint_dir: str = "checkpoints"

    # === Logging ===
    log_interval: int = 10            # Log every N iters
    wandb_project: Optional[str] = "tinystories-gpt"
    wandb_run_name: Optional[str] = None

    # === Data ===
    batch_size: int = 64
    block_size: int = 256
    max_train_stories: Optional[int] = None  # None = use all

    # === Device & Performance ===
    device: str = "auto"              # "auto", "cuda", "cpu", "mps"
    compile_model: bool = True        # torch.compile (PyTorch 2.0+)
    mixed_precision: bool = True      # AMP (FP16 on CUDA)
