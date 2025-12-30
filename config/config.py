import torch
import numpy as np
import random
from dataclasses import dataclass


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class GPTConfig:
    """Configuration for GPT model."""

    # Model architecture
    vocab_size: int = 50304      # GPT-2 vocab (50257) padded to nearest multiple of 64
    block_size: int = 256        # Maximum context length
    n_layer: int = 6             # Number of transformer blocks
    n_head: int = 6              # Number of attention heads
    n_embd: int = 384            # Embedding dimension
    dropout: float = 0.1         # Dropout rate
    bias: bool = False           # Use bias in Linear and LayerNorm layers

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        emb_params = self.vocab_size * self.n_embd + self.block_size * self.n_embd

        # Per transformer block
        attn_params = 4 * self.n_embd * self.n_embd  # Q, K, V, O projections
        ffn_params = 2 * self.n_embd * (4 * self.n_embd)  # Up and down projections
        ln_params = 4 * self.n_embd  # 2 layer norms per block
        block_params = attn_params + ffn_params + ln_params

        # Total
        total = emb_params + self.n_layer * block_params + 2 * self.n_embd  # Final LN
        return total


# Preset configurations
TINY_CONFIG = GPTConfig(n_layer=4, n_head=4, n_embd=256)    # ~3M params
SMALL_CONFIG = GPTConfig(n_layer=6, n_head=6, n_embd=384)   # ~10M params
MEDIUM_CONFIG = GPTConfig(n_layer=8, n_head=8, n_embd=512)  # ~25M params
LARGE_CONFIG = GPTConfig(n_layer=12, n_head=12, n_embd=768) # ~85M params
