"""Transformer block combining attention and MLP."""
import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.mlp import MLP


class Block(nn.Module):
    """
    Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP

    Uses pre-norm architecture (LayerNorm before attention/MLP).
    """

    def __init__(self, config):
        super().__init__()

        # Pre-attention layer norm
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Self-attention
        self.attn = CausalSelfAttention(config)

        # Pre-MLP layer norm
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Feed-forward network
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        # Self-attention with residual
        x = x + self.attn(self.ln_1(x))

        # MLP with residual
        x = x + self.mlp(self.ln_2(x))

        return x
