"""Feed-forward network (MLP) for transformer."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.

    Expands to 4x hidden dimension then projects back.
    """

    def __init__(self, config):
        super().__init__()

        # Up projection: n_embd -> 4 * n_embd
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)

        # GELU activation (used in GPT-2)
        self.gelu = nn.GELU()

        # Down projection: 4 * n_embd -> n_embd
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
