"""Complete GPT language model."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

from model.block import Block

logger = logging.getLogger(__name__)


class GPT(nn.Module):
    """
    GPT Language Model.

    A decoder-only transformer for autoregressive language modeling.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: vocab_size -> n_embd
            wte=nn.Embedding(config.vocab_size, config.n_embd),

            # Position embeddings: block_size -> n_embd
            wpe=nn.Embedding(config.block_size, config.n_embd),

            # Dropout after embeddings
            drop=nn.Dropout(config.dropout),

            # Transformer blocks
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer norm
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language model head: n_embd -> vocab_size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        # This reduces parameters and often improves performance
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Log parameter count
        n_params = self.get_num_params()
        logger.info(f"Model initialized with {n_params/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters.

        Args:
            non_embedding: If True, exclude position embeddings
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices of shape (batch, seq_len), optional

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, \
            f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Create position indices: [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Get embeddings
        tok_emb = self.transformer.wte(idx)   # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (t, n_embd)

        # Combine and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (b, t, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens if any
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1 = more focused, >1 = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this probability mass

        Returns:
            Token indices of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
