# Phase 2: Transformer Model Architecture

## Overview

Build a GPT-style decoder-only transformer from scratch. This phase creates the core neural network components.

**Estimated Time**: 5-6 hours
**Dependencies**: Phase 0 & 1 complete, all tests passing

---

## Architecture Diagram

```
Input Token IDs [batch, seq_len]
        │
        ▼
┌───────────────────────────────────────┐
│  Token Embedding    (vocab → n_embd)  │
│  + Position Embedding (pos → n_embd)  │
│  + Dropout                            │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     Transformer Block (× n_layer)     │
│  ┌─────────────────────────────────┐  │
│  │ LayerNorm                       │  │
│  │ CausalSelfAttention             │  │
│  │ + Residual Connection           │  │
│  ├─────────────────────────────────┤  │
│  │ LayerNorm                       │  │
│  │ MLP (FFN)                       │  │
│  │ + Residual Connection           │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Final LayerNorm                      │
│  Linear Head (n_embd → vocab_size)    │
└───────────────────────────────────────┘
        │
        ▼
Output Logits [batch, seq_len, vocab_size]
```

---

## Step 1: CausalSelfAttention (`model/attention.py`)

### Purpose
Multi-head self-attention with causal masking to prevent attending to future tokens.

### Key Concepts
- **Query, Key, Value**: Three projections of input, used to compute attention
- **Scaled Dot-Product**: `softmax(QK^T / sqrt(d_k)) × V`
- **Causal Mask**: Lower triangular matrix, blocks future positions
- **Multi-Head**: Split into `n_head` parallel attention operations

### Implementation Details

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # Combined Q, K, V projection: (n_embd) → (3 * n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # Output projection: (n_embd) → (n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Causal mask (registered as buffer, not parameter)
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer("mask", torch.tril(...))

    def forward(self, x):  # x: (B, T, C)
        # 1. Project to Q, K, V
        # 2. Reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
        # 3. Compute attention scores: Q @ K^T / sqrt(head_dim)
        # 4. Apply causal mask (set future to -inf)
        # 5. Softmax + dropout
        # 6. Apply to values: att @ V
        # 7. Reshape back: (B, n_head, T, head_dim) → (B, T, C)
        # 8. Output projection
```

### Shapes Flow
```
Input:  (B, T, n_embd)
Q,K,V:  (B, T, n_embd) each
Split:  (B, n_head, T, head_dim)
Scores: (B, n_head, T, T)
Output: (B, T, n_embd)
```

---

## Step 2: MLP (`model/mlp.py`)

### Purpose
Position-wise feed-forward network. Processes each position independently.

### Architecture
```
Input (n_embd) → Linear (4 * n_embd) → GELU → Linear (n_embd) → Dropout → Output
```

### Implementation Details

```python
class MLP(nn.Module):
    def __init__(self, config):
        # Expand: n_embd → 4 * n_embd
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)

        # GELU activation (smoother than ReLU)
        self.gelu = nn.GELU()

        # Contract: 4 * n_embd → n_embd
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

### Why 4x Expansion?
Standard transformer design. Allows the network to learn more complex transformations before projecting back.

---

## Step 3: Transformer Block (`model/block.py`)

### Purpose
Single transformer layer combining attention and MLP with residual connections.

### Architecture (Pre-Norm)
```
x ──┬── LayerNorm → Attention ──┬── + ──┬── LayerNorm → MLP ──┬── + ── output
    │                           │       │                     │
    └───────── (residual) ──────┘       └───── (residual) ────┘
```

### Implementation Details

```python
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Attention with residual
        x = x + self.mlp(self.ln_2(x))   # MLP with residual
        return x
```

### Why Pre-Norm?
- LayerNorm before (not after) sublayers
- More stable training, especially for deep networks
- Used in GPT-2 and most modern transformers

---

## Step 4: GPT Model (`model/gpt.py`)

### Purpose
Complete language model: embeddings, transformer blocks, output head, generation.

### Components

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),    # Token embeddings
            wpe = nn.Embedding(block_size, n_embd),   # Position embeddings
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(config) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=False),  # Final layer norm
        ))

        # Output head (shares weights with token embeddings)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying!
```

### Forward Pass

```python
def forward(self, idx, targets=None):
    # idx: (B, T) token indices

    # 1. Get embeddings
    tok_emb = self.transformer.wte(idx)      # (B, T, n_embd)
    pos_emb = self.transformer.wpe(positions) # (T, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)

    # 2. Apply transformer blocks
    for block in self.transformer.h:
        x = block(x)

    # 3. Final layer norm + output projection
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    # 4. Compute loss if targets provided
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    return logits, loss
```

### Generate Method

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = idx[:, -block_size:]

        # Get predictions
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # Last position only

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat([idx, idx_next], dim=1)

    return idx
```

### Weight Initialization

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
```

### Special: Scaled Initialization for Residual Projections
```python
# Scale down residual projections by 1/sqrt(2*n_layer)
# Prevents activations from exploding in deep networks
for name, p in self.named_parameters():
    if name.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
```

---

## Step 5: Model Exports (`model/__init__.py`)

```python
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.block import Block
from model.gpt import GPT

__all__ = ["CausalSelfAttention", "MLP", "Block", "GPT"]
```

---

## Step 6: Tests (`tests/test_model.py`)

### Required Tests

| Test | What It Verifies |
|------|------------------|
| `test_attention_output_shape` | Attention preserves shape (B, T, C) |
| `test_attention_causal_mask` | Future positions are masked |
| `test_mlp_output_shape` | MLP preserves shape (B, T, C) |
| `test_block_output_shape` | Block preserves shape (B, T, C) |
| `test_gpt_output_shape` | GPT outputs (B, T, vocab_size) |
| `test_gpt_loss_computation` | Loss is computed correctly |
| `test_gpt_generate` | Generation produces valid tokens |
| `test_weight_tying` | Embedding and output weights are same object |
| `test_parameter_count` | Actual params close to estimate |

---

## File Checklist

### Files to Create
- [x] `model/attention.py` - CausalSelfAttention class
- [x] `model/mlp.py` - MLP class
- [x] `model/block.py` - Block class
- [x] `model/gpt.py` - GPT class with generate()
- [x] `model/__init__.py` - Update exports
- [x] `tests/test_model.py` - All model tests (16 tests)

### Implementation Checklist

#### Attention (`model/attention.py`)
- [x] Combined Q, K, V projection (c_attn)
- [x] Output projection (c_proj)
- [x] Causal mask registered as buffer
- [x] Multi-head reshape logic
- [x] Scaled dot-product attention
- [x] Attention dropout
- [x] Residual dropout

#### MLP (`model/mlp.py`)
- [x] Up projection (c_fc): n_embd → 4*n_embd
- [x] GELU activation
- [x] Down projection (c_proj): 4*n_embd → n_embd
- [x] Dropout

#### Block (`model/block.py`)
- [x] Pre-attention LayerNorm (ln_1)
- [x] CausalSelfAttention
- [x] Pre-MLP LayerNorm (ln_2)
- [x] MLP
- [x] Residual connections (both)

#### GPT (`model/gpt.py`)
- [x] Token embeddings (wte)
- [x] Position embeddings (wpe)
- [x] Embedding dropout
- [x] ModuleList of Blocks
- [x] Final LayerNorm (ln_f)
- [x] Output head (lm_head)
- [x] Weight tying (wte ↔ lm_head)
- [x] Weight initialization
- [x] Scaled init for residual projections
- [x] forward() with optional loss
- [x] generate() with temperature, top_k, top_p
- [x] get_num_params() helper

#### Tests (`tests/test_model.py`)
- [x] All shape tests pass
- [x] Loss computation test passes
- [x] Generation test passes (3 variants: basic, top_k, top_p)
- [x] Weight tying test passes
- [x] Parameter count test passes
- [x] Gradient flow test passes

---

## Verification Commands

```bash
# Run all model tests
pytest tests/test_model.py -v

# Quick sanity check
python -c "
from config import SMALL_CONFIG
from model import GPT
import torch

model = GPT(SMALL_CONFIG)
print(f'Parameters: {model.get_num_params()/1e6:.2f}M')

x = torch.randint(0, 1000, (2, 128))
logits, loss = model(x, x)
print(f'Logits shape: {logits.shape}')
print(f'Loss: {loss.item():.4f}')

out = model.generate(x[:, :10], max_new_tokens=20)
print(f'Generated shape: {out.shape}')
print('All checks passed!')
"
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Shape mismatch in attention | Check reshape: (B, T, C) → (B, nh, T, hd) |
| NaN in attention scores | Ensure mask applied before softmax |
| Loss not decreasing | Check weight initialization, learning rate |
| OOM errors | Reduce batch_size or block_size |
| generate() repeats tokens | Check temperature > 0, try top_k |

---

## Success Criteria

Phase 2 is complete when:
1. All 9+ tests in `test_model.py` pass
2. Model instantiates with correct parameter count
3. Forward pass produces correct shapes
4. Loss is computed (positive scalar)
5. Generation produces valid token sequences
6. Weight tying is verified

---

## Next Phase

After Phase 2, proceed to **Phase 3: Training Loop & Evaluation** which covers:
- Training configuration
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- Mixed precision training
- Checkpointing
- Wandb logging
