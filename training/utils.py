"""Training utilities: LR scheduler and optimizer configuration."""
import math
import logging
import torch

logger = logging.getLogger(__name__)


def get_lr(
    it: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Schedule:
    1. Linear warmup: 0 -> learning_rate over warmup_iters
    2. Cosine decay: learning_rate -> min_lr over (lr_decay_iters - warmup_iters)
    3. Constant: min_lr after lr_decay_iters

    Args:
        it: Current iteration (0-indexed)
        learning_rate: Peak learning rate
        min_lr: Minimum learning rate
        warmup_iters: Number of warmup iterations
        lr_decay_iters: Total decay iterations

    Returns:
        Learning rate for current iteration
    """
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, f"decay_ratio {decay_ratio} out of bounds"

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1..0
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple,
    device_type: str
) -> torch.optim.Optimizer:
    """
    Configure AdamW optimizer with weight decay only on weight tensors.

    Weight decay is applied only to weights of Linear layers.
    Biases, LayerNorm parameters, and Embeddings have no weight decay.

    Args:
        model: The model to optimize
        weight_decay: Weight decay coefficient
        learning_rate: Learning rate
        betas: Adam beta parameters (beta1, beta2)
        device_type: "cuda" or "cpu"

    Returns:
        Configured AdamW optimizer
    """
    # Separate parameters into those that should and shouldn't have weight decay
    decay = set()
    no_decay = set()

    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # Full parameter name

            if pn.endswith('bias'):
                # All biases: no decay
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # Weights of whitelist modules: decay
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # Weights of blacklist modules: no decay
                no_decay.add(fpn)

    # Get actual parameter names from model
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # Filter to only include parameters that actually exist in model.named_parameters()
    # This handles weight tying where lm_head.weight and wte.weight are the same tensor
    decay = decay & param_dict.keys()
    no_decay = no_decay & param_dict.keys()

    # Validate no overlap
    inter_params = decay & no_decay
    assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"

    # Validate all parameters accounted for
    assert len(param_dict.keys() - (decay | no_decay)) == 0, \
        f"Parameters not in either decay/no_decay: {param_dict.keys() - (decay | no_decay)}"

    # Create optimizer groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]

    # Use fused AdamW if available (faster on CUDA)
    use_fused = (
        device_type == 'cuda'
        and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args
    )

    logger.info(f"Using fused AdamW: {use_fused}")
    logger.info(f"Num decayed parameter tensors: {len(decay)}")
    logger.info(f"Num non-decayed parameter tensors: {len(no_decay)}")

    return optimizer
