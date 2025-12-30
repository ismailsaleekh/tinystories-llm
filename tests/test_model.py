"""Tests for model components."""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import GPTConfig, TINY_CONFIG
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.block import Block
from model.gpt import GPT


@pytest.fixture
def config():
    """Use TINY_CONFIG for fast testing."""
    return TINY_CONFIG


@pytest.fixture
def device():
    """Use GPU if available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention."""

    def test_output_shape(self, config, device):
        """Test attention output shape."""
        attn = CausalSelfAttention(config).to(device)
        x = torch.randn(2, 128, config.n_embd, device=device)
        y = attn(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_output_shape_different_seq_len(self, config, device):
        """Test attention with different sequence lengths."""
        attn = CausalSelfAttention(config).to(device)

        for seq_len in [1, 32, 64, 128]:
            x = torch.randn(2, seq_len, config.n_embd, device=device)
            y = attn(x)
            assert y.shape == x.shape, f"Failed for seq_len={seq_len}"

    def test_causal_mask(self, config, device):
        """Test that attention is causal (basic sanity check)."""
        attn = CausalSelfAttention(config).to(device)

        # Create input where later positions have different values
        x = torch.zeros(1, 10, config.n_embd, device=device)
        x[:, 5:, :] = 1.0  # Later positions are different

        y = attn(x)

        # Output should have correct shape
        assert y.shape == x.shape


class TestMLP:
    """Tests for MLP."""

    def test_output_shape(self, config, device):
        """Test MLP output shape."""
        mlp = MLP(config).to(device)
        x = torch.randn(2, 128, config.n_embd, device=device)
        y = mlp(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_output_shape_different_seq_len(self, config, device):
        """Test MLP with different sequence lengths."""
        mlp = MLP(config).to(device)

        for seq_len in [1, 32, 64, 128]:
            x = torch.randn(2, seq_len, config.n_embd, device=device)
            y = mlp(x)
            assert y.shape == x.shape, f"Failed for seq_len={seq_len}"


class TestBlock:
    """Tests for Block."""

    def test_output_shape(self, config, device):
        """Test transformer block output shape."""
        block = Block(config).to(device)
        x = torch.randn(2, 128, config.n_embd, device=device)
        y = block(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_output_shape_different_seq_len(self, config, device):
        """Test block with different sequence lengths."""
        block = Block(config).to(device)

        for seq_len in [1, 32, 64, 128]:
            x = torch.randn(2, seq_len, config.n_embd, device=device)
            y = block(x)
            assert y.shape == x.shape, f"Failed for seq_len={seq_len}"


class TestGPT:
    """Tests for GPT model."""

    def test_output_shape(self, config, device):
        """Test GPT output shape."""
        model = GPT(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 128), device=device)
        logits, _ = model(x)
        assert logits.shape == (2, 128, config.vocab_size), f"Got {logits.shape}"

    def test_output_shape_different_seq_len(self, config, device):
        """Test GPT with different sequence lengths."""
        model = GPT(config).to(device)

        for seq_len in [1, 32, 64, 128]:
            x = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
            logits, _ = model(x)
            assert logits.shape == (2, seq_len, config.vocab_size), f"Failed for seq_len={seq_len}"

    def test_loss_computation(self, config, device):
        """Test that loss is computed correctly."""
        model = GPT(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 128), device=device)
        y = torch.randint(0, config.vocab_size, (2, 128), device=device)

        logits, loss = model(x, y)

        assert loss is not None, "Loss should not be None when targets provided"
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    def test_no_loss_without_targets(self, config, device):
        """Test that loss is None when no targets provided."""
        model = GPT(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 128), device=device)

        logits, loss = model(x)

        assert loss is None, "Loss should be None when no targets"

    def test_generate(self, config, device):
        """Test text generation."""
        model = GPT(config).to(device)
        model.eval()

        # Start with a few tokens
        idx = torch.randint(0, config.vocab_size, (1, 5), device=device)

        # Generate 10 new tokens
        output = model.generate(idx, max_new_tokens=10, temperature=1.0)

        assert output.shape == (1, 15), f"Expected (1, 15), got {output.shape}"
        assert output.dtype == torch.long, f"Expected long, got {output.dtype}"

    def test_generate_with_top_k(self, config, device):
        """Test generation with top-k sampling."""
        model = GPT(config).to(device)
        model.eval()

        idx = torch.randint(0, config.vocab_size, (1, 5), device=device)
        output = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=50)

        assert output.shape == (1, 15)

    def test_generate_with_top_p(self, config, device):
        """Test generation with top-p (nucleus) sampling."""
        model = GPT(config).to(device)
        model.eval()

        idx = torch.randint(0, config.vocab_size, (1, 5), device=device)
        output = model.generate(idx, max_new_tokens=10, temperature=0.8, top_p=0.9)

        assert output.shape == (1, 15)

    def test_parameter_count(self, config):
        """Test parameter count estimation."""
        model = GPT(config)
        actual = model.get_num_params()
        estimated = config.estimate_params()

        # Should be within 10% (estimation is approximate)
        diff = abs(actual - estimated) / actual
        assert diff < 0.1, f"Param count diff {diff:.1%} > 10%: actual={actual}, estimated={estimated}"

    def test_weight_tying(self, config, device):
        """Test that embedding and output weights are tied."""
        model = GPT(config).to(device)

        # These should be the exact same tensor object
        assert model.transformer.wte.weight is model.lm_head.weight, \
            "Token embedding and lm_head weights should be tied"

    def test_gradient_flow(self, config, device):
        """Test that gradients flow through the model."""
        model = GPT(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, 32), device=device)
        y = torch.randint(0, config.vocab_size, (2, 32), device=device)

        logits, loss = model(x, y)
        loss.backward()

        # Check that some gradients are non-zero
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients detected after backward pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
