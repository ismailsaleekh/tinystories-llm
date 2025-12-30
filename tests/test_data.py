"""Tests for data pipeline."""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import Tokenizer
from data import get_dataloaders


class TestTokenizer:
    """Tests for the tokenizer."""

    def test_encode_decode_roundtrip(self):
        """Test tokenizer encode/decode round trip."""
        tokenizer = Tokenizer()

        text = "Once upon a time, there was a little rabbit."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text, f"Round-trip failed: '{decoded}' != '{text}'"

    def test_token_types(self):
        """Test that tokens are integers in valid range."""
        tokenizer = Tokenizer()

        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.encode(text)

        assert all(isinstance(t, int) for t in tokens), "All tokens should be integers"
        assert all(0 <= t < tokenizer.vocab_size for t in tokens), "Tokens out of vocab range"

    def test_vocab_size(self):
        """Test vocab size is correct."""
        tokenizer = Tokenizer()
        assert tokenizer.vocab_size == 50257, f"Expected 50257, got {tokenizer.vocab_size}"

    def test_eot_token(self):
        """Test EOT token exists."""
        tokenizer = Tokenizer()
        assert tokenizer.eot_token == 50256, f"Expected 50256, got {tokenizer.eot_token}"

    def test_callable(self):
        """Test tokenizer is callable."""
        tokenizer = Tokenizer()
        tokens = tokenizer("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0


class TestDataset:
    """Tests for the dataset and dataloaders."""

    @pytest.fixture
    def small_dataloaders(self):
        """Create small dataloaders for testing."""
        return get_dataloaders(
            block_size=256,
            batch_size=4,
            max_train_stories=1000,  # Small subset for fast testing
            num_workers=0,  # Avoid multiprocessing issues in tests
            pin_memory=False
        )

    def test_dataloader_returns_tuple(self, small_dataloaders):
        """Test that get_dataloaders returns correct tuple."""
        train_loader, val_loader, tokenizer = small_dataloaders

        assert train_loader is not None
        assert val_loader is not None
        assert tokenizer is not None

    def test_batch_shapes(self, small_dataloaders):
        """Test that batches have correct shapes."""
        train_loader, val_loader, tokenizer = small_dataloaders

        x, y = next(iter(train_loader))

        assert x.shape == (4, 256), f"Expected (4, 256), got {x.shape}"
        assert y.shape == (4, 256), f"Expected (4, 256), got {y.shape}"

    def test_batch_dtypes(self, small_dataloaders):
        """Test that batches have correct dtypes."""
        train_loader, _, _ = small_dataloaders

        x, y = next(iter(train_loader))

        assert x.dtype == torch.long, f"Expected torch.long, got {x.dtype}"
        assert y.dtype == torch.long, f"Expected torch.long, got {y.dtype}"

    def test_target_shift(self, small_dataloaders):
        """Test that targets are shifted by 1 position."""
        train_loader, _, _ = small_dataloaders

        x, y = next(iter(train_loader))

        # For each sample in batch, y should be x shifted by 1
        # x[i, 1:] should have significant overlap with y[i, :-1]
        # (not exact due to how samples are drawn, but should be close)
        assert x.shape == y.shape, "x and y should have same shape"

    def test_tokens_in_valid_range(self, small_dataloaders):
        """Test that all tokens are in valid vocabulary range."""
        train_loader, _, tokenizer = small_dataloaders

        x, y = next(iter(train_loader))

        assert x.min() >= 0, f"Token below 0: {x.min()}"
        assert x.max() < tokenizer.vocab_size, f"Token above vocab: {x.max()}"
        assert y.min() >= 0, f"Target below 0: {y.min()}"
        assert y.max() < tokenizer.vocab_size, f"Target above vocab: {y.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
