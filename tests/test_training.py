#!/usr/bin/env python
"""Tests for Phase 3 training components."""
import pytest
import torch
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import GPTConfig, TINY_CONFIG, set_seed
from config.training_config import TrainingConfig
from training.utils import get_lr, configure_optimizer
from model.gpt import GPT


# ============================================================================
# TrainingConfig Tests
# ============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        # Optimization
        assert config.learning_rate == 3e-4
        assert config.min_lr == 3e-5
        assert config.weight_decay == 0.1
        assert config.beta1 == 0.9
        assert config.beta2 == 0.95
        assert config.grad_clip == 1.0

        # Schedule
        assert config.max_iters == 5000
        assert config.warmup_iters == 100
        assert config.lr_decay_iters == 5000

        # Evaluation
        assert config.eval_interval == 500
        assert config.eval_iters == 200

        # Checkpointing
        assert config.checkpoint_interval == 1000
        assert config.checkpoint_dir == "checkpoints"

        # Data
        assert config.batch_size == 64
        assert config.block_size == 256

        # Device
        assert config.device == "auto"
        assert config.compile_model == True
        assert config.mixed_precision == True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=1e-3,
            max_iters=1000,
            batch_size=32,
            wandb_project=None,
        )

        assert config.learning_rate == 1e-3
        assert config.max_iters == 1000
        assert config.batch_size == 32
        assert config.wandb_project is None

    def test_min_lr_ratio(self):
        """Test that min_lr is typically 10% of learning_rate."""
        config = TrainingConfig()
        # Default ratio should be ~10%
        assert config.min_lr / config.learning_rate == pytest.approx(0.1, rel=0.01)


# ============================================================================
# Learning Rate Schedule Tests
# ============================================================================

class TestLearningRateSchedule:
    """Tests for get_lr function."""

    def test_warmup_start(self):
        """Test LR at start of warmup."""
        lr = get_lr(
            it=0,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At it=0: lr = 3e-4 * (0+1) / 100 = 3e-6
        assert lr == pytest.approx(3e-6, rel=0.01)

    def test_warmup_middle(self):
        """Test LR in middle of warmup."""
        lr = get_lr(
            it=50,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At it=50: lr = 3e-4 * 51 / 100 = 1.53e-4
        expected = 3e-4 * 51 / 100
        assert lr == pytest.approx(expected, rel=0.01)

    def test_warmup_end(self):
        """Test LR at end of warmup (should be peak)."""
        lr = get_lr(
            it=99,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At it=99: lr = 3e-4 * 100 / 100 = 3e-4
        assert lr == pytest.approx(3e-4, rel=0.01)

    def test_decay_start(self):
        """Test LR at start of decay (just after warmup)."""
        lr = get_lr(
            it=100,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At start of decay, should be close to max LR
        assert lr == pytest.approx(3e-4, rel=0.01)

    def test_decay_middle(self):
        """Test LR in middle of decay."""
        lr = get_lr(
            it=550,  # Midpoint: (100 + 1000) / 2 = 550
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At midpoint, cosine should give approximately (max + min) / 2
        expected_mid = (3e-4 + 3e-5) / 2
        assert lr == pytest.approx(expected_mid, rel=0.1)

    def test_decay_end(self):
        """Test LR at end of decay."""
        lr = get_lr(
            it=1000,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # At end of decay, should be at min_lr
        assert lr == pytest.approx(3e-5, rel=0.01)

    def test_after_decay(self):
        """Test LR after decay (should stay at min)."""
        lr = get_lr(
            it=2000,
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=100,
            lr_decay_iters=1000
        )
        # After decay, should be min_lr
        assert lr == pytest.approx(3e-5, rel=0.01)

    def test_lr_monotonically_increases_during_warmup(self):
        """Test LR increases during warmup."""
        prev_lr = 0
        for it in range(100):
            lr = get_lr(it, 3e-4, 3e-5, 100, 1000)
            assert lr > prev_lr, f"LR should increase at it={it}"
            prev_lr = lr

    def test_lr_monotonically_decreases_during_decay(self):
        """Test LR decreases during decay."""
        prev_lr = float('inf')
        for it in range(100, 1001, 10):
            lr = get_lr(it, 3e-4, 3e-5, 100, 1000)
            assert lr <= prev_lr, f"LR should decrease at it={it}"
            prev_lr = lr


# ============================================================================
# Optimizer Configuration Tests
# ============================================================================

class TestOptimizerConfig:
    """Tests for configure_optimizer function."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return GPT(TINY_CONFIG)

    def test_creates_adamw_optimizer(self, model):
        """Test that AdamW optimizer is created."""
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type='cpu'
        )
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_two_param_groups(self, model):
        """Test that optimizer has two parameter groups (decay and no-decay)."""
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type='cpu'
        )
        assert len(optimizer.param_groups) == 2

    def test_weight_decay_separation(self, model):
        """Test that weight decay is applied correctly."""
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type='cpu'
        )

        # One group should have weight_decay=0.1, other should have 0.0
        weight_decays = [pg['weight_decay'] for pg in optimizer.param_groups]
        assert 0.1 in weight_decays
        assert 0.0 in weight_decays

    def test_all_parameters_included(self, model):
        """Test that all parameters are in optimizer."""
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type='cpu'
        )

        # Count parameters in optimizer
        opt_params = sum(len(pg['params']) for pg in optimizer.param_groups)

        # Count unique parameters in model (accounting for weight tying)
        model_params = len(list(model.parameters()))

        # Should match (weight tying means some params are shared)
        assert opt_params == model_params

    def test_learning_rate_set(self, model):
        """Test that learning rate is set correctly."""
        lr = 1e-3
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=lr,
            betas=(0.9, 0.95),
            device_type='cpu'
        )

        for pg in optimizer.param_groups:
            assert pg['lr'] == lr

    def test_betas_set(self, model):
        """Test that betas are set correctly."""
        betas = (0.9, 0.95)
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=betas,
            device_type='cpu'
        )

        for pg in optimizer.param_groups:
            assert pg['betas'] == betas


# ============================================================================
# Trainer Component Tests
# ============================================================================

class TestTrainerComponents:
    """Tests for Trainer class components."""

    @pytest.fixture
    def model(self):
        """Create a small model."""
        set_seed(42)
        return GPT(TINY_CONFIG)

    @pytest.fixture
    def training_config(self):
        """Create a minimal training config."""
        return TrainingConfig(
            max_iters=10,
            batch_size=2,
            block_size=32,
            eval_interval=5,
            eval_iters=2,
            checkpoint_interval=5,
            log_interval=2,
            warmup_iters=2,
            lr_decay_iters=10,
            wandb_project=None,  # Disable wandb for testing
            compile_model=False,  # Disable compile for faster tests
            mixed_precision=False,  # Disable AMP for CPU testing
            device='cpu',
        )

    def test_trainer_import(self):
        """Test that Trainer can be imported."""
        from training.trainer import Trainer
        assert Trainer is not None

    def test_trainer_init(self, model, training_config):
        """Test Trainer initialization."""
        from training.trainer import Trainer
        from data.tokenizer import Tokenizer

        # Create mock dataloaders
        mock_train_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
            for _ in range(10)
        ]
        mock_val_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
            for _ in range(5)
        ]

        tokenizer = Tokenizer()

        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            tokenizer=tokenizer,
            config=training_config,
        )

        assert trainer.device == 'cpu'
        assert trainer.iter_num == 0
        assert trainer.best_val_loss == float('inf')

    def test_trainer_get_batch(self, model, training_config):
        """Test Trainer._get_batch method."""
        from training.trainer import Trainer
        from data.tokenizer import Tokenizer

        batch_data = (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
        mock_train_loader = [batch_data]
        mock_val_loader = [batch_data]

        tokenizer = Tokenizer()

        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            tokenizer=tokenizer,
            config=training_config,
        )

        x, y = trainer._get_batch('train')
        assert x.shape == (2, 32)
        assert y.shape == (2, 32)

    def test_trainer_estimate_loss(self, model, training_config):
        """Test Trainer.estimate_loss method."""
        from training.trainer import Trainer
        from data.tokenizer import Tokenizer

        mock_train_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
            for _ in range(5)
        ]
        mock_val_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
            for _ in range(5)
        ]

        tokenizer = Tokenizer()

        trainer = Trainer(
            model=model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader,
            tokenizer=tokenizer,
            config=training_config,
        )

        losses = trainer.estimate_loss()

        assert 'train' in losses
        assert 'val' in losses
        assert losses['train'] > 0
        assert losses['val'] > 0

    def test_trainer_generate_sample(self, model, training_config):
        """Test Trainer.generate_sample method."""
        from training.trainer import Trainer
        from data.tokenizer import Tokenizer

        mock_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
        ]

        tokenizer = Tokenizer()

        trainer = Trainer(
            model=model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            tokenizer=tokenizer,
            config=training_config,
        )

        sample = trainer.generate_sample("Once upon a time", max_tokens=20)

        assert isinstance(sample, str)
        assert len(sample) > 0
        assert sample.startswith("Once upon a time")


# ============================================================================
# Integration Tests
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_full_training_step(self):
        """Test a complete training step (forward + backward + optimizer step)."""
        set_seed(42)
        model = GPT(TINY_CONFIG)

        config = TrainingConfig(
            learning_rate=3e-4,
            weight_decay=0.1,
            device='cpu',
        )

        optimizer = configure_optimizer(
            model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type='cpu'
        )

        # Create fake batch
        x = torch.randint(0, TINY_CONFIG.vocab_size, (2, 32))
        y = torch.randint(0, TINY_CONFIG.vocab_size, (2, 32))

        # Initial loss
        model.train()
        _, loss1 = model(x, y)

        # Training step
        optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Loss after step
        _, loss2 = model(x, y)

        # Loss should generally decrease (or at least not explode)
        assert not torch.isnan(loss2)
        assert not torch.isinf(loss2)

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        from training.trainer import Trainer
        from data.tokenizer import Tokenizer

        set_seed(42)
        model = GPT(TINY_CONFIG)

        config = TrainingConfig(
            max_iters=5,
            batch_size=2,
            block_size=32,
            checkpoint_dir=str(tmp_path),
            wandb_project=None,
            compile_model=False,
            mixed_precision=False,
            device='cpu',
        )

        mock_loader = [
            (torch.randint(0, 1000, (2, 32)), torch.randint(0, 1000, (2, 32)))
            for _ in range(5)
        ]

        tokenizer = Tokenizer()

        trainer = Trainer(
            model=model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            tokenizer=tokenizer,
            config=config,
        )

        # Simulate some training
        trainer.iter_num = 100
        trainer.best_val_loss = 1.5

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint.pt")

        # Verify file exists
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint['iter_num'] == 100
        assert checkpoint['best_val_loss'] == 1.5
        assert 'model' in checkpoint
        assert 'optimizer' in checkpoint
        assert 'config' in checkpoint


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and potential bugs."""

    def test_lr_with_zero_warmup(self):
        """Test LR schedule with zero warmup."""
        # Should start directly with cosine decay
        lr = get_lr(0, 3e-4, 3e-5, warmup_iters=0, lr_decay_iters=1000)
        # With warmup_iters=0, it should be in decay phase
        assert lr == pytest.approx(3e-4, rel=0.01)

    def test_lr_with_equal_warmup_and_decay(self):
        """Test LR when warmup_iters equals lr_decay_iters."""
        # Edge case: no decay phase
        lr = get_lr(50, 3e-4, 3e-5, warmup_iters=100, lr_decay_iters=100)
        # Should still be in warmup
        expected = 3e-4 * 51 / 100
        assert lr == pytest.approx(expected, rel=0.01)

    def test_optimizer_with_no_bias_model(self):
        """Test optimizer with model that has no biases."""
        config = GPTConfig(
            n_layer=2, n_head=2, n_embd=64,
            bias=False  # No biases
        )
        model = GPT(config)

        # Should not raise
        optimizer = configure_optimizer(
            model,
            weight_decay=0.1,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            device_type='cpu'
        )
        assert optimizer is not None

    def test_very_small_batch(self):
        """Test with batch size of 1."""
        model = GPT(TINY_CONFIG)
        x = torch.randint(0, TINY_CONFIG.vocab_size, (1, 32))
        y = torch.randint(0, TINY_CONFIG.vocab_size, (1, 32))

        logits, loss = model(x, y)
        assert not torch.isnan(loss)

    def test_sequence_length_1(self):
        """Test with sequence length of 1."""
        model = GPT(TINY_CONFIG)
        x = torch.randint(0, TINY_CONFIG.vocab_size, (2, 1))
        y = torch.randint(0, TINY_CONFIG.vocab_size, (2, 1))

        logits, loss = model(x, y)
        assert logits.shape == (2, 1, TINY_CONFIG.vocab_size)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
