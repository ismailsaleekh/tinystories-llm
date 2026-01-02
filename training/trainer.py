"""GPT Trainer with mixed precision, gradient clipping, and wandb logging."""
import math
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from contextlib import nullcontext

import torch

from config.training_config import TrainingConfig
from model.gpt import GPT
from training.utils import get_lr, configure_optimizer

logger = logging.getLogger(__name__)


class Trainer:
    """
    GPT Training class with support for:
    - Mixed precision training (FP16)
    - Gradient clipping
    - Learning rate scheduling (warmup + cosine decay)
    - Checkpointing
    - Wandb logging
    - Sample generation during training
    """

    def __init__(
        self,
        model: GPT,
        train_loader,
        val_loader,
        tokenizer,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Compile model (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = configure_optimizer(
            self.model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=self.device_type
        )

        # Setup mixed precision
        self.scaler = torch.amp.GradScaler(
            device=self.device,
            enabled=config.mixed_precision and self.device_type == "cuda"
        )

        if config.mixed_precision and self.device_type == "cuda":
            self.ctx = torch.amp.autocast(device_type=self.device_type, dtype=torch.float16)
        else:
            self.ctx = nullcontext()

        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')

        # DataLoader iterators
        self._train_iter = None
        self._val_iter = None

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _get_batch(self, split: str):
        """Get a batch from train or val loader, handling iterator exhaustion."""
        loader = self.train_loader if split == 'train' else self.val_loader

        try:
            if split == 'train':
                x, y = next(self._train_iter)
            else:
                x, y = next(self._val_iter)
        except (StopIteration, TypeError):
            # Reset iterator when exhausted
            if split == 'train':
                self._train_iter = iter(loader)
                x, y = next(self._train_iter)
            else:
                self._val_iter = iter(loader)
                x, y = next(self._val_iter)

        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.model.eval()
        losses = {}

        for split in ['train', 'val']:
            loader = self.train_loader if split == 'train' else self.val_loader
            total_loss = 0.0
            num_batches = 0

            # Reset iterator for clean evaluation
            data_iter = iter(loader)

            for i in range(self.config.eval_iters):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    break

                x, y = x.to(self.device), y.to(self.device)

                with self.ctx:
                    _, loss = self.model(x, y)

                total_loss += loss.item()
                num_batches += 1

            losses[split] = total_loss / max(num_batches, 1)

        self.model.train()
        return losses

    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time", max_tokens: int = 100) -> str:
        """Generate a sample text from the model."""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)

        y = self.model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50
        )

        self.model.train()
        return self.tokenizer.decode(y[0].tolist())

    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.iter_num}.pt"

        # Handle compiled model
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod

        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': model_to_save.config,
            'training_config': self.config,
        }

        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Handle compiled model
        model_to_load = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_load = self.model._orig_mod

        model_to_load.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Resumed from iteration {self.iter_num}")

    def train(self, resume_from: str = None):
        """
        Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        # Initialize wandb
        wandb = None
        if self.config.wandb_project:
            try:
                import wandb as wb
                wandb = wb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config={
                        **vars(self.config),
                        'model_params': self.model.get_num_params() if hasattr(self.model, 'get_num_params')
                                       else sum(p.numel() for p in self.model.parameters()),
                    }
                )
                logger.info("Wandb initialized")
            except ImportError:
                logger.warning("Wandb not installed, skipping logging")
                wandb = None
            except Exception as e:
                logger.warning(f"Wandb init failed: {e}, skipping logging")
                wandb = None

        # Training loop
        self.model.train()
        self._train_iter = iter(self.train_loader)

        t0 = time.time()
        local_iter_num = 0

        logger.info(f"Starting training from iteration {self.iter_num}")
        logger.info(f"Training for {self.config.max_iters} iterations")

        while self.iter_num < self.config.max_iters:
            # Update learning rate
            lr = get_lr(
                self.iter_num,
                self.config.learning_rate,
                self.config.min_lr,
                self.config.warmup_iters,
                self.config.lr_decay_iters
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Get batch
            x, y = self._get_batch('train')

            # Forward pass with mixed precision
            with self.ctx:
                logits, loss = self.model(x, y)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Timing
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Logging
            if self.iter_num % self.config.log_interval == 0:
                loss_val = loss.item()
                logger.info(
                    f"iter {self.iter_num}: loss {loss_val:.4f}, "
                    f"lr {lr:.2e}, time {dt*1000:.0f}ms"
                )

                if wandb:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/lr": lr,
                        "train/iter_time_ms": dt * 1000,
                    }, step=self.iter_num)

            # Evaluation
            if self.iter_num > 0 and self.iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss()

                train_ppl = math.exp(losses['train'])
                val_ppl = math.exp(losses['val'])

                logger.info(
                    f"step {self.iter_num}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
                logger.info(f"train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")

                if wandb:
                    wandb.log({
                        "eval/train_loss": losses['train'],
                        "eval/val_loss": losses['val'],
                        "eval/train_ppl": train_ppl,
                        "eval/val_ppl": val_ppl,
                    }, step=self.iter_num)

                # Generate sample
                sample = self.generate_sample()
                logger.info(f"Sample: {sample[:200]}...")

                if wandb:
                    wandb.log({
                        "samples": wandb.Html(f"<pre>{sample}</pre>")
                    }, step=self.iter_num)

                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint(is_best=True)

            # Checkpointing
            if self.iter_num > 0 and self.iter_num % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            self.iter_num += 1
            local_iter_num += 1

        # Final save
        self.save_checkpoint(filename="final_model.pt")

        if wandb:
            wandb.finish()

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
