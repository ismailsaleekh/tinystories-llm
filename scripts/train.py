#!/usr/bin/env python
"""Training script for TinyStories GPT."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GPTConfig,
    set_seed,
    TINY_CONFIG,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    LARGE_CONFIG,
    TrainingConfig,
)
from model import GPT
from data import get_dataloaders
from training import Trainer


# Model configuration presets
MODEL_CONFIGS = {
    "tiny": TINY_CONFIG,      # ~3M params
    "small": SMALL_CONFIG,    # ~10M params
    "medium": MEDIUM_CONFIG,  # ~25M params
    "large": LARGE_CONFIG,    # ~85M params
}


def setup_logging():
    """Configure logging to console and file."""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Train TinyStories GPT')

    # Training arguments
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-iters', type=int, default=5000,
        help='Maximum training iterations'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--max-stories', type=int, default=None,
        help='Maximum number of training stories (None for all)'
    )

    # Model arguments
    parser.add_argument(
        '--model-size', type=str, default='small',
        choices=['tiny', 'small', 'medium', 'large'],
        help='Model size preset'
    )

    # Logging arguments
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--wandb-run-name', type=str, default=None,
        help='Wandb run name'
    )

    # Performance arguments
    parser.add_argument(
        '--no-compile', action='store_true',
        help='Disable torch.compile'
    )
    parser.add_argument(
        '--no-amp', action='store_true',
        help='Disable automatic mixed precision'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Get model config
    model_config = MODEL_CONFIGS[args.model_size]
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Estimated parameters: {model_config.estimate_params()/1e6:.1f}M")

    # Create training config
    training_config = TrainingConfig(
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=model_config.block_size,
        max_train_stories=args.max_stories,
        wandb_project=None if args.no_wandb else "tinystories-gpt",
        wandb_run_name=args.wandb_run_name,
        compile_model=not args.no_compile,
        mixed_precision=not args.no_amp,
    )

    logger.info(f"Max iterations: {training_config.max_iters}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Block size: {training_config.block_size}")
    if training_config.max_train_stories:
        logger.info(f"Max training stories: {training_config.max_train_stories}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, tokenizer = get_dataloaders(
        block_size=training_config.block_size,
        batch_size=training_config.batch_size,
        max_train_stories=training_config.max_train_stories,
    )

    # Create model
    logger.info("Creating model...")
    model = GPT(model_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=training_config,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from=args.resume)

    logger.info("Done!")


if __name__ == "__main__":
    main()
