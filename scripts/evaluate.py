#!/usr/bin/env python
"""Evaluation script for trained TinyStories GPT models."""
import argparse
import math
import logging
import sys
from pathlib import Path
from tqdm import tqdm

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT
from data import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_perplexity(model, data_loader, device):
    """
    Calculate perplexity on a dataset.

    Args:
        model: The GPT model
        data_loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        avg_loss: Average cross-entropy loss
        perplexity: exp(avg_loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Evaluate TinyStories GPT')
    parser.add_argument(
        'checkpoint', type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto, cuda, cpu)'
    )
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create model from saved config
    config = checkpoint['config']
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model'])

    num_params = model.get_num_params()
    logger.info(f"Model parameters: {num_params/1e6:.2f}M")

    # Get training info if available
    if 'iter_num' in checkpoint:
        logger.info(f"Checkpoint iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        logger.info(f"Best val loss at save: {checkpoint['best_val_loss']:.4f}")

    # Create validation dataloader
    logger.info("Loading validation data...")
    _, val_loader, _ = get_dataloaders(
        block_size=config.block_size,
        batch_size=args.batch_size,
    )

    # Evaluate
    logger.info("Calculating perplexity...")
    loss, ppl = calculate_perplexity(model, val_loader, device)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Checkpoint:            {args.checkpoint}")
    print(f"Model Parameters:      {num_params/1e6:.2f}M")
    print("-" * 50)
    print(f"Validation Loss:       {loss:.4f}")
    print(f"Validation Perplexity: {ppl:.2f}")
    print("=" * 50)

    # Quality assessment
    if ppl < 5:
        quality = "EXCELLENT - Model generates very coherent text"
    elif ppl < 10:
        quality = "GOOD - Model generates coherent text with minor issues"
    elif ppl < 20:
        quality = "FAIR - Model generates somewhat coherent text"
    elif ppl < 50:
        quality = "POOR - Model needs more training"
    else:
        quality = "UNTRAINED - Model produces mostly random text"

    print(f"Quality: {quality}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
