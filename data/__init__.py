"""Data module with dataset and dataloader utilities."""
from torch.utils.data import DataLoader
from datasets import load_dataset
import logging

from data.dataset import TinyStoriesDataset
from data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def get_dataloaders(
    block_size: int = 256,
    batch_size: int = 64,
    max_train_stories: int = None,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Create train and validation data loaders.

    Args:
        block_size: Context window size
        batch_size: Batch size for training
        max_train_stories: Limit training stories (None for all)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, tokenizer
    """
    logger.info("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")

    logger.info("Creating training dataset...")
    train_dataset = TinyStoriesDataset(
        dataset['train'],
        block_size=block_size,
        max_stories=max_train_stories
    )

    logger.info("Creating validation dataset...")
    val_dataset = TinyStoriesDataset(
        dataset['validation'],
        block_size=block_size,
        max_stories=None  # Use full validation set
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(f"Train batches: {len(train_loader):,}")
    logger.info(f"Val batches: {len(val_loader):,}")

    return train_loader, val_loader, train_dataset.tokenizer


__all__ = [
    "get_dataloaders",
    "TinyStoriesDataset",
    "Tokenizer",
]
