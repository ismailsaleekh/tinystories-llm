"""TinyStories dataset with tokenization and caching."""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib

from data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset with tokenization and caching.

    Args:
        data: HuggingFace dataset split
        block_size: Context window size
        cache_dir: Directory to cache tokenized data
        max_stories: Maximum number of stories to use (None for all)
    """

    def __init__(
        self,
        data,
        block_size: int = 256,
        cache_dir: str = "data/cache",
        max_stories: int = None
    ):
        self.block_size = block_size
        self.tokenizer = Tokenizer()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Limit stories if specified
        if max_stories is not None:
            data = data.select(range(min(max_stories, len(data))))

        # Try to load from cache
        cache_file = self._get_cache_path(data, block_size, max_stories)

        if cache_file.exists():
            logger.info(f"Loading tokenized data from cache: {cache_file}")
            self.tokens = np.load(cache_file)
        else:
            logger.info("Tokenizing dataset (this may take a while)...")
            self.tokens = self._tokenize_and_cache(data, cache_file)

        logger.info(f"Total tokens: {len(self.tokens):,}")
        logger.info(f"Number of samples: {len(self):,}")

    def _get_cache_path(self, data, block_size, max_stories) -> Path:
        """Generate unique cache filename based on dataset parameters."""
        params = f"{len(data)}_{block_size}_{max_stories}"
        hash_str = hashlib.md5(params.encode()).hexdigest()[:8]
        return self.cache_dir / f"tokens_{hash_str}.npy"

    def _tokenize_and_cache(self, data, cache_file: Path) -> np.ndarray:
        """Tokenize all stories and cache to disk."""
        all_tokens = []
        invalid_count = 0

        for story in tqdm(data, desc="Tokenizing"):
            text = story['text']

            # Skip empty or invalid stories
            if not text or len(text.strip()) == 0:
                invalid_count += 1
                continue

            tokens = self.tokenizer.encode(text)
            tokens.append(self.tokenizer.eot_token)  # Add end-of-text token
            all_tokens.extend(tokens)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} empty/invalid stories")

        tokens_array = np.array(all_tokens, dtype=np.uint16)  # Use uint16 to save memory

        # Save to cache
        np.save(cache_file, tokens_array)
        logger.info(f"Cached tokenized data to: {cache_file}")

        return tokens_array

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int):
        """
        Get a training sample.

        Returns:
            x: Input tokens [block_size]
            y: Target tokens [block_size] (shifted by 1)
        """
        x = torch.tensor(self.tokens[idx:idx + self.block_size].astype(np.int64), dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1].astype(np.int64), dtype=torch.long)
        return x, y
