"""Dataset download and exploration utilities."""
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_tinystories():
    """Download TinyStories dataset from HuggingFace."""
    logger.info("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")

    logger.info(f"Training stories: {len(dataset['train']):,}")
    logger.info(f"Validation stories: {len(dataset['validation']):,}")

    return dataset


def explore_dataset(dataset, num_samples=5):
    """Print sample stories from the dataset."""
    print("\n" + "=" * 50)
    print("Sample Stories")
    print("=" * 50)

    for i in range(min(num_samples, len(dataset['train']))):
        story = dataset['train'][i]['text']
        print(f"\n--- Story {i + 1} ---")
        print(story[:500] + "..." if len(story) > 500 else story)


if __name__ == "__main__":
    dataset = download_tinystories()
    explore_dataset(dataset, num_samples=3)
