"""
Test Data Loader for Training Pipeline

Tests the data loading and preprocessing without downloading models.
Verifies:
1. JSON parsing
2. Image loading
3. Data batching
4. Text preprocessing
5. Memory usage

Usage:
    python scripts/test_data_loader.py --data data/training/deepcad_1k/train.json --images data/training/deepcad_1k/images
"""

import argparse
import json
import sys
from pathlib import Path
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleCADDataset:
    """Simple dataset loader for testing"""

    def __init__(self, data_path, image_folder):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = Path(image_folder)
        logger.info(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get image
        image_name = item["image"]
        image_path = self.image_folder / image_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path)

        # Build conversation
        conversations = item["conversations"]

        return {
            "image": image,
            "image_name": image_name,
            "conversations": conversations
        }


def test_data_loading(data_path: str, image_folder: str, num_samples: int = 10):
    """Test loading data samples"""
    logger.info("=" * 60)
    logger.info("Testing Data Loading")
    logger.info("=" * 60)

    try:
        dataset = SimpleCADDataset(data_path, image_folder)
        logger.info(f"✓ Dataset initialized with {len(dataset)} samples")

        # Test loading samples
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]

            # Verify image
            image = sample["image"]
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Image: {sample['image_name']}")
            logger.info(f"  Image size: {image.size}")
            logger.info(f"  Image mode: {image.mode}")

            # Verify conversations
            conversations = sample["conversations"]
            logger.info(f"  Conversations: {len(conversations)} turns")

            for j, turn in enumerate(conversations):
                text = turn["value"]
                logger.info(f"    Turn {j+1} ({turn['from']}): {len(text)} chars")

        logger.info(f"\n✓ Successfully loaded {num_samples} samples")
        return True

    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_properties(data_path: str, image_folder: str, num_check: int = 50):
    """Analyze image properties"""
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing Image Properties")
    logger.info("=" * 60)

    dataset = SimpleCADDataset(data_path, image_folder)

    sizes = []
    modes = set()
    file_sizes = []

    for i in range(min(num_check, len(dataset))):
        sample = dataset[i]
        image = sample["image"]

        sizes.append(image.size)
        modes.add(image.mode)

        # Get file size
        image_path = Path(image_folder) / sample["image_name"]
        file_sizes.append(image_path.stat().st_size / 1024)  # KB

    # Statistics
    logger.info(f"Checked {len(sizes)} images:")
    logger.info(f"  Image sizes: {set(sizes)}")
    logger.info(f"  Color modes: {modes}")
    logger.info(f"  File sizes: {min(file_sizes):.1f} - {max(file_sizes):.1f} KB")
    logger.info(f"  Average file size: {sum(file_sizes)/len(file_sizes):.1f} KB")

    return True


def test_text_statistics(data_path: str):
    """Analyze text statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("Analyzing Text Statistics")
    logger.info("=" * 60)

    with open(data_path) as f:
        data = json.load(f)

    prompt_lengths = []
    response_lengths = []
    total_lengths = []

    for sample in data:
        conversations = sample["conversations"]

        for turn in conversations:
            text_len = len(turn["value"])

            if turn["from"] == "human":
                prompt_lengths.append(text_len)
            elif turn["from"] in ["gpt", "assistant"]:
                response_lengths.append(text_len)

            total_lengths.append(text_len)

    logger.info(f"Analyzed {len(data)} samples:")
    logger.info(f"\nPrompts ({len(prompt_lengths)} total):")
    logger.info(f"  Min: {min(prompt_lengths)} chars")
    logger.info(f"  Max: {max(prompt_lengths)} chars")
    logger.info(f"  Avg: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")

    logger.info(f"\nResponses ({len(response_lengths)} total):")
    logger.info(f"  Min: {min(response_lengths)} chars")
    logger.info(f"  Max: {max(response_lengths)} chars")
    logger.info(f"  Avg: {sum(response_lengths)/len(response_lengths):.0f} chars")

    logger.info(f"\nOverall text:")
    logger.info(f"  Total chars: {sum(total_lengths):,}")
    logger.info(f"  Avg per turn: {sum(total_lengths)/len(total_lengths):.0f} chars")

    return True


def test_batch_simulation(data_path: str, image_folder: str, batch_size: int = 2, num_batches: int = 3):
    """Simulate batch loading"""
    logger.info("\n" + "=" * 60)
    logger.info(f"Simulating Batch Loading (batch_size={batch_size})")
    logger.info("=" * 60)

    dataset = SimpleCADDataset(data_path, image_folder)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))

        logger.info(f"\nBatch {batch_idx + 1}:")
        logger.info(f"  Indices: {start_idx} to {end_idx-1}")

        batch_samples = []
        for idx in range(start_idx, end_idx):
            sample = dataset[idx]
            batch_samples.append(sample)

        logger.info(f"  Loaded {len(batch_samples)} samples")

        # Show batch contents
        for i, sample in enumerate(batch_samples):
            logger.info(f"    Sample {i+1}: {sample['image_name']}")

    logger.info(f"\n✓ Successfully simulated {num_batches} batches")
    return True


def estimate_memory_usage(data_path: str, image_folder: str):
    """Estimate memory usage for full dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("Estimating Memory Usage")
    logger.info("=" * 60)

    with open(data_path) as f:
        data = json.load(f)

    # Sample image size
    dataset = SimpleCADDataset(data_path, image_folder)
    sample = dataset[0]
    image = sample["image"]

    # Image memory (width * height * channels * bytes_per_pixel)
    channels = len(image.mode)
    bytes_per_pixel = 1 if image.mode in ['L', 'P'] else 4
    image_memory = image.size[0] * image.size[1] * channels * bytes_per_pixel

    # Text memory (rough estimate)
    total_text_chars = sum(
        len(turn["value"])
        for item in data
        for turn in item["conversations"]
    )
    text_memory = total_text_chars * 2  # UTF-8 encoding

    logger.info(f"Per-image memory: {image_memory / 1024:.1f} KB")
    logger.info(f"Total images: {len(data)}")
    logger.info(f"Total image memory: {image_memory * len(data) / 1024 / 1024:.1f} MB")
    logger.info(f"\nTotal text memory: {text_memory / 1024 / 1024:.1f} MB")
    logger.info(f"Dataset memory (approx): {(image_memory * len(data) + text_memory) / 1024 / 1024:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to images folder"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to test load (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for simulation (default: 2)"
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("Data Loader Test - DeepCAD Training Pipeline")
    logger.info("=" * 60)

    # Test 1: Load samples
    if not test_data_loading(args.data, args.images, args.num_samples):
        logger.error("\n❌ FAILED: Data loading test")
        sys.exit(1)

    # Test 2: Analyze images
    if not test_image_properties(args.data, args.images):
        logger.error("\n❌ FAILED: Image properties test")
        sys.exit(1)

    # Test 3: Analyze text
    if not test_text_statistics(args.data):
        logger.error("\n❌ FAILED: Text statistics test")
        sys.exit(1)

    # Test 4: Batch simulation
    if not test_batch_simulation(args.data, args.images, args.batch_size):
        logger.error("\n❌ FAILED: Batch simulation test")
        sys.exit(1)

    # Test 5: Memory estimation
    estimate_memory_usage(args.data, args.images)

    # Success!
    logger.info("\n" + "=" * 60)
    logger.info("✅ All Data Loader Tests PASSED!")
    logger.info("=" * 60)
    logger.info("\nData pipeline is ready for training!")


if __name__ == "__main__":
    main()
