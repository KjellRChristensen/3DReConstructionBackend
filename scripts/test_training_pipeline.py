"""
Dry Run Test for Training Pipeline

Tests the complete training pipeline without downloading models or doing full training.
Verifies:
1. Dataset loading
2. Data format validation
3. Image file accessibility
4. Configuration parsing
5. Basic data pipeline

Usage:
    python scripts/test_training_pipeline.py --config config/deepcad_1k_dryrun.yaml
"""

import argparse
import json
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_dataset_files(train_data_path: str, val_data_path: str, image_folder: str):
    """Test that dataset files exist and are readable"""
    logger.info("=" * 60)
    logger.info("Testing Dataset Files")
    logger.info("=" * 60)

    # Check train data
    train_path = Path(train_data_path)
    if not train_path.exists():
        logger.error(f"❌ Train data not found: {train_data_path}")
        return False
    logger.info(f"✓ Train data exists: {train_data_path}")

    # Check val data
    val_path = Path(val_data_path)
    if not val_path.exists():
        logger.error(f"❌ Validation data not found: {val_data_path}")
        return False
    logger.info(f"✓ Validation data exists: {val_data_path}")

    # Check image folder
    image_dir = Path(image_folder)
    if not image_dir.exists():
        logger.error(f"❌ Image folder not found: {image_folder}")
        return False
    logger.info(f"✓ Image folder exists: {image_folder}")

    return True


def test_data_format(train_data_path: str, val_data_path: str, image_folder: str):
    """Test that data format is correct"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Data Format")
    logger.info("=" * 60)

    # Load train data
    with open(train_data_path) as f:
        train_data = json.load(f)

    logger.info(f"✓ Train data loaded: {len(train_data)} samples")

    # Load val data
    with open(val_data_path) as f:
        val_data = json.load(f)

    logger.info(f"✓ Validation data loaded: {len(val_data)} samples")

    # Validate format
    if len(train_data) == 0:
        logger.error("❌ Train data is empty")
        return False

    # Check first sample
    sample = train_data[0]
    required_fields = ["image", "conversations"]

    for field in required_fields:
        if field not in sample:
            logger.error(f"❌ Missing required field '{field}' in sample")
            return False

    logger.info(f"✓ Required fields present: {required_fields}")

    # Check conversations format
    conversations = sample["conversations"]
    if len(conversations) == 0:
        logger.error("❌ Conversations list is empty")
        return False

    conv = conversations[0]
    if "from" not in conv or "value" not in conv:
        logger.error("❌ Invalid conversation format")
        return False

    logger.info(f"✓ Conversation format valid")

    # Show sample
    logger.info("\nSample conversation:")
    logger.info(f"  Image: {sample['image']}")
    for i, turn in enumerate(conversations[:2]):
        logger.info(f"  Turn {i+1} ({turn['from']}): {turn['value'][:80]}...")

    return True


def test_image_files(train_data_path: str, image_folder: str, max_check: int = 10):
    """Test that referenced images exist"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Image Files")
    logger.info("=" * 60)

    with open(train_data_path) as f:
        train_data = json.load(f)

    image_dir = Path(image_folder)

    # Check first N images
    missing = []
    found = 0

    for i, sample in enumerate(train_data[:max_check]):
        image_name = sample["image"]
        image_path = image_dir / image_name

        if image_path.exists():
            found += 1
            if i < 3:
                logger.info(f"  ✓ {image_name}")
        else:
            missing.append(image_name)
            logger.warning(f"  ✗ {image_name}")

    if missing:
        logger.error(f"❌ {len(missing)} images missing out of {max_check} checked")
        return False

    logger.info(f"✓ All {found} checked images exist")

    # Count total images
    all_images = list(image_dir.glob("*.png"))
    logger.info(f"✓ Total images in folder: {len(all_images)}")

    return True


def test_config_loading(config_path: str):
    """Test configuration loading"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Configuration Loading")
    logger.info("=" * 60)

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"✓ Config loaded from: {config_path}")
        logger.info(f"  Base model: {config.get('base_model')}")
        logger.info(f"  Train data: {config.get('train_data')}")
        logger.info(f"  Val data: {config.get('val_data')}")
        logger.info(f"  Image folder: {config.get('image_folder')}")
        logger.info(f"  Output dir: {config.get('output_dir')}")

        if 'training' in config:
            logger.info(f"  Epochs: {config['training'].get('num_epochs')}")
            logger.info(f"  Batch size: {config['training'].get('per_device_batch_size')}")
            logger.info(f"  Max steps: {config['training'].get('max_steps', 'unlimited')}")

        return config

    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return None


def test_data_statistics(train_data_path: str, val_data_path: str):
    """Show dataset statistics"""
    logger.info("\n" + "=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)

    with open(train_data_path) as f:
        train_data = json.load(f)

    with open(val_data_path) as f:
        val_data = json.load(f)

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Total conversations: {len(train_data) + len(val_data)}")

    # Analyze conversation lengths
    train_lengths = []
    for sample in train_data[:100]:
        for turn in sample["conversations"]:
            train_lengths.append(len(turn["value"]))

    if train_lengths:
        avg_length = sum(train_lengths) / len(train_lengths)
        max_length = max(train_lengths)
        min_length = min(train_lengths)

        logger.info(f"\nConversation text statistics (first 100 samples):")
        logger.info(f"  Average length: {avg_length:.0f} chars")
        logger.info(f"  Max length: {max_length} chars")
        logger.info(f"  Min length: {min_length} chars")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--max-image-check",
        type=int,
        default=20,
        help="Max number of images to verify (default: 20)"
    )

    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("DeepCAD Training Pipeline - Dry Run Test")
    logger.info("=" * 60)

    # Test 1: Load configuration
    config = test_config_loading(args.config)
    if config is None:
        logger.error("\n❌ Dry run FAILED: Config loading failed")
        sys.exit(1)

    train_data = config.get('train_data')
    val_data = config.get('val_data')
    image_folder = config.get('image_folder')

    # Test 2: Check files exist
    if not test_dataset_files(train_data, val_data, image_folder):
        logger.error("\n❌ Dry run FAILED: Dataset files missing")
        sys.exit(1)

    # Test 3: Validate data format
    if not test_data_format(train_data, val_data, image_folder):
        logger.error("\n❌ Dry run FAILED: Data format invalid")
        sys.exit(1)

    # Test 4: Check image files
    if not test_image_files(train_data, image_folder, args.max_image_check):
        logger.error("\n❌ Dry run FAILED: Image files missing")
        sys.exit(1)

    # Test 5: Show statistics
    test_data_statistics(train_data, val_data)

    # Success!
    logger.info("\n" + "=" * 60)
    logger.info("✅ Dry Run PASSED - All Tests Successful!")
    logger.info("=" * 60)
    logger.info("\nDataset is ready for training!")
    logger.info(f"\nTo start training, run:")
    logger.info(f"  python -m training.finetune --config {args.config}")
    logger.info("\nNote: This was a dry run. Actual training requires:")
    logger.info("  1. Downloading the base model (~2-3 GB)")
    logger.info("  2. GPU/MPS acceleration for reasonable speed")
    logger.info("  3. 8-16 GB RAM/VRAM")


if __name__ == "__main__":
    main()
