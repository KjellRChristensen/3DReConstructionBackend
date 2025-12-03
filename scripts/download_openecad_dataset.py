"""
Download and prepare OpenECAD dataset for training

This script downloads the OpenECAD dataset from HuggingFace and converts it
to the format expected by the training pipeline.

Usage:
    python scripts/download_openecad_dataset.py --output data/training/openecad_full
    python scripts/download_openecad_dataset.py --subset 10000 --output data/training/openecad_10k
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_prepare_openecad(
    output_dir: str = "data/training/openecad_full",
    subset: Optional[int] = None,
    train_split: float = 0.9,
):
    """
    Download OpenECAD dataset and convert to training format

    Args:
        output_dir: Directory to save the processed dataset
        subset: If specified, only use first N samples (for testing)
        train_split: Fraction of data to use for training (rest is validation)
    """
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        logger.error("Required packages not installed. Run: pip install datasets pillow")
        return

    # Download from HuggingFace
    logger.info("Downloading OpenECAD dataset from HuggingFace...")
    if subset:
        logger.info(f"Loading subset: {subset} samples")
        dataset = load_dataset("Yuan-Che/OpenECAD-Dataset", split=f"train[:{subset}]")
    else:
        logger.info("Loading full dataset (919K samples - this may take a while)")
        dataset = load_dataset("Yuan-Che/OpenECAD-Dataset", split="train")

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Create directory structure
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    train_dir = output_path / "train"
    val_dir = output_path / "val"

    images_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Split dataset
    split_idx = int(len(dataset) * train_split)
    logger.info(f"Splitting: {split_idx} train, {len(dataset) - split_idx} val")

    # Process training samples
    logger.info("Processing training samples...")
    train_samples = []
    for i in range(split_idx):
        if i % 1000 == 0:
            logger.info(f"  Processed {i}/{split_idx} training samples")

        sample = dataset[i]

        # Save image (handle both PIL Image and string path)
        img_filename = f"train_{i:06d}.jpg"
        img_path = images_dir / img_filename

        image = sample["image"]
        if isinstance(image, str):
            # If it's a string path, assume it's already in the right format
            # and the dataset will be structured differently
            logger.warning(f"Image is string path, not PIL Image: {image}")
            # Skip image processing for now - dataset may need different handling
            continue
        else:
            # PIL Image object
            image.save(img_path, quality=95)

        # Create conversation format
        train_samples.append({
            "id": f"train_{i:06d}",
            "image": img_filename,
            "conversations": sample["conversations"]
        })

    # Process validation samples
    logger.info("Processing validation samples...")
    val_samples = []
    for i in range(split_idx, len(dataset)):
        if (i - split_idx) % 1000 == 0:
            logger.info(f"  Processed {i - split_idx}/{len(dataset) - split_idx} validation samples")

        sample = dataset[i]

        # Save image
        img_filename = f"val_{i - split_idx:06d}.jpg"
        img_path = images_dir / img_filename
        sample["image"].save(img_path, quality=95)

        # Create conversation format
        val_samples.append({
            "id": f"val_{i - split_idx:06d}",
            "image": img_filename,
            "conversations": sample["conversations"]
        })

    # Save JSON files
    logger.info("Saving dataset JSON files...")
    train_json_path = train_dir / "train.json"
    with open(train_json_path, 'w') as f:
        json.dump(train_samples, f, indent=2)

    val_json_path = val_dir / "val.json"
    with open(val_json_path, 'w') as f:
        json.dump(val_samples, f, indent=2)

    # Create config file
    config = {
        "dataset_name": "OpenECAD",
        "source": "Yuan-Che/OpenECAD-Dataset",
        "total_samples": len(dataset),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_split": train_split,
        "image_folder": str(images_dir),
        "train_json": str(train_json_path),
        "val_json": str(val_json_path),
    }

    config_path = output_path / "dataset_info.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("✅ Dataset preparation complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Train samples:    {len(train_samples)}")
    logger.info(f"Val samples:      {len(val_samples)}")
    logger.info(f"Image folder:     {images_dir}")
    logger.info(f"Train JSON:       {train_json_path}")
    logger.info(f"Val JSON:         {val_json_path}")
    logger.info(f"Config:           {config_path}")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Update config/finetune_config.yaml with these paths:")
    logger.info(f"   train_data: \"{train_json_path}\"")
    logger.info(f"   val_data: \"{val_json_path}\"")
    logger.info(f"   image_folder: \"{images_dir}\"")
    logger.info("\n2. Start training:")
    logger.info("   python -m training.finetune --config config/finetune_config.yaml")
    logger.info("\n3. Or use the API:")
    logger.info(f"   POST /training/finetune/start with dataset_name=\"{output_path.name}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare OpenECAD dataset for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/openecad_full",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only download first N samples (for testing)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)"
    )

    args = parser.parse_args()

    logger.info("OpenECAD Dataset Download and Preparation")
    logger.info("="*60)
    logger.info(f"Output directory: {args.output}")
    if args.subset:
        logger.info(f"Subset size:      {args.subset}")
    else:
        logger.info(f"Subset size:      Full dataset (919K samples)")
    logger.info(f"Train split:      {args.train_split:.1%}")
    logger.info("="*60 + "\n")

    download_and_prepare_openecad(
        output_dir=args.output,
        subset=args.subset,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
