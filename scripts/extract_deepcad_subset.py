"""
Extract a subset of DeepCAD dataset for training

This script extracts a specified number of samples from the DeepCAD dataset
and copies them to a new directory for processing.

Usage:
    python scripts/extract_deepcad_subset.py --size 1000 --output data/training/deepcad_1k
    python scripts/extract_deepcad_subset.py --size 10000 --output data/training/deepcad_10k
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_split_file(split_path: Path) -> dict:
    """Load the train/val/test split JSON file"""
    with open(split_path) as f:
        return json.load(f)


def extract_subset(
    source_dir: Path,
    output_dir: Path,
    sample_ids: List[str],
    split_name: str = "train"
):
    """
    Extract a subset of CAD JSON files

    Args:
        source_dir: Source directory containing cad_json subdirs
        output_dir: Output directory for subset
        sample_ids: List of sample IDs to extract
        split_name: Name of the split (train/val/test)
    """
    output_json_dir = output_dir / "cad_json"
    output_json_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = 0

    for sample_id in sample_ids:
        # DeepCAD structure: cad_json/0000/00001234.json
        # Sample ID format: 0000/00001234 (already includes subdirectory path)

        source_file = source_dir / "cad_json" / f"{sample_id}.json"

        if not source_file.exists():
            logger.warning(f"File not found: {source_file}")
            failed += 1
            continue

        # Extract subdirectory and filename from sample_id
        subdir = sample_id.split('/')[0]
        filename = sample_id.split('/')[1]

        # Create output subdirectory
        output_subdir = output_json_dir / subdir
        output_subdir.mkdir(exist_ok=True)

        # Copy file
        output_file = output_subdir / f"{filename}.json"
        shutil.copy2(source_file, output_file)
        copied += 1

        if copied % 100 == 0:
            logger.info(f"Copied {copied}/{len(sample_ids)} files...")

    logger.info(f"✓ Extraction complete: {copied} files copied, {failed} failed")

    return copied, failed


def create_subset_split(
    original_split: dict,
    train_size: int,
    val_size: int,
    output_dir: Path
):
    """
    Create new train/val split files for the subset

    Args:
        original_split: Original split dictionary
        train_size: Number of training samples
        val_size: Number of validation samples
        output_dir: Output directory
    """
    subset_split = {
        "train": original_split["train"][:train_size],
        "validation": original_split["validation"][:val_size],
        "test": []  # We don't need test set for training
    }

    split_path = output_dir / "train_val_split.json"
    with open(split_path, 'w') as f:
        json.dump(subset_split, f, indent=2)

    logger.info(f"✓ Created split file: {split_path}")
    logger.info(f"  Train: {len(subset_split['train'])} samples")
    logger.info(f"  Val: {len(subset_split['validation'])} samples")

    return subset_split


def create_dataset_info(
    output_dir: Path,
    train_size: int,
    val_size: int,
    source: str = "DeepCAD"
):
    """Create dataset info file"""
    info = {
        "name": output_dir.name,
        "source": source,
        "train_samples": train_size,
        "val_samples": val_size,
        "total_samples": train_size + val_size,
        "format": "DeepCAD JSON",
        "status": "extracted",
        "next_step": "Render orthographic views using render_deepcad_views.py"
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"✓ Created dataset info: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract subset of DeepCAD dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/input/deepcad/data",
        help="Source DeepCAD data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for subset"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of validation samples (default: 100)"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    logger.info("DeepCAD Subset Extraction")
    logger.info("=" * 60)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Val size: {args.val_size}")
    logger.info("=" * 60)

    # Load original split
    split_file = source_dir / "train_val_test_split.json"
    logger.info(f"\nLoading split file: {split_file}")
    original_split = load_split_file(split_file)

    logger.info(f"Original dataset:")
    logger.info(f"  Train: {len(original_split['train'])} samples")
    logger.info(f"  Val: {len(original_split['validation'])} samples")
    logger.info(f"  Test: {len(original_split['test'])} samples")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subset split
    logger.info(f"\nCreating subset split...")
    subset_split = create_subset_split(
        original_split,
        args.train_size,
        args.val_size,
        output_dir
    )

    # Extract training samples
    logger.info(f"\nExtracting training samples...")
    train_copied, train_failed = extract_subset(
        source_dir,
        output_dir,
        subset_split["train"],
        "train"
    )

    # Extract validation samples
    logger.info(f"\nExtracting validation samples...")
    val_copied, val_failed = extract_subset(
        source_dir,
        output_dir,
        subset_split["validation"],
        "validation"
    )

    # Create dataset info
    create_dataset_info(
        output_dir,
        train_copied,
        val_copied
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Subset extraction complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training samples: {train_copied}")
    logger.info(f"Validation samples: {val_copied}")
    logger.info(f"Total: {train_copied + val_copied}")
    logger.info(f"Failed: {train_failed + val_failed}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Render orthographic views:")
    logger.info(f"   python scripts/render_deepcad_views.py --input {output_dir}")
    logger.info("\n2. Or use API:")
    logger.info(f"   POST /training/datasets/create name={output_dir.name}")


if __name__ == "__main__":
    main()
