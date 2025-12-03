"""
Convert DeepCAD dataset to TinyLLaVA conversation format

This script takes the rendered CAD images and corresponding JSON files
and converts them into the TinyLLaVA training format with conversations.

Usage:
    python scripts/convert_to_tinyllava.py --input data/training/deepcad_1k
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Conversation templates for CAD generation
CONVERSATION_TEMPLATES = [
    {
        "human": "Generate the CAD construction sequence for this design.",
        "system_name": "full_generation"
    },
    {
        "human": "What is the CAD code for this 3D model?",
        "system_name": "code_request"
    },
    {
        "human": "Create parametric CAD code for the design shown in the image.",
        "system_name": "parametric_request"
    },
    {
        "human": "Describe this CAD model using construction operations.",
        "system_name": "construction_description"
    },
    {
        "human": "Convert this image into a CAD construction sequence.",
        "system_name": "image_to_cad"
    },
]


def simplify_cad_json(cad_data: Dict) -> str:
    """
    Convert DeepCAD JSON to a simplified readable CAD code format

    This is a placeholder - in a real implementation, you would:
    1. Parse the construction sequence
    2. Convert to a domain-specific language (DSL)
    3. Format as human-readable code

    For now, we'll create a simplified representation
    """
    entities = cad_data.get('entities', {})

    code_lines = ["# CAD Construction Sequence\n"]

    for entity_id, entity in entities.items():
        entity_type = entity.get('type', 'Unknown')

        if entity_type == 'Sketch':
            # Extract sketch info
            name = entity.get('name', 'Sketch')
            profiles = entity.get('profiles', {})
            num_profiles = len(profiles)

            code_lines.append(f"# Create sketch: {name}")
            code_lines.append(f"sketch = Sketch()")

            # Extract curves from profiles
            for profile_id, profile in profiles.items():
                loops = profile.get('loops', [])
                for loop in loops:
                    curves = loop.get('profile_curves', [])
                    for curve in curves:
                        curve_type = curve.get('type', 'Unknown')
                        start = curve.get('start_point', {})
                        end = curve.get('end_point', {})

                        if curve_type == 'Line3D' and start and end:
                            code_lines.append(
                                f"sketch.add_line("
                                f"start=({start.get('x', 0):.3f}, {start.get('y', 0):.3f}, {start.get('z', 0):.3f}), "
                                f"end=({end.get('x', 0):.3f}, {end.get('y', 0):.3f}, {end.get('z', 0):.3f}))"
                            )
                        elif curve_type == 'Arc3D':
                            code_lines.append(f"sketch.add_arc(...)")
                        elif curve_type == 'Circle3D':
                            code_lines.append(f"sketch.add_circle(...)")

            code_lines.append("")

        elif entity_type == 'ExtrudeFeature':
            # Extract extrusion info
            name = entity.get('name', 'Extrusion')
            extent_one = entity.get('extent_one', {})
            distance = extent_one.get('distance', {}).get('value', 0)

            code_lines.append(f"# Extrude: {name}")
            code_lines.append(f"extrude = Extrude(sketch, distance={distance:.3f})")
            code_lines.append("")

    if len(code_lines) == 1:
        return "# Empty or unsupported CAD model\npass"

    return "\n".join(code_lines)


def create_conversation_entry(
    image_filename: str,
    cad_json_path: Path,
    template_idx: int = 0
) -> Dict[str, Any]:
    """Create a single conversation entry in TinyLLaVA format"""

    # Load CAD JSON
    with open(cad_json_path) as f:
        cad_data = json.load(f)

    # Generate simplified CAD code
    cad_code = simplify_cad_json(cad_data)

    # Select conversation template
    template = CONVERSATION_TEMPLATES[template_idx % len(CONVERSATION_TEMPLATES)]

    # Create conversation
    conversation = {
        "image": image_filename,
        "conversations": [
            {
                "from": "human",
                "value": template["human"]
            },
            {
                "from": "gpt",
                "value": cad_code
            }
        ]
    }

    return conversation


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    split: str = "train",
    randomize_templates: bool = True
):
    """Convert DeepCAD dataset to TinyLLaVA format"""

    # Load split file
    split_file = input_dir / "train_val_split.json"
    with open(split_file) as f:
        splits = json.load(f)

    split_key = "train" if split == "train" else "validation"
    sample_ids = splits.get(split_key, [])

    logger.info(f"Converting {len(sample_ids)} samples for {split} split")

    # Prepare output
    conversations = []

    # Process each sample
    for i, sample_id in enumerate(sample_ids):
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(sample_ids)} samples...")

        # Get paths
        # Sample ID format: "0067/00675619" -> filename is "00675619"
        filename_id = sample_id.split('/')[-1]
        cad_json_path = input_dir / "cad_json" / f"{sample_id}.json"

        # Check if CAD JSON exists
        if not cad_json_path.exists():
            logger.warning(f"CAD JSON not found: {cad_json_path}")
            continue

        # Process each view (front, top, right)
        for view in ["front", "top", "right"]:
            image_filename = f"{filename_id}_{view}.png"

            # Check if image exists
            image_path = input_dir / "images" / image_filename
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            # Select template (random or sequential)
            template_idx = random.randint(0, len(CONVERSATION_TEMPLATES) - 1) if randomize_templates else (i % len(CONVERSATION_TEMPLATES))

            # Create conversation entry
            entry = create_conversation_entry(
                image_filename,
                cad_json_path,
                template_idx
            )

            conversations.append(entry)

    logger.info(f"Created {len(conversations)} conversation entries")

    # Save output
    output_file = output_dir / f"{split}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)

    logger.info(f"✓ Saved {split} data: {output_file}")

    return len(conversations)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepCAD dataset to TinyLLaVA format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (e.g., data/training/deepcad_1k)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--randomize-templates",
        action="store_true",
        default=True,
        help="Randomize conversation templates"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    logger.info("DeepCAD to TinyLLaVA Format Converter")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Convert training set
    logger.info("\nConverting training set...")
    train_count = convert_dataset(
        input_dir,
        output_dir,
        split="train",
        randomize_templates=args.randomize_templates
    )

    # Convert validation set
    logger.info("\nConverting validation set...")
    val_count = convert_dataset(
        input_dir,
        output_dir,
        split="val",
        randomize_templates=args.randomize_templates
    )

    # Create metadata
    metadata = {
        "dataset_name": input_dir.name,
        "format": "TinyLLaVA conversation format",
        "train_samples": train_count,
        "val_samples": val_count,
        "total_conversations": train_count + val_count,
        "views_per_model": 3,
        "conversation_templates": len(CONVERSATION_TEMPLATES),
        "files": {
            "train": str(output_dir / "train.json"),
            "val": str(output_dir / "val.json"),
            "images": str(output_dir / "images")
        }
    }

    metadata_file = output_dir / "tinyllava_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✓ Saved metadata: {metadata_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Conversion complete!")
    logger.info("=" * 60)
    logger.info(f"Training conversations: {train_count}")
    logger.info(f"Validation conversations: {val_count}")
    logger.info(f"Total: {train_count + val_count}")
    logger.info("=" * 60)
    logger.info("\nNext step: Start fine-tuning")
    logger.info(f"  python -m training.finetune \\")
    logger.info(f"    --train_data {output_dir}/train.json \\")
    logger.info(f"    --val_data {output_dir}/val.json \\")
    logger.info(f"    --image_folder {output_dir}/images")


if __name__ == "__main__":
    main()
