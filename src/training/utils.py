"""
Utility functions for training
"""
from pathlib import Path
import os
from typing import Optional


def is_model_cached(huggingface_id: str) -> bool:
    """
    Check if a HuggingFace model is already downloaded/cached locally.

    Args:
        huggingface_id: HuggingFace model ID (e.g., "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B")

    Returns:
        True if model is cached, False otherwise
    """
    # Get HuggingFace cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    # Convert model ID to cache directory name
    # e.g., "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" -> "models--tinyllava--TinyLLaVA-Phi-2-SigLIP-3.1B"
    model_dir_name = "models--" + huggingface_id.replace("/", "--")
    model_cache_path = cache_dir / model_dir_name

    # Check if directory exists
    if not model_cache_path.exists():
        return False

    # Check if snapshots directory exists (indicates successful download)
    snapshots_dir = model_cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    # Check if there's at least one snapshot with files
    # (incomplete downloads won't have snapshot directories)
    try:
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            return False

        # Check if the first snapshot has actual files
        first_snapshot = snapshots[0]
        if first_snapshot.is_dir():
            files = list(first_snapshot.iterdir())
            # Should have at least config.json and model files
            return len(files) > 0

    except (PermissionError, OSError):
        return False

    return False


def get_model_cache_info(huggingface_id: str) -> dict:
    """
    Get detailed cache information for a model.

    Args:
        huggingface_id: HuggingFace model ID

    Returns:
        Dictionary with cache information:
        - is_cached: bool
        - cache_size_bytes: int (0 if not cached)
        - cache_path: str or None
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + huggingface_id.replace("/", "--")
    model_cache_path = cache_dir / model_dir_name

    info = {
        "is_cached": False,
        "cache_size_bytes": 0,
        "cache_path": None,
    }

    if not model_cache_path.exists():
        return info

    # Calculate directory size
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_cache_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)

        info["cache_size_bytes"] = total_size
        info["cache_path"] = str(model_cache_path)
        info["is_cached"] = is_model_cached(huggingface_id)

    except (PermissionError, OSError):
        pass

    return info


def format_bytes(bytes_size: int) -> str:
    """Format bytes in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"
