"""
Training Module for VLM CAD Fine-tuning

This module provides tools for:
1. Generating orthographic projections from 3D models
2. Creating training datasets (2D views + CAD code)
3. Fine-tuning VLM models with LoRA
4. Batch processing for dataset creation

Usage:
    # Generate training dataset
    from training import DatasetGenerator, DatasetConfig
    config = DatasetConfig(output_dir="data/training")
    generator = DatasetGenerator(config)
    stats = generator.generate_from_models(model_paths)

    # Fine-tune model
    from training import FineTuner, FineTuneConfig
    config = FineTuneConfig.from_yaml("config/finetune_config.yaml")
    tuner = FineTuner(config)
    result = tuner.train()
"""

from .orthographic_renderer import (
    OrthographicRenderer,
    ViewType,
    RenderConfig,
    OrthographicView,
    TrainingPair,
)

from .dataset import (
    DatasetGenerator,
    DatasetConfig,
    TrainingSample,
    CADCodeExtractor,
    CADCodeFormat,
    create_dataset_from_directory,
)

from .finetune import (
    FineTuner,
    FineTuneConfig,
    LoRAConfig,
    TrainingConfig,
    create_default_config,
)

from .worker import TrainingWorker
from .trainer import VLMTrainer
from .utils import is_model_cached, get_model_cache_info, format_bytes

__all__ = [
    # Orthographic rendering
    "OrthographicRenderer",
    "ViewType",
    "RenderConfig",
    "OrthographicView",
    "TrainingPair",

    # Dataset generation
    "DatasetGenerator",
    "DatasetConfig",
    "TrainingSample",
    "CADCodeExtractor",
    "CADCodeFormat",
    "create_dataset_from_directory",

    # Fine-tuning
    "FineTuner",
    "FineTuneConfig",
    "LoRAConfig",
    "TrainingConfig",
    "create_default_config",

    # Training worker
    "TrainingWorker",
    "VLMTrainer",

    # Utilities
    "is_model_cached",
    "get_model_cache_info",
    "format_bytes",
]
