"""
Fine-tuning Script for VLM CAD Models

Fine-tunes OpenECAD/TinyLLaVA models on custom CAD datasets using LoRA.

Usage:
    # From command line
    python -m training.finetune --config config/finetune_config.yaml

    # From Python
    from training.finetune import FineTuner, FineTuneConfig
    config = FineTuneConfig(...)
    tuner = FineTuner(config)
    tuner.train()

Requirements:
    - TinyLLaVA Factory: pip install -e git+https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
    - PEFT (for LoRA): pip install peft
    - DeepSpeed (optional): pip install deepspeed
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA configuration following OpenECAD paper settings"""
    r: int = 128                    # LoRA rank
    alpha: int = 256                # LoRA alpha
    dropout: float = 0.05           # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3


@dataclass
class FineTuneConfig:
    """Complete fine-tuning configuration"""
    # Model
    base_model: str = "Yuan-Che/OpenECADv2-SigLIP-0.89B"
    model_type: str = "openecad"  # openecad, internvl

    # Data
    train_data: str = ""           # Path to train.json
    val_data: Optional[str] = None  # Path to val.json
    image_folder: str = ""          # Path to images directory

    # Output
    output_dir: str = "./checkpoints"
    run_name: Optional[str] = None

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    use_lora: bool = True

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Hardware
    device: str = "auto"            # auto, cuda, mps, cpu
    deepspeed_config: Optional[str] = None

    # Conversation
    conv_mode: str = "phi"          # phi, llama, gemma

    def __post_init__(self):
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.base_model.split("/")[-1]
            self.run_name = f"{model_name}_{timestamp}"

    @classmethod
    def from_yaml(cls, path: str) -> "FineTuneConfig":
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        if 'lora' in data:
            data['lora'] = LoRAConfig(**data['lora'])
        if 'training' in data:
            data['training'] = TrainingConfig(**data['training'])

        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file"""
        data = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)


class FineTuner:
    """
    Fine-tunes VLM models for CAD code generation.

    Supports:
    - OpenECAD models (TinyLLaVA-based)
    - InternVL models
    - LoRA and full fine-tuning
    """

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self._check_dependencies()
        self._setup_output_dir()

    def _check_dependencies(self):
        """Check required dependencies"""
        self.has_tinyllava = False
        self.has_peft = False
        self.has_deepspeed = False

        try:
            import tinyllava
            self.has_tinyllava = True
        except ImportError:
            logger.warning(
                "TinyLLaVA not installed. Install from: "
                "https://github.com/TinyLLaVA/TinyLLaVA_Factory"
            )

        try:
            import peft
            self.has_peft = True
        except ImportError:
            logger.warning("PEFT not installed. Install: pip install peft")

        try:
            import deepspeed
            self.has_deepspeed = True
        except ImportError:
            logger.debug("DeepSpeed not installed (optional)")

        try:
            import torch
            self.has_torch = True
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            raise RuntimeError("PyTorch is required for fine-tuning")

    def _setup_output_dir(self):
        """Create output directory structure"""
        self.output_dir = Path(self.config.output_dir) / self.config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(str(self.output_dir / "config.yaml"))

    def train(self, progress_callback=None) -> Dict[str, Any]:
        """
        Run fine-tuning.

        Args:
            progress_callback: Optional callback function(epoch, step, loss, total_steps)
                               for progress reporting

        Returns:
            Training results and metrics
        """
        if not self.has_tinyllava:
            return self._train_with_transformers(progress_callback)
        else:
            return self._train_with_tinyllava(progress_callback)

    def _train_with_tinyllava(self, progress_callback=None) -> Dict[str, Any]:
        """Train using TinyLLaVA's native training"""
        logger.info("Training with TinyLLaVA Factory")

        # Import TinyLLaVA training utilities
        from tinyllava.train import train as tinyllava_train

        # Prepare training arguments
        training_args = self._prepare_tinyllava_args()

        # Run training
        result = tinyllava_train(training_args)

        return {
            "success": True,
            "status": "completed",
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.output_dir),
            "result": result,
        }

    def _prepare_tinyllava_args(self):
        """Prepare arguments for TinyLLaVA training"""
        from argparse import Namespace

        args = Namespace(
            # Model
            model_name_or_path=self.config.base_model,
            version=self.config.conv_mode,

            # Data
            data_path=self.config.train_data,
            image_folder=self.config.image_folder,

            # Training
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,

            # LoRA
            lora_enable=self.config.use_lora,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,

            # Misc
            seed=42,
            report_to="tensorboard",
        )

        return args

    def _train_with_transformers(self, progress_callback=None) -> Dict[str, Any]:
        """Fallback training using transformers + PEFT directly"""
        logger.info("Training with Transformers + PEFT")

        if not self.has_peft:
            raise RuntimeError("PEFT required for LoRA training. Install: pip install peft")

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            TrainerCallback,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Load model
        logger.info(f"Loading model: {self.config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=False,
        )

        # Configure LoRA
        if self.config.use_lora:
            logger.info("Configuring LoRA")
            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
                task_type=self.config.lora.task_type,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Load dataset
        train_dataset = self._load_dataset(self.config.train_data)
        eval_dataset = None
        if self.config.val_data:
            eval_dataset = self._load_dataset(self.config.val_data)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            weight_decay=self.config.training.weight_decay,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard",
            run_name=self.config.run_name,
        )

        # Create progress callback if provided
        callbacks = []
        if progress_callback:
            num_epochs = self.config.training.num_epochs

            class ProgressCallback(TrainerCallback):
                def __init__(self, callback, epochs):
                    self.callback = callback
                    self.epochs = epochs

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and self.callback:
                        loss = logs.get("loss", 0.0)
                        epoch = int(state.epoch) if state.epoch else 0
                        step = state.global_step
                        total_steps = state.max_steps
                        self.callback(epoch, step, loss, total_steps)

            callbacks.append(ProgressCallback(progress_callback, num_epochs))

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save
        trainer.save_model()
        trainer.save_state()

        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return {
            "success": True,
            "status": "completed",
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.output_dir),
            "metrics": metrics,
        }

    def _load_dataset(self, data_path: str):
        """Load dataset from JSON file"""
        from torch.utils.data import Dataset
        import torch

        class CADDataset(Dataset):
            def __init__(self, data_path, image_folder, tokenizer, max_length=2048):
                with open(data_path, 'r') as f:
                    self.data = json.load(f)
                self.image_folder = Path(image_folder)
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]

                # Build conversation text
                text_parts = []
                for turn in item["conversations"]:
                    if turn["from"] == "human":
                        text_parts.append(f"Human: {turn['value']}")
                    else:
                        text_parts.append(f"Assistant: {turn['value']}")

                text = "\n".join(text_parts)

                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                }

        # Note: This is simplified - real implementation needs image processing
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=False,
        )

        return CADDataset(data_path, self.config.image_folder, tokenizer)

    def load_checkpoint(self, checkpoint_path: str) -> Any:
        """Load a fine-tuned checkpoint"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.config.use_lora:
            from peft import PeftModel

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            model = model.merge_and_unload()  # Optional: merge for inference

        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            use_fast=False,
        )

        return model, tokenizer


def create_default_config(output_path: str = "config/finetune_config.yaml"):
    """Create a default configuration file"""
    config = FineTuneConfig(
        base_model="Yuan-Che/OpenECADv2-SigLIP-0.89B",
        train_data="data/training/train/train.json",
        val_data="data/training/val/val.json",
        image_folder="data/training/images",
        output_dir="checkpoints",
    )
    config.to_yaml(output_path)
    logger.info(f"Created default config at {output_path}")
    return config


def main():
    """Command-line interface for fine-tuning"""
    parser = argparse.ArgumentParser(description="Fine-tune VLM CAD models")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default config file"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Yuan-Che/OpenECADv2-SigLIP-0.89B",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        help="Path to images folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=128,
        help="LoRA rank"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.create_config:
        create_default_config()
        return

    # Load or create config
    if args.config:
        config = FineTuneConfig.from_yaml(args.config)
    else:
        config = FineTuneConfig(
            base_model=args.base_model,
            train_data=args.train_data or "",
            image_folder=args.image_folder or "",
            output_dir=args.output_dir,
        )
        config.training.num_epochs = args.epochs
        config.training.per_device_batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
        config.lora.r = args.lora_r
        config.lora.alpha = args.lora_r * 2

    if not config.train_data:
        parser.error("--train-data is required")

    # Run fine-tuning
    tuner = FineTuner(config)
    result = tuner.train()

    logger.info(f"Training completed: {result}")


if __name__ == "__main__":
    main()
