"""
VLM Trainer for CAD reconstruction fine-tuning
"""
import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Disable all parallelism to avoid semaphore leaks and multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# MPS (Metal Performance Shaders) stability settings for macOS
# These help prevent crashes on newer macOS versions (26+/Tahoe)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable aggressive memory caching
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for unsupported MPS ops
os.environ["MTL_DEBUG_LAYER"] = "0"  # Disable Metal debug validation layer
os.environ["MTL_SHADER_VALIDATION"] = "0"  # Disable Metal shader validation

# Force multiprocessing to use the correct Python and start method
import sys
import multiprocessing

# Ensure our venv's bin directory is first in PATH
venv_bin = os.path.dirname(sys.executable)
current_path = os.environ.get("PATH", "")
if not current_path.startswith(venv_bin):
    os.environ["PATH"] = f"{venv_bin}:{current_path}"

# Set executable environment variables
os.environ["PYTHONEXECUTABLE"] = sys.executable
os.environ["__PYVENV_LAUNCHER__"] = sys.executable

# Set multiprocessing start method (spawn is default on macOS, but be explicit)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from ..database import get_db, TrainingJob, TrainingResult, Model

logger = logging.getLogger(__name__)


def check_mps_stability() -> bool:
    """
    Test if MPS (Metal Performance Shaders) is stable on this system.

    Runs a small test computation to detect potential Metal heap allocator
    crashes that occur on newer macOS versions (26+/Tahoe) with Python 3.13+.

    Returns:
        True if MPS is stable and can be used, False otherwise
    """
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False

    # Check macOS version - MPS is unstable on macOS 26+ (Tahoe)
    # due to Metal API changes that cause heap allocator crashes
    import platform
    try:
        macos_version = platform.mac_ver()[0]
        major_version = int(macos_version.split('.')[0])
        if major_version >= 26:
            logger.warning(
                f"macOS {macos_version} detected. MPS is unstable on macOS 26+ "
                "due to Metal heap allocator crashes. Forcing CPU mode."
            )
            return False
    except (ValueError, IndexError):
        pass  # Can't parse version, continue with stability test

    try:
        logger.info("Testing MPS stability...")

        # Test 1: Basic tensor operations
        device = torch.device("mps")
        x = torch.randn(64, 64, device=device)
        y = torch.randn(64, 64, device=device)
        z = torch.matmul(x, y)

        # Test 2: Linear layer (this is where the crash typically occurs)
        linear = torch.nn.Linear(64, 64).to(device)
        input_tensor = torch.randn(4, 64, device=device)
        output = linear(input_tensor)

        # Test 3: Synchronize and clear cache to check memory management
        torch.mps.synchronize()
        torch.mps.empty_cache()

        # Test 4: Multiple allocations and deallocations (stress test memory)
        tensors = []
        for i in range(10):
            t = torch.randn(128, 128, device=device)
            tensors.append(t)
        del tensors
        torch.mps.synchronize()
        torch.mps.empty_cache()

        # Test 5: Larger operation similar to training
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
        ).to(device)

        test_input = torch.randn(8, 256, device=device)
        test_output = model(test_input)
        loss = test_output.mean()
        loss.backward()

        torch.mps.synchronize()
        torch.mps.empty_cache()

        # Cleanup
        del model, x, y, z, linear, input_tensor, output, test_input, test_output, loss
        torch.mps.empty_cache()

        logger.info("MPS stability check passed")
        return True

    except Exception as e:
        logger.warning(f"MPS stability check failed: {e}")
        logger.warning("Falling back to CPU for training")

        # Try to clean up MPS state
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        return False


class CADDataset(Dataset):
    """Dataset for CAD image-to-code pairs"""

    def __init__(self, dataset_path: str, split: str = "train"):
        """
        Initialize CAD dataset

        Args:
            dataset_path: Path to dataset directory
            split: "train" or "val"
        """
        self.dataset_path = Path(dataset_path)
        self.split = split

        # Load conversation data
        # Try multiple file name patterns
        possible_files = [
            self.dataset_path / f"{split}_conversations.json",
            self.dataset_path / f"{split}.json",
        ]

        conv_file = None
        for file_path in possible_files:
            if file_path.exists():
                conv_file = file_path
                break

        if not conv_file:
            raise FileNotFoundError(f"Conversation file not found. Tried: {possible_files}")

        with open(conv_file) as f:
            self.conversations = json.load(f)

        logger.info(f"Loaded {len(self.conversations)} {split} samples from {dataset_path}")

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        """Get a single training sample"""
        sample = self.conversations[idx]

        # Load image - try multiple possible locations
        image_filename = sample["image"]
        possible_paths = [
            self.dataset_path / image_filename,  # Direct path
            self.dataset_path / "images" / image_filename,  # images subdirectory
        ]

        image = None
        image_path = None
        for path in possible_paths:
            try:
                image = Image.open(path).convert("RGB")
                image_path = path
                break
            except Exception:
                continue

        if image is None:
            logger.warning(f"Failed to load image {image_filename}, using blank image")
            # Return a blank image as fallback
            image = Image.new("RGB", (512, 512), color=(255, 255, 255))
            image_path = possible_paths[0]

        # Get conversations
        conversations = sample.get("conversations", [])

        return {
            "image": image,
            "conversations": conversations,
            "image_path": str(image_path),
        }


class ProgressCallback(TrainerCallback):
    """Callback to update training progress in database"""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update progress at epoch start"""
        try:
            with get_db() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()

                # Check if job was cancelled or deleted
                if not job:
                    logger.warning(f"Job {self.job_id} not found - stopping training")
                    control.should_training_stop = True
                    return control

                if job.status == "failed" and job.error_message and "Stopped by user request" in job.error_message:
                    logger.warning(f"Job {self.job_id} was cancelled - stopping training")
                    control.should_training_stop = True
                    return control

                if job:
                    epoch = int(state.epoch) if state.epoch else 0
                    job.current_stage = f"Training Epoch {epoch + 1}/{args.num_train_epochs}"
                    db.commit()
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update progress and metrics on log"""
        if not logs:
            return

        try:
            # Log available keys for debugging
            logger.info(f"Training logs: {list(logs.keys())}")
            if "loss" in logs:
                logger.info(f"Loss value: {logs['loss']}")

            with get_db() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()

                # Check if job was cancelled or deleted
                if not job:
                    logger.warning(f"Job {self.job_id} not found - stopping training")
                    control.should_training_stop = True
                    return control

                if job.status == "failed" and job.error_message and "Stopped by user request" in job.error_message:
                    logger.warning(f"Job {self.job_id} was cancelled - stopping training")
                    control.should_training_stop = True
                    return control

                # Update progress (0.0 to 1.0)
                if state.max_steps > 0:
                    progress = state.global_step / state.max_steps
                    job.progress = min(progress, 1.0)
                    job.total_steps = state.max_steps
                    job.current_step = state.global_step

                # Update current epoch
                job.current_epoch = int(state.epoch) if state.epoch else 0

                # Save training result - look for various loss keys
                loss_value = logs.get("loss")
                if loss_value is None:
                    loss_value = logs.get("train_loss") or logs.get("train/loss")

                if loss_value is not None:
                    # Update current loss on job record (for real-time display)
                    job.current_loss = float(loss_value)

                    # Also save to training_results table for history
                    result = TrainingResult(
                        job_id=job.id,
                        epoch=int(state.epoch) if state.epoch else None,
                        train_loss=float(loss_value),
                        val_loss=logs.get("eval_loss") or logs.get("eval/loss"),
                        learning_rate=logs.get("learning_rate"),
                        timestamp=datetime.utcnow(),
                    )
                    db.add(result)

                db.commit()

        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}")


class VLMTrainer:
    """Trainer for Vision-Language Models on CAD data"""

    def __init__(
        self,
        job_id: str,
        dataset_path: str,
        model_name: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        use_lora: bool = True,
        device: str = "auto",
    ):
        """
        Initialize VLM trainer

        Args:
            job_id: Training job ID
            dataset_path: Path to prepared dataset
            model_name: HuggingFace model ID (e.g., "bczhou/TinyLLaVA-1.1B")
            output_dir: Directory to save model checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            use_lora: Whether to use LoRA for efficient fine-tuning
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        self.job_id = job_id
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_lora = use_lora

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device selection: use specified device or auto-detect
        # For MPS, we run a stability check first due to crashes on macOS 26+/Tahoe
        if device and device != "auto":
            self.device = device
            # If user explicitly requested MPS, still check stability
            if device == "mps" and not check_mps_stability():
                logger.warning("MPS explicitly requested but stability check failed, falling back to CPU")
                self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            # MPS available but may be unstable on newer macOS versions
            if check_mps_stability():
                self.device = "mps"
            else:
                logger.warning("MPS available but unstable, using CPU instead")
                self.device = "cpu"
        else:
            self.device = "cpu"
        print(f"[TRAINER] Using device: {self.device}")
        logger.info(f"Using device: {self.device}")

    def _load_model_and_processor(self):
        """Load VLM model and processor from HuggingFace"""
        from .download_tracker import get_download_tracker
        from .utils import is_model_cached

        logger.info(f"Loading model: {self.model_name}")

        try:
            # Start download tracking if model not cached
            tracker = get_download_tracker()
            if not is_model_cached(self.model_name):
                logger.info(f"Model not cached, starting download tracking...")
                tracker.start_tracking(self.model_name, estimated_size_gb=6.0)

            # Load processor/tokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Load config first and set attention implementation
            from transformers import AutoConfig
            from transformers import PreTrainedModel

            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            # Force eager attention to avoid SDPA compatibility issues
            config._attn_implementation = "eager"
            config._attn_implementation_internal = "eager"
            # Disable KV cache for training (prevents memory accumulation and slowdown)
            config.use_cache = False

            # Load model
            # Use float16 for CUDA only, float32 for MPS and CPU
            # MPS has precision issues with float16 that cause NaN in complex models
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Monkey-patch PreTrainedModel to handle missing _supports_sdpa gracefully
            original_getattr = PreTrainedModel.__getattribute__

            def patched_getattr(self, name):
                if name == "_supports_sdpa":
                    return False  # Default to False if not defined
                return original_getattr(self, name)

            PreTrainedModel.__getattribute__ = patched_getattr

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    config=config,
                    dtype=dtype,  # Changed from torch_dtype (deprecated)
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    attn_implementation="eager",  # Use eager attention (compatible with all models)
                )
            finally:
                # Restore original __getattribute__
                PreTrainedModel.__getattribute__ = original_getattr

            # Stop download tracking
            tracker.stop_tracking(self.model_name)

            # Move model to the target device
            print(f"[TRAINER] Moving model to device: {self.device}")
            logger.info(f"Moving model to device: {self.device}")
            self.model = self.model.to(self.device)
            print(f"[TRAINER] Model moved to {self.device} successfully")

            # Apply LoRA if requested
            if self.use_lora:
                logger.info("Applying LoRA configuration")
                lora_config = LoraConfig(
                    r=16,  # LoRA rank
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],  # Attention layers
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.model = get_peft_model(self.model, lora_config)
                # Enable input gradients for gradient checkpointing compatibility
                self.model.enable_input_require_grads()
                self.model.print_trainable_parameters()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _prepare_datasets(self):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets...")

        self.train_dataset = CADDataset(self.dataset_path, split="train")
        self.val_dataset = CADDataset(self.dataset_path, split="val")

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")

    def _collate_fn(self, batch):
        """Collate function for DataLoader - compatible with TinyLLaVA"""
        images = [item["image"] for item in batch]
        conversations = [item["conversations"] for item in batch]

        # Format conversations as text with IMAGE token
        # TinyLLaVA expects <image> token to indicate where vision features should be inserted
        IMAGE_TOKEN = "<image>"

        texts = []
        assistant_masks = []  # Track where assistant responses are

        for convs in conversations:
            # Build full conversation text with image token at the beginning
            text = f"{IMAGE_TOKEN}\n"  # Add image token first

            for conv in convs:
                role = conv.get("from", "unknown")
                value = conv.get("value", "")

                if role == "human":
                    text += f"USER: {value}\n"
                elif role == "gpt":
                    text += f"ASSISTANT: {value}\n"

            texts.append(text.strip())

        try:
            # Use tokenizer directly (not processor)
            # TinyLLaVA's processor may not support both text and images together
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor

            # Tokenize text (with image tokens)
            text_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            # Process images with proper processor
            # Get the vision model's expected input size from the processor
            if hasattr(self.processor, 'image_processor'):
                image_processor = self.processor.image_processor
            elif hasattr(self.processor, 'feature_extractor'):
                image_processor = self.processor.feature_extractor
            else:
                # Fallback - read image processor config from database
                image_processor = None
                try:
                    with get_db() as db:
                        # Query the model table for image processor configuration
                        model_record = db.query(Model).filter(
                            Model.huggingface_id == self.model_name
                        ).first()

                        if model_record and model_record.image_processor_id:
                            processor_id = model_record.image_processor_id
                            image_size = model_record.image_size or 224
                            logger.info(f"Using image processor from database: {processor_id} (size: {image_size})")

                            # Load the appropriate processor based on the ID
                            if 'siglip' in processor_id.lower():
                                from transformers import SiglipImageProcessor
                                image_processor = SiglipImageProcessor.from_pretrained(processor_id)
                            else:
                                from transformers import CLIPImageProcessor
                                image_processor = CLIPImageProcessor.from_pretrained(processor_id)
                except Exception as e:
                    logger.warning(f"Failed to load image processor from database: {e}")

                # If database lookup failed, use hardcoded fallback
                if image_processor is None:
                    model_name_lower = self.model_name.lower()
                    if 'siglip' in model_name_lower or 'phi-2-siglip' in model_name_lower:
                        from transformers import SiglipImageProcessor
                        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
                        logger.info(f"Using SigLIP fallback processor for {self.model_name}")
                    else:
                        # Default to CLIP processor (224x224) for most TinyLLaVA models
                        from transformers import CLIPImageProcessor
                        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
                        logger.info(f"Using CLIP fallback processor (224x224) for {self.model_name}")

            # Process all images
            image_inputs = image_processor(
                images=images,
                return_tensors="pt"
            )

            # Build final input dict
            inputs = {
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
            }

            # Add image tensor with correct key name
            # TinyLLaVA expects 'images' (not 'pixel_values')
            if "pixel_values" in image_inputs:
                inputs["images"] = image_inputs["pixel_values"]
            else:
                inputs["images"] = image_inputs.get("images")

            # Create labels for causal LM training
            # CRITICAL: We need to mask everything EXCEPT assistant responses
            labels = inputs["input_ids"].clone()

            # First, mask ALL tokens
            labels[:, :] = -100

            # Then, unmask only the ASSISTANT responses
            # Use character offset mapping for reliable token position finding
            for idx, text in enumerate(texts):
                input_ids = inputs["input_ids"][idx]

                # Tokenize with offset mapping to get char->token positions
                tokenized_with_offsets = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=2048,
                )
                offset_mapping = tokenized_with_offsets.get("offset_mapping", [])

                # Find all "ASSISTANT:" positions in the text
                search_start = 0
                assistant_marker = "ASSISTANT:"

                while True:
                    pos = text.find(assistant_marker, search_start)
                    if pos == -1:
                        break

                    # Find the end of this assistant response (next USER: or end of text)
                    response_start_char = pos + len(assistant_marker)
                    next_user = text.find("USER:", response_start_char)
                    response_end_char = next_user if next_user != -1 else len(text)

                    # Convert character positions to token positions using offset mapping
                    token_start = None
                    token_end = None

                    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
                        # Find first token that starts at or after response_start_char
                        if token_start is None and char_start >= response_start_char:
                            token_start = tok_idx
                        # Find last token before response_end_char
                        if char_end <= response_end_char and char_start < response_end_char:
                            token_end = tok_idx + 1

                    # Unmask the response tokens (skip the "ASSISTANT:" prefix)
                    if token_start is not None and token_end is not None and token_start < token_end:
                        # Ensure we don't exceed the actual input length
                        token_end = min(token_end, len(input_ids))
                        labels[idx, token_start:token_end] = input_ids[token_start:token_end]
                        logger.debug(f"Unmasked tokens {token_start}:{token_end} for assistant response")

                    search_start = response_end_char

            # Also mask padding tokens
            labels[inputs["attention_mask"] == 0] = -100

            inputs["labels"] = labels

            # Log label statistics for debugging
            non_masked = (labels != -100).sum().item()
            total = labels.numel()
            logger.info(f"Batch labels - Total tokens: {total}, Training tokens: {non_masked} ({100*non_masked/total:.1f}%), Masked: {total - non_masked}")

            # Log tensor shapes for debugging
            logger.debug(f"Batch shapes - input_ids: {inputs['input_ids'].shape}, "
                        f"images: {inputs['images'].shape}, labels: {labels.shape}")

            return inputs

        except Exception as e:
            logger.error(f"Collate error: {e}", exc_info=True)
            # Return minimal valid batch
            return {
                "input_ids": torch.tensor([[self.processor.tokenizer.eos_token_id]]),
                "attention_mask": torch.tensor([[1]]),
                "images": torch.zeros((1, 3, 384, 384)),  # SigLIP default size
                "labels": torch.tensor([[-100]]),
            }

    def train(self):
        """Run training"""
        logger.info(f"Starting training for job {self.job_id}")

        # Update job status - mark as loading model
        with get_db() as db:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()
            if job:
                job.current_stage = "Loading model"
                job.is_loading_model = True
                db.commit()

        # Load model and datasets
        self._load_model_and_processor()
        self._prepare_datasets()

        # Mark model loading complete
        with get_db() as db:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()
            if job:
                job.is_loading_model = False
                job.current_stage = "Preparing training"
                db.commit()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",  # Changed from evaluation_strategy for transformers v4.20+
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.device == "cuda",  # Only use fp16 for CUDA (MPS handles dtype automatically)
            dataloader_num_workers=0,  # Disable multiprocessing (causes pickle issues with monkey-patched models and MPS)
            dataloader_pin_memory=False,  # MPS doesn't support pinned memory
            gradient_checkpointing=False,  # Disabled - causes Metal GPU issues on MPS
            remove_unused_columns=False,
            report_to=["none"],  # Disable wandb/tensorboard
            use_cpu=(self.device == "cpu"),  # Explicitly set CPU mode if needed
        )

        # Ensure model is in training mode
        self.model.train()

        # Verify some parameters require gradients
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")

        if trainable_params == 0:
            raise ValueError("No trainable parameters! Model won't train.")

        # Create custom compute_loss function with essential validation
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                """Custom loss computation with validation"""
                # Ensure inputs have required keys
                if "labels" not in inputs:
                    raise ValueError("Labels not found in inputs!")

                labels = inputs.get("labels")

                # Check if all labels are masked
                if labels is not None:
                    non_masked = (labels != -100).sum().item()
                    if non_masked == 0:
                        logger.warning("All labels are masked (-100)! Returning zero loss.")
                        return torch.tensor(0.0, device=model.device, requires_grad=True)

                # Forward pass
                outputs = model(**inputs)

                # Extract loss from model output
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                    logits = getattr(outputs, 'logits', None)
                elif isinstance(outputs, dict):
                    loss = outputs.get("loss")
                    logits = outputs.get("logits")
                else:
                    loss = outputs
                    logits = None

                # If model didn't return loss, compute manually
                if loss is None:
                    if logits is None:
                        raise ValueError("Model outputs don't contain 'logits' or 'loss'")

                    import torch.nn.functional as F
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )

                # Validate loss - warn if issues detected
                if loss is not None:
                    if not torch.isfinite(loss):
                        logger.warning(f"Loss is NaN or Inf: {loss.item()}")
                    elif loss.item() == 0.0:
                        logger.warning("Loss is exactly 0.0")

                return (loss, outputs) if return_outputs else loss

        # MPS memory management callback to prevent Metal heap issues
        class MPSMemoryCallback(TrainerCallback):
            """Periodically clear MPS cache to prevent Metal heap allocation crashes."""

            def __init__(self, device: str, clear_every_n_steps: int = 50):
                self.device = device
                self.clear_every_n_steps = clear_every_n_steps

            def on_step_end(self, args, state, control, **kwargs):
                if self.device == "mps" and state.global_step % self.clear_every_n_steps == 0:
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception as e:
                        logger.warning(f"Failed to clear MPS cache: {e}")

            def on_epoch_end(self, args, state, control, **kwargs):
                if self.device == "mps":
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                        logger.info("Cleared MPS cache at end of epoch")
                    except Exception as e:
                        logger.warning(f"Failed to clear MPS cache: {e}")

        # Build callbacks list
        callbacks = [ProgressCallback(self.job_id)]
        if self.device == "mps":
            callbacks.append(MPSMemoryCallback(self.device))

        # Create trainer
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self._collate_fn,
            callbacks=callbacks,
        )

        # Update status
        with get_db() as db:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()
            if job:
                job.current_stage = "Training"
                db.commit()

        # Train with MPS crash recovery
        print(f"[TRAINER] Starting training loop on device: {self.device}")
        logger.info("Starting training loop...")

        try:
            trainer.train()
            print("[TRAINER] Training loop completed")
        except Exception as e:
            error_msg = str(e).lower()
            is_mps_error = (
                self.device == "mps" and
                any(x in error_msg for x in ['mps', 'metal', 'heap', 'gpu'])
            )

            if is_mps_error:
                logger.error(f"MPS training crashed: {e}")
                logger.info("Attempting to recover by switching to CPU...")

                # Clean up MPS state
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

                # Update job status
                with get_db() as db:
                    job = db.query(TrainingJob).filter(TrainingJob.job_id == self.job_id).first()
                    if job:
                        job.current_stage = "Recovering (switching to CPU)"
                        db.commit()

                # Move model to CPU and retry
                self.device = "cpu"
                self.model = self.model.to("cpu")

                # Update training args for CPU
                training_args = TrainingArguments(
                    output_dir=str(self.output_dir),
                    num_train_epochs=self.epochs,
                    per_device_train_batch_size=self.batch_size,
                    per_device_eval_batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    warmup_steps=100,
                    logging_steps=10,
                    save_steps=500,
                    eval_steps=500,
                    eval_strategy="steps",
                    save_strategy="steps",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    fp16=False,
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                    gradient_checkpointing=False,
                    remove_unused_columns=False,
                    report_to=["none"],
                    use_cpu=True,
                )

                # Recreate trainer for CPU
                trainer = CustomTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.val_dataset,
                    data_collator=self._collate_fn,
                    callbacks=[ProgressCallback(self.job_id)],
                )

                logger.info("Retrying training on CPU...")
                trainer.train()
                print("[TRAINER] Training loop completed on CPU after MPS failure")
            else:
                # Re-raise non-MPS errors
                raise

        # Save final model
        final_model_path = self.output_dir / "final_model"
        logger.info(f"Saving final model to {final_model_path}")
        trainer.save_model(str(final_model_path))

        if self.use_lora:
            # Save LoRA adapters separately
            self.model.save_pretrained(str(self.output_dir / "lora_adapters"))

        logger.info("Training completed successfully")
