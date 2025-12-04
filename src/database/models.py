"""
Database models for training data management
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Model(Base):
    """Available VLM models for training"""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Model specifications
    architecture = Column(String(100), nullable=True)  # e.g., "llava", "tinyllava", "instructblip"
    base_model = Column(String(255), nullable=True)  # e.g., "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B"
    vision_model = Column(String(255), nullable=True)  # e.g., "openai/clip-vit-large-patch14"
    parameters = Column(String(50), nullable=True)  # e.g., "1.1B", "7B"

    # Capabilities
    supports_lora = Column(Boolean, default=True)
    supports_full_finetune = Column(Boolean, default=True)
    recommended_for_cad = Column(Boolean, default=False)

    # Requirements
    min_vram_gb = Column(Integer, nullable=True)
    min_ram_gb = Column(Integer, nullable=True)

    # Status
    available = Column(Boolean, default=True)
    verified = Column(Boolean, default=False)

    # Metadata
    huggingface_id = Column(String(512), nullable=True)
    config_file = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Image processor configuration
    image_processor_id = Column(String(255), nullable=True)  # HuggingFace ID for image processor
    image_size = Column(Integer, default=224)  # Expected image size (224, 336, 384, etc.)

    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="model")

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "architecture": self.architecture,
            "base_model": self.base_model,
            "vision_model": self.vision_model,
            "parameters": self.parameters,
            "supports_lora": self.supports_lora,
            "supports_full_finetune": self.supports_full_finetune,
            "recommended_for_cad": self.recommended_for_cad,
            "min_vram_gb": self.min_vram_gb,
            "min_ram_gb": self.min_ram_gb,
            "available": self.available,
            "verified": self.verified,
            "huggingface_id": self.huggingface_id,
            "image_processor_id": self.image_processor_id,
            "image_size": self.image_size,
            "created": self.created_at.isoformat() if self.created_at else None,
            "updated": self.updated_at.isoformat() if self.updated_at else None,
        }


class Dataset(Base):
    """Training dataset metadata"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    path = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)

    # Dataset statistics
    total_models = Column(Integer, default=0)
    train_samples = Column(Integer, default=0)
    val_samples = Column(Integer, default=0)
    images = Column(Integer, default=0)
    conversations = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)

    # Status and timestamps
    status = Column(String(50), default="ready")  # ready, processing, error
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="dataset", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "models": self.total_models,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "images": self.images,
            "conversations": self.conversations,
            "size_bytes": self.size_bytes,
            "size": self._format_size(self.size_bytes),
            "status": self.status,
            "created": self.created_at.isoformat() if self.created_at else None,
            "updated": self.updated_at.isoformat() if self.updated_at else None,
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


class TrainingJob(Base):
    """Training job tracking"""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)

    # Training configuration
    epochs = Column(Integer, default=3)
    batch_size = Column(Integer, default=2)
    learning_rate = Column(Float, nullable=True)
    use_lora = Column(Boolean, default=True)
    device = Column(String(20), default="auto")  # "auto", "mps", "cpu", "cuda"

    # Job status
    status = Column(String(50), default="pending")  # pending, running, completed, failed, halted
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    current_stage = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    is_loading_model = Column(Boolean, default=False)  # True while downloading/loading model

    # Current training metrics (updated during training)
    current_loss = Column(Float, nullable=True)
    current_epoch = Column(Integer, nullable=True)
    current_step = Column(Integer, nullable=True)
    total_steps = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_jobs")
    results = relationship("TrainingResult", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert to dictionary for API response"""
        # Build metrics object if we have training data
        metrics = None
        if self.current_loss is not None or self.current_epoch is not None:
            metrics = {
                "epoch": self.current_epoch,
                "step": self.current_step,
                "total_steps": self.total_steps,
                "loss": self.current_loss,
                "learning_rate": self.learning_rate,
            }

        return {
            "id": self.id,
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else None,
            "model_id": self.model_id,
            "model_name": self.model.name if self.model else None,
            "model_display_name": self.model.display_name if self.model else None,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "use_lora": self.use_lora,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "is_loading_model": self.is_loading_model or False,  # True while loading/downloading model
            "loss": self.current_loss,  # Current training loss
            "metrics": metrics,  # Detailed metrics object
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TrainingResult(Base):
    """Training results and metrics"""
    __tablename__ = "training_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    # Training metrics
    epoch = Column(Integer, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    # Performance metrics
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    samples_per_second = Column(Float, nullable=True)

    # Model artifacts
    checkpoint_path = Column(String(512), nullable=True)
    model_size_bytes = Column(Integer, nullable=True)

    # Metadata
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("TrainingJob", back_populates="results")

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "samples_per_second": self.samples_per_second,
            "checkpoint_path": self.checkpoint_path,
            "model_size_bytes": self.model_size_bytes,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
