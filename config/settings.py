"""
Application settings and configuration
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration"""

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    input_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "input")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "output")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 7001
    api_workers: int = 4
    cors_origins: list = ["*"]

    # Processing
    default_dpi: int = 300
    use_gpu: bool = True
    max_image_size: int = 8192  # Max dimension in pixels

    # Model defaults
    default_wall_height: float = 2.8  # meters
    default_floor_thickness: float = 0.3
    default_door_height: float = 2.1
    default_window_height: float = 1.2
    default_window_sill: float = 0.9

    # Detection
    detection_model: str = "yolov8"
    detection_confidence: float = 0.5

    # Redis (for task queue)
    redis_url: str = "redis://localhost:6379/0"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    class Config:
        env_prefix = "RECON_"
        env_file = ".env"


settings = Settings()
