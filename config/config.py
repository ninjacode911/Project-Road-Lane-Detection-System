"""
Configuration management for Lane Detection System.
Uses pydantic-settings for type-safe configuration from environment variables.
"""

from pathlib import Path
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model Configuration
    model_type: Literal["yolo", "segformer", "ensemble", "traditional"] = "yolo"
    model_path: Path = Path("models/weights/yolov8n-seg.pt")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # Processing Configuration
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    batch_size: int = Field(default=1, ge=1)
    image_size: int = Field(default=640, ge=32)
    use_fp16: bool = True
    num_workers: int = Field(default=4, ge=1)

    # Video Processing
    video_fps: int = Field(default=30, ge=1)
    skip_frames: int = Field(default=0, ge=0)
    max_duration: int = Field(default=0, ge=0)  # 0 = no limit

    # Web Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_reload: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path = Path("results/logs/app.log")
    log_to_console: bool = True

    # Paths
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    models_dir: Path = Path("models/weights")
    upload_dir: Path = Path("data/uploads")
    temp_dir: Path = Path("data/temp")

    # Advanced Options
    enable_temporal_smoothing: bool = True
    kalman_filter_enabled: bool = True
    polynomial_order: int = Field(default=2, ge=1, le=3)
    enable_multi_lane: bool = True

    # Performance Monitoring
    enable_profiling: bool = False
    benchmark_mode: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.results_dir,
            self.models_dir,
            self.upload_dir,
            self.temp_dir,
            self.results_dir / "images",
            self.results_dir / "videos",
            self.results_dir / "masks",
            self.results_dir / "logs",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
