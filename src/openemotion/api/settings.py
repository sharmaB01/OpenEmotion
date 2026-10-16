from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from openemotion.config import DEFAULT_MODEL_PATH


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(min_length=1)
    model_path: Path = DEFAULT_MODEL_PATH
    device: str = "cpu"
    # Reported in every result (contract v1). Empty = the checkpoint file name without extension.
    model_version: str = Field(default="", max_length=64)
    cache_path: Path = Path("outputs/ai-jobs.sqlite3")
    download_hosts: list[str] = ["api.cloudinary.com", "res.cloudinary.com"]
    max_video_bytes: int = Field(default=90 * 1024 * 1024, ge=1024)
    max_frame_bytes: int = Field(default=512 * 1024, ge=128)
    max_video_seconds: float = Field(default=7200, gt=0)
    processing_timeout_seconds: float = Field(default=110, gt=0)
    download_timeout_seconds: float = Field(default=30, gt=0)
    low_confidence_threshold: float = Field(default=0.4, ge=0, le=1)
