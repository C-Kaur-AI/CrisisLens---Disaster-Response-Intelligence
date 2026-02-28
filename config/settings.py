"""
CrisisLens Configuration
Central settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
ROOT_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    # ─── Project ───
    app_name: str = "CrisisLens"
    app_version: str = "0.1.0"
    debug: bool = False

    # ─── Model Names (HuggingFace) ───
    relevance_model: str = "facebook/bart-large-mnli"
    relevance_finetuned_path: str = str(ROOT_DIR / "models" / "finetuned")
    ner_model: str = "Davlan/xlm-roberta-base-ner-hrl"
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fasttext_model_path: str = str(ROOT_DIR / "models" / "lid.176.bin")

    # ─── Classification Labels ───
    crisis_types: list[str] = Field(default=[
        "rescue and evacuation request",
        "infrastructure and utility damage",
        "medical emergency and health crisis",
        "food water and supply request",
        "casualty and death report",
        "volunteer and donation offer",
        "situational update and warning",
        "displaced people and shelter need",
    ])

    crisis_type_short_labels: list[str] = Field(default=[
        "RESCUE_REQUEST",
        "INFRASTRUCTURE_DAMAGE",
        "MEDICAL_EMERGENCY",
        "SUPPLY_REQUEST",
        "CASUALTY_REPORT",
        "VOLUNTEER_OFFER",
        "SITUATIONAL_UPDATE",
        "DISPLACEMENT",
    ])

    urgency_levels: list[str] = Field(default=[
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ])

    # ─── Thresholds ───
    relevance_threshold: float = 0.65
    dedup_similarity_threshold: float = 0.85
    type_confidence_threshold: float = 0.40
    urgency_critical_threshold: float = 0.85
    urgency_high_threshold: float = 0.65
    urgency_medium_threshold: float = 0.40

    # ─── Geocoding ───
    geocoding_user_agent: str = "crisislens-app"
    geocoding_timeout: int = 10

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    # CORS: use specific origins in production; "*" is for demo/development only
    cors_origins: str = "*"

    # ─── Dashboard ───
    dashboard_port: int = 8501

    # ─── Deduplication ───
    dedup_window_size: int = 500  # Number of recent messages to check against

    class Config:
        env_file = str(ROOT_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()
