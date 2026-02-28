"""
CrisisLens — Shared BART Model
Single zero-shot classification pipeline shared by relevance, type, and urgency classifiers.
Avoids 3× memory usage (~5GB → ~1.7GB) from loading BART-large-MNLI multiple times.
"""

import logging
from typing import Optional, Any

import torch
from transformers import pipeline as hf_pipeline

from config.settings import settings

logger = logging.getLogger(__name__)

_shared_pipeline: Optional[Any] = None


def get_shared_bart_pipeline(device: Optional[int] = None):
    """Load and return the shared BART zero-shot pipeline. Singleton pattern."""
    global _shared_pipeline
    if _shared_pipeline is None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading shared BART model: {settings.relevance_model}")
        _shared_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=settings.relevance_model,
            device=device,
        )
        logger.info("Shared BART model loaded (reused by relevance, type, urgency)")
    return _shared_pipeline


def reset_shared_bart():
    """Clear the shared pipeline (for testing)."""
    global _shared_pipeline
    _shared_pipeline = None
