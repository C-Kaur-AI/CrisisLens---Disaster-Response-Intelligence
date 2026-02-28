"""
CrisisLens — Urgency Scorer
Scores the urgency level of crisis messages (CRITICAL / HIGH / MEDIUM / LOW).
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch

from config.settings import settings
from src.pipeline.shared_bart import get_shared_bart_pipeline

logger = logging.getLogger(__name__)


@dataclass
class UrgencyScore:
    """Result of urgency scoring."""
    level: str          # CRITICAL, HIGH, MEDIUM, LOW
    score: float        # 0.0 - 1.0, higher = more urgent
    keyword_boost: float  # Additional score from keyword matching


class UrgencyScorer:
    """
    Scores the urgency of crisis messages using a combination of:
    1. Zero-shot classification for semantic urgency understanding
    2. Keyword-based boosting for critical signal words
    
    Urgency Levels:
    - CRITICAL (0.85-1.0): Imminent threat to life, people trapped
    - HIGH (0.65-0.85): Significant damage, urgent resource needs
    - MEDIUM (0.40-0.65): Infrastructure damage, displacement
    - LOW (0.0-0.40): Situational updates, volunteer offers
    """

    # Critical urgency keywords with their boost weights
    CRITICAL_KEYWORDS = {
        # English
        'trapped': 0.25, 'dying': 0.30, 'drowning': 0.30, 'buried': 0.25,
        'collapsed': 0.20, 'fire': 0.15, 'bleeding': 0.25, 'unconscious': 0.25,
        'children': 0.10, 'baby': 0.15, 'pregnant': 0.15, 'elderly': 0.10,
        'help us': 0.20, 'please help': 0.15, 'sos': 0.25, 'emergency': 0.15,
        'urgent': 0.15, 'immediately': 0.10, 'critical': 0.15, 'life threatening': 0.25,
        # Common in other languages
        'socorro': 0.20, 'ayuda': 0.20, 'urgente': 0.15,  # Spanish
        'au secours': 0.20, 'aide': 0.15,  # French
        'مساعدة': 0.20, 'طوارئ': 0.15,  # Arabic
        'बचाओ': 0.20, 'मदद': 0.15,  # Hindi
    }

    HIGH_KEYWORDS = {
        'injured': 0.10, 'damage': 0.08, 'destroyed': 0.10, 'flood': 0.08,
        'earthquake': 0.08, 'casualties': 0.12, 'missing': 0.10,
        'evacuation': 0.10, 'rescue': 0.10, 'survivors': 0.08,
        'medical': 0.08, 'hospital': 0.08, 'ambulance': 0.10,
    }

    URGENCY_HYPOTHESES = [
        "This is an extremely urgent life-threatening emergency requiring immediate rescue",
        "This reports significant damage or urgent need for resources",
        "This is a moderate update about the disaster situation",
        "This is a general informational update or offer of help",
    ]

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None):
        self.model_name = model_name or settings.relevance_model

        if device is None:
            self._device = 0 if torch.cuda.is_available() else -1
        else:
            self._device = device

        self._classifier = None

    def load(self):
        """Use shared BART pipeline (single instance across relevance/type/urgency)."""
        if self._classifier is None:
            self._classifier = get_shared_bart_pipeline(device=self._device)

    def score(self, text: str) -> UrgencyScore:
        """
        Score the urgency of a crisis message.
        
        Args:
            text: Preprocessed and relevance-verified message text
            
        Returns:
            UrgencyScore with level, numeric score, and keyword boost
        """
        self.load()

        if not text or not text.strip():
            return UrgencyScore(level="LOW", score=0.0, keyword_boost=0.0)

        try:
            # Step 1: Zero-shot semantic scoring
            result = self._classifier(
                text,
                candidate_labels=self.URGENCY_HYPOTHESES,
                multi_label=False,
            )

            # Weighted score: critical=1.0, high=0.75, medium=0.5, low=0.25
            weights = [1.0, 0.75, 0.5, 0.25]
            semantic_score = 0.0
            for label, conf in zip(result["labels"], result["scores"]):
                idx = self.URGENCY_HYPOTHESES.index(label)
                semantic_score += conf * weights[idx]

            # Step 2: Keyword boost
            keyword_boost = self._compute_keyword_boost(text)

            # Step 3: Combine scores (capped at 1.0)
            final_score = min(1.0, semantic_score * 0.7 + keyword_boost * 0.3)

            # Step 4: Map to level
            level = self._score_to_level(final_score)

            return UrgencyScore(
                level=level,
                score=round(final_score, 4),
                keyword_boost=round(keyword_boost, 4),
            )

        except Exception as e:
            logger.error(f"Urgency scoring failed: {e}")
            return UrgencyScore(level="MEDIUM", score=0.5, keyword_boost=0.0)

    def _compute_keyword_boost(self, text: str) -> float:
        """Calculate urgency boost from keyword matches."""
        text_lower = text.lower()
        boost = 0.0

        for keyword, weight in self.CRITICAL_KEYWORDS.items():
            if keyword in text_lower:
                boost += weight

        for keyword, weight in self.HIGH_KEYWORDS.items():
            if keyword in text_lower:
                boost += weight

        # Cap at 1.0
        return min(1.0, boost)

    def _score_to_level(self, score: float) -> str:
        """Map numeric score to urgency level."""
        if score >= settings.urgency_critical_threshold:
            return "CRITICAL"
        elif score >= settings.urgency_high_threshold:
            return "HIGH"
        elif score >= settings.urgency_medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    def batch_score(self, texts: list[str]) -> list[UrgencyScore]:
        """Score urgency for a batch of texts."""
        return [self.score(t) for t in texts]
