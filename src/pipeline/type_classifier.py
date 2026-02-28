"""
CrisisLens — Multi-Label Event Type Classifier
Classifies crisis messages into one or more event categories.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from config.settings import settings
from src.pipeline.shared_bart import get_shared_bart_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TypeClassification:
    """Result of event type classification."""
    labels: list[str]           # e.g., ["RESCUE_REQUEST", "MEDICAL_EMERGENCY"]
    scores: dict[str, float]    # All label scores
    top_label: str              # Highest confidence label
    top_score: float            # Highest confidence score


class TypeClassifier:
    """
    Multi-label classifier for crisis event types.
    
    Uses zero-shot classification with multi_label=True so multiple
    categories can be assigned to a single message. For example, a message
    like "Family trapped on roof, grandmother needs insulin" would be
    classified as both RESCUE_REQUEST and MEDICAL_EMERGENCY.
    
    Labels:
    - RESCUE_REQUEST: People trapped, evacuation needed
    - INFRASTRUCTURE_DAMAGE: Roads, bridges, buildings destroyed
    - MEDICAL_EMERGENCY: Health crisis, medical supplies needed
    - SUPPLY_REQUEST: Food, water, shelter materials needed
    - CASUALTY_REPORT: Deaths, injuries reported
    - VOLUNTEER_OFFER: People offering help, donations
    - SITUATIONAL_UPDATE: General status, warnings, weather
    - DISPLACEMENT: People displaced, seeking shelter
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None):
        self.model_name = model_name or settings.relevance_model
        self.threshold = settings.type_confidence_threshold
        self.hypothesis_labels = settings.crisis_types
        self.short_labels = settings.crisis_type_short_labels

        if device is None:
            self._device = 0 if torch.cuda.is_available() else -1
        else:
            self._device = device

        self._classifier = None

    def load(self):
        """Use shared BART pipeline (single instance across relevance/type/urgency)."""
        if self._classifier is None:
            self._classifier = get_shared_bart_pipeline(device=self._device)

    def classify(self, text: str) -> TypeClassification:
        """
        Classify the crisis event type(s) of the message.
        
        Args:
            text: Preprocessed message text
            
        Returns:
            TypeClassification with labels above threshold
        """
        self.load()

        if not text or not text.strip():
            return TypeClassification(
                labels=[],
                scores={},
                top_label="UNKNOWN",
                top_score=0.0,
            )

        try:
            result = self._classifier(
                text,
                candidate_labels=self.hypothesis_labels,
                multi_label=True,
            )

            # Map hypothesis labels to short labels and filter by threshold
            label_scores = {}
            active_labels = []

            for hyp_label, score in zip(result["labels"], result["scores"]):
                idx = self.hypothesis_labels.index(hyp_label)
                short = self.short_labels[idx]
                label_scores[short] = round(score, 4)

                if score >= self.threshold:
                    active_labels.append(short)

            # Determine top label
            if active_labels:
                top_label = active_labels[0]
                top_score = label_scores[top_label]
            else:
                # If nothing passes threshold, take the best one anyway
                top_label = max(label_scores, key=label_scores.get)
                top_score = label_scores[top_label]
                active_labels = [top_label]

            return TypeClassification(
                labels=active_labels,
                scores=label_scores,
                top_label=top_label,
                top_score=round(top_score, 4),
            )

        except Exception as e:
            logger.error(f"Type classification failed: {e}")
            return TypeClassification(
                labels=["UNKNOWN"],
                scores={},
                top_label="UNKNOWN",
                top_score=0.0,
            )

    def batch_classify(self, texts: list[str]) -> list[TypeClassification]:
        """Classify a batch of texts."""
        return [self.classify(t) for t in texts]
