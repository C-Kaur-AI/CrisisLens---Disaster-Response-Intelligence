"""
CrisisLens — Relevance Classifier
Determines whether a message is related to an active disaster/crisis.
Uses fine-tuned XLM-RoBERTa by default; falls back to BART zero-shot if unavailable.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from config.settings import settings
from src.pipeline.shared_bart import get_shared_bart_pipeline

logger = logging.getLogger(__name__)


@dataclass
class RelevanceResult:
    """Result of relevance classification."""
    is_relevant: bool
    confidence: float
    label: str  # "RELEVANT" or "NOT_RELEVANT"


class RelevanceClassifier:
    """
    Binary classifier to determine if a message is disaster/crisis-related.

    Default: Fine-tuned XLM-RoBERTa (HumAID, ~0.96 F1) if models/finetuned exists.
    Fallback: BART-large-MNLI zero-shot (~0.76–0.82 F1).
    """

    # Candidate labels for BART zero-shot (used only when fallback)
    HYPOTHESIS_LABELS = [
        "natural disaster emergency crisis",
        "normal everyday activity",
    ]

    def __init__(self, model_path: Optional[str] = None, device: Optional[int] = None):
        self.finetuned_path = Path(model_path or settings.relevance_finetuned_path)
        self.bart_model = settings.relevance_model
        self.threshold = settings.relevance_threshold

        # Auto-detect device
        if device is None:
            self._device = 0 if torch.cuda.is_available() else -1
        else:
            self._device = device

        self._finetuned_model = None
        self._finetuned_tokenizer = None
        self._bart_classifier = None
        self._backend: Optional[str] = None  # "finetuned" or "bart"

    def _finetuned_available(self) -> bool:
        """Check if fine-tuned model directory exists and has required files."""
        if not self.finetuned_path.exists() or not self.finetuned_path.is_dir():
            return False
        # Need config and at least one of pytorch_model.bin or model.safetensors
        has_config = (self.finetuned_path / "config.json").exists()
        has_weights = (
            (self.finetuned_path / "pytorch_model.bin").exists()
            or (self.finetuned_path / "model.safetensors").exists()
        )
        return has_config and has_weights

    def load(self):
        """Load fine-tuned model if available; otherwise use shared BART."""
        if self._backend is not None:
            return

        if self._finetuned_available():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                self._finetuned_tokenizer = AutoTokenizer.from_pretrained(str(self.finetuned_path))
                self._finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.finetuned_path)
                )
                self._finetuned_model.eval()
                if self._device >= 0:
                    self._finetuned_model = self._finetuned_model.cuda(self._device)
                self._backend = "finetuned"
                logger.info("Relevance: using fine-tuned XLM-RoBERTa model")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}. Falling back to BART.")
                self._backend = "bart"
        else:
            logger.info("Fine-tuned model not found. Relevance: using BART zero-shot")
            self._backend = "bart"

        if self._backend == "bart":
            self._bart_classifier = get_shared_bart_pipeline(device=self._device)

    def _classify_finetuned(self, text: str) -> RelevanceResult:
        """Classify using fine-tuned XLM-RoBERTa. Class 1 = relevant."""
        enc = self._finetuned_tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        if self._device >= 0:
            enc = {k: v.cuda(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._finetuned_model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(logits.argmax(dim=1).item())
        confidence = float(probs[1])  # prob of class 1 (relevant)

        is_relevant = pred == 1
        return RelevanceResult(
            is_relevant=is_relevant,
            confidence=round(confidence, 4),
            label="RELEVANT" if is_relevant else "NOT_RELEVANT",
        )

    def _classify_bart(self, text: str) -> RelevanceResult:
        """Classify using BART zero-shot."""
        result = self._bart_classifier(
            text,
            candidate_labels=self.HYPOTHESIS_LABELS,
            multi_label=False,
        )
        crisis_idx = result["labels"].index(self.HYPOTHESIS_LABELS[0])
        crisis_score = result["scores"][crisis_idx]
        is_relevant = crisis_score >= self.threshold
        return RelevanceResult(
            is_relevant=is_relevant,
            confidence=round(crisis_score, 4),
            label="RELEVANT" if is_relevant else "NOT_RELEVANT",
        )

    def classify(self, text: str) -> RelevanceResult:
        """
        Classify whether the text is related to a disaster/crisis.
        """
        self.load()

        if not text or not text.strip():
            return RelevanceResult(
                is_relevant=False,
                confidence=0.0,
                label="NOT_RELEVANT",
            )

        try:
            if self._backend == "finetuned":
                return self._classify_finetuned(text)
            return self._classify_bart(text)
        except Exception as e:
            logger.error(f"Relevance classification failed: {e}")
            if self._backend == "finetuned":
                logger.info("Falling back to BART for this request")
                self._bart_classifier = get_shared_bart_pipeline(device=self._device)
                return self._classify_bart(text)
            return RelevanceResult(
                is_relevant=False,
                confidence=0.0,
                label="NOT_RELEVANT",
            )

    def batch_classify(self, texts: list[str]) -> list[RelevanceResult]:
        """Classify a batch of texts."""
        self.load()

        if not texts:
            return []

        if not all(text and str(text).strip() for text in texts):
            return [self.classify(t) for t in texts]

        try:
            if self._backend == "finetuned":
                enc = self._finetuned_tokenizer(
                    texts,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                )
                if self._device >= 0:
                    enc = {k: v.cuda(self._device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self._finetuned_model(**enc).logits
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                return [
                    RelevanceResult(
                        is_relevant=int(p) == 1,
                        confidence=round(float(probs[i, 1]), 4),
                        label="RELEVANT" if int(p) == 1 else "NOT_RELEVANT",
                    )
                    for i, p in enumerate(preds)
                ]
        except Exception as e:
            logger.warning(f"Batch fine-tuned failed: {e}, falling back to per-sample")
        return [self.classify(t) for t in texts]
