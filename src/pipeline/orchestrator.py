"""
CrisisLens — Pipeline Orchestrator
End-to-end coordinator that chains all NLP components together.
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

from src.pipeline.preprocessor import TextPreprocessor, PreprocessedMessage
from src.pipeline.language_detector import LanguageDetector, LanguageDetection
from src.pipeline.relevance_classifier import RelevanceClassifier, RelevanceResult
from src.pipeline.type_classifier import TypeClassifier, TypeClassification
from src.pipeline.urgency_scorer import UrgencyScorer, UrgencyScore
from src.pipeline.geo_ner import GeoNER, LocationEntity
from src.pipeline.geocoder import Geocoder, GeocodedLocation
from src.pipeline.deduplicator import Deduplicator, DeduplicationResult

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GeocodedEntity:
    """A location entity with geocoding results."""
    text: str
    label: str
    confidence: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    display_name: Optional[str] = None
    country: Optional[str] = None


@dataclass
class CrisisAnalysisResult:
    """Complete analysis result for a single message."""
    # Input
    original_text: str
    cleaned_text: str
    
    # Language
    language: LanguageDetection
    
    # Relevance
    is_relevant: bool
    relevance_confidence: float
    
    # Classification (only if relevant)
    event_types: list[str] = field(default_factory=list)
    type_scores: dict[str, float] = field(default_factory=dict)
    
    # Urgency (only if relevant)
    urgency_level: str = "LOW"
    urgency_score: float = 0.0
    
    # Locations (only if relevant)
    locations: list[GeocodedEntity] = field(default_factory=list)
    
    # Deduplication
    is_duplicate: bool = False
    cluster_id: Optional[str] = None
    
    # Metadata
    processing_time_ms: float = 0.0
    pipeline_version: str = settings.app_version
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "language": {
                "code": self.language.lang_code,
                "confidence": self.language.confidence,
                "method": self.language.method,
            },
            "is_relevant": self.is_relevant,
            "relevance_confidence": self.relevance_confidence,
            "event_types": self.event_types,
            "type_scores": self.type_scores,
            "urgency": {
                "level": self.urgency_level,
                "score": self.urgency_score,
            },
            "locations": [
                {
                    "text": loc.text,
                    "label": loc.label,
                    "confidence": loc.confidence,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "display_name": loc.display_name,
                    "country": loc.country,
                }
                for loc in self.locations
            ],
            "deduplication": {
                "is_duplicate": self.is_duplicate,
                "cluster_id": self.cluster_id,
            },
            "processing_time_ms": self.processing_time_ms,
            "pipeline_version": self.pipeline_version,
        }


class CrisisLensPipeline:
    """
    End-to-end CrisisLens NLP Pipeline.
    
    Processing flow:
    1. Preprocess raw text (clean URLs, emojis, mentions)
    2. Detect language
    3. Classify relevance (is this disaster-related?)
    4. If relevant:
       a. Classify event types (rescue, damage, medical, etc.)
       b. Score urgency (CRITICAL → LOW)
       c. Extract location entities (GeoNER)
       d. Geocode extracted locations
    5. Check for semantic duplicates
    6. Return complete CrisisAnalysisResult
    """

    def __init__(self):
        # Initialize all pipeline components
        self.preprocessor = TextPreprocessor()
        self.language_detector = LanguageDetector(
            fasttext_model_path=settings.fasttext_model_path
        )
        self.relevance_classifier = RelevanceClassifier()
        self.type_classifier = TypeClassifier()
        self.urgency_scorer = UrgencyScorer()
        self.geo_ner = GeoNER()
        self.geocoder = Geocoder()
        self.deduplicator = Deduplicator()

        self._loaded = False
        self._stats = {
            "total_processed": 0,
            "total_relevant": 0,
            "total_critical": 0,
            "total_duplicates": 0,
        }

    def load_models(self):
        """Pre-load all ML models. Call this during startup to avoid first-request latency."""
        logger.info("Pre-loading all CrisisLens models...")
        start = time.time()

        self.relevance_classifier.load()
        self.type_classifier.load()
        self.urgency_scorer.load()
        self.geo_ner.load()
        self.deduplicator.load()

        elapsed = time.time() - start
        logger.info(f"All models loaded in {elapsed:.1f}s")
        self._loaded = True

    def analyze(self, text: str, skip_dedup: bool = False) -> CrisisAnalysisResult:
        """
        Run the full analysis pipeline on a single message.
        
        Args:
            text: Raw message text (any language)
            skip_dedup: If True, skip deduplication check
            
        Returns:
            CrisisAnalysisResult with all analysis fields populated
        """
        start_time = time.time()

        # ── Step 1: Preprocess ──
        preprocessed = self.preprocessor.preprocess(text)
        clean_text = preprocessed.cleaned_text

        # ── Step 2: Language Detection ──
        language = self.language_detector.detect(clean_text)

        # ── Step 3: Relevance Classification ──
        relevance = self.relevance_classifier.classify(clean_text)

        # Build the base result
        result = CrisisAnalysisResult(
            original_text=preprocessed.original_text,
            cleaned_text=clean_text,
            language=language,
            is_relevant=relevance.is_relevant,
            relevance_confidence=relevance.confidence,
        )

        # ── Steps 4a-4d: Only process if relevant ──
        if relevance.is_relevant:
            # Type Classification
            type_result = self.type_classifier.classify(clean_text)
            result.event_types = type_result.labels
            result.type_scores = type_result.scores

            # Urgency Scoring
            urgency = self.urgency_scorer.score(clean_text)
            result.urgency_level = urgency.level
            result.urgency_score = urgency.score

            # GeoNER — Extract location entities
            location_entities = self.geo_ner.extract(clean_text)

            # Geocode each location entity
            geocoded_locations = []
            for entity in location_entities:
                geo = self.geocoder.geocode(entity.text)
                geocoded_locations.append(GeocodedEntity(
                    text=entity.text,
                    label=entity.label,
                    confidence=entity.confidence,
                    latitude=geo.latitude if geo else None,
                    longitude=geo.longitude if geo else None,
                    display_name=geo.display_name if geo else None,
                    country=geo.country if geo else None,
                ))
            result.locations = geocoded_locations

            # Update stats
            self._stats["total_relevant"] += 1
            if urgency.level == "CRITICAL":
                self._stats["total_critical"] += 1

        # ── Step 5: Deduplication ──
        if not skip_dedup:
            dedup = self.deduplicator.check(clean_text)
            result.is_duplicate = dedup.is_duplicate
            result.cluster_id = dedup.cluster_id
            if dedup.is_duplicate:
                self._stats["total_duplicates"] += 1

        # ── Finalize ──
        elapsed_ms = (time.time() - start_time) * 1000
        result.processing_time_ms = round(elapsed_ms, 2)

        self._stats["total_processed"] += 1

        logger.info(
            f"Analyzed message: relevant={result.is_relevant}, "
            f"types={result.event_types}, urgency={result.urgency_level}, "
            f"locations={[l.text for l in result.locations]}, "
            f"duplicate={result.is_duplicate}, "
            f"time={result.processing_time_ms}ms"
        )

        return result

    def analyze_batch(self, texts: list[str], skip_dedup: bool = False) -> list[CrisisAnalysisResult]:
        """Analyze a batch of messages."""
        return [self.analyze(t, skip_dedup=skip_dedup) for t in texts]

    @property
    def stats(self) -> dict:
        """Pipeline processing statistics."""
        return {
            **self._stats,
            "dedup_window_size": self.deduplicator.window_count,
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self._stats = {
            "total_processed": 0,
            "total_relevant": 0,
            "total_critical": 0,
            "total_duplicates": 0,
        }
        self.deduplicator.reset()
