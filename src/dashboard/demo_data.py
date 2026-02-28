"""
CrisisLens — Pre-computed demo results for instant load.
Used when HR/demo users want to see outputs without waiting for model load.
Results are representative of pipeline output (not live model inference).
"""

from dataclasses import dataclass, field
from typing import Optional

# Simple structure for demo display (matches CrisisAnalysisResult fields we render)
@dataclass
class DemoLanguage:
    lang_code: str
    confidence: float
    method: str = "demo"

@dataclass
class DemoLocation:
    text: str
    label: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    display_name: Optional[str] = None
    country: Optional[str] = None

@dataclass
class DemoResult:
    """Demo result matching display format of CrisisAnalysisResult."""
    original_text: str
    cleaned_text: str
    language: DemoLanguage
    is_relevant: bool
    relevance_confidence: float
    event_types: list[str] = field(default_factory=list)
    urgency_level: str = "LOW"
    urgency_score: float = 0.0
    locations: list[DemoLocation] = field(default_factory=list)
    is_duplicate: bool = False
    cluster_id: Optional[str] = None
    processing_time_ms: float = 0.0


def get_demo_result_for_text(text: str, sample_key: str) -> DemoResult:
    """
    Return pre-computed demo result for a sample.
    Keys by sample_key (e.g. "🆘 Rescue (English)") for consistent mapping.
    """
    # Map sample keys to expected outputs (model would infer these from the text)
    DEMO_MAP = {
        "🆘 Rescue (English)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("en", 0.98), is_relevant=True, relevance_confidence=0.94,
            event_types=["RESCUE_REQUEST"], urgency_level="CRITICAL", urgency_score=0.91,
            locations=[DemoLocation("Hatay", "LOC", 36.2, 36.16, "Hatay, Turkey", "TR")],
            processing_time_ms=0.0,
        ),
        "🏥 Medical (Spanish)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("es", 0.99), is_relevant=True, relevance_confidence=0.96,
            event_types=["MEDICAL_EMERGENCY"], urgency_level="CRITICAL", urgency_score=0.89,
            locations=[DemoLocation("San Pedro", "LOC", None, None, None, None)],
            processing_time_ms=0.0,
        ),
        "🏚️ Damage (Hindi)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("hi", 0.97), is_relevant=True, relevance_confidence=0.92,
            event_types=["INFRASTRUCTURE_DAMAGE"], urgency_level="HIGH", urgency_score=0.78,
            locations=[DemoLocation("दिल्ली", "LOC", 28.61, 77.21, "Delhi, India", "IN")],
            processing_time_ms=0.0,
        ),
        "📢 Update (French)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("fr", 0.99), is_relevant=True, relevance_confidence=0.88,
            event_types=["SITUATIONAL_UPDATE"], urgency_level="MEDIUM", urgency_score=0.65,
            locations=[DemoLocation("Lyon", "LOC", 45.76, 4.84, "Lyon, France", "FR")],
            processing_time_ms=0.0,
        ),
        "🍽️ Supply (Arabic)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("ar", 0.98), is_relevant=True, relevance_confidence=0.93,
            event_types=["SUPPLY_REQUEST"], urgency_level="HIGH", urgency_score=0.82,
            locations=[DemoLocation("حلب", "LOC", 36.2, 37.16, "Aleppo, Syria", "SY")],
            processing_time_ms=0.0,
        ),
        "🚑 Medical (German)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("de", 0.99), is_relevant=True, relevance_confidence=0.95,
            event_types=["MEDICAL_EMERGENCY", "CASUALTY_REPORT"], urgency_level="CRITICAL", urgency_score=0.90,
            locations=[DemoLocation("Köln", "LOC", 50.94, 6.96, "Cologne, Germany", "DE")],
            processing_time_ms=0.0,
        ),
        "🏠 Displacement (Punjabi)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("pa", 0.95), is_relevant=True, relevance_confidence=0.89,
            event_types=["DISPLACEMENT", "SUPPLY_REQUEST"], urgency_level="HIGH", urgency_score=0.75,
            locations=[DemoLocation("Amritsar", "LOC", 31.63, 74.87, "Amritsar, India", "IN")],
            processing_time_ms=0.0,
        ),
        "🌊 Flood (Gujarati)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("gu", 0.96), is_relevant=True, relevance_confidence=0.91,
            event_types=["INFRASTRUCTURE_DAMAGE", "RESCUE_REQUEST"], urgency_level="HIGH", urgency_score=0.80,
            locations=[DemoLocation("Ahmedabad", "LOC", 23.02, 72.57, "Ahmedabad, India", "IN")],
            processing_time_ms=0.0,
        ),
        "🔥 Fire (Polish)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("pl", 0.99), is_relevant=True, relevance_confidence=0.94,
            event_types=["RESCUE_REQUEST", "INFRASTRUCTURE_DAMAGE"], urgency_level="CRITICAL", urgency_score=0.88,
            locations=[DemoLocation("Warszawa", "LOC", 52.23, 21.01, "Warsaw, Poland", "PL")],
            processing_time_ms=0.0,
        ),
        "❌ Not Crisis": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("en", 0.99), is_relevant=False, relevance_confidence=0.12,
            event_types=[], urgency_level="LOW", urgency_score=0.0,
            locations=[], processing_time_ms=0.0,
        ),
        "🔍 Implicit rescue (EN)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("en", 0.97), is_relevant=True, relevance_confidence=0.89,
            event_types=["RESCUE_REQUEST"], urgency_level="CRITICAL", urgency_score=0.85,
            locations=[], processing_time_ms=0.0,
        ),
        "🔍 Implicit medical (EN)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("en", 0.98), is_relevant=True, relevance_confidence=0.91,
            event_types=["MEDICAL_EMERGENCY"], urgency_level="CRITICAL", urgency_score=0.88,
            locations=[], processing_time_ms=0.0,
        ),
        "🔍 Implicit damage (EN)": DemoResult(
            original_text=text, cleaned_text=text,
            language=DemoLanguage("en", 0.99), is_relevant=True, relevance_confidence=0.93,
            event_types=["INFRASTRUCTURE_DAMAGE"], urgency_level="HIGH", urgency_score=0.82,
            locations=[DemoLocation("Port-au-Prince", "LOC", 18.59, -72.31, "Port-au-Prince, Haiti", "HT")],
            processing_time_ms=0.0,
        ),
    }
    # Default for any sample not in map
    default = DemoResult(
        original_text=text, cleaned_text=text,
        language=DemoLanguage("und", 0.8), is_relevant=True, relevance_confidence=0.85,
        event_types=["SITUATIONAL_UPDATE"], urgency_level="MEDIUM", urgency_score=0.6,
        locations=[], processing_time_ms=0.0,
    )
    return DEMO_MAP.get(sample_key, default)
