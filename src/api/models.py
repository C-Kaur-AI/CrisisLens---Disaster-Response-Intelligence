"""
CrisisLens — API Pydantic Models
Request and response schemas for the FastAPI endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Request Models ───

class AnalyzeRequest(BaseModel):
    """Request to analyze a single message."""
    text: str = Field(..., description="Raw message text to analyze", min_length=1, max_length=5000)
    lang: Optional[str] = Field(None, description="Override language code (ISO 639-1)")
    skip_dedup: bool = Field(False, description="Skip deduplication check")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "URGENT: Family of 4 trapped on 2nd floor in Hatay district, water rising fast. Please send rescue team! #TurkeyEarthquake",
                    "lang": None,
                    "skip_dedup": False,
                }
            ]
        }
    }


class BatchAnalyzeRequest(BaseModel):
    """Request to analyze multiple messages."""
    texts: list[str] = Field(..., description="List of raw message texts", min_length=1, max_length=100)
    skip_dedup: bool = Field(False, description="Skip deduplication check")


# ─── Response Models ───

class LanguageInfo(BaseModel):
    """Language detection result."""
    code: str = Field(..., description="ISO language code")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    method: str = Field(..., description="Detection method used")


class UrgencyInfo(BaseModel):
    """Urgency scoring result."""
    level: str = Field(..., description="Urgency level: CRITICAL, HIGH, MEDIUM, LOW")
    score: float = Field(..., description="Numeric urgency score (0-1)")


class LocationInfo(BaseModel):
    """Geocoded location entity."""
    text: str = Field(..., description="Location text as found in message")
    label: str = Field(..., description="Entity type: LOC, FACILITY")
    confidence: float = Field(..., description="NER model confidence")
    latitude: Optional[float] = Field(None, description="Geocoded latitude")
    longitude: Optional[float] = Field(None, description="Geocoded longitude")
    display_name: Optional[str] = Field(None, description="Full display name from geocoder")
    country: Optional[str] = Field(None, description="Country code")


class DeduplicationInfo(BaseModel):
    """Deduplication result."""
    is_duplicate: bool = Field(..., description="Whether this is a duplicate message")
    cluster_id: Optional[str] = Field(None, description="Deduplication cluster ID")


class AnalyzeResponse(BaseModel):
    """Complete analysis response for a single message."""
    original_text: str
    cleaned_text: str
    language: LanguageInfo
    is_relevant: bool
    relevance_confidence: float
    event_types: list[str] = Field(default_factory=list)
    type_scores: dict[str, float] = Field(default_factory=dict)
    urgency: UrgencyInfo
    locations: list[LocationInfo] = Field(default_factory=list)
    deduplication: DeduplicationInfo
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "original_text": "Family trapped on 2nd floor Hatay district, water rising!",
                    "cleaned_text": "Family trapped on 2nd floor Hatay district, water rising!",
                    "language": {"code": "en", "confidence": 0.99, "method": "langdetect"},
                    "is_relevant": True,
                    "relevance_confidence": 0.95,
                    "event_types": ["RESCUE_REQUEST"],
                    "type_scores": {"RESCUE_REQUEST": 0.92, "SITUATIONAL_UPDATE": 0.31},
                    "urgency": {"level": "CRITICAL", "score": 0.94},
                    "locations": [
                        {
                            "text": "Hatay", "label": "LOC", "confidence": 0.96,
                            "latitude": 36.4, "longitude": 36.34,
                            "display_name": "Hatay, Turkey", "country": "TR"
                        }
                    ],
                    "deduplication": {"is_duplicate": False, "cluster_id": "cluster_000001"},
                    "processing_time_ms": 423.5,
                }
            ]
        }
    }


class BatchAnalyzeResponse(BaseModel):
    """Response for batch analysis."""
    results: list[AnalyzeResponse]
    total_processed: int
    total_relevant: int
    total_critical: int


class StatsResponse(BaseModel):
    """Pipeline statistics response."""
    total_processed: int
    total_relevant: int
    total_critical: int
    total_duplicates: int
    dedup_window_size: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
