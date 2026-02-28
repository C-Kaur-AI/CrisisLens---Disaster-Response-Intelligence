"""
CrisisLens — API Routes
FastAPI endpoint definitions for crisis message analysis.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    StatsResponse,
    HealthResponse,
    LanguageInfo,
    UrgencyInfo,
    LocationInfo,
    DeduplicationInfo,
)
from src.pipeline.orchestrator import CrisisLensPipeline, CrisisAnalysisResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Global pipeline instance (initialized in main.py lifespan)
pipeline: Optional[CrisisLensPipeline] = None


def get_pipeline() -> CrisisLensPipeline:
    """Get the global pipeline instance."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Server is starting up.",
        )
    return pipeline


def result_to_response(result: CrisisAnalysisResult) -> AnalyzeResponse:
    """Convert a CrisisAnalysisResult to an AnalyzeResponse."""
    return AnalyzeResponse(
        original_text=result.original_text,
        cleaned_text=result.cleaned_text,
        language=LanguageInfo(
            code=result.language.lang_code,
            confidence=result.language.confidence,
            method=result.language.method,
        ),
        is_relevant=result.is_relevant,
        relevance_confidence=result.relevance_confidence,
        event_types=result.event_types,
        type_scores=result.type_scores,
        urgency=UrgencyInfo(
            level=result.urgency_level,
            score=result.urgency_score,
        ),
        locations=[
            LocationInfo(
                text=loc.text,
                label=loc.label,
                confidence=loc.confidence,
                latitude=loc.latitude,
                longitude=loc.longitude,
                display_name=loc.display_name,
                country=loc.country,
            )
            for loc in result.locations
        ],
        deduplication=DeduplicationInfo(
            is_duplicate=result.is_duplicate,
            cluster_id=result.cluster_id,
        ),
        processing_time_ms=result.processing_time_ms,
    )


# ─── Endpoints ───

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a single crisis message",
    description="Process a single text message through the full CrisisLens NLP pipeline.",
    tags=["Analysis"],
)
async def analyze_message(request: AnalyzeRequest):
    """
    Analyze a single message for crisis relevance, type, urgency, and location.
    """
    pipe = get_pipeline()

    try:
        result = await asyncio.to_thread(
            pipe.analyze, request.text, skip_dedup=request.skip_dedup
        )
        return result_to_response(result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    summary="Analyze multiple crisis messages",
    description="Process a batch of messages through the CrisisLens pipeline.",
    tags=["Analysis"],
)
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze a batch of messages (max 100) for crisis intelligence.
    """
    pipe = get_pipeline()

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 100 messages.",
        )

    try:
        # Run blocking batch in thread pool to avoid blocking event loop
        results = await asyncio.to_thread(
            pipe.analyze_batch, request.texts, skip_dedup=request.skip_dedup
        )

        responses = [result_to_response(r) for r in results]
        relevant = sum(1 for r in results if r.is_relevant)
        critical = sum(1 for r in results if r.urgency_level == "CRITICAL")

        return BatchAnalyzeResponse(
            results=responses,
            total_processed=len(results),
            total_relevant=relevant,
            total_critical=critical,
        )

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get pipeline statistics",
    description="Returns processing statistics since server start.",
    tags=["Monitoring"],
)
async def get_stats():
    """Get pipeline processing statistics."""
    pipe = get_pipeline()
    stats = pipe.stats
    return StatsResponse(**stats)


@router.post(
    "/reset",
    summary="Reset pipeline state",
    description="Resets deduplication window and statistics.",
    tags=["Monitoring"],
)
async def reset_pipeline():
    """Reset pipeline statistics and deduplication window."""
    pipe = get_pipeline()
    pipe.reset_stats()
    return {"status": "reset", "message": "Pipeline state cleared"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
async def health_check():
    """Check if the service is healthy and models are loaded."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "starting",
        version="0.1.0",
        models_loaded=pipeline is not None and pipeline._loaded,
    )
