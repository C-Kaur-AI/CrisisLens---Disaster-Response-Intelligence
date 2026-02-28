"""
CrisisLens — FastAPI Application Entry Point
Main application factory with CORS, lifespan management, and router includes.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api import routes
from src.pipeline.orchestrator import CrisisLensPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: load models on startup, cleanup on shutdown.
    """
    logger.info("=" * 60)
    logger.info("🚀 CrisisLens API starting up...")
    logger.info("=" * 60)

    # Initialize and load the pipeline
    pipeline = CrisisLensPipeline()

    logger.info("Loading NLP models (this may take a minute on first run)...")
    pipeline.load_models()

    # Set the pipeline in the routes module
    routes.pipeline = pipeline

    logger.info("=" * 60)
    logger.info("✅ CrisisLens API ready to receive requests!")
    logger.info(f"📖 Swagger docs: http://localhost:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield  # Application is running

    # Cleanup on shutdown
    logger.info("CrisisLens API shutting down...")
    routes.pipeline = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="🌍 CrisisLens API",
        description=(
            "**Multilingual Crisis & Disaster Response NLP Pipeline**\n\n"
            "CrisisLens processes social media messages during natural disasters "
            "to extract actionable intelligence for first responders.\n\n"
            "**Capabilities:**\n"
            "- 🌐 Multilingual language detection (176 languages)\n"
            "- 🎯 Crisis relevance classification\n"
            "- 📋 Multi-label event type classification\n"
            "- 🚨 Urgency scoring (CRITICAL → LOW)\n"
            "- 📍 Geographic entity extraction & geocoding\n"
            "- 🔁 Semantic deduplication\n\n"
            "**UN SDGs:** #11 Sustainable Cities, #13 Climate Action"
        ),
        version=settings.app_version,
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
        lifespan=lifespan,
    )

    # CORS middleware
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(routes.router, prefix="/api/v1")

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "app": "CrisisLens",
            "version": settings.app_version,
            "docs": "/docs",
            "api": "/api/v1",
        }

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
