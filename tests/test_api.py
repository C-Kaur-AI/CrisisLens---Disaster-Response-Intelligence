"""
CrisisLens — API Endpoint Tests
Tests for the FastAPI endpoints using httpx TestClient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


class TestAPIEndpoints:
    """Tests for the CrisisLens API."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns app info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "CrisisLens"
        assert "version" in data

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "starting"]

    def test_analyze_crisis_message(self, client):
        """Test analyzing a crisis message."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": "URGENT: Building collapsed in Hatay, people trapped under rubble!",
                "skip_dedup": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "is_relevant" in data
        assert "language" in data
        assert "urgency" in data
        assert "locations" in data
        assert "deduplication" in data
        assert "processing_time_ms" in data

    def test_analyze_non_crisis_message(self, client):
        """Test analyzing a non-crisis message."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": "Had a great coffee this morning! ☕"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_relevant" in data

    def test_analyze_empty_text_rejected(self, client):
        """Test that empty text is rejected."""
        response = client.post(
            "/api/v1/analyze",
            json={"text": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_batch_analyze(self, client):
        """Test batch analysis endpoint."""
        response = client.post(
            "/api/v1/analyze/batch",
            json={
                "texts": [
                    "Earthquake in Turkey, buildings collapsed",
                    "Beautiful day at the park with family",
                ],
                "skip_dedup": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total_processed"] == 2

    def test_stats_endpoint(self, client):
        """Test the stats endpoint."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_processed" in data
        assert "total_relevant" in data

    def test_reset_endpoint(self, client):
        """Test the reset endpoint."""
        response = client.post("/api/v1/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"

    def test_analyze_multilingual(self, client):
        """Test analyzing a non-English message."""
        response = client.post(
            "/api/v1/analyze",
            json={
                "text": "Terremoto fuerte en México! Edificios destruidos, ayuda urgente!",
                "skip_dedup": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"]["code"] in ["es", "pt", "it"]
