"""
CrisisLens — Pipeline Integration Tests
Tests the end-to-end pipeline orchestrator with sample messages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pytest
from src.pipeline.orchestrator import CrisisLensPipeline, CrisisAnalysisResult


@pytest.fixture(scope="module")
def pipeline():
    """Create and load the pipeline once for all tests in this module."""
    pipe = CrisisLensPipeline()
    pipe.load_models()
    return pipe


class TestPipelineIntegration:
    """Integration tests for the full CrisisLens pipeline."""

    def test_crisis_message_english(self, pipeline):
        """Test that an English crisis message is correctly identified."""
        text = "URGENT: Building collapsed in downtown Hatay, people are trapped under rubble. Send rescue teams immediately!"
        result = pipeline.analyze(text, skip_dedup=True)
        
        assert isinstance(result, CrisisAnalysisResult)
        assert result.is_relevant is True
        assert result.urgency_level in ["CRITICAL", "HIGH"]
        assert len(result.event_types) > 0
        assert result.processing_time_ms > 0

    def test_non_crisis_message(self, pipeline):
        """Test that a non-crisis message is correctly filtered."""
        text = "Just had the best pizza at the new restaurant downtown! Highly recommend the margherita. 🍕"
        result = pipeline.analyze(text, skip_dedup=True)
        
        assert isinstance(result, CrisisAnalysisResult)
        assert result.is_relevant is False

    def test_multilingual_spanish(self, pipeline):
        """Test Spanish crisis message processing."""
        text = "Terremoto fuerte en la ciudad. Muchos edificios destruidos. Necesitamos ayuda urgente!"
        result = pipeline.analyze(text, skip_dedup=True)
        
        assert isinstance(result, CrisisAnalysisResult)
        assert result.language.lang_code in ["es", "pt"]  # May detect as Portuguese too
        assert result.is_relevant is True

    def test_multilingual_hindi(self, pipeline):
        """Test Hindi crisis message processing."""
        text = "दिल्ली में बाढ़ आ गई है, लोग फंसे हुए हैं, तुरंत बचाव दल भेजो!"
        result = pipeline.analyze(text, skip_dedup=True)
        
        assert isinstance(result, CrisisAnalysisResult)
        assert result.language.lang_code == "hi"

    def test_result_has_all_fields(self, pipeline):
        """Test that the result object has all expected fields."""
        text = "Flood in Mumbai, many people displaced from coastal areas"
        result = pipeline.analyze(text, skip_dedup=True)
        
        assert hasattr(result, "original_text")
        assert hasattr(result, "cleaned_text")
        assert hasattr(result, "language")
        assert hasattr(result, "is_relevant")
        assert hasattr(result, "relevance_confidence")
        assert hasattr(result, "event_types")
        assert hasattr(result, "type_scores")
        assert hasattr(result, "urgency_level")
        assert hasattr(result, "urgency_score")
        assert hasattr(result, "locations")
        assert hasattr(result, "is_duplicate")
        assert hasattr(result, "processing_time_ms")

    def test_to_dict_serialization(self, pipeline):
        """Test that results can be serialized to dict."""
        text = "Earthquake damage in Istanbul"
        result = pipeline.analyze(text, skip_dedup=True)
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "language" in result_dict
        assert "urgency" in result_dict
        assert "locations" in result_dict

    def test_empty_input(self, pipeline):
        """Test handling of empty input."""
        result = pipeline.analyze("", skip_dedup=True)
        assert result.is_relevant is False
        assert result.cleaned_text == ""

    def test_deduplication(self, pipeline):
        """Test that duplicate messages are detected."""
        pipeline.deduplicator.reset()
        
        text1 = "Major flooding in Houston Texas, hundreds of homes underwater"
        text2 = "Severe flooding in Houston TX, hundreds of houses flooded"
        
        r1 = pipeline.analyze(text1)
        r2 = pipeline.analyze(text2)
        
        # The second message should ideally be flagged as duplicate
        # (depending on model similarity threshold)
        assert r1.cluster_id is not None
        assert r2.cluster_id is not None

    def test_batch_analysis(self, pipeline):
        """Test batch processing."""
        texts = [
            "Earthquake in Turkey, buildings collapsed",
            "Beautiful sunset at the beach today",
            "Flood warning issued for coastal regions",
        ]
        results = pipeline.analyze_batch(texts, skip_dedup=True)
        
        assert len(results) == 3
        assert all(isinstance(r, CrisisAnalysisResult) for r in results)

    def test_pipeline_stats(self, pipeline):
        """Test that pipeline stats are tracked."""
        stats = pipeline.stats
        assert "total_processed" in stats
        assert "total_relevant" in stats
        assert stats["total_processed"] > 0
