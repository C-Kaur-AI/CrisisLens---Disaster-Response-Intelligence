"""
CrisisLens — Preprocessor Unit Tests
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import pytest
from src.pipeline.preprocessor import TextPreprocessor, PreprocessedMessage


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestTextPreprocessor:
    """Tests for the TextPreprocessor."""

    def test_empty_input(self, preprocessor):
        result = preprocessor.preprocess("")
        assert result.cleaned_text == ""
        assert result.original_text == ""

    def test_none_input(self, preprocessor):
        result = preprocessor.preprocess(None)
        assert result.cleaned_text == ""

    def test_url_removal(self, preprocessor):
        text = "Flood in city https://example.com/photo.jpg please help"
        result = preprocessor.preprocess(text)
        assert "https://example.com" not in result.cleaned_text
        assert "please help" in result.cleaned_text
        assert len(result.urls) == 1

    def test_mention_removal(self, preprocessor):
        text = "@rescueteam Please help us in downtown area"
        result = preprocessor.preprocess(text)
        assert "@rescueteam" not in result.cleaned_text
        assert "Please help" in result.cleaned_text
        assert "@rescueteam" in result.mentions

    def test_hashtag_segmentation(self, preprocessor):
        text = "#FloodAlert in the eastern region"
        result = preprocessor.preprocess(text)
        assert "Flood Alert" in result.cleaned_text
        assert "FloodAlert" in result.hashtags

    def test_rt_removal(self, preprocessor):
        text = "RT @user123: Major earthquake just hit the city!"
        result = preprocessor.preprocess(text)
        assert not result.cleaned_text.startswith("RT")
        assert "earthquake" in result.cleaned_text

    def test_emoji_conversion(self, preprocessor):
        text = "Building on fire 🔥 need help 🆘"
        result = preprocessor.preprocess(text)
        assert "🔥" not in result.cleaned_text
        assert "🆘" not in result.cleaned_text
        # Emoji should be converted to text descriptions
        assert "fire" in result.cleaned_text.lower()

    def test_whitespace_cleanup(self, preprocessor):
        text = "Too   much    space   here"
        result = preprocessor.preprocess(text)
        assert "   " not in result.cleaned_text

    def test_metadata_extraction(self, preprocessor):
        text = "@user1 Flood in city https://t.co/abc #Rescue #HelpNeeded"
        result = preprocessor.preprocess(text)
        assert len(result.mentions) == 1
        assert len(result.urls) == 1
        assert len(result.hashtags) == 2

    def test_batch_preprocess(self, preprocessor):
        texts = ["Hello world", "Test message #Two"]
        results = preprocessor.batch_preprocess(texts)
        assert len(results) == 2
        assert all(isinstance(r, PreprocessedMessage) for r in results)

    def test_unicode_handling(self, preprocessor):
        text = "Earthquake in Türkiye\u200b caused massive damage"
        result = preprocessor.preprocess(text)
        # Zero-width characters should be removed
        assert "\u200b" not in result.cleaned_text

    def test_preserves_multilingual(self, preprocessor):
        text = "भूकंप से भारी तबाही, मदद चाहिए"
        result = preprocessor.preprocess(text)
        assert "भूकंप" in result.cleaned_text

    def test_preserves_arabic(self, preprocessor):
        text = "زلزال قوي ضرب المنطقة ونحتاج مساعدة"
        result = preprocessor.preprocess(text)
        assert "زلزال" in result.cleaned_text
