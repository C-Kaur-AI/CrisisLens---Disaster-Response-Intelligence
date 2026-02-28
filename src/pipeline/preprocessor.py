"""
CrisisLens — Text Preprocessor
Cleans and normalizes raw social media text for NLP processing.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import emoji


@dataclass
class PreprocessedMessage:
    """Result of preprocessing a raw message."""
    original_text: str
    cleaned_text: str
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    has_media: bool = False


class TextPreprocessor:
    """
    Preprocesses raw social media text for downstream NLP tasks.
    
    Handles:
    - URL removal and extraction
    - @mention removal and extraction
    - Emoji → text conversion (🔥 → "fire")
    - Hashtag segmentation (#FloodAlert → "Flood Alert")
    - Unicode normalization
    - Whitespace cleanup
    - RT prefix removal
    """

    # Compiled regex patterns for performance
    URL_PATTERN = re.compile(
        r'https?://\S+|www\.\S+', re.IGNORECASE
    )
    MENTION_PATTERN = re.compile(r'@[\w]+')
    HASHTAG_PATTERN = re.compile(r'#([\w]+)')
    RT_PATTERN = re.compile(r'^RT\s+@[\w]+:\s*', re.IGNORECASE)
    MULTI_SPACE = re.compile(r'\s+')
    CAMEL_CASE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 convert_emojis: bool = True,
                 segment_hashtags: bool = True):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.convert_emojis = convert_emojis
        self.segment_hashtags = segment_hashtags

    def preprocess(self, text: str) -> PreprocessedMessage:
        """
        Full preprocessing pipeline for a single message.
        
        Args:
            text: Raw message text
            
        Returns:
            PreprocessedMessage with cleaned text and extracted metadata
        """
        if not text or not text.strip():
            return PreprocessedMessage(
                original_text=text or "",
                cleaned_text="",
            )

        original = text

        # Extract metadata before removal
        urls = self.URL_PATTERN.findall(text)
        mentions = self.MENTION_PATTERN.findall(text)
        hashtags = self.HASHTAG_PATTERN.findall(text)

        # Remove RT prefix
        text = self.RT_PATTERN.sub('', text)

        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub('', text)

        # Remove mentions
        if self.remove_mentions:
            text = self.MENTION_PATTERN.sub('', text)

        # Convert emojis to text
        if self.convert_emojis:
            text = self._convert_emojis(text)

        # Segment hashtags
        if self.segment_hashtags:
            text = self._segment_hashtags(text)

        # Unicode normalization
        text = self._normalize_unicode(text)

        # Clean whitespace
        text = self.MULTI_SPACE.sub(' ', text).strip()

        return PreprocessedMessage(
            original_text=original,
            cleaned_text=text,
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            has_media=any(
                ext in url.lower()
                for url in urls
                for ext in ['.jpg', '.png', '.gif', '.mp4', '.video']
            ),
        )

    def _convert_emojis(self, text: str) -> str:
        """Convert emoji characters to their text descriptions."""
        return emoji.demojize(text, delimiters=(" ", " "))

    def _segment_hashtags(self, text: str) -> str:
        """
        Segment CamelCase hashtags into separate words.
        #FloodAlert → Flood Alert
        #HELP → HELP
        """
        def _split_hashtag(match: re.Match) -> str:
            tag = match.group(1)
            # Split CamelCase
            segmented = self.CAMEL_CASE.sub(' ', tag)
            return segmented

        return self.HASHTAG_PATTERN.sub(_split_hashtag, text)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters (NFC form)."""
        text = unicodedata.normalize('NFC', text)
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        return text

    def batch_preprocess(self, texts: list[str]) -> list[PreprocessedMessage]:
        """Preprocess a batch of messages."""
        return [self.preprocess(t) for t in texts]
