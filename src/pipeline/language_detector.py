"""
CrisisLens — Language Detector
Identifies the language of incoming messages using fastText or fallback.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """Result of language detection."""
    lang_code: str
    confidence: float
    method: str  # "fasttext" or "langdetect"


class LanguageDetector:
    """
    Detects the language of text using fastText's lid.176.bin model.
    Falls back to `langdetect` library if fastText model is unavailable.
    
    Supports 176 languages with fastText and ~55 with langdetect.
    """

    def __init__(self, fasttext_model_path: Optional[str] = None):
        self._fasttext_model = None
        self._use_fallback = False

        # Try to load fastText model
        if fasttext_model_path:
            model_path = Path(fasttext_model_path)
            if model_path.exists():
                try:
                    import fasttext
                    # Suppress fasttext warnings about loading
                    fasttext.FastText.eprint = lambda x: None
                    self._fasttext_model = fasttext.load_model(str(model_path))
                    logger.info("FastText language model loaded successfully")
                except ImportError:
                    logger.warning("fasttext not installed, falling back to langdetect")
                    self._use_fallback = True
                except Exception as e:
                    logger.warning(f"Failed to load fastText model: {e}")
                    self._use_fallback = True
            else:
                logger.warning(
                    f"FastText model not found at {model_path}. "
                    "Download from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                )
                self._use_fallback = True
        else:
            self._use_fallback = True

    def detect(self, text: str) -> LanguageDetection:
        """
        Detect the language of the given text.
        
        Args:
            text: Input text to detect language for
            
        Returns:
            LanguageDetection with language code, confidence, and method used
        """
        if not text or not text.strip():
            return LanguageDetection(lang_code="und", confidence=0.0, method="none")

        # Clean text for detection (single line, no URLs)
        clean = text.replace('\n', ' ').strip()

        if self._fasttext_model and not self._use_fallback:
            return self._detect_fasttext(clean)
        else:
            return self._detect_langdetect(clean)

    def _detect_fasttext(self, text: str) -> LanguageDetection:
        """Detect language using fastText."""
        try:
            predictions = self._fasttext_model.predict(text, k=1)
            label = predictions[0][0]  # e.g., '__label__en'
            confidence = float(predictions[1][0])

            # Extract language code from label
            lang_code = label.replace('__label__', '')

            return LanguageDetection(
                lang_code=lang_code,
                confidence=round(confidence, 4),
                method="fasttext",
            )
        except Exception as e:
            logger.error(f"FastText detection failed: {e}")
            return self._detect_langdetect(text)

    def _detect_langdetect(self, text: str) -> LanguageDetection:
        """Fallback language detection using langdetect library."""
        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if results:
                top = results[0]
                return LanguageDetection(
                    lang_code=top.lang,
                    confidence=round(top.prob, 4),
                    method="langdetect",
                )
        except ImportError:
            logger.error("langdetect not installed. Install with: pip install langdetect")
        except Exception as e:
            logger.error(f"Language detection failed: {e}")

        return LanguageDetection(lang_code="und", confidence=0.0, method="failed")

    def batch_detect(self, texts: list[str]) -> list[LanguageDetection]:
        """Detect languages for a batch of texts."""
        return [self.detect(t) for t in texts]
