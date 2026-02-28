"""
CrisisLens — Semantic Deduplicator
Identifies and clusters near-duplicate crisis messages using sentence embeddings.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""
    is_duplicate: bool
    cluster_id: Optional[str]        # ID of the matching cluster
    similarity_score: float          # Highest similarity to existing messages
    matched_text: Optional[str]      # The text it was matched against


class Deduplicator:
    """
    Semantic deduplication using multilingual sentence embeddings.
    
    Maintains a sliding window of recent message embeddings and checks
    incoming messages against them using cosine similarity.
    
    Messages with similarity > threshold are flagged as duplicates
    and assigned to the same cluster.
    
    Uses 'paraphrase-multilingual-MiniLM-L12-v2' which supports 50+ languages
    and produces 384-dimensional embeddings.
    """

    def __init__(self, 
                 model_name: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 window_size: Optional[int] = None):
        self.model_name = model_name or settings.sentence_model
        self.threshold = similarity_threshold or settings.dedup_similarity_threshold
        self.window_size = window_size or settings.dedup_window_size

        self._model: Optional[SentenceTransformer] = None
        
        # Sliding window of recent embeddings and their metadata
        self._embeddings: deque = deque(maxlen=self.window_size)
        self._texts: deque = deque(maxlen=self.window_size)
        self._cluster_ids: deque = deque(maxlen=self.window_size)
        self._next_cluster_id = 0

    def load(self):
        """Load the sentence transformer model lazily."""
        if self._model is None:
            logger.info(f"Loading sentence model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Sentence model loaded successfully")

    def check(self, text: str) -> DeduplicationResult:
        """
        Check if the message is a duplicate of a recent message.
        
        If duplicate: returns the existing cluster ID.
        If new: creates a new cluster, stores the embedding, returns the new ID.
        
        Args:
            text: Preprocessed message text
            
        Returns:
            DeduplicationResult with duplicate flag and cluster info
        """
        self.load()

        if not text or not text.strip():
            return DeduplicationResult(
                is_duplicate=False,
                cluster_id=None,
                similarity_score=0.0,
                matched_text=None,
            )

        # Encode the new message
        embedding = self._model.encode(text, normalize_embeddings=True)

        # Compare against existing embeddings in the window
        if len(self._embeddings) > 0:
            # Stack all embeddings for vectorized cosine similarity
            existing = np.array(list(self._embeddings))
            similarities = np.dot(existing, embedding)

            max_idx = np.argmax(similarities)
            max_sim = float(similarities[max_idx])

            if max_sim >= self.threshold:
                # Duplicate found!
                cluster_id = self._cluster_ids[max_idx]
                matched_text = self._texts[max_idx]

                # Store this embedding too (for future matching)
                self._embeddings.append(embedding)
                self._texts.append(text)
                self._cluster_ids.append(cluster_id)

                return DeduplicationResult(
                    is_duplicate=True,
                    cluster_id=cluster_id,
                    similarity_score=round(max_sim, 4),
                    matched_text=matched_text,
                )

        # New unique message — create a new cluster
        # Compute max similarity to existing BEFORE appending (avoids off-by-one)
        max_sim = 0.0
        if len(self._embeddings) > 0:
            existing = np.array(list(self._embeddings))
            max_sim = float(np.max(np.dot(existing, embedding)))

        cluster_id = f"cluster_{self._next_cluster_id:06d}"
        self._next_cluster_id += 1

        self._embeddings.append(embedding)
        self._texts.append(text)
        self._cluster_ids.append(cluster_id)

        return DeduplicationResult(
            is_duplicate=False,
            cluster_id=cluster_id,
            similarity_score=round(max_sim, 4),
            matched_text=None,
        )

    def batch_check(self, texts: list[str]) -> list[DeduplicationResult]:
        """Check deduplication for a batch of texts (processed sequentially)."""
        return [self.check(t) for t in texts]

    def reset(self):
        """Clear all stored embeddings and reset cluster counter."""
        self._embeddings.clear()
        self._texts.clear()
        self._cluster_ids.clear()
        self._next_cluster_id = 0

    @property
    def window_count(self) -> int:
        """Number of messages currently in the dedup window."""
        return len(self._embeddings)
