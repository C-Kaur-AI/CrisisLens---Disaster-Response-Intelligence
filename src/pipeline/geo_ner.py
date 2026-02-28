"""
CrisisLens — Geographic Named Entity Recognition (GeoNER)
Extracts location entities from crisis messages using multilingual NER.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class LocationEntity:
    """A single extracted location entity."""
    text: str           # The location string as found in text
    label: str          # Entity type: LOC, GPE, FACILITY, etc.
    confidence: float   # Model confidence
    start: int          # Start character offset in original text
    end: int            # End character offset in original text


class GeoNER:
    """
    Extracts geographic/location named entities from text using
    a multilingual NER model (XLM-RoBERTa fine-tuned on WikiANN).
    
    Supported entity types mapped to locations:
    - LOC: Geographic locations (rivers, mountains, regions)
    - PER: Filtered out (person names)
    - ORG: Filtered out unless facility-like (hospitals, schools)
    
    The model supports 100+ languages out of the box.
    """

    # Organization keywords that are actually facilities/locations
    FACILITY_KEYWORDS = {
        'hospital', 'school', 'university', 'airport', 'station',
        'bridge', 'church', 'mosque', 'temple', 'shelter',
        'camp', 'center', 'centre', 'building', 'tower',
        'stadium', 'park', 'market', 'port', 'base',
    }

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None):
        self.model_name = model_name or settings.ner_model

        if device is None:
            self._device = 0 if torch.cuda.is_available() else -1
        else:
            self._device = device

        self._ner = None

    def load(self):
        """Load the NER model lazily."""
        if self._ner is None:
            logger.info(f"Loading NER model: {self.model_name}")
            self._ner = hf_pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                device=self._device,
            )
            logger.info("NER model loaded successfully")

    def extract(self, text: str) -> list[LocationEntity]:
        """
        Extract location entities from text.
        
        Args:
            text: Input text (can be any language)
            
        Returns:
            List of LocationEntity objects found in the text
        """
        self.load()

        if not text or not text.strip():
            return []

        try:
            entities = self._ner(text)

            locations = []
            for ent in entities:
                entity_label = ent.get("entity_group", ent.get("entity", ""))

                # Keep LOC entities directly
                if "LOC" in entity_label:
                    locations.append(LocationEntity(
                        text=ent["word"].strip(),
                        label="LOC",
                        confidence=round(float(ent["score"]), 4),
                        start=ent.get("start", 0),
                        end=ent.get("end", 0),
                    ))

                # Check if ORG entity is actually a facility/location
                elif "ORG" in entity_label:
                    word_lower = ent["word"].strip().lower()
                    if any(kw in word_lower for kw in self.FACILITY_KEYWORDS):
                        locations.append(LocationEntity(
                            text=ent["word"].strip(),
                            label="FACILITY",
                            confidence=round(float(ent["score"]), 4),
                            start=ent.get("start", 0),
                            end=ent.get("end", 0),
                        ))

            # Merge adjacent location tokens that might have been split
            locations = self._merge_adjacent(locations)

            return locations

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def _merge_adjacent(self, entities: list[LocationEntity]) -> list[LocationEntity]:
        """Merge adjacent location entities that form a single place name."""
        if len(entities) <= 1:
            return entities

        merged = [entities[0]]
        for ent in entities[1:]:
            prev = merged[-1]
            # If this entity starts right after the previous one (with small gap)
            if ent.start - prev.end <= 2 and ent.label == prev.label:
                # Merge into previous
                merged[-1] = LocationEntity(
                    text=f"{prev.text} {ent.text}".strip(),
                    label=prev.label,
                    confidence=min(prev.confidence, ent.confidence),
                    start=prev.start,
                    end=ent.end,
                )
            else:
                merged.append(ent)

        return merged

    def batch_extract(self, texts: list[str]) -> list[list[LocationEntity]]:
        """Extract locations from a batch of texts."""
        return [self.extract(t) for t in texts]
