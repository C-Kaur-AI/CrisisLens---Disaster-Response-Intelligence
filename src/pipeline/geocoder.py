"""
CrisisLens — Geocoder
Converts location entity strings to geographic coordinates (lat/lng).
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class GeocodedLocation:
    """A geocoded location with coordinates."""
    query: str          # Original location string
    display_name: str   # Full display name from geocoder
    latitude: float
    longitude: float
    confidence: float   # How confident we are in the geocoding
    country: str        # Country code if available
    raw: Optional[dict] = None


class Geocoder:
    """
    Converts location strings to latitude/longitude coordinates
    using OpenStreetMap's Nominatim service via geopy.
    
    Features:
    - In-memory LRU caching to minimize API calls
    - Rate limiting (1 request/second for Nominatim TOS compliance)
    - Graceful fallback on failure
    - Country/region context hints for better accuracy
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: Optional[int] = None):
        self.user_agent = user_agent or settings.geocoding_user_agent
        self.timeout = timeout or settings.geocoding_timeout
        self._geocoder = Nominatim(
            user_agent=self.user_agent,
            timeout=self.timeout,
        )
        self._last_request_time = 0.0
        self._cache: dict[str, Optional[GeocodedLocation]] = {}
        self._cache_max_size = 2000  # Prevent unbounded growth in batch processing

    def geocode(self, location_text: str, context_country: Optional[str] = None) -> Optional[GeocodedLocation]:
        """
        Geocode a location string to coordinates.
        
        Args:
            location_text: Location name/description to geocode
            context_country: ISO country code to bias results (e.g., 'TR' for Turkey)
            
        Returns:
            GeocodedLocation if successful, None if geocoding fails
        """
        if not location_text or not location_text.strip():
            return None

        # Normalize the query
        query = location_text.strip()
        cache_key = f"{query}|{context_country or ''}"

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Rate limiting (Nominatim requires 1 req/sec)
        self._rate_limit()

        try:
            # Add country bias if provided
            kwargs = {}
            if context_country:
                kwargs['country_codes'] = context_country

            result = self._geocoder.geocode(
                query,
                exactly_one=True,
                language='en',
                addressdetails=True,
                **kwargs,
            )

            if result:
                address = result.raw.get('address', {})
                country_code = address.get('country_code', '').upper()

                geocoded = GeocodedLocation(
                    query=query,
                    display_name=result.address,
                    latitude=round(result.latitude, 6),
                    longitude=round(result.longitude, 6),
                    confidence=self._estimate_confidence(result),
                    country=country_code,
                    raw=result.raw,
                )
                if len(self._cache) >= self._cache_max_size:
                    # Evict oldest ~10% (simple FIFO via pop)
                    for _ in range(self._cache_max_size // 10):
                        self._cache.pop(next(iter(self._cache)), None)
                self._cache[cache_key] = geocoded
                return geocoded
            else:
                logger.debug(f"No geocoding result for: {query}")
                self._cache[cache_key] = None
                return None

        except GeocoderTimedOut:
            logger.warning(f"Geocoding timed out for: {query}")
            return None
        except GeocoderUnavailable:
            logger.warning("Geocoding service unavailable")
            return None
        except Exception as e:
            logger.error(f"Geocoding error for '{query}': {e}")
            return None

    def _rate_limit(self):
        """Ensure we don't exceed Nominatim's 1 request/second limit."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _estimate_confidence(self, result) -> float:
        """
        Estimate geocoding confidence based on result quality.
        Considers the importance score and bounding box specificity.
        """
        raw = result.raw
        importance = float(raw.get('importance', 0.5))

        # Check if the result has a tight bounding box (more specific = better)
        bbox = raw.get('boundingbox', [])
        if len(bbox) == 4:
            lat_range = abs(float(bbox[1]) - float(bbox[0]))
            lng_range = abs(float(bbox[3]) - float(bbox[2]))
            # Smaller bounding box = more specific = higher confidence
            specificity = max(0.0, 1.0 - (lat_range * lng_range) / 100.0)
        else:
            specificity = 0.5

        confidence = (importance * 0.6 + specificity * 0.4)
        return round(min(1.0, confidence), 4)

    def batch_geocode(self, locations: list[str], context_country: Optional[str] = None) -> list[Optional[GeocodedLocation]]:
        """Geocode a batch of location strings."""
        return [self.geocode(loc, context_country) for loc in locations]

    def clear_cache(self):
        """Clear the geocoding cache."""
        self._cache.clear()
