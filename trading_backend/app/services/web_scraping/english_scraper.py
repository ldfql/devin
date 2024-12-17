"""Base class for English platform scrapers with rate limiting and caching."""
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import hashlib
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)

class BaseEnglishScraper(ABC):
    """Base class for English platform scrapers with rate limiting and caching."""

    _rate_limit_lock = asyncio.Lock()
    _last_request_time: Dict[str, datetime] = {}
    _rate_limits: Dict[str, timedelta] = {
        'default': timedelta(seconds=1),
        'youtube': timedelta(seconds=2),
        'twitter': timedelta(seconds=5)
    }

    # Sentiment analysis patterns
    POSITIVE_PATTERNS = [
        r'bull(?:ish)?', r'moon', r'pump', r'long', r'buy', r'support',
        r'break(?:out)?', r'surge', r'gain', r'profit', r'growth'
    ]
    NEGATIVE_PATTERNS = [
        r'bear(?:ish)?', r'dump', r'short', r'sell', r'resistance',
        r'crash', r'drop', r'loss', r'decline', r'fall'
    ]

    def __init__(self, platform: str):
        """Initialize scraper with platform-specific settings.

        Args:
            platform: Platform identifier ('youtube' or 'twitter')
        """
        self.platform = platform
        self._cache_lock = asyncio.Lock()
        self._cache_dir = Path("cache/english_platforms") / platform
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and parameters."""
        key_parts = [url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        async with self._cache_lock:
            try:
                data = json.loads(cache_file.read_text())
                if datetime.fromisoformat(data['timestamp']) + timedelta(hours=1) < datetime.now():
                    return None
                return data['content']
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Cache read error for {cache_key}: {e}")
                return None

    async def _cache_response(self, cache_key: str, content: Dict) -> None:
        """Cache response with timestamp."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'content': content
        }

        async with self._cache_lock:
            try:
                cache_file.write_text(json.dumps(cache_data))
            except Exception as e:
                logger.error(f"Cache write error for {cache_key}: {e}")

    async def _wait_for_rate_limit(self) -> None:
        """Wait for rate limit based on platform."""
        async with self._rate_limit_lock:
            now = datetime.now()
            last_request = self._last_request_time.get(self.platform, now - timedelta(days=1))
            wait_time = (last_request + self._rate_limits.get(
                self.platform, self._rate_limits['default'])) - now

            if wait_time.total_seconds() > 0:
                await asyncio.sleep(wait_time.total_seconds())

            self._last_request_time[self.platform] = datetime.now()

    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of English text using pattern matching.

        Args:
            text: Text to analyze

        Returns:
            Dictionary containing sentiment score and label
        """
        text = text.lower()
        positive_count = sum(len(re.findall(pattern, text)) for pattern in self.POSITIVE_PATTERNS)
        negative_count = sum(len(re.findall(pattern, text)) for pattern in self.NEGATIVE_PATTERNS)

        total = positive_count + negative_count
        if total == 0:
            return {"sentiment": "neutral", "score": 0.0}

        score = (positive_count - negative_count) / (positive_count + negative_count)
        return {
            "sentiment": "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral",
            "score": score
        }

    async def _make_request(
        self,
        url: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache: bool = True
    ) -> Dict[str, Any]:
        """Make rate-limited HTTP request with caching.

        Args:
            url: Request URL
            method: HTTP method
            params: Query parameters
            headers: Request headers
            cache: Whether to use cache

        Returns:
            Response data as dictionary
        """
        cache_key = self._get_cache_key(url, params)

        if cache:
            cached_response = await self._get_cached_response(cache_key)
            if cached_response is not None:
                return cached_response

        await self._wait_for_rate_limit()

        try:
            session = await self._get_session()
            async with session.request(
                method,
                url,
                params=params,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if cache:
                    await self._cache_response(cache_key, data)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"Request error for {url}: {e}")
            raise

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def get_content(self, *args, **kwargs) -> Dict[str, Any]:
        """Get content from platform. Must be implemented by subclasses."""
        pass
