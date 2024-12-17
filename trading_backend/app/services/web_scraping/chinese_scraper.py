"""Chinese social media platform scraping service."""
import asyncio
from typing import Dict, List, Optional
import aiohttp
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePlatformScraper:
    """Base class for Chinese platform scrapers."""

    def __init__(self, rate_limit: int = 60):
        """Initialize base scraper with rate limiting."""
        self.rate_limit = rate_limit  # requests per minute
        self.last_request_time: Dict[str, datetime] = {}
        self.cache_dir = Path("cache/chinese_platforms")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def _rate_limit_wait(self, endpoint: str):
        """Implement rate limiting."""
        now = datetime.now()
        if endpoint in self.last_request_time:
            time_since_last = now - self.last_request_time[endpoint]
            wait_time = timedelta(minutes=1) / self.rate_limit - time_since_last
            if wait_time.total_seconds() > 0:
                await asyncio.sleep(wait_time.total_seconds())
        self.last_request_time[endpoint] = now

    def _get_cache_path(self, platform: str, endpoint: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{platform}_{endpoint}.json"

    async def _cached_request(self, platform: str, endpoint: str, url: str, headers: Dict) -> Optional[Dict]:
        """Make a cached request."""
        cache_path = self._get_cache_path(platform, endpoint)

        # Check cache first
        if cache_path.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(minutes=15):  # 15-minute cache
                try:
                    return json.loads(cache_path.read_text())
                except json.JSONDecodeError:
                    logger.warning(f"Invalid cache file for {platform}_{endpoint}")

        # Make actual request
        await self._rate_limit_wait(endpoint)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Cache the response
                        cache_path.write_text(json.dumps(data))
                        return data
                    else:
                        logger.error(f"Request failed for {platform}_{endpoint}: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {platform}_{endpoint}: {str(e)}")
            return None

class XiaohongshuScraper(BasePlatformScraper):
    """Xiaohongshu platform scraper."""

    def __init__(self):
        """Initialize Xiaohongshu scraper."""
        super().__init__(rate_limit=30)  # More conservative rate limit
        self.base_url = "https://api.xiaohongshu.com"

    async def get_crypto_posts(self, keyword: str) -> List[Dict]:
        """Get cryptocurrency-related posts."""
        endpoint = f"search/notes/{keyword}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        data = await self._cached_request(
            "xiaohongshu",
            endpoint,
            f"{self.base_url}/{endpoint}",
            headers
        )
        return data.get("data", []) if data else []

class DouyinScraper(BasePlatformScraper):
    """Douyin platform scraper."""

    def __init__(self):
        """Initialize Douyin scraper."""
        super().__init__(rate_limit=30)
        self.base_url = "https://api.douyin.com"

    async def get_crypto_posts(self, keyword: str) -> List[Dict]:
        """Get cryptocurrency-related posts."""
        endpoint = f"search/videos/{keyword}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        data = await self._cached_request(
            "douyin",
            endpoint,
            f"{self.base_url}/{endpoint}",
            headers
        )
        return data.get("data", []) if data else []

class ChinesePlatformScraper:
    """Main scraper class for Chinese social media platforms."""

    def __init__(self):
        """Initialize Chinese platform scraper."""
        self.platforms = {
            'xiaohongshu': XiaohongshuScraper(),
            'douyin': DouyinScraper()
        }

    async def get_market_sentiment(self, keyword: str) -> Dict[str, List[Dict]]:
        """Get market sentiment from all platforms."""
        tasks = []
        for platform_name, scraper in self.platforms.items():
            tasks.append(scraper.get_crypto_posts(keyword))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiment_data = {}
        for platform_name, result in zip(self.platforms.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching from {platform_name}: {str(result)}")
                sentiment_data[platform_name] = []
            else:
                sentiment_data[platform_name] = result

        return sentiment_data
