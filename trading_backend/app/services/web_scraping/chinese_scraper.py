"""Chinese social media platform scraping service."""
import asyncio
from typing import Dict, List, Optional, Callable, Any
import aiohttp
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePlatformScraper:
    """Base class for platform scrapers with rate limiting."""

    _rate_limit_lock = asyncio.Lock()
    _last_request_time = datetime.now() - timedelta(seconds=10)
    _rate_limit = 10.2  # Add buffer to ensure we exceed 10 seconds

    def __init__(self):
        """Initialize base scraper."""
        self._cache_lock = asyncio.Lock()

    async def _rate_limit_wait(self):
        """Wait for rate limit."""
        async with self._rate_limit_lock:
            time_since_last = (datetime.now() - self._last_request_time).total_seconds()
            if time_since_last < self._rate_limit:
                wait_time = self._rate_limit - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
            self._last_request_time = datetime.now()

    def _get_cache_path(self, platform: str, endpoint: str) -> Path:
        """Get cache file path."""
        cache_dir = Path("cache") / platform / Path(endpoint).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{Path(endpoint).name}.json"

    async def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make HTTP request with rate limiting."""
        await self._rate_limit_wait()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

    async def _cached_request(self, platform: str, endpoint: str, fetch_func: Callable) -> Dict:
        """Make a cached request."""
        cache_path = self._get_cache_path(platform, endpoint)

        # Check cache first
        if cache_path.exists():
            try:
                cached_data = cache_path.read_text()
                if cached_data:
                    logger.debug(f"Using cached data for {platform}/{endpoint}")
                    return json.loads(cached_data)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Cache error for {platform}/{endpoint}: {str(e)}")

        # If no cache or cache error, fetch fresh data
        logger.debug(f"No cache found for {platform}/{endpoint}, fetching fresh data")
        try:
            # Always enforce rate limit, even if the request fails
            await self._rate_limit_wait()
            data = await fetch_func()
            # Cache the result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data))
            return data
        except Exception as e:
            logger.error(f"Request error for {platform}/{endpoint}: {str(e)}")
            return {"data": []}

class XiaohongshuScraper(BasePlatformScraper):
    """Xiaohongshu platform scraper."""

    def __init__(self):
        """Initialize Xiaohongshu scraper."""
        super().__init__()

    async def get_crypto_posts(self, symbol: str) -> List[Dict]:
        """Get cryptocurrency-related posts."""
        endpoint = f"search/notes/{symbol}"

        try:
            data = await self._cached_request(
                platform="xiaohongshu",
                endpoint=endpoint,
                fetch_func=lambda: self._make_request(
                    url=f"https://api.xiaohongshu.com/api/sns/v1/search/notes",
                    params={
                        "keyword": symbol,
                        "page": 1,
                        "page_size": 20,
                        "sort": "time_desc"
                    }
                )
            )
            return data.get("data", []) if data else []
        except Exception as e:
            logger.error(f"Error fetching Xiaohongshu posts for {symbol}: {str(e)}")
            return []

class DouyinScraper(BasePlatformScraper):
    """Douyin platform scraper."""

    def __init__(self):
        """Initialize Douyin scraper."""
        super().__init__()

    async def get_crypto_posts(self, symbol: str) -> List[Dict]:
        """Get cryptocurrency-related posts."""
        endpoint = f"search/videos/{symbol}"

        try:
            data = await self._cached_request(
                platform="douyin",
                endpoint=endpoint,
                fetch_func=lambda: self._make_request(
                    url=f"https://api.douyin.com/api/v1/search/videos",
                    params={
                        "keyword": symbol,
                        "page": 1,
                        "page_size": 20,
                        "sort": "time_desc"
                    }
                )
            )
            return data.get("data", []) if data else []
        except Exception as e:
            logger.error(f"Error fetching Douyin posts for {symbol}: {str(e)}")
            return []

class ChinesePlatformScraper:
    """Scraper for multiple Chinese platforms."""

    def __init__(self):
        """Initialize platform scrapers."""
        self.xiaohongshu = XiaohongshuScraper()
        self.douyin = DouyinScraper()
        self._cache_lock = asyncio.Lock()
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = datetime.now() - timedelta(seconds=10)
        self._rate_limit = 10.2  # Add buffer to ensure we exceed 10 seconds
        self._cache_dir = Path("cache/chinese_platforms/market_sentiment")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def _enforce_rate_limit(self):
        """Enforce rate limiting at the aggregator level."""
        async with self._rate_limit_lock:
            time_since_last = (datetime.now() - self._last_request_time).total_seconds()
            if time_since_last < self._rate_limit:
                wait_time = self._rate_limit - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
            self._last_request_time = datetime.now()

    async def get_market_sentiment(self, symbol: str) -> Dict[str, List[Dict]]:
        """Get market sentiment from all platforms."""
        # Validate symbol format first
        valid_symbols = ["BTC", "ETH", "USDT"]
        if symbol not in valid_symbols:
            logger.warning(f"Invalid symbol: {symbol}")
            return {"xiaohongshu": [], "douyin": []}

        try:
            # Check cache first
            cache_file = self._cache_dir / f"{symbol}.json"
            if cache_file.exists():
                try:
                    cached_data = json.loads(cache_file.read_text())
                    if isinstance(cached_data, dict) and "xiaohongshu" in cached_data and "douyin" in cached_data:
                        logger.debug(f"Using cached market sentiment for {symbol}")
                        return cached_data
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.error(f"Cache error for market sentiment {symbol}: {str(e)}")

            # Always enforce rate limit before making requests
            await self._enforce_rate_limit()

            async with self._cache_lock:
                # Create tasks but don't await them yet
                tasks = [
                    self.xiaohongshu.get_crypto_posts(symbol),
                    self.douyin.get_crypto_posts(symbol)
                ]

                # Now await all tasks together
                results = await asyncio.gather(*tasks, return_exceptions=True)

                response = {
                    "xiaohongshu": results[0] if not isinstance(results[0], Exception) else [],
                    "douyin": results[1] if not isinstance(results[1], Exception) else []
                }

                # Cache the results
                cache_file.write_text(json.dumps(response))
                return response

        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {str(e)}")
            return {"xiaohongshu": [], "douyin": []}

class ChineseSentimentAnalyzer:
    """Analyzer for Chinese platform sentiment."""

    def __init__(self):
        """Initialize the analyzer."""
        self.platform_scraper = ChinesePlatformScraper()
        self._cache_dir = Path("cache/chinese_sentiment")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of Chinese text."""
        # Basic sentiment analysis using keyword matching
        positive_keywords = ["看涨", "利好", "突破", "上升", "增长", "牛市"]
        negative_keywords = ["看跌", "利空", "下跌", "下降", "减少", "熊市"]

        text = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text)
        negative_count = sum(1 for word in negative_keywords if word in text)

        total = positive_count + negative_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 1.0}

        positive_score = positive_count / total
        negative_score = negative_count / total
        neutral_score = 1.0 - abs(positive_score - negative_score)

        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }

    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get aggregated market sentiment for a symbol."""
        try:
            posts = await self.platform_scraper.get_market_sentiment(symbol)


            # Analyze sentiment for each platform's posts
            sentiments = {
                platform: [
                    await self.analyze_sentiment(post.get("content", ""))
                    for post in platform_posts
                ]
                for platform, platform_posts in posts.items()
            }

            # Calculate average sentiment
            total_posts = sum(len(platform_posts) for platform_posts in posts.values())
            if total_posts == 0:
                return {
                    "overall_sentiment": "neutral",
                    "confidence": 0.5,
                    "platform_breakdown": {},
                    "post_count": 0
                }

            # Aggregate sentiments across platforms
            total_positive = 0
            total_negative = 0
            total_neutral = 0

            for platform_sentiments in sentiments.values():
                for sentiment in platform_sentiments:
                    total_positive += sentiment["positive"]
                    total_negative += sentiment["negative"]
                    total_neutral += sentiment["neutral"]

            avg_positive = total_positive / total_posts
            avg_negative = total_negative / total_posts
            avg_neutral = total_neutral / total_posts

            # Determine overall sentiment
            if avg_neutral > 0.6:
                overall = "neutral"
            elif avg_positive > avg_negative:
                overall = "positive"
            else:
                overall = "negative"

            # Calculate confidence based on sentiment strength and post count
            confidence = (1 - avg_neutral) * min(1.0, total_posts / 20)

            return {
                "overall_sentiment": overall,
                "confidence": confidence,
                "platform_breakdown": sentiments,
                "post_count": total_posts
            }

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "platform_breakdown": {},
                "post_count": 0
            }
