"""Twitter scraper for market signals and financial news."""
import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from .english_scraper import BaseEnglishScraper

logger = logging.getLogger(__name__)

class TwitterScraper(BaseEnglishScraper):
    """Twitter/X scraper for market signals and financial news."""

    INFLUENTIAL_ACCOUNTS = [
        "elonmusk",  # Elon Musk
        "cz_binance",  # Binance CEO
        "SBF_FTX",  # FTX insights
        "VitalikButerin",  # Ethereum founder
        "CryptoCapo_",  # Crypto analyst
    ]

    FINANCIAL_NEWS_ACCOUNTS = [
        "Cointelegraph",
        "CoinDesk",
        "TheBlock__",
        "DocumentingBTC",
        "BitcoinMagazine",
    ]

    def __init__(self):
        """Initialize Twitter scraper with rate limiting."""
        super().__init__(platform="twitter")
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Twitter API v2 client with bearer token."""
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            logger.error("Twitter bearer token not found")
            raise ValueError("Bearer token must be configured")

    async def get_content(self, *args, **kwargs) -> Dict[str, Any]:
        """Get content from Twitter platform.

        This method implements the abstract method from BaseEnglishScraper.
        It serves as a unified interface for getting Twitter content.
        """
        content_type = kwargs.get("content_type", "tweets")
        username = kwargs.get("username")

        if content_type == "tweets" and username:
            return {"tweets": await self.get_user_tweets(username)}
        elif content_type == "influential":
            return {"tweets": await self.get_influential_tweets()}
        elif content_type == "news":
            return {"tweets": await self.get_financial_news()}
        else:
            raise ValueError(f"Invalid content type: {content_type}")

    async def get_user_tweets(self, username: str, limit: int = 100) -> List[Dict]:
        """Get latest tweets from specified user.

        Args:
            username: Twitter username without @ symbol
            limit: Maximum number of tweets to retrieve

        Returns:
            List of tweet dictionaries containing id, text, and created_at
        """
        await self._wait_for_rate_limit()
        # Mock implementation for testing
        return [
            {
                "id": "123",
                "text": f"Mock tweet from {username}",
                "created_at": datetime.now().isoformat()
            }
        ]

    async def get_influential_tweets(self) -> List[Dict]:
        """Get latest tweets from influential crypto figures."""
        all_tweets = []
        for account in self.INFLUENTIAL_ACCOUNTS:
            try:
                tweets = await self.get_user_tweets(account, limit=20)
                all_tweets.extend(tweets)
            except Exception as e:
                logger.error(f"Error fetching tweets for {account}: {e}")
        return sorted(all_tweets, key=lambda x: x["created_at"], reverse=True)

    async def get_financial_news(self) -> List[Dict]:
        """Get latest crypto news from financial news accounts."""
        all_news = []
        for account in self.FINANCIAL_NEWS_ACCOUNTS:
            try:
                tweets = await self.get_user_tweets(account, limit=20)
                all_news.extend(tweets)
            except Exception as e:
                logger.error(f"Error fetching news from {account}: {e}")
        return sorted(all_news, key=lambda x: x["created_at"], reverse=True)

    async def discover_related_accounts(self, seed_account: str) -> List[str]:
        """Discover related accounts based on interactions and following patterns."""
        await self._wait_for_rate_limit()
        # Mock implementation for testing
        return [
            f"related_account_{i}" for i in range(5)
        ]

    async def analyze_market_sentiment(self, tweets: List[Dict]) -> Dict:
        """Analyze market sentiment from a list of tweets."""
        sentiments = []
        for tweet in tweets:
            sentiment = await self.analyze_text_sentiment(tweet["text"])
            sentiments.append(sentiment)

        # Calculate average sentiment
        if not sentiments:
            return {"sentiment": "neutral", "score": 0.0}

        avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
        return {
            "sentiment": "bullish" if avg_score > 0.2 else "bearish" if avg_score < -0.2 else "neutral",
            "score": avg_score
        }
