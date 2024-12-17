"""Tests for Twitter scraper functionality."""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.services.web_scraping.twitter_scraper import TwitterScraper

@pytest.fixture
def twitter_scraper():
    """Create TwitterScraper instance for testing."""
    with patch.dict('os.environ', {'TWITTER_BEARER_TOKEN': 'test_token'}):
        return TwitterScraper()

@pytest.mark.asyncio
async def test_get_user_tweets(twitter_scraper):
    """Test retrieving tweets from a specific user."""
    content = await twitter_scraper.get_content(content_type="tweets", username="elonmusk")
    tweets = content["tweets"]
    assert len(tweets) > 0
    assert all(isinstance(t, dict) for t in tweets)
    assert all(k in tweets[0] for k in ["id", "text", "created_at"])

@pytest.mark.asyncio
async def test_get_influential_tweets(twitter_scraper):
    """Test retrieving tweets from influential accounts."""
    content = await twitter_scraper.get_content(content_type="influential")
    tweets = content["tweets"]
    assert len(tweets) > 0
    assert all(isinstance(t, dict) for t in tweets)

@pytest.mark.asyncio
async def test_get_financial_news(twitter_scraper):
    """Test retrieving tweets from financial news accounts."""
    content = await twitter_scraper.get_content(content_type="news")
    news = content["tweets"]
    assert len(news) > 0
    assert all(isinstance(n, dict) for n in news)

@pytest.mark.asyncio
async def test_discover_related_accounts(twitter_scraper):
    """Test discovering related accounts."""
    related = await twitter_scraper.discover_related_accounts("elonmusk")
    assert len(related) > 0
    assert all(isinstance(a, str) for a in related)

@pytest.mark.asyncio
async def test_analyze_market_sentiment(twitter_scraper):
    """Test market sentiment analysis from tweets."""
    tweets = [
        {"text": "Bitcoin is going to the moon! 🚀", "created_at": datetime.now().isoformat()},
        {"text": "Bearish on crypto markets today", "created_at": datetime.now().isoformat()},
    ]
    sentiment = await twitter_scraper.analyze_market_sentiment(tweets)
    assert "sentiment" in sentiment
    assert "score" in sentiment
    assert sentiment["sentiment"] in ["bullish", "bearish", "neutral"]
    assert isinstance(sentiment["score"], float)

@pytest.mark.asyncio
async def test_rate_limiting(twitter_scraper):
    """Test rate limiting functionality."""
    for _ in range(5):
        await twitter_scraper.get_content(content_type="tweets", username="test_user")
    # Should not raise rate limit exception
    assert True

@pytest.mark.asyncio
async def test_error_handling(twitter_scraper):
    """Test error handling for API failures."""
    with patch.object(twitter_scraper, '_wait_for_rate_limit', side_effect=Exception("API Error")):
        with pytest.raises(Exception):
            await twitter_scraper.get_content(content_type="tweets", username="nonexistent_user")
