"""Tests for Chinese platform integration."""
import pytest
from app.services.web_scraping.chinese_scraper import ChinesePlatformScraper
from app.services.web_scraping.sentiment_analyzer import ChineseSentimentAnalyzer
import json
from pathlib import Path
import asyncio

@pytest.fixture
def sentiment_analyzer():
    """Create sentiment analyzer instance."""
    return ChineseSentimentAnalyzer()

@pytest.fixture
def platform_scraper():
    """Create platform scraper instance."""
    return ChinesePlatformScraper()

def test_sentiment_analysis(sentiment_analyzer):
    """Test Chinese text sentiment analysis."""
    # Test bullish text
    bullish_text = "比特币突破新高，市场看多情绪强烈，建议建仓。"
    bullish_result = sentiment_analyzer.analyze_text(bullish_text)
    assert bullish_result["sentiment"] > 0
    assert bullish_result["confidence"] > 0.5
    assert bullish_result["crypto_relevance"] > 0.3

    # Test bearish text
    bearish_text = "市场风险加大，比特币可能回调，建议清仓。"
    bearish_result = sentiment_analyzer.analyze_text(bearish_text)
    assert bearish_result["sentiment"] < 0
    assert bearish_result["confidence"] > 0.5
    assert bearish_result["crypto_relevance"] > 0.3

    # Test neutral text
    neutral_text = "比特币价格波动，市场观望。"
    neutral_result = sentiment_analyzer.analyze_text(neutral_text)
    assert -0.3 <= neutral_result["sentiment"] <= 0.3
    assert neutral_result["confidence"] > 0.5
    assert neutral_result["crypto_relevance"] > 0.3

@pytest.mark.asyncio
async def test_platform_scraping(platform_scraper):
    """Test Chinese platform scraping."""
    # Test scraping with caching
    symbol = "BTC"
    result = await platform_scraper.get_market_sentiment(symbol)

    # Verify structure
    assert isinstance(result, dict)
    assert "xiaohongshu" in result
    assert "douyin" in result

    # Test rate limiting
    start_time = asyncio.get_event_loop().time()
    await platform_scraper.get_market_sentiment(symbol)
    end_time = asyncio.get_event_loop().time()

    # Should respect rate limit (at least 2 seconds between requests)
    assert end_time - start_time >= 2

@pytest.mark.asyncio
async def test_cache_mechanism(platform_scraper):
    """Test caching mechanism."""
    symbol = "ETH"

    # First request should create cache
    result1 = await platform_scraper.get_market_sentiment(symbol)

    # Second request should use cache
    start_time = asyncio.get_event_loop().time()
    result2 = await platform_scraper.get_market_sentiment(symbol)
    end_time = asyncio.get_event_loop().time()

    # Cache hit should be fast (less than 0.1 seconds)
    assert end_time - start_time < 0.1
    assert result1 == result2  # Results should be identical when using cache

@pytest.mark.asyncio
async def test_error_handling(platform_scraper):
    """Test error handling in scraping."""
    # Test with invalid symbol
    result = await platform_scraper.get_market_sentiment("INVALID_SYMBOL")

    # Should return empty lists for invalid data but not crash
    assert isinstance(result, dict)
    assert all(isinstance(posts, list) for posts in result.values())
    assert all(len(posts) == 0 for posts in result.values())
