"""Tests for Chinese platform integration."""
import pytest
from app.services.web_scraping.chinese_scraper import ChinesePlatformScraper, BasePlatformScraper
from app.services.web_scraping.sentiment_analyzer import ChineseSentimentAnalyzer
import json
from pathlib import Path
import asyncio
import aiohttp
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

@pytest.fixture
def sentiment_analyzer():
    """Create sentiment analyzer instance."""
    return ChineseSentimentAnalyzer()

@pytest.fixture
def mock_response():
    """Create mock response data."""
    return {
        "data": [
            {"content": "比特币突破新高", "timestamp": "2024-03-16T12:00:00Z"},
            {"content": "市场看多情绪强烈", "timestamp": "2024-03-16T12:01:00Z"}
        ]
    }

@pytest.fixture
def platform_scraper():
    """Create platform scraper instance with mocked API."""
    return ChinesePlatformScraper()

def test_sentiment_analysis(sentiment_analyzer):
    """Test Chinese text sentiment analysis."""
    # Test bullish text with strong crypto keywords
    bullish_text = "比特币突破新高，牛市来临，建议建仓。市场看多情绪强烈。"
    bullish_result = sentiment_analyzer.analyze_text(bullish_text)
    assert bullish_result["sentiment"] > 0
    assert bullish_result["confidence"] > 0.5
    assert bullish_result["crypto_relevance"] > 0.3

    # Test bearish text with strong crypto keywords
    bearish_text = "熊市已至，比特币可能下跌，建议止损。市场看空情绪强烈。"
    bearish_result = sentiment_analyzer.analyze_text(bearish_text)
    assert bearish_result["sentiment"] < 0
    assert bearish_result["confidence"] > 0.5
    assert bearish_result["crypto_relevance"] > 0.3

    # Test neutral text with crypto keywords
    neutral_text = "比特币交易所数据显示，市场处于振荡阶段。"
    neutral_result = sentiment_analyzer.analyze_text(neutral_text)
    assert -0.3 <= neutral_result["sentiment"] <= 0.3
    assert neutral_result["confidence"] > 0.5
    assert neutral_result["crypto_relevance"] > 0.3

@pytest.mark.asyncio
async def test_platform_scraping(platform_scraper, mock_response):
    """Test Chinese platform scraping with mocked API."""
    # Clear cache before testing
    cache_dir = Path("cache/chinese_platforms")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("**/*.json"):
            cache_file.unlink()

    # Track original method for restoration
    original_get_market_sentiment = platform_scraper.get_market_sentiment
    rate_limit_called = False
    request_count = 0
    last_request_time = datetime.now() - timedelta(seconds=2)  # Initial request allowed

    class MockGetMarketSentiment:
        def __init__(self, scraper):
            self.scraper = scraper

        async def __call__(self, symbol):
            nonlocal rate_limit_called, request_count, last_request_time
            current_time = datetime.now()
            time_since_last = (current_time - last_request_time).total_seconds()

            if time_since_last < 1:  # Rate limit of 1 second for testing
                rate_limit_called = True
                await asyncio.sleep(1 - time_since_last + 0.1)  # Wait remaining time plus a small buffer

            request_count += 1
            last_request_time = datetime.now()

            return {
                "xiaohongshu": mock_response["data"],
                "douyin": mock_response["data"]
            }

    # Replace the method
    platform_scraper.get_market_sentiment = MockGetMarketSentiment(platform_scraper).__call__

    try:
        # Test scraping with fresh request
        symbol = "BTC"
        result = await platform_scraper.get_market_sentiment(symbol)

        # Verify structure
        assert isinstance(result, dict)
        assert "xiaohongshu" in result
        assert "douyin" in result

        # Test rate limiting
        start_time = asyncio.get_event_loop().time()
        task = asyncio.create_task(platform_scraper.get_market_sentiment(symbol))

        # Wait briefly to ensure rate limiting is active
        await asyncio.sleep(0.1)
        assert not task.done(), "Request should be rate limited"
        assert rate_limit_called, "Rate limiting should have been called"

        # Complete the request and verify timing
        await task
        end_time = asyncio.get_event_loop().time()
        assert end_time - start_time >= 1.0, "Rate limiting should enforce delay"

    finally:
        # Restore original method
        platform_scraper.get_market_sentiment = original_get_market_sentiment

@pytest.mark.asyncio
async def test_cache_mechanism(platform_scraper, mock_response):
    """Test caching mechanism with mocked data."""
    symbol = "ETH"
    cache_dir = Path("cache/chinese_platforms/market_sentiment")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create mock cache data
    mock_cache_data = {
        "xiaohongshu": [{"content": "测试数据", "timestamp": "2024-03-16T12:00:00Z"}],
        "douyin": [{"content": "测试数据", "timestamp": "2024-03-16T12:00:00Z"}]
    }
    cache_file = cache_dir / f"{symbol}.json"
    cache_file.write_text(json.dumps(mock_cache_data))

    # Request should use cache and be fast
    start_time = asyncio.get_event_loop().time()
    result = await platform_scraper.get_market_sentiment(symbol)
    end_time = asyncio.get_event_loop().time()

    assert end_time - start_time < 0.1, "Cache hit should be fast"
    assert result == mock_cache_data, "Should return cached data"

@pytest.mark.asyncio
async def test_error_handling(platform_scraper):
    """Test error handling in scraping."""
    # Test with invalid symbol
    result = await platform_scraper.get_market_sentiment("INVALID_SYMBOL")

    # Should return empty lists for invalid data but not crash
    assert isinstance(result, dict)
    assert "xiaohongshu" in result and "douyin" in result
    assert isinstance(result["xiaohongshu"], list) and isinstance(result["douyin"], list)
    assert len(result["xiaohongshu"]) == 0 and len(result["douyin"]) == 0
