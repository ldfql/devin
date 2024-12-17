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
async def test_platform_scraping(platform_scraper):
    """Test Chinese platform scraping."""
    # Clear cache before testing
    cache_dir = Path("cache/chinese_platforms")
    if cache_dir.exists():
        for cache_file in cache_dir.glob("**/*.json"):
            cache_file.unlink()

    # Test scraping with fresh request
    symbol = "BTC"
    result = await platform_scraper.get_market_sentiment(symbol)

    # Verify structure
    assert isinstance(result, dict)
    assert "xiaohongshu" in result
    assert "douyin" in result

    # Test rate limiting by forcing a sleep
    await asyncio.sleep(0.1)  # Small delay for test stability
    start_time = asyncio.get_event_loop().time()

    # Create a task for the second request
    task = asyncio.create_task(platform_scraper.get_market_sentiment(symbol))

    # Wait to check if task is still running (should be due to rate limiting)
    await asyncio.sleep(8)  # Increased to 8 seconds to ensure task is still running
    assert not task.done(), "Request should be rate limited"

    # Wait for completion and verify timing
    await task
    end_time = asyncio.get_event_loop().time()
    assert end_time - start_time >= 10, "Rate limiting should enforce 10 second delay"

@pytest.mark.asyncio
async def test_cache_mechanism(platform_scraper):
    """Test caching mechanism."""
    symbol = "ETH"
    cache_dir = Path("cache/chinese_platforms")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create mock cache data
    mock_data = {"data": [{"content": "测试数据", "timestamp": "2024-03-16T12:00:00Z"}]}
    cache_path = cache_dir / "xiaohongshu_search" / "notes" / f"{symbol}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mock_data))

    # Request should use cache and be fast
    start_time = asyncio.get_event_loop().time()
    result = await platform_scraper.get_market_sentiment(symbol)
    end_time = asyncio.get_event_loop().time()

    assert end_time - start_time < 0.1  # Cache hit should be fast

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
