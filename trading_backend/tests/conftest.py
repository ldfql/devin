"""Test configuration and fixtures."""
import pytest
import asyncio
from unittest.mock import patch
from pathlib import Path
import json

from .data.mock_responses import (
    get_mock_twitter_response,
    get_mock_youtube_response,
    get_mock_market_data
)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_twitter_api():
    """Mock Twitter API responses."""
    with patch('app.services.web_scraping.twitter_scraper.TwitterScraper._make_request') as mock:
        mock.side_effect = get_mock_twitter_response
        yield mock

@pytest.fixture
def mock_youtube_api():
    """Mock YouTube API responses."""
    with patch('app.services.web_scraping.youtube_scraper.YoutubeScraper._get_channel_videos') as mock:
        mock.side_effect = get_mock_youtube_response
        yield mock

@pytest.fixture
def mock_market_api():
    """Mock market data API responses."""
    with patch('app.services.market_analysis.market_cycle.MarketCycleAnalyzer._get_market_data') as mock:
        mock.side_effect = get_mock_market_data
        yield mock

@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / 'data'

@pytest.fixture
def sample_signals():
    """Get sample market signals data."""
    return {
        'twitter': [
            {
                'source': 'twitter',
                'text': 'BTC showing strong bullish signals with golden cross forming.',
                'sentiment': 'bullish',
                'confidence': 0.92,
                'timestamp': '2024-01-15T10:00:00Z'
            }
        ],
        'youtube': [
            {
                'source': 'youtube',
                'text': 'Bitcoin Bullish Breakout Imminent - Technical Analysis',
                'sentiment': 'bullish',
                'confidence': 0.88,
                'timestamp': '2024-01-15T09:00:00Z'
            }
        ],
        'screenshot': [
            {
                'source': 'screenshot',
                'text': 'Clear bullish flag pattern forming',
                'sentiment': 'bullish',
                'confidence': 0.95,
                'timestamp': '2024-01-15T10:05:00Z'
            }
        ]
    }
