"""Tests for English platform scraper base class."""
import pytest
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import aiohttp
from app.services.web_scraping.english_scraper import BaseEnglishScraper

class TestScraper(BaseEnglishScraper):
    """Test implementation of BaseEnglishScraper."""

    def __init__(self):
        """Initialize test scraper."""
        super().__init__("test")

    async def get_content(self, *args, **kwargs):
        """Test implementation of abstract method."""
        return await self._make_request(
            "https://api.example.com/test",
            params={"test": "data"}
        )

@pytest.fixture
def scraper():
    """Create test scraper instance."""
    return TestScraper()

@pytest.fixture
def mock_response():
    """Create mock response data."""
    return {"data": "test_content"}

@pytest.mark.asyncio
async def test_rate_limiting(scraper, mocker):
    """Test rate limiting functionality."""
    # Mock the _make_request method
    mock_request = mocker.patch.object(
        scraper,
        '_make_request',
        return_value={"data": "test"}
    )

    # Make multiple requests
    start_time = datetime.now()
    await asyncio.gather(
        scraper.get_content(),
        scraper.get_content(),
        scraper.get_content()
    )
    end_time = datetime.now()

    # Verify rate limiting
    time_diff = (end_time - start_time).total_seconds()
    assert time_diff >= 2.0  # Should wait at least 1 second between requests
    assert mock_request.call_count == 3

@pytest.mark.asyncio
async def test_caching(scraper, mock_response, mocker):
    """Test response caching."""
    # Mock aiohttp.ClientSession
    mock_session = mocker.AsyncMock()
    mock_session.request.return_value.__aenter__.return_value.json.return_value = mock_response
    mock_session.request.return_value.__aenter__.return_value.raise_for_status = mocker.AsyncMock()
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)

    # First request should hit the API
    response1 = await scraper.get_content()
    assert response1 == mock_response
    assert mock_session.request.call_count == 1

    # Second request should use cache
    response2 = await scraper.get_content()
    assert response2 == mock_response
    assert mock_session.request.call_count == 1  # No additional API call

@pytest.mark.asyncio
async def test_error_handling(scraper, mocker):
    """Test error handling."""
    # Mock aiohttp.ClientSession to raise an error
    mock_session = mocker.AsyncMock()
    mock_session.request.side_effect = aiohttp.ClientError("Test error")
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)

    with pytest.raises(aiohttp.ClientError):
        await scraper.get_content()

@pytest.mark.asyncio
async def test_platform_specific_rate_limits(scraper):
    """Test platform-specific rate limiting."""
    assert scraper._rate_limits['youtube'] == timedelta(seconds=2)
    assert scraper._rate_limits['twitter'] == timedelta(seconds=5)
    assert scraper._rate_limits['default'] == timedelta(seconds=1)

@pytest.mark.asyncio
async def test_cache_expiration(scraper, mock_response, mocker):
    """Test cache expiration."""
    # Mock aiohttp.ClientSession
    mock_session = mocker.AsyncMock()
    mock_session.request.return_value.__aenter__.return_value.json.return_value = mock_response
    mock_session.request.return_value.__aenter__.return_value.raise_for_status = mocker.AsyncMock()
    mocker.patch('aiohttp.ClientSession', return_value=mock_session)

    # First request
    await scraper.get_content()

    # Modify cache file timestamp to be old
    cache_key = scraper._get_cache_key("https://api.example.com/test", {"test": "data"})
    cache_file = scraper._cache_dir / f"{cache_key}.json"
    data = json.loads(cache_file.read_text())
    data['timestamp'] = (datetime.now() - timedelta(hours=2)).isoformat()
    cache_file.write_text(json.dumps(data))

    # Second request should hit API again due to expired cache
    await scraper.get_content()
    assert mock_session.request.call_count == 2
