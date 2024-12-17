"""Tests for YouTube scraper."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from app.services.web_scraping.youtube_scraper import YouTubeScraper

@pytest.fixture
def mock_youtube_client():
    """Create mock YouTube client."""
    return MagicMock()

@pytest.fixture
def scraper(mock_youtube_client):
    """Create YouTubeScraper instance with mocked client."""
    with patch('app.services.web_scraping.youtube_scraper.build') as mock_build:
        mock_build.return_value = mock_youtube_client
        scraper = YouTubeScraper()
        scraper.youtube = mock_youtube_client
        return scraper

@pytest.fixture
def mock_video_response():
    """Create mock video response."""
    return {
        "items": [
            {
                "id": {"videoId": "test_video_1"},
                "snippet": {
                    "title": "Trading Strategy: Order Flow Analysis",
                    "description": "Learn about drawdown mitigation techniques",
                    "publishedAt": "2024-01-20T10:00:00Z"
                }
            }
        ]
    }

@pytest.fixture
def mock_video_details():
    """Create mock video details."""
    return {
        "items": [
            {
                "id": "test_video_1",
                "snippet": {
                    "title": "Trading Strategy: Order Flow Analysis",
                    "description": "Learn about drawdown mitigation techniques",
                    "publishedAt": "2024-01-20T10:00:00Z"
                },
                "statistics": {
                    "viewCount": "1000",
                    "likeCount": "100"
                },
                "contentDetails": {
                    "duration": "PT15M33S"
                }
            }
        ]
    }

@pytest.mark.asyncio
async def test_get_channel_id(scraper, mock_youtube_client):
    """Test getting channel ID from username."""
    mock_youtube_client.search().list().execute.return_value = {
        "items": [
            {
                "snippet": {
                    "channelId": "test_channel_id"
                }
            }
        ]
    }

    channel_id = await scraper.get_channel_id("daytradewarrior")
    assert channel_id == "test_channel_id"
    mock_youtube_client.search().list.assert_called_with(
        part="snippet",
        q="daytradewarrior",
        type="channel",
        maxResults=1
    )

@pytest.mark.asyncio
async def test_get_latest_videos(scraper, mock_youtube_client, mock_video_response, mock_video_details):
    """Test getting latest videos from channel."""
    mock_youtube_client.search().list().execute.return_value = mock_video_response
    mock_youtube_client.videos().list().execute.return_value = mock_video_details

    videos = await scraper.get_latest_videos("test_channel_id")
    assert len(videos) == 1
    assert videos[0]["id"] == "test_video_1"
    assert "order flow" in videos[0]["title"].lower()

@pytest.mark.asyncio
async def test_extract_trading_insights(scraper):
    """Test extracting trading insights from text."""
    text = "This video covers order flow analysis and drawdown mitigation strategies"
    insights = scraper._extract_trading_insights(text)

    assert len(insights) == 2
    assert any(insight["keyword"] == "order flow" for insight in insights)
    assert any(insight["keyword"] == "drawdown" for insight in insights)

@pytest.mark.asyncio
async def test_get_trading_insights(scraper, mock_youtube_client, mock_video_response, mock_video_details):
    """Test getting trading insights from channel."""
    mock_youtube_client.search().list().execute.side_effect = [
        {"items": [{"snippet": {"channelId": "test_channel_id"}}]},
        mock_video_response
    ]
    mock_youtube_client.videos().list().execute.return_value = mock_video_details

    insights = await scraper.get_trading_insights("daytradewarrior")
    assert insights["username"] == "daytradewarrior"
    assert insights["channel_id"] == "test_channel_id"
    assert len(insights["insights"]) > 0

@pytest.mark.asyncio
async def test_discover_related_channels(scraper, mock_youtube_client):
    """Test discovering related channels."""
    mock_youtube_client.playlists().list().execute.return_value = {
        "items": [{"id": "playlist_1"}]
    }
    mock_youtube_client.playlistItems().list().execute.return_value = {
        "items": [{"snippet": {"videoOwnerChannelId": "related_channel_1"}}]
    }
    mock_youtube_client.channels().list().execute.return_value = {
        "items": [{
            "snippet": {
                "title": "Related Trading Channel",
                "description": "Trading strategies and analysis"
            },
            "statistics": {
                "subscriberCount": "10000",
                "videoCount": "100"
            }
        }]
    }

    related_channels = await scraper.discover_related_channels("test_channel_id")
    assert len(related_channels) == 1
    assert related_channels[0]["id"] == "related_channel_1"
