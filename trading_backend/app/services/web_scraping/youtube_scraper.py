"""YouTube channel scraper for trading insights."""
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime, timedelta
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .english_scraper import BaseEnglishScraper

logger = logging.getLogger(__name__)

class YouTubeScraper(BaseEnglishScraper):
    """YouTube channel scraper for trading insights."""

    TRADING_KEYWORDS = [
        "order flow", "drawdown", "mitigation", "trading strategy",
        "market analysis", "price action", "support resistance",
        "trading setup", "market structure", "risk management"
    ]

    def __init__(self, api_key: str):
        """Initialize YouTube scraper with API credentials."""
        super().__init__("youtube")
        try:
            self.youtube = build(
                "youtube", "v3",
                developerKey=api_key,
                cache_discovery=False
            )
            logger.info("YouTube scraper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube client: {e}")
            raise

    async def get_channel_id(self, username: str) -> Optional[str]:
        """Get channel ID from username."""
        cache_key = f"channel_id_{username}"
        cached_data = await self._get_cached_response(cache_key)
        if cached_data:
            return cached_data.get("channel_id")

        try:
            response = self.youtube.search().list(
                part="snippet",
                q=username,
                type="channel",
                maxResults=1
            ).execute()

            if response.get("items"):
                channel_id = response["items"][0]["snippet"]["channelId"]
                await self._cache_response(cache_key, {"channel_id": channel_id})
                return channel_id
            return None
        except HttpError as e:
            logger.error(f"Error getting channel ID for {username}: {e}")
            return None

    async def get_latest_videos(self, channel_id: str, max_results: int = 10) -> List[Dict]:
        """Get latest videos from a channel."""
        cache_key = f"videos_{channel_id}_{max_results}"
        cached_data = await self._get_cached_response(cache_key)
        if cached_data:
            return cached_data.get("videos", [])

        try:
            response = self.youtube.search().list(
                part="snippet",
                channelId=channel_id,
                order="date",
                type="video",
                maxResults=max_results
            ).execute()

            videos = []
            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                video_data = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=video_id
                ).execute()

                if video_data.get("items"):
                    video_info = video_data["items"][0]
                    videos.append({
                        "id": video_id,
                        "title": video_info["snippet"]["title"],
                        "description": video_info["snippet"]["description"],
                        "published_at": video_info["snippet"]["publishedAt"],
                        "view_count": video_info["statistics"].get("viewCount", 0),
                        "like_count": video_info["statistics"].get("likeCount", 0),
                        "duration": video_info["contentDetails"]["duration"]
                    })

            await self._cache_response(cache_key, {"videos": videos})
            return videos
        except HttpError as e:
            logger.error(f"Error getting videos for channel {channel_id}: {e}")
            return []

    def _extract_trading_insights(self, text: str) -> List[str]:
        """Extract trading-related insights from text."""
        insights = []
        text_lower = text.lower()

        for keyword in self.TRADING_KEYWORDS:
            if keyword in text_lower:
                # Find the sentence containing the keyword
                start = text_lower.find(keyword)
                # Get surrounding context (200 characters)
                context_start = max(0, start - 100)
                context_end = min(len(text), start + 100)
                context = text[context_start:context_end].strip()
                insights.append({
                    "keyword": keyword,
                    "context": context
                })

        return insights

    async def get_trading_insights(self, username: str) -> Dict[str, Any]:
        """Get trading insights from a channel's recent videos."""
        channel_id = await self.get_channel_id(username)
        if not channel_id:
            logger.error(f"Could not find channel ID for username: {username}")
            return {"username": username, "insights": []}

        videos = await self.get_latest_videos(channel_id)
        all_insights = []

        for video in videos:
            video_insights = self._extract_trading_insights(
                f"{video['title']} {video['description']}"
            )
            if video_insights:
                all_insights.append({
                    "video_id": video["id"],
                    "title": video["title"],
                    "published_at": video["published_at"],
                    "insights": video_insights
                })

        return {
            "username": username,
            "channel_id": channel_id,
            "total_videos_analyzed": len(videos),
            "insights": all_insights
        }

    async def discover_related_channels(self, seed_channel_id: str) -> List[Dict]:
        """Discover related trading channels."""
        cache_key = f"related_channels_{seed_channel_id}"
        cached_data = await self._get_cached_response(cache_key)
        if cached_data:
            return cached_data.get("channels", [])

        try:
            # Get channel's playlists
            playlists = self.youtube.playlists().list(
                part="snippet",
                channelId=seed_channel_id,
                maxResults=50
            ).execute()

            related_channels = set()
            for playlist in playlists.get("items", []):
                # Get videos in playlist
                playlist_items = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist["id"],
                    maxResults=50
                ).execute()

                # Extract unique channel IDs
                for item in playlist_items.get("items", []):
                    channel_id = item["snippet"]["videoOwnerChannelId"]
                    if channel_id != seed_channel_id:
                        related_channels.add(channel_id)

            # Get channel details
            channels_info = []
            for channel_id in related_channels:
                channel_data = self.youtube.channels().list(
                    part="snippet,statistics",
                    id=channel_id
                ).execute()

                if channel_data.get("items"):
                    channel = channel_data["items"][0]
                    channels_info.append({
                        "id": channel_id,
                        "title": channel["snippet"]["title"],
                        "description": channel["snippet"]["description"],
                        "subscriber_count": channel["statistics"].get("subscriberCount", 0),
                        "video_count": channel["statistics"].get("videoCount", 0)
                    })

            await self._cache_response(cache_key, {"channels": channels_info})
            return channels_info
        except HttpError as e:
            logger.error(f"Error discovering related channels: {e}")
            return []

    async def get_content(self, username: str) -> Dict[str, Any]:
        """Implement abstract method from BaseEnglishScraper."""
        return await self.get_trading_insights(username)
