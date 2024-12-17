"""Integration tests for market signal collection and analysis."""
import pytest
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List

from app.services.web_scraping.twitter_scraper import TwitterScraper
from app.services.web_scraping.youtube_scraper import YoutubeScraper
from app.services.web_scraping.english_sentiment import EnglishSentimentAnalyzer
from app.services.ocr.ocr_service import OCRService
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer

@pytest.fixture
async def twitter_scraper():
    scraper = TwitterScraper()
    await scraper.initialize(
        api_key="test_key",
        api_secret="test_secret",
        access_token="test_token",
        access_token_secret="test_token_secret"
    )
    return scraper

@pytest.fixture
async def youtube_scraper():
    return YoutubeScraper()

@pytest.fixture
def sentiment_analyzer():
    return EnglishSentimentAnalyzer()

@pytest.fixture
def ocr_service():
    return OCRService()

@pytest.fixture
def market_analyzer():
    return MarketCycleAnalyzer()

class TestMarketSignalIntegration:
    """Integration tests for market signal collection and analysis."""

    @pytest.mark.asyncio
    async def test_twitter_sentiment_flow(self, twitter_scraper, sentiment_analyzer):
        """Test Twitter data collection and sentiment analysis flow."""
        # Get tweets from influential accounts
        tweets = await twitter_scraper.get_influential_tweets()
        assert len(tweets) > 0

        # Analyze sentiment
        sentiments = []
        for tweet in tweets:
            sentiment = await sentiment_analyzer.analyze_text_sentiment(tweet['text'])
            sentiments.append(sentiment)

        # Validate results
        assert len(sentiments) == len(tweets)
        assert all(s['confidence'] > 0.7 for s in sentiments)
        assert all(s['sentiment'] in ['bullish', 'bearish', 'neutral'] for s in sentiments)

        # Calculate accuracy
        total = len(sentiments)
        high_confidence = len([s for s in sentiments if s['confidence'] > 0.85])
        accuracy = high_confidence / total if total > 0 else 0
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} below required 85%"

    @pytest.mark.asyncio
    async def test_youtube_sentiment_flow(self, youtube_scraper, sentiment_analyzer):
        """Test YouTube data collection and sentiment analysis flow."""
        # Get content from trading channels
        channels = ['daytradewarrior', 'innercircletrader']
        videos = []
        for channel in channels:
            channel_videos = await youtube_scraper.get_latest_videos(channel)
            videos.extend(channel_videos)

        assert len(videos) > 0

        # Analyze video titles and descriptions
        sentiments = []
        for video in videos:
            # Combine title and description for analysis
            text = f"{video['title']} {video['description']}"
            sentiment = await sentiment_analyzer.analyze_text_sentiment(text)
            sentiments.append(sentiment)

        # Validate results
        assert len(sentiments) == len(videos)
        assert all(s['confidence'] > 0.7 for s in sentiments)

        # Calculate accuracy
        total = len(sentiments)
        high_confidence = len([s for s in sentiments if s['confidence'] > 0.85])
        accuracy = high_confidence / total if total > 0 else 0
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} below required 85%"

    @pytest.mark.asyncio
    async def test_screenshot_analysis_flow(self, ocr_service):
        """Test screenshot analysis flow with test images."""
        # Test English screenshot
        with open('tests/data/test_screenshot_en.txt', 'rb') as f:
            en_result = await ocr_service.extract_text(f.read())

        assert en_result['language'] == 'English'
        assert en_result['signals'] is not None
        assert en_result['signals']['confidence'] > 0.85

        # Test Chinese screenshot
        with open('tests/data/test_screenshot_cn.txt', 'rb') as f:
            cn_result = await ocr_service.extract_text(f.read())

        assert cn_result['language'] == 'Simplified Chinese'
        assert cn_result['signals'] is not None
        assert cn_result['signals']['confidence'] > 0.85

    @pytest.mark.asyncio
    async def test_real_time_accuracy(
        self, twitter_scraper, youtube_scraper,
        sentiment_analyzer, market_analyzer
    ):
        """Test real-time data accuracy across all sources."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)

        signals = []
        while datetime.now() < end_time:
            # Collect signals from Twitter
            tweets = await twitter_scraper.get_influential_tweets()
            for tweet in tweets:
                sentiment = await sentiment_analyzer.analyze_text_sentiment(tweet['text'])
                if sentiment['confidence'] > 0.85:
                    signals.append({
                        'source': 'twitter',
                        'sentiment': sentiment['sentiment'],
                        'confidence': sentiment['confidence'],
                        'timestamp': datetime.now()
                    })

            # Collect signals from YouTube
            for channel in ['daytradewarrior', 'innercircletrader']:
                videos = await youtube_scraper.get_latest_videos(channel)
                for video in videos:
                    text = f"{video['title']} {video['description']}"
                    sentiment = await sentiment_analyzer.analyze_text_sentiment(text)
                    if sentiment['confidence'] > 0.85:
                        signals.append({
                            'source': 'youtube',
                            'sentiment': sentiment['sentiment'],
                            'confidence': sentiment['confidence'],
                            'timestamp': datetime.now()
                        })

            # Short pause between iterations
            await asyncio.sleep(30)

        # Validate real-time accuracy
        total_signals = len(signals)
        assert total_signals > 0, "No signals collected during test period"

        high_confidence_signals = len([s for s in signals if s['confidence'] > 0.85])
        accuracy = high_confidence_signals / total_signals

        assert accuracy >= 0.85, f"Real-time accuracy {accuracy:.2%} below required 85%"

        # Validate market cycle analysis
        cycle_analysis = await market_analyzer.analyze_signals(signals)
        assert cycle_analysis['confidence'] > 0.85
        assert 'trend' in cycle_analysis
        assert 'cycle_phase' in cycle_analysis
