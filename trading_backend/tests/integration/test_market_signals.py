"""Integration tests for market signal collection and analysis."""
import pytest
import pytest_asyncio
from typing import Dict, List

from app.services.web_scraping.twitter_scraper import TwitterScraper
from app.services.web_scraping.youtube_scraper import YouTubeScraper
from app.services.web_scraping.english_sentiment import EnglishSentimentAnalyzer
from app.services.ocr.ocr_service import OCRService
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer

@pytest_asyncio.fixture
async def twitter_scraper():
    """Twitter scraper fixture."""
    return TwitterScraper(
        api_key="test_key",
        api_secret="test_secret",
        access_token="test_token",
        access_token_secret="test_token_secret"
    )

@pytest_asyncio.fixture
async def youtube_scraper():
    """YouTube scraper fixture."""
    return YouTubeScraper(api_key="test_key")

@pytest.fixture
def sentiment_analyzer():
    """Sentiment analyzer fixture."""
    return EnglishSentimentAnalyzer()

@pytest.fixture
def ocr_service():
    """OCR service fixture."""
    return OCRService()

@pytest.fixture
def market_analyzer():
    """Market analyzer fixture."""
    return MarketCycleAnalyzer()

class TestMarketSignalIntegration:
    """Integration tests for market signal collection and analysis."""

    @pytest.mark.asyncio
    async def test_twitter_sentiment_flow(self, twitter_scraper, sentiment_analyzer):
        """Test Twitter data collection and sentiment analysis flow."""
        # Get tweets from influential accounts
        tweets = await twitter_scraper.get_influential_tweets()
        assert len(tweets) > 0, "No tweets collected"

        # Analyze sentiment
        sentiments = []
        for tweet in tweets:
            sentiment = await sentiment_analyzer.analyze_text(tweet['text'])
            sentiments.append(sentiment)

        # Verify sentiment analysis results
        assert len(sentiments) == len(tweets), "Not all tweets analyzed"
        for sentiment in sentiments:
            assert 'sentiment' in sentiment, "Missing sentiment in analysis"
            assert 'confidence' in sentiment, "Missing confidence score"
            assert sentiment['confidence'] >= 0.85, "Confidence below threshold"

    @pytest.mark.asyncio
    async def test_youtube_sentiment_flow(self, youtube_scraper, sentiment_analyzer):
        """Test YouTube data collection and sentiment analysis flow."""
        # Get content from trading channels
        channels = ['daytradewarrior', 'innercircletrader']
        videos = []
        for channel in channels:
            channel_videos = await youtube_scraper.get_latest_videos(channel)
            videos.extend(channel_videos)

        assert len(videos) > 0, "No videos collected"

        # Analyze video titles and descriptions
        sentiments = []
        for video in videos:
            title_sentiment = await sentiment_analyzer.analyze_text(video['title'])
            desc_sentiment = await sentiment_analyzer.analyze_text(video['description'])
            sentiments.extend([title_sentiment, desc_sentiment])

        # Verify sentiment analysis results
        assert len(sentiments) > 0, "No sentiments analyzed"
        for sentiment in sentiments:
            assert sentiment['confidence'] >= 0.85, "Confidence below threshold"

    @pytest.mark.asyncio
    async def test_screenshot_analysis_flow(self, ocr_service):
        """Test screenshot analysis flow with test images."""
        # Test English screenshot
        with open('tests/data/test_screenshot_en.txt', 'r') as f:
            en_text = f.read()
            en_result = await ocr_service.extract_text_from_string(en_text)

        assert en_result['language'] == 'English'
        assert 'sentiment' in en_result
        assert en_result['confidence'] >= 0.85

        # Test Chinese screenshot
        with open('tests/data/test_screenshot_cn.txt', 'r') as f:
            cn_text = f.read()
            cn_result = await ocr_service.extract_text_from_string(cn_text)

        assert cn_result['language'] == 'Chinese'
        assert 'sentiment' in cn_result
        assert cn_result['confidence'] >= 0.85

    @pytest.mark.asyncio
    async def test_real_time_accuracy(
        self,
        twitter_scraper,
        youtube_scraper,
        sentiment_analyzer,
        market_analyzer
    ):
        """Test real-time accuracy of market analysis."""
        # Collect signals from all sources
        signals = []

        # Twitter signals
        tweets = await twitter_scraper.get_influential_tweets()
        for tweet in tweets:
            sentiment = await sentiment_analyzer.analyze_text(tweet['text'])
            signals.append({
                'source': 'twitter',
                'text': tweet['text'],
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence']
            })

        # YouTube signals
        channels = ['daytradewarrior', 'innercircletrader']
        for channel in channels:
            videos = await youtube_scraper.get_latest_videos(channel)
            for video in videos:
                title_sentiment = await sentiment_analyzer.analyze_text(video['title'])
                signals.append({
                    'source': 'youtube',
                    'text': video['title'],
                    'sentiment': title_sentiment['sentiment'],
                    'confidence': title_sentiment['confidence']
                })

        # Analyze market signals
        analysis = await market_analyzer.analyze_signals(signals)

        # Verify analysis results
        assert analysis['confidence'] >= 0.85, "Overall confidence below threshold"
        assert 'trend' in analysis, "Missing trend in analysis"
        assert 'cycle_phase' in analysis, "Missing cycle phase"
        assert analysis['sentiment_distribution'], "Missing sentiment distribution"

        # Verify accuracy tracking
        accuracies = []
        for signal in signals:
            accuracies.append(signal['confidence'])

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        assert avg_accuracy >= 0.85, "Average accuracy below threshold"
