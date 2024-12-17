import pytest
import asyncio
from app.services.web_scraping.twitter_scraper import TwitterScraper
from app.services.web_scraping.youtube_scraper import YouTubeScraper
from app.services.web_scraping.account_discovery import AccountDiscoveryService

@pytest.fixture
async def twitter_scraper():
    return TwitterScraper()

@pytest.fixture
async def youtube_scraper():
    return YouTubeScraper()

@pytest.fixture
async def account_discovery():
    return AccountDiscoveryService()

@pytest.mark.asyncio
async def test_twitter_expert_discovery(account_discovery):
    """Test discovering crypto experts on Twitter."""
    experts = await account_discovery.find_twitter_accounts(
        keywords=["crypto trading", "bitcoin analysis"],
        min_followers=10000,
        max_accounts=5
    )

    assert len(experts) > 0, "Should find crypto experts"
    for expert in experts:
        assert expert["followers_count"] >= 10000
        assert "crypto" in expert["description"].lower() or "trading" in expert["description"].lower()

@pytest.mark.asyncio
async def test_youtube_channel_discovery(account_discovery):
    """Test discovering crypto YouTube channels."""
    channels = await account_discovery.find_youtube_channels(
        keywords=["crypto trading", "bitcoin analysis"],
        min_subscribers=5000,
        max_channels=5
    )

    assert len(channels) > 0, "Should find crypto channels"
    for channel in channels:
        assert channel["subscriber_count"] >= 5000
        assert "crypto" in channel["description"].lower() or "trading" in channel["description"].lower()

@pytest.mark.asyncio
async def test_twitter_rate_limiting(twitter_scraper):
    """Test Twitter API rate limiting."""
    # Make multiple requests in quick succession
    requests = []
    for _ in range(5):
        requests.append(twitter_scraper.get_user_tweets("binance"))

    # Should not raise rate limit errors
    results = await asyncio.gather(*requests, return_exceptions=True)
    assert not any(isinstance(r, Exception) for r in results), "Rate limiting should prevent exceptions"

@pytest.mark.asyncio
async def test_youtube_rate_limiting(youtube_scraper):
    """Test YouTube API rate limiting."""
    # Make multiple requests in quick succession
    requests = []
    for _ in range(5):
        requests.append(youtube_scraper.get_channel_videos("UCWH7F2aQ2aFH5Uih1Ge9_SA"))

    # Should not raise rate limit errors
    results = await asyncio.gather(*requests, return_exceptions=True)
    assert not any(isinstance(r, Exception) for r in results), "Rate limiting should prevent exceptions"

@pytest.mark.asyncio
async def test_twitter_sentiment_tracking(twitter_scraper):
    """Test tracking Twitter sentiment over time."""
    # Get tweets from multiple crypto experts
    experts = ["binance", "cz_binance", "SBF_FTX"]
    sentiments = []

    for expert in experts:
        tweets = await twitter_scraper.get_user_tweets(expert, limit=10)
        for tweet in tweets:
            sentiment = await twitter_scraper.analyze_tweet_sentiment(tweet["text"])
            sentiments.append(sentiment)

    # Verify sentiment analysis
    assert len(sentiments) > 0, "Should analyze multiple tweets"
    for sentiment in sentiments:
        assert "label" in sentiment
        assert "confidence" in sentiment
        assert sentiment["confidence"] >= 0.85, "Sentiment confidence below threshold"

@pytest.mark.asyncio
async def test_youtube_content_analysis(youtube_scraper):
    """Test analyzing YouTube video content."""
    # Get videos from crypto channels
    channels = ["UCWH7F2aQ2aFH5Uih1Ge9_SA"]  # Example channel ID
    video_analyses = []

    for channel in channels:
        videos = await youtube_scraper.get_channel_videos(channel, limit=5)
        for video in videos:
            analysis = await youtube_scraper.analyze_video_content(video["id"])
            video_analyses.append(analysis)

    # Verify content analysis
    assert len(video_analyses) > 0, "Should analyze multiple videos"
    for analysis in video_analyses:
        assert "market_sentiment" in analysis
        assert "confidence" in analysis
        assert analysis["confidence"] >= 0.85, "Analysis confidence below threshold"

@pytest.mark.asyncio
async def test_expert_verification(account_discovery):
    """Test verification of discovered experts."""
    experts = await account_discovery.find_twitter_accounts(
        keywords=["crypto trading"],
        min_followers=50000,
        max_accounts=3
    )

    for expert in experts:
        verification = await account_discovery.verify_expert(expert["id"])
        assert verification["is_verified"], f"Expert {expert['username']} should be verified"
        assert verification["credibility_score"] >= 0.85, "Expert credibility below threshold"

@pytest.mark.asyncio
async def test_content_relevance_filtering(twitter_scraper, youtube_scraper):
    """Test filtering relevant content from social media."""
    # Test Twitter content filtering
    tweets = await twitter_scraper.get_filtered_tweets(
        keywords=["bitcoin", "trading"],
        min_engagement=100,
        max_tweets=10
    )

    assert len(tweets) > 0, "Should find relevant tweets"
    for tweet in tweets:
        assert tweet["relevance_score"] >= 0.85, "Tweet relevance below threshold"

    # Test YouTube content filtering
    videos = await youtube_scraper.get_filtered_videos(
        keywords=["crypto analysis"],
        min_views=1000,
        max_videos=10
    )

    assert len(videos) > 0, "Should find relevant videos"
    for video in videos:
        assert video["relevance_score"] >= 0.85, "Video relevance below threshold"
