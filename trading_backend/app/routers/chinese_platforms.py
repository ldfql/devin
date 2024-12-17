"""Router for Chinese platform integration."""
from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging
from ..services.web_scraping.chinese_scraper import ChinesePlatformScraper
from ..services.web_scraping.sentiment_analyzer import ChineseSentimentAnalyzer

router = APIRouter(prefix="/api/chinese-platforms", tags=["chinese-platforms"])
logger = logging.getLogger(__name__)

scraper = ChinesePlatformScraper()
analyzer = ChineseSentimentAnalyzer()

@router.get("/sentiment/{symbol}")
async def get_market_sentiment(symbol: str) -> Dict:
    """Get market sentiment for a cryptocurrency symbol."""
    try:
        # Get posts from Chinese platforms
        platform_data = await scraper.get_market_sentiment(symbol)

        # Analyze sentiment for each platform
        sentiment_data = {}
        for platform, posts in platform_data.items():
            sentiment = analyzer.get_aggregated_sentiment(posts)
            sentiment_data[platform] = sentiment

        # Calculate overall sentiment across platforms
        all_sentiments = [
            data['overall_sentiment'] * data['confidence']
            for data in sentiment_data.values()
            if data['post_count'] > 0
        ]

        if all_sentiments:
            overall_sentiment = sum(all_sentiments) / len(all_sentiments)
        else:
            overall_sentiment = 0.0

        return {
            "symbol": symbol,
            "platform_sentiments": sentiment_data,
            "overall_sentiment": overall_sentiment
        }

    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing market sentiment: {str(e)}"
        )
