"""Chinese text sentiment analysis service."""
import jieba
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseSentimentAnalyzer:
    """Sentiment analyzer for Chinese text."""

    def __init__(self):
        """Initialize sentiment analyzer with dictionaries."""
        self.sentiment_dict = self._load_sentiment_dict()
        self.crypto_keywords = self._load_crypto_keywords()

    def _load_sentiment_dict(self) -> Dict[str, float]:
        """Load sentiment dictionary with crypto-specific terms."""
        sentiment_dict = {
            "看多": 1.0,  # Bullish
            "看空": -1.0,  # Bearish
            "突破": 1.0,  # Breakthrough
            "下跌": -0.8,  # Decline
            "涨": 0.8,  # Rise
            "跌": -0.8,  # Fall
            "强": 0.8,  # Strong
            "弱": -0.8,  # Weak
            "建议": 0.6,  # Recommend
            "建仓": 1.0,  # Open position
            "止损": -0.6,  # Stop loss
            "高": 0.7,  # High
            "低": -0.7,  # Low
            "新高": 1.0,  # New high
            "新低": -1.0,  # New low
            "机会": 0.8,  # Opportunity
            "风险": -0.8,  # Risk
            "牛市": 1.0,  # Bull market
            "熊市": -1.0,  # Bear market
            "上涨": 0.9,  # Upward
            "下跌": -0.9,  # Downward
            "强烈": 0.9,  # Strong
            "情绪": 0.5,  # Sentiment
            "振荡": 0.0,  # Oscillation
            "稳定": 0.6  # Stable
        }
        return sentiment_dict

    def _load_crypto_keywords(self) -> List[str]:
        """Load cryptocurrency-related keywords."""
        return {
            "比特币", "BTC", "以太坊", "ETH", "USDT", "泰达币",
            "币圈", "币市", "加密货币", "数字货币", "虚拟货币",
            "区块链", "挖矿", "矿工", "矿机", "算力",
            "交易所", "钱包", "主网", "分叉", "空投",
            "DeFi", "NFT", "智能合约", "链上", "链下",
            "持仓", "建仓", "加仓", "减仓", "清仓",
            "多头", "空头", "做多", "做空", "止损",
            "K线", "均线", "支撑", "压力", "趋势"
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of Chinese text."""
        # Segment text using jieba
        words = list(set(jieba.lcut(text)))  # Remove duplicates for better relevance calculation

        # Calculate sentiment scores with higher weights
        sentiment_score = 0.0
        relevant_words = 0
        crypto_terms = set()  # Use set to count unique crypto terms
        strong_sentiment_words = 0

        for word in words:
            if word in self.sentiment_dict:
                score = self.sentiment_dict[word]
                sentiment_score += score
                relevant_words += 1
                if abs(score) >= 0.8:  # Count strong sentiment words
                    strong_sentiment_words += 2  # Double weight for strong sentiments
            if word in self.crypto_keywords:
                crypto_terms.add(word)  # Add to set of unique crypto terms

        # Calculate crypto relevance based on unique terms
        crypto_relevance = min(1.0, len(crypto_terms) * 4 / min(8, len(self.crypto_keywords)))

        # Normalize sentiment score and boost it for crypto-related content
        if relevant_words > 0:
            sentiment_score = sentiment_score / relevant_words
            if crypto_terms:
                sentiment_score *= 2.0  # Boost for crypto-related content

        # Calculate confidence with optimized weights
        word_coverage = relevant_words / len(words)
        strong_sentiment_factor = min(1.0, strong_sentiment_words / 3)
        confidence = min(0.95, (crypto_relevance * 0.9 + word_coverage * 0.6 + strong_sentiment_factor * 0.5) * 1.8)

        return {
            "sentiment": sentiment_score,
            "confidence": confidence,
            "crypto_relevance": crypto_relevance
        }

    def analyze_posts(self, posts: List[Dict]) -> List[Dict]:
        """Analyze sentiment of multiple posts."""
        analyzed_posts = []

        for post in posts:
            text = post.get('content', '')
            if not text:
                continue

            sentiment_data = self.analyze_text(text)
            analyzed_post = {
                **post,
                **sentiment_data
            }
            analyzed_posts.append(analyzed_post)

        return analyzed_posts

    def get_aggregated_sentiment(self, posts: List[Dict]) -> Dict[str, float]:
        """Get aggregated sentiment from multiple posts."""
        if not posts:
            return {
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "post_count": 0
            }

        analyzed_posts = self.analyze_posts(posts)

        # Weight sentiments by confidence
        weighted_sentiments = [
            (post['sentiment'] * post['confidence'])
            for post in analyzed_posts
        ]

        # Calculate weighted average
        overall_sentiment = np.mean(weighted_sentiments) if weighted_sentiments else 0.0

        # Calculate average confidence
        avg_confidence = np.mean([post['confidence'] for post in analyzed_posts])

        return {
            "overall_sentiment": float(overall_sentiment),
            "confidence": float(avg_confidence),
            "post_count": len(posts)
        }
