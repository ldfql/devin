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
        """Load sentiment dictionary."""
        # Default sentiment dictionary with common terms
        default_dict = {
            "看多": 1.0,  # bullish
            "看空": -1.0,  # bearish
            "上涨": 0.8,  # rise
            "下跌": -0.8,  # fall
            "突破": 0.6,  # breakthrough
            "回调": -0.4,  # pullback
            "强势": 0.7,  # strong
            "弱势": -0.7,  # weak
            "建仓": 0.5,  # open position
            "清仓": -0.5,  # close position
            "机会": 0.6,  # opportunity
            "风险": -0.6,  # risk
            "稳定": 0.4,  # stable
            "波动": -0.3,  # volatile
            "利好": 0.8,  # positive news
            "利空": -0.8,  # negative news
        }

        dict_path = Path("data/sentiment_dict.json")
        if dict_path.exists():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    custom_dict = json.load(f)
                default_dict.update(custom_dict)
            except Exception as e:
                logger.error(f"Error loading custom sentiment dictionary: {e}")

        return default_dict

    def _load_crypto_keywords(self) -> List[str]:
        """Load cryptocurrency-related keywords."""
        default_keywords = [
            "比特币", "BTC", "以太坊", "ETH",
            "币圈", "区块链", "加密货币", "数字货币",
            "交易所", "挖矿", "持仓", "空投",
            "合约", "现货", "杠杆", "做多",
            "做空", "趋势", "行情", "币价"
        ]

        keywords_path = Path("data/crypto_keywords.json")
        if keywords_path.exists():
            try:
                with open(keywords_path, 'r', encoding='utf-8') as f:
                    custom_keywords = json.load(f)
                default_keywords.extend(custom_keywords)
            except Exception as e:
                logger.error(f"Error loading custom crypto keywords: {e}")

        return list(set(default_keywords))

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of Chinese text."""
        # Segment text using jieba
        words = jieba.lcut(text)

        # Calculate sentiment scores
        sentiment_score = 0.0
        relevant_words = 0

        for word in words:
            if word in self.sentiment_dict:
                sentiment_score += self.sentiment_dict[word]
                relevant_words += 1

        # Calculate crypto relevance
        crypto_relevance = sum(1 for keyword in self.crypto_keywords if keyword in text) / len(self.crypto_keywords)

        # Normalize sentiment score
        if relevant_words > 0:
            sentiment_score = sentiment_score / relevant_words

        # Calculate confidence based on relevance and word count
        confidence = min(0.85, (crypto_relevance + (relevant_words / len(words))) / 2)

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
