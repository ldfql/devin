"""Mock API responses for integration tests."""
from typing import Dict, List
import json
from datetime import datetime, timedelta

MOCK_TWITTER_RESPONSES = {
    'influential_tweets': [
        {
            'id': '1234567890',
            'text': 'BTC showing strong bullish signals with golden cross forming. Support at $45k looking solid. Multiple technical patterns confirm uptrend. #Bitcoin #Trading',
            'created_at': '2024-01-15T10:00:00Z',
            'user': {
                'id': '98765432',
                'username': 'cryptoanalyst',
                'followers_count': 50000
            }
        },
        {
            'id': '1234567891',
            'text': 'Clear head and shoulders pattern on ETH/USD 4h chart. Resistance at $2,800 crucial. Volume profile suggests distribution. #Ethereum #Crypto',
            'created_at': '2024-01-15T10:05:00Z',
            'user': {
                'id': '98765433',
                'username': 'tradingexpert',
                'followers_count': 75000
            }
        },
        {
            'id': '1234567892',
            'text': '比特币突破关键阻力位，上升三角形形态确认。目标价位50000美元。 #比特币 #加密货币',
            'created_at': '2024-01-15T10:10:00Z',
            'user': {
                'id': '98765434',
                'username': 'crypto_cn',
                'followers_count': 60000
            }
        }
    ]
}

MOCK_YOUTUBE_RESPONSES = {
    'channel_videos': {
        'daytradewarrior': [
            {
                'id': 'video123',
                'title': 'Bitcoin Bullish Breakout Imminent - Technical Analysis',
                'description': 'In this video, we analyze the current BTC setup showing multiple bullish signals including golden cross and higher lows. Support at $45,000 holding strong with increasing buy pressure.',
                'published_at': '2024-01-15T09:00:00Z',
                'view_count': 15000,
                'likes': 1200,
                'comments': 300
            }
        ],
        'innercircletrader': [
            {
                'id': 'video456',
                'title': 'Major Support Level Holding - BTC Analysis',
                'description': 'Analysis of current market structure showing strong support at $45,000 with increasing buy pressure. Multiple technical indicators suggesting bullish continuation.',
                'published_at': '2024-01-15T08:30:00Z',
                'view_count': 20000,
                'likes': 1500,
                'comments': 400
            }
        ]
    }
}

MOCK_MARKET_DATA = {
    'BTC/USD': [
        {'timestamp': '2024-01-15T10:00:00Z', 'price': 45000, 'volume': 1000},
        {'timestamp': '2024-01-15T10:05:00Z', 'price': 45100, 'volume': 1200},
        {'timestamp': '2024-01-15T10:10:00Z', 'price': 45200, 'volume': 1500}
    ],
    'ETH/USD': [
        {'timestamp': '2024-01-15T10:00:00Z', 'price': 2800, 'volume': 800},
        {'timestamp': '2024-01-15T10:05:00Z', 'price': 2790, 'volume': 900},
        {'timestamp': '2024-01-15T10:10:00Z', 'price': 2780, 'volume': 1100}
    ]
}

def get_mock_twitter_response(endpoint: str) -> Dict:
    """Get mock Twitter API response."""
    return MOCK_TWITTER_RESPONSES.get(endpoint, {})

def get_mock_youtube_response(channel: str) -> List[Dict]:
    """Get mock YouTube API response."""
    return MOCK_YOUTUBE_RESPONSES['channel_videos'].get(channel, [])

def get_mock_market_data(symbol: str, timeframe: str = '5m') -> List[Dict]:
    """Get mock market price data."""
    return MOCK_MARKET_DATA.get(symbol, [])
