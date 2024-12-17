"""Signal extractor for OCR text analysis."""
import re
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalExtractor:
    """Extract trading signals from OCR text."""

    def __init__(self):
        """Initialize signal patterns."""
        # Technical patterns in English
        self.en_patterns = {
            'bullish': [
                r'bull(?:ish)?\s*flag',
                r'double\s*bottom',
                r'inverse\s*head\s*and\s*shoulders?',
                r'golden\s*cross',
                r'(higher|ascending)\s*triangle',
                r'cup\s*and\s*handle',
                r'breakout',
                r'support\s*(?:level|zone|line)',
                r'uptrend\s*(?:line|channel)',
                r'accumulation',
            ],
            'bearish': [
                r'bear(?:ish)?\s*flag',
                r'double\s*top',
                r'head\s*and\s*shoulders?',
                r'death\s*cross',
                r'(descending|falling)\s*triangle',
                r'breakdown',
                r'resistance\s*(?:level|zone|line)',
                r'downtrend\s*(?:line|channel)',
                r'distribution',
            ]
        }

        # Technical patterns in Chinese
        self.cn_patterns = {
            'bullish': [
                r'多头旗形',
                r'双底形态',
                r'反头肩形态',
                r'黄金交叉',
                r'上升三角形',
                r'杯柄形态',
                r'突破',
                r'支撑位',
                r'上升趋势',
                r'筑底',
            ],
            'bearish': [
                r'空头旗形',
                r'双顶形态',
                r'头肩形态',
                r'死亡交叉',
                r'下降三角形',
                r'跌破',
                r'阻力位',
                r'下降趋势',
                r'出货',
            ]
        }

        # Price level patterns
        self.price_patterns = {
            'en': r'(?:price|level|target|stop)\s*(?:at|:)?\s*(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            'cn': r'(?:价格|目标|止损)\s*(?:位|点|在)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        }

        # Cryptocurrency patterns
        self.crypto_patterns = {
            'en': [
                r'(?:btc|bitcoin|eth|ethereum|bnb|sol|xrp)',
                r'(?:doge|ada|dot|link|uni|aave)',
            ],
            'cn': [
                r'(?:比特币|以太坊|币安币|瑞波币)',
                r'(?:狗狗币|艾达币|波卡币|链链|悠尼币)',
            ]
        }

    def extract_signals(self, text: str, lang: str = 'en') -> Dict:
        """Extract trading signals from text."""
        try:
            # Normalize text
            text = text.lower().strip()

            # Select patterns based on language
            patterns = self.en_patterns if lang == 'en' else self.cn_patterns
            price_pattern = self.price_patterns['en'] if lang == 'en' else self.price_patterns['cn']
            crypto_patterns = self.crypto_patterns['en'] if lang == 'en' else self.crypto_patterns['cn']

            # Extract patterns
            bullish_matches = self._find_patterns(text, patterns['bullish'])
            bearish_matches = self._find_patterns(text, patterns['bearish'])
            prices = self._extract_prices(text, price_pattern)
            cryptos = self._extract_cryptocurrencies(text, crypto_patterns)

            # Calculate confidence based on pattern matches
            signal_strength = len(bullish_matches) - len(bearish_matches)
            confidence = min(0.95, (abs(signal_strength) * 0.2 + 0.3))

            # Determine overall sentiment
            if signal_strength > 1:
                sentiment = 'bullish'
            elif signal_strength < -1:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
                confidence = min(0.7, confidence)  # Lower confidence for neutral signals

            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'patterns': {
                    'bullish': bullish_matches,
                    'bearish': bearish_matches
                },
                'prices': prices,
                'cryptocurrencies': cryptos,
                'timestamp': datetime.now().isoformat()
            }


        except Exception as e:
            logger.error(f"Error extracting signals: {str(e)}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'patterns': {'bullish': [], 'bearish': []},
                'prices': [],
                'cryptocurrencies': [],
                'timestamp': datetime.now().isoformat()
            }

    def _find_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matching patterns in text."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(pattern)
        return matches

    def _extract_prices(self, text: str, pattern: str) -> List[float]:
        """Extract price levels from text."""
        matches = re.finditer(pattern, text)
        prices = []
        for match in matches:
            try:
                price_str = match.group(1).replace('$', '').replace(',', '')
                price = float(price_str)
                if 0 < price < 1000000:  # Basic validation
                    prices.append(price)
            except (ValueError, IndexError):
                continue
        return prices

    def _extract_cryptocurrencies(self, text: str, patterns: List[str]) -> List[str]:
        """Extract mentioned cryptocurrencies."""
        cryptos = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                cryptos.add(match.group())
        return list(cryptos)

    def get_confidence_explanation(self, result: Dict) -> str:
        """Get human-readable explanation of confidence score."""
        if not result or 'confidence' not in result:
            return "No confidence score available"

        confidence = result['confidence']
        patterns = result.get('patterns', {})
        bull_count = len(patterns.get('bullish', []))
        bear_count = len(patterns.get('bearish', []))

        if confidence > 0.8:
            return f"High confidence ({confidence:.2f}) based on {bull_count + bear_count} clear technical patterns"
        elif confidence > 0.5:
            return f"Moderate confidence ({confidence:.2f}) with mixed signals"
        else:
            return f"Low confidence ({confidence:.2f}) due to insufficient clear patterns"
