"""Tests for the signal extractor service."""
import pytest
from datetime import datetime
from app.services.ocr.signal_extractor import SignalExtractor

@pytest.fixture
def signal_extractor():
    return SignalExtractor()

def test_english_bullish_patterns(signal_extractor):
    """Test detection of English bullish patterns."""
    text = """
    BTC showing a clear bullish flag pattern with support at $45,000.
    Looking for a golden cross on the 4h chart.
    Higher triangle forming with accumulation zone.
    """
    result = signal_extractor.extract_signals(text, 'en')

    assert result['sentiment'] == 'bullish'
    assert result['confidence'] > 0.7
    assert len(result['patterns']['bullish']) >= 3
    assert 'bull.*flag' in result['patterns']['bullish']
    assert 'golden.*cross' in result['patterns']['bullish']
    assert '.*triangle' in result['patterns']['bullish']
    assert 45000 in result['prices']

def test_english_bearish_patterns(signal_extractor):
    """Test detection of English bearish patterns."""
    text = """
    Clear head and shoulders pattern forming.
    Death cross imminent with resistance at $42,500.
    Seeing a descending triangle with distribution phase.
    """
    result = signal_extractor.extract_signals(text, 'en')

    assert result['sentiment'] == 'bearish'
    assert result['confidence'] > 0.7
    assert len(result['patterns']['bearish']) >= 3
    assert 'head.*shoulders' in result['patterns']['bearish']
    assert 'death.*cross' in result['patterns']['bearish']
    assert '.*triangle' in result['patterns']['bearish']
    assert 42500 in result['prices']

def test_chinese_bullish_patterns(signal_extractor):
    """Test detection of Chinese bullish patterns."""
    text = """
    比特币形成多头旗形，支撑位在45000。
    4小时图即将出现黄金交叉。
    上升三角形伴随筑底形态。
    """
    result = signal_extractor.extract_signals(text, 'cn')

    assert result['sentiment'] == 'bullish'
    assert result['confidence'] > 0.7
    assert len(result['patterns']['bullish']) >= 3
    assert '多头旗形' in result['patterns']['bullish']
    assert '黄金交叉' in result['patterns']['bullish']
    assert '上升三角形' in result['patterns']['bullish']
    assert 45000 in result['prices']

def test_chinese_bearish_patterns(signal_extractor):
    """Test detection of Chinese bearish patterns."""
    text = """
    形成头肩形态，
    即将出现死亡交叉，阻力位在42500。
    下降三角形伴随出货阶段。
    """
    result = signal_extractor.extract_signals(text, 'cn')

    assert result['sentiment'] == 'bearish'
    assert result['confidence'] > 0.7
    assert len(result['patterns']['bearish']) >= 3
    assert '头肩形态' in result['patterns']['bearish']
    assert '死亡交叉' in result['patterns']['bearish']
    assert '下降三角形' in result['patterns']['bearish']
    assert 42500 in result['prices']

def test_cryptocurrency_detection(signal_extractor):
    """Test cryptocurrency mention detection."""
    text = """
    BTC and ETH showing strength.
    Looking at SOL, DOGE, and ADA.
    比特币和以太坊走势良好。
    关注波卡币和狗狗币。
    """
    en_result = signal_extractor.extract_signals(text, 'en')
    cn_result = signal_extractor.extract_signals(text, 'cn')

    # Check English detection
    assert 'btc' in en_result['cryptocurrencies']
    assert 'eth' in en_result['cryptocurrencies']
    assert 'sol' in en_result['cryptocurrencies']
    assert 'doge' in en_result['cryptocurrencies']
    assert 'ada' in en_result['cryptocurrencies']

    # Check Chinese detection
    assert '比特币' in cn_result['cryptocurrencies']
    assert '以太坊' in cn_result['cryptocurrencies']
    assert '波卡币' in cn_result['cryptocurrencies']
    assert '狗狗币' in cn_result['cryptocurrencies']

def test_neutral_sentiment(signal_extractor):
    """Test neutral sentiment detection."""
    text = """
    Market showing mixed signals.
    Some support at $40,000 but also resistance at $41,000.
    Waiting for clearer direction.
    """
    result = signal_extractor.extract_signals(text, 'en')

    assert result['sentiment'] == 'neutral'
    assert result['confidence'] < 0.8  # Lower confidence for neutral signals
    assert 40000 in result['prices']
    assert 41000 in result['prices']

def test_confidence_explanation(signal_extractor):
    """Test confidence score explanation generation."""
    # Test high confidence case
    high_conf_text = """
    Clear bullish flag pattern.
    Golden cross forming.
    Strong support at $45,000.
    Higher triangle with accumulation.
    """
    high_conf_result = signal_extractor.extract_signals(high_conf_text, 'en')
    explanation = signal_extractor.get_confidence_explanation(high_conf_result)
    assert 'High confidence' in explanation
    assert str(round(high_conf_result['confidence'], 2)) in explanation

    # Test low confidence case
    low_conf_text = "Market is moving sideways."
    low_conf_result = signal_extractor.extract_signals(low_conf_text, 'en')
    explanation = signal_extractor.get_confidence_explanation(low_conf_result)
    assert 'Low confidence' in explanation
    assert str(round(low_conf_result['confidence'], 2)) in explanation

def test_price_extraction(signal_extractor):
    """Test price level extraction."""
    text = """
    Support at $45,000 and $46,500
    阻力位在48000和49500
    Target: 50,000 USD
    止损位：42500
    """
    en_result = signal_extractor.extract_signals(text, 'en')
    cn_result = signal_extractor.extract_signals(text, 'cn')

    # Check English price extraction
    assert 45000 in en_result['prices']
    assert 46500 in en_result['prices']
    assert 50000 in en_result['prices']

    # Check Chinese price extraction
    assert 48000 in cn_result['prices']
    assert 49500 in cn_result['prices']
    assert 42500 in cn_result['prices']
