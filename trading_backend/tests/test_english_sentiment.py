import pytest
import pytest_asyncio
from app.services.web_scraping.english_sentiment import EnglishSentimentAnalyzer
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def sentiment_analyzer():
    analyzer = EnglishSentimentAnalyzer()
    await analyzer.initialize()
    return analyzer

@pytest.mark.asyncio
async def test_initialization(sentiment_analyzer):
    assert sentiment_analyzer.initialized
    assert sentiment_analyzer.nlp is not None
    assert sentiment_analyzer.finbert is not None
    assert sentiment_analyzer.tokenizer is not None
    assert sentiment_analyzer.accuracy_metrics is not None

@pytest.mark.asyncio
async def test_sentiment_analysis_bullish(sentiment_analyzer):
    text = "Bitcoin showing strong bullish momentum with increasing volume and breaking key resistance levels. Technical indicators suggest further upside potential."
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['sentiment'] == 'bullish'
    assert result['confidence'] > 0.8
    assert all(comp['confidence'] > 0 for comp in result['components'].values())

@pytest.mark.asyncio
async def test_sentiment_analysis_bearish(sentiment_analyzer):
    text = "Market showing weakness with declining volume and breaking below support. Bears taking control with increasing selling pressure."
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['sentiment'] == 'bearish'
    assert result['confidence'] > 0.8
    assert all(comp['confidence'] > 0 for comp in result['components'].values())

@pytest.mark.asyncio
async def test_sentiment_analysis_neutral(sentiment_analyzer):
    text = "Bitcoin trading sideways with mixed signals. Volume remains average with no clear direction."
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['sentiment'] == 'neutral'
    assert 'confidence' in result
    assert all(comp['confidence'] > 0 for comp in result['components'].values())

@pytest.mark.asyncio
async def test_accuracy_validation(sentiment_analyzer):
    test_cases = [
        ("Bitcoin surges 15% with massive institutional buying and golden cross formation on daily chart", "bullish"),
        ("Major support level breached with death cross pattern and heavy selling volume from whales", "bearish"),
        ("Price consolidating within range as volume decreases, awaiting next move", "neutral"),
        ("Multiple bullish divergences on RSI with strong accumulation from institutional wallets", "bullish"),
        ("Sharp bearish reversal with multiple technical supports broken and increasing sell orders", "bearish"),
        ("Strong breakout confirmed with 3x average volume and institutional accumulation signals", "bullish"),
        ("Critical support zones failing with bearish divergence and increasing short positions", "bearish"),
        ("Clear bull flag pattern with rising buy pressure and decreasing sell orders", "bullish"),
        ("Bearish engulfing pattern with major resistance rejection and volume spike", "bearish"),
        ("Decisive breakout above key resistance with institutional buying confirmed", "bullish")
    ]

    correct_predictions = 0
    total_cases = len(test_cases)

    for text, expected in test_cases:
        try:
            result = await sentiment_analyzer.analyze_sentiment(text)
            is_correct = result['sentiment'] == expected
            if is_correct:
                correct_predictions += 1
            logger.info(f"Text: {text}")
            logger.info(f"Expected: {expected}, Got: {result['sentiment']}")
            logger.info(f"Confidence: {result['confidence']:.2f}")
            logger.info(f"Current accuracy: {correct_predictions}/{total_cases}")
        except Exception as e:
            logger.error(f"Error processing case: {str(e)}")

    accuracy = correct_predictions / total_cases
    assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} is below required 85%"

@pytest.mark.asyncio
async def test_ensemble_agreement(sentiment_analyzer):
    text = "Massive bullish breakout with 5x volume spike and strong institutional buying confirmed by on-chain data"
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['components']['finbert']['score'] > 0
    assert result['components']['pattern']['score'] > 0
    assert result['components']['spacy']['score'] > 0
    assert result['confidence'] > 0.9

@pytest.mark.asyncio
async def test_confidence_calculation(sentiment_analyzer):
    text = "Strong bearish reversal confirmed with death cross pattern and massive whale selling pressure"
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['confidence'] > 0.8
    assert all(0 <= comp['confidence'] <= 1.0 for comp in result['components'].values())

@pytest.mark.asyncio
async def test_real_time_accuracy_tracking(sentiment_analyzer):
    test_texts = [
        "Bitcoin breaks above key resistance with strong volume",
        "Market showing signs of weakness and potential reversal",
        "Trading volume remains steady with price consolidation"
    ]

    for text in test_texts:
        result = await sentiment_analyzer.analyze_sentiment(text)
        assert 'confidence' in result
        assert 'last_100_predictions' in sentiment_analyzer.accuracy_metrics

    current_accuracy = sentiment_analyzer.get_current_accuracy()
    assert current_accuracy >= 0, "Real-time accuracy tracking should be functioning"
