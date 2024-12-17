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
    assert sentiment_analyzer.finbert_model is not None
    assert sentiment_analyzer.finbert_tokenizer is not None

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
        ("Strong buy signal with multiple indicators confirming uptrend", "bullish"),
        ("Market showing weakness, likely to test lower support", "bearish"),
        ("Consolidation phase with balanced buying and selling", "neutral"),
        ("Massive volume spike with institutional buying pressure", "bullish"),
        ("Death cross forming with declining volume and weak bounces", "bearish"),
        ("Price action suggests continuation of bullish trend", "bullish"),
        ("Bears taking control as support levels break down", "bearish"),
        ("Multiple technical indicators showing strong buy signals", "bullish"),
        ("Market sentiment shifting bearish with increasing sell orders", "bearish"),
        ("Clear breakout above resistance with strong volume", "bullish")
    ]

    correct_predictions = 0
    for text, actual in test_cases:
        is_correct = await sentiment_analyzer.validate_accuracy(text, actual)
        if is_correct:
            correct_predictions += 1
        logger.info(f"Prediction accuracy: {correct_predictions}/{len(test_cases)}")

    accuracy = correct_predictions / len(test_cases)
    assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} is below required 85%"

@pytest.mark.asyncio
async def test_ensemble_agreement(sentiment_analyzer):
    text = "Bitcoin price surges 10% with massive buying volume and institutional adoption news"
    result = await sentiment_analyzer.analyze_sentiment(text)

    assert result['components']['finbert']['score'] > 0
    assert result['components']['pattern']['score'] > 0
    assert result['components']['spacy']['score'] > 0
    assert result['confidence'] > 0.9

@pytest.mark.asyncio
async def test_confidence_calculation(sentiment_analyzer):
    text = "Clear bearish signal with death cross formation and declining volume"
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
        assert len(sentiment_analyzer.accuracy_metrics['confidence_scores']) > 0

    current_accuracy = sentiment_analyzer.get_current_accuracy()
    assert current_accuracy > 0, "Real-time accuracy tracking should be functioning"
