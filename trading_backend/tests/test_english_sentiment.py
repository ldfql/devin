import pytest
import pytest_asyncio
from app.services.web_scraping.english_sentiment import EnglishSentimentAnalyzer
import asyncio

@pytest_asyncio.fixture
async def sentiment_analyzer():
    analyzer = EnglishSentimentAnalyzer()
    return analyzer

@pytest.mark.asyncio
async def test_sentiment_analysis_bullish(sentiment_analyzer):
    text = "Bitcoin showing strong bullish momentum with increasing volume and breaking key resistance levels. Technical indicators suggest further upside potential."
    result = await sentiment_analyzer.analyze_sentiment(text)
    assert result['sentiment'] == 'bullish'
    assert result['confidence'] > 0.8

@pytest.mark.asyncio
async def test_sentiment_analysis_bearish(sentiment_analyzer):
    text = "Market showing weakness with declining volume and breaking below support. Bears taking control with increasing selling pressure."
    result = await sentiment_analyzer.analyze_sentiment(text)
    assert result['sentiment'] == 'bearish'
    assert result['confidence'] > 0.8

@pytest.mark.asyncio
async def test_sentiment_analysis_neutral(sentiment_analyzer):
    text = "Bitcoin trading sideways with mixed signals. Volume remains average with no clear direction."
    result = await sentiment_analyzer.analyze_sentiment(text)
    assert result['sentiment'] == 'neutral'

@pytest.mark.asyncio
async def test_accuracy_validation(sentiment_analyzer):
    test_cases = [
        ("Strong buy signal with multiple indicators confirming uptrend", "bullish"),
        ("Market showing weakness, likely to test lower support", "bearish"),
        ("Consolidation phase with balanced buying and selling", "neutral")
    ]

    correct_predictions = 0
    for text, actual in test_cases:
        is_correct = await sentiment_analyzer.validate_accuracy(text, actual)
        if is_correct:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_cases)
    assert accuracy >= 0.85, f"Accuracy {accuracy:.2%} is below required 85%"

@pytest.mark.asyncio
async def test_ensemble_agreement(sentiment_analyzer):
    text = "Bitcoin price surges 10% with massive buying volume and institutional adoption news"
    result = await sentiment_analyzer.analyze_sentiment(text)

    # Check if all components agree on bullish sentiment
    assert result['components']['finbert']['score'] > 0
    assert result['components']['pattern']['score'] > 0
    assert result['components']['spacy']['score'] > 0
    assert result['confidence'] > 0.9

@pytest.mark.asyncio
async def test_confidence_calculation(sentiment_analyzer):
    text = "Clear bearish signal with death cross formation and declining volume"
    result = await sentiment_analyzer.analyze_sentiment(text)

    # High confidence for clear signals
    assert result['confidence'] > 0.8

    # Test confidence components
    assert all(0 <= comp['confidence'] <= 1.0 for comp in result['components'].values())
