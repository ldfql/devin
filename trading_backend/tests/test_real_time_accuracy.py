import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer
from app.services.web_scraping.english_sentiment import EnglishSentimentAnalyzer
from app.services.web_scraping.chinese_scraper import ChineseSentimentAnalyzer

@pytest.fixture
async def market_analyzer():
    return MarketCycleAnalyzer()

@pytest.fixture
async def english_sentiment():
    return EnglishSentimentAnalyzer()

@pytest.fixture
async def chinese_sentiment():
    return ChineseSentimentAnalyzer()

@pytest.mark.asyncio
async def test_market_prediction_accuracy(market_analyzer):
    """Test market prediction accuracy with real-time data."""
    # Get historical data for the last 30 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    predictions = []
    actuals = []

    # Analyze each day's data
    current_date = start_date
    while current_date < end_date:
        # Get market data for the period
        market_data = await market_analyzer.get_market_data(
            symbol="BTCUSDT",
            start_time=current_date,
            end_time=current_date + timedelta(days=1)
        )

        # Make prediction
        prediction = await market_analyzer.predict_market_direction(market_data)
        predictions.append(prediction["is_bullish"])

        # Get actual result (next day's close > current close = bullish)
        next_day_data = await market_analyzer.get_market_data(
            symbol="BTCUSDT",
            start_time=current_date + timedelta(days=1),
            end_time=current_date + timedelta(days=2)
        )

        is_actually_bullish = next_day_data["close"][-1] > market_data["close"][-1]
        actuals.append(is_actually_bullish)

        current_date += timedelta(days=1)

    # Calculate accuracy
    accuracy = np.mean([p == a for p, a in zip(predictions, actuals)])
    assert accuracy >= 0.85, f"Market prediction accuracy {accuracy:.2%} below required 85%"

@pytest.mark.asyncio
async def test_sentiment_analysis_accuracy(english_sentiment, chinese_sentiment):
    """Test sentiment analysis accuracy across languages."""
    # Test data with known sentiment
    english_test_data = [
        ("Bitcoin price surges to new highs as institutional adoption grows", "positive"),
        ("Market crash wipes out billions in crypto value", "negative"),
        ("Bitcoin trading sideways as market participants await Fed decision", "neutral")
    ]

    chinese_test_data = [
        ("比特币价格创新高，机构采用率持续增长", "positive"),
        ("市场崩盘导致加密货币市值蒸发数十亿", "negative"),
        ("比特币横盘整理，市场参与者等待美联储决议", "neutral")
    ]

    # Test English sentiment accuracy
    english_correct = 0
    for text, expected in english_test_data:
        sentiment = await english_sentiment.analyze_sentiment(text)
        if sentiment["label"] == expected:
            english_correct += 1

    english_accuracy = english_correct / len(english_test_data)
    assert english_accuracy >= 0.85, f"English sentiment accuracy {english_accuracy:.2%} below required 85%"

    # Test Chinese sentiment accuracy
    chinese_correct = 0
    for text, expected in chinese_test_data:
        sentiment = await chinese_sentiment.analyze_sentiment(text)
        if sentiment["label"] == expected:
            chinese_correct += 1

    chinese_accuracy = chinese_correct / len(chinese_test_data)
    assert chinese_accuracy >= 0.85, f"Chinese sentiment accuracy {chinese_accuracy:.2%} below required 85%"

@pytest.mark.asyncio
async def test_confidence_scores(english_sentiment, chinese_sentiment):
    """Test confidence scores for sentiment predictions."""
    # Test high confidence predictions
    high_confidence_text = "Bitcoin price skyrockets 20% in massive rally"
    result = await english_sentiment.analyze_sentiment(high_confidence_text)
    assert result["confidence"] >= 0.85, "Confidence score below threshold for clear sentiment"

    # Test low confidence predictions
    ambiguous_text = "Bitcoin price movement catches attention"
    result = await english_sentiment.analyze_sentiment(ambiguous_text)
    assert result["confidence"] < 0.85, "Confidence score too high for ambiguous text"

@pytest.mark.asyncio
async def test_real_time_sentiment_tracking():
    """Test real-time sentiment tracking accuracy."""
    analyzer = MarketCycleAnalyzer()

    # Track sentiment changes over time
    sentiment_changes = []
    start_time = datetime.utcnow() - timedelta(hours=24)
    end_time = datetime.utcnow()

    # Get sentiment data in 1-hour intervals
    current_time = start_time
    while current_time < end_time:
        sentiment = await analyzer.get_aggregated_sentiment(
            start_time=current_time,
            end_time=current_time + timedelta(hours=1)
        )
        sentiment_changes.append(sentiment)
        current_time += timedelta(hours=1)

    # Calculate sentiment change accuracy
    correct_predictions = 0
    total_predictions = len(sentiment_changes) - 1

    for i in range(total_predictions):
        predicted_direction = sentiment_changes[i]["predicted_direction"]
        actual_direction = sentiment_changes[i + 1]["actual_direction"]
        if predicted_direction == actual_direction:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    assert accuracy >= 0.85, f"Real-time sentiment tracking accuracy {accuracy:.2%} below required 85%"

@pytest.mark.asyncio
async def test_combined_analysis_accuracy(market_analyzer):
    """Test accuracy of combined market and sentiment analysis."""
    # Get real-time market data and sentiment
    market_data = await market_analyzer.get_market_data("BTCUSDT")
    sentiment_data = await market_analyzer.get_aggregated_sentiment()

    # Make prediction using combined analysis
    prediction = await market_analyzer.predict_with_sentiment(
        market_data=market_data,
        sentiment_data=sentiment_data
    )

    # Verify prediction confidence
    assert prediction["confidence"] >= 0.85, "Combined analysis confidence below threshold"

    # Track prediction accuracy
    accurate_predictions = []
    for _ in range(10):  # Test 10 consecutive predictions
        prediction = await market_analyzer.predict_with_sentiment(
            market_data=market_data,
            sentiment_data=sentiment_data
        )
        accurate_predictions.append(prediction["is_accurate"])

        # Wait for next market update
        await asyncio.sleep(60)

        # Update data
        market_data = await market_analyzer.get_market_data("BTCUSDT")
        sentiment_data = await market_analyzer.get_aggregated_sentiment()

    accuracy = np.mean(accurate_predictions)
    assert accuracy >= 0.85, f"Combined analysis accuracy {accuracy:.2%} below required 85%"
